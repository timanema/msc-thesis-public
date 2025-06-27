#include <compressors/compactionv2/compaction-compressor.cuh>
#include <compressors/compactionv2/compaction-defines.cuh>
#include <compressors/compactionv2/compaction-encode.cuh>
#include <compressors/compactionv2/compaction-transpose.cuh>
#include <compressors/shared.cuh>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gtsst::compressors {
    CompressionConfiguration CompactionV2Compressor::configure_compression(const size_t buf_size) {
        return CompressionConfiguration{.input_buffer_size = buf_size,
                                        .compression_buffer_size = buf_size,
                                        .temp_buffer_size = buf_size * 3,
                                        .min_alignment_input = compactionv2::WORD_ALIGNMENT,
                                        .min_alignment_output = compactionv2::WORD_ALIGNMENT,
                                        .min_alignment_temp = compactionv2::TMP_WORD_ALIGNMENT,
                                        .must_pad_alignment = true,
                                        .block_size = compactionv2::BLOCK_SIZE,
                                        .table_range = compactionv2::BLOCK_SIZE * compactionv2::SUPER_BLOCK_SIZE,
                                        .must_pad_block = true,

                                        .escape_symbol = fsst::Symbol::escape,
                                        .padding_symbol = fsst::Symbol::ignore,
                                        .padding_enabled = true,

                                        .device_buffers = true};
    }

    GTSSTStatus CompactionV2Compressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                                     CompressionConfiguration& config) {
        if (config.input_buffer_size > compactionv2::BLOCK_SIZE * (size_t)0xFFFFFFFF) {
            return gtsstErrorTooBig;
        }

        if (config.block_size != compactionv2::BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        if (config.min_alignment_input != compactionv2::WORD_ALIGNMENT ||
            config.min_alignment_output != compactionv2::WORD_ALIGNMENT ||
            config.min_alignment_temp != compactionv2::TMP_WORD_ALIGNMENT) {
            return gtsstErrorBadBlockSize;
        }

        if ((uintptr_t)src % compactionv2::WORD_ALIGNMENT != 0 || (uintptr_t)dst % compactionv2::WORD_ALIGNMENT != 0 ||
            (uintptr_t)tmp % compactionv2::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorAlignment;
        }

        if (config.input_buffer_size % compactionv2::BLOCK_SIZE != 0 ||
            config.temp_buffer_size % compactionv2::TMP_OUT_BLOCK_SIZE != 0) {
            return gtsstErrorBlockAlignment;
        }

        if (config.input_buffer_size % compactionv2::WORD_ALIGNMENT != 0 ||
            config.temp_buffer_size % compactionv2::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorWordAlignment;
        }

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV2Compressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src,
                                                 uint8_t* tmp, CompressionConfiguration& config, size_t* out_size,
                                                 CompressionStatistics& stats) {
        if (const GTSSTStatus buffer_validation = validate_compression_buffers(src, dst, tmp, config);
            buffer_validation != gtsstSuccess) {
            return buffer_validation;
        }

        if (config.input_buffer_size == 0) {
            return gtsstSuccess;
        }

        // TODO: remove this assertion
        assert(!data_contains(sample_src, 254, config.input_buffer_size));
        assert(!data_contains(sample_src, 255, config.input_buffer_size));

        // Some bookkeeping
        const uint64_t number_of_blocks = config.input_buffer_size / compactionv2::BLOCK_SIZE;
        const uint64_t working_block_size = number_of_blocks * compactionv2::TMP_OUT_BLOCK_SIZE;
        const uint64_t number_of_tables = (number_of_blocks - 1) / compactionv2::SUPER_BLOCK_SIZE + 1;
        const uint64_t metadata_mem_size = sizeof(compactionv2::GCompactionMetadata) * number_of_tables;
        const uint64_t block_headers_mem_size = sizeof(BlockHeader) * number_of_blocks;

        compactionv2::GCompactionMetadata* metadata_host;
        GBaseHeader* table_headers_host;
        BlockHeader* block_headers_host;
        safeCUDACall(cudaMallocHost(&metadata_host, metadata_mem_size));
        safeCUDACall(cudaMallocHost(&table_headers_host, sizeof(GBaseHeader) * number_of_tables));
        safeCUDACall(cudaMallocHost(&block_headers_host, block_headers_mem_size));

        // Some CUDA bookkeeping
        compactionv2::GCompactionMetadata* metadata_gpu;
        BlockHeader* block_headers_gpu;

        // Allocate some CUDA buffers
        safeCUDACall(cudaMalloc(&metadata_gpu, metadata_mem_size));
        safeCUDACall(cudaMalloc(&block_headers_gpu, block_headers_mem_size));

        // Set temp_dst to all ignores, so all unused data is filtered in a later stage
        safeCUDACall(cudaMemsetAsync(tmp, 254, working_block_size));

        // Start transpose while table is being generated
        compactionv2::shared_transpose<uint64_t, compactionv2::BLOCK_SIZE, compactionv2::n_words_per_tile,
                                       compactionv2::THREAD_COUNT><<<number_of_blocks, 32>>>(src, dst);

        // Phase 1: Symbol generation (CPU for now)
        const auto symbol_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(number_of_tables);
        for (uint32_t i = 0; i < number_of_tables; i++) {
            threads.emplace_back(gpu_create_metadata<symbols::SmallSymbolMatchTableData>, i,
                                 compactionv2::BLOCK_SIZE * compactionv2::SUPER_BLOCK_SIZE, metadata_host,
                                 table_headers_host, sample_src, config.input_buffer_size);
        }
        for (std::thread& t : threads) {
            t.join();
        }

        // Phase 2: Precomputation
        const auto precomputation_start = std::chrono::high_resolution_clock::now();
        // Copy metadata to GPU memory
        safeCUDACall(cudaMemcpyAsync(metadata_gpu, metadata_host, metadata_mem_size, cudaMemcpyHostToDevice));

        // compactionv2::shared_transpose<<<number_of_blocks, 32>>>(src, dst, compactionv2::BLOCK_SIZE);
        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Phase 3: Encoding (GPU)
        const auto encoding_start = std::chrono::high_resolution_clock::now();

        // Run all blocks
        compactionv2::gpu_compaction<<<number_of_blocks, compactionv2::THREAD_COUNT>>>(metadata_gpu, block_headers_gpu,
            dst, tmp);
        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Phase 4: Postprocessing (Partial CPU for now)
        const auto post_start = std::chrono::high_resolution_clock::now();

        // Copy comp headers & temp_dst to CPU
        safeCUDACall(cudaMemcpy(block_headers_host, block_headers_gpu, block_headers_mem_size, cudaMemcpyDeviceToHost));

        // Gather total output size
        uint64_t total_data_size = 0;
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            total_data_size += block_headers_host[block_id].compressed_size;
        }

        // Calculate header positions
        FileHeader file_header{
            .compressed_size = total_data_size + block_headers_mem_size + sizeof(FileHeader),
            .uncompressed_size = config.input_buffer_size, // TODO: would have to change this to support input padding
            .num_tables = (uint32_t)number_of_tables,
            .table_size = 0,
            .num_blocks = (uint32_t)number_of_blocks,
        };
        size_t header_size = sizeof(FileHeader);

        // Copy tables
        for (int table_id = 0; table_id < number_of_tables; table_id++) {
            safeCUDACall(cudaMemcpyAsync(dst + header_size, &table_headers_host[table_id],
                metadata_host[table_id].header_offset, cudaMemcpyHostToDevice));

            header_size += metadata_host[table_id].header_offset;
            file_header.table_size += metadata_host[table_id].header_offset;
        }

        // Copy block headers
        safeCUDACall(
            cudaMemcpyAsync(dst + header_size, block_headers_host, block_headers_mem_size, cudaMemcpyHostToDevice));
        header_size += block_headers_mem_size;

        // Copy file header
        file_header.compressed_size += file_header.table_size;
        safeCUDACall(cudaMemcpyAsync(dst, &file_header, sizeof(FileHeader), cudaMemcpyHostToDevice));

        // Then do stream compaction on the actual data
        const thrust::device_ptr<uint8_t> thrust_gpu_in = thrust::device_pointer_cast(tmp);
        const thrust::device_ptr<uint8_t> thrust_gpu_out = thrust::device_pointer_cast(dst + header_size);
        const thrust::device_ptr<uint8_t> thrust_new_end =
            copy_if(thrust::device, thrust_gpu_in, thrust_gpu_in + working_block_size, thrust_gpu_out, is_not_ignore());
        const size_t thrust_out_size = thrust_new_end - thrust_gpu_out;
        const size_t out = thrust_out_size + header_size;

        // Finally, free buffers
        safeCUDACall(cudaFreeHost(metadata_host));
        safeCUDACall(cudaFreeHost(table_headers_host));
        safeCUDACall(cudaFreeHost(block_headers_host));

        // And free cuda buffers
        safeCUDACall(cudaFree(metadata_gpu));
        safeCUDACall(cudaFree(block_headers_gpu));

        // Check and update output size
        assert(file_header.compressed_size - sizeof(FileHeader) - file_header.table_size -
            block_headers_mem_size ==
            total_data_size);
        assert(thrust_out_size == total_data_size);
        *out_size = out;

        // Update statistics
        stats.table_generation =
            std::chrono::duration_cast<std::chrono::microseconds>(precomputation_start - symbol_start);
        stats.precomputation =
            std::chrono::duration_cast<std::chrono::microseconds>(encoding_start - precomputation_start);
        stats.encoding = std::chrono::duration_cast<std::chrono::microseconds>(post_start - encoding_start);
        stats.postprocessing = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - post_start);

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
