#include <compressors/compactionv2/compaction-transpose.cuh>
#include <compressors/compactionv3t/compaction-compressor.cuh>
#include <compressors/compactionv3t/compaction-defines.cuh>
#include <compressors/compactionv3t/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fstream>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gtsst::compressors {
    CompressionConfiguration CompactionV3TCompressor::configure_compression(const size_t buf_size) {
        return CompressionConfiguration{.input_buffer_size = buf_size,
                                        .compression_buffer_size = buf_size,
                                        .temp_buffer_size = buf_size * 2,
                                        .min_alignment_input = compactionv3t::WORD_ALIGNMENT,
                                        .min_alignment_output = compactionv3t::WORD_ALIGNMENT,
                                        .min_alignment_temp = compactionv3t::TMP_WORD_ALIGNMENT,
                                        .must_pad_alignment = true,
                                        .block_size = compactionv3t::BLOCK_SIZE,
                                        .table_range = compactionv3t::BLOCK_SIZE * compactionv3t::SUPER_BLOCK_SIZE,
                                        .must_pad_block = true,

                                        .escape_symbol = fsst::Symbol::escape,
                                        .padding_symbol = fsst::Symbol::ignore,
                                        .padding_enabled = true,

                                        .device_buffers = true};
    }

    GTSSTStatus CompactionV3TCompressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                                      CompressionConfiguration& config) {
        if (config.input_buffer_size > compactionv3t::BLOCK_SIZE * (size_t)0xFFFFFFFF) {
            return gtsstErrorTooBig;
        }

        if (config.block_size != compactionv3t::BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        if (config.min_alignment_input != compactionv3t::WORD_ALIGNMENT ||
            config.min_alignment_output != compactionv3t::WORD_ALIGNMENT ||
            config.min_alignment_temp != compactionv3t::TMP_WORD_ALIGNMENT) {
            return gtsstErrorBadBlockSize;
        }

        if ((uintptr_t)src % compactionv3t::WORD_ALIGNMENT != 0 || (uintptr_t)dst % compactionv3t::WORD_ALIGNMENT != 0
            ||
            (uintptr_t)tmp % compactionv3t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorAlignment;
        }

        if (config.input_buffer_size % compactionv3t::BLOCK_SIZE != 0 ||
            config.temp_buffer_size % compactionv3t::TMP_OUT_BLOCK_SIZE != 0) {
            return gtsstErrorBlockAlignment;
        }

        if (config.input_buffer_size % compactionv3t::WORD_ALIGNMENT != 0 ||
            config.temp_buffer_size % compactionv3t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorWordAlignment;
        }

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV3TCompressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src,
                                                  uint8_t* tmp, CompressionConfiguration& config, size_t* out_size,
                                                  CompressionStatistics& stats) {
        if (const GTSSTStatus buffer_validation = validate_compression_buffers(src, dst, tmp, config);
            buffer_validation != gtsstSuccess) {
            return buffer_validation;
        }

        // TODO: remove this assertion
        assert(!data_contains(sample_src, 254, config.input_buffer_size));

        // Some bookkeeping
        const uint64_t number_of_blocks = config.input_buffer_size / compactionv3t::BLOCK_SIZE;
        const uint64_t working_block_size = number_of_blocks * compactionv3t::TMP_OUT_BLOCK_SIZE;
        const uint64_t number_of_tables = (number_of_blocks - 1) / compactionv3t::SUPER_BLOCK_SIZE + 1;
        const uint64_t metadata_mem_size = sizeof(compactionv3t::GCompactionMetadata) * number_of_tables;
        const uint64_t block_headers_mem_size = sizeof(CompactionV3TBlockHeader<compactionv3t::SUB_BLOCKS>) *
            number_of_blocks;

        compactionv3t::GCompactionMetadata* metadata_host;
        GBaseHeader* table_headers_host;
        CompactionV3TBlockHeader<compactionv3t::SUB_BLOCKS>* block_headers_host;
        safeCUDACall(cudaMallocHost(&metadata_host, metadata_mem_size));
        safeCUDACall(cudaMallocHost(&table_headers_host, sizeof(GBaseHeader) * number_of_tables));
        safeCUDACall(cudaMallocHost(&block_headers_host, block_headers_mem_size));

        // Some CUDA bookkeeping
        compactionv3t::GCompactionMetadata* metadata_gpu;
        CompactionV3TBlockHeader<compactionv3t::SUB_BLOCKS>* block_headers_gpu;

        // Allocate some CUDA buffers
        safeCUDACall(cudaMalloc(&metadata_gpu, metadata_mem_size));
        safeCUDACall(cudaMalloc(&block_headers_gpu, block_headers_mem_size));

        // Set temp_dst to all ignores, so all unused data is filtered in a later stage
        safeCUDACall(cudaMemsetAsync(tmp, 0xFE, working_block_size));

        // Start transpose while table is being generated
        compactionv2::shared_transpose<uint32_t, compactionv3t::BLOCK_SIZE, compactionv3t::n_words_per_tile,
                                       compactionv3t::THREAD_COUNT><<<number_of_blocks, 32>>>(src, dst);
        // compactionv1::basic_transpose<uint32_t, compactionv2::BLOCK_SIZE,
        // compactionv2::n_words_per_tile><<<number_of_blocks, compactionv2::THREAD_COUNT>>>(src, dst);

        // Phase 1: Symbol generation (CPU for now)
        const auto symbol_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(number_of_tables);
        for (uint32_t i = 0; i < number_of_tables; i++) {
            threads.emplace_back(gpu_create_metadata<symbols::SmallSymbolMatchTableData>, i,
                                 compactionv3t::BLOCK_SIZE * compactionv3t::SUPER_BLOCK_SIZE, metadata_host,
                                 table_headers_host, sample_src, config.input_buffer_size);
        }
        for (std::thread& t : threads) {
            t.join();
        }

        // Phase 2: Precomputation
        const auto precomputation_start = std::chrono::high_resolution_clock::now();
        // Copy metadata to GPU memory
        safeCUDACall(cudaMemcpyAsync(metadata_gpu, metadata_host, metadata_mem_size, cudaMemcpyHostToDevice));
        // compactionv1::shared_transpose<<<number_of_blocks, 32>>>(src, dst, compactionv1::BLOCK_SIZE);

        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Phase 3: Encoding (GPU)
        const auto encoding_start = std::chrono::high_resolution_clock::now();

        // Run all blocks
        compactionv3t::gpu_compaction<<<number_of_blocks, compactionv3t::THREAD_COUNT>>>(
            metadata_gpu, block_headers_gpu,
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
        CompactionV3TFileHeader file_header{
            {
                total_data_size + block_headers_mem_size + sizeof(CompactionV3TFileHeader),
                config.input_buffer_size, // TODO: would have to change this to support input padding
                (uint32_t)number_of_tables,
                0,
                (uint32_t)number_of_blocks,
            },
            compactionv3t::SUB_BLOCKS,
        };
        size_t header_size = sizeof(CompactionV3TFileHeader);

        // Copy tables
        for (int table_id = 0; table_id < number_of_tables; table_id++) {
            safeCUDACall(cudaMemcpyAsync(dst + header_size, &table_headers_host[table_id],
                metadata_host[table_id].header_offset, cudaMemcpyHostToDevice));

            header_size += metadata_host[table_id].header_offset;
            file_header.table_size += metadata_host[table_id].header_offset;
        }

        // Useful output debug
        // auto x = (char*) malloc(compactionv2::BLOCK_SIZE/2);
        // safeCUDACall(cudaMemcpy(x, tmp, compactionv2::BLOCK_SIZE/2, cudaMemcpyDeviceToHost));
        // std::ofstream out_file("/home/tim/test.out", std::ios::binary);
        // out_file.write(x, compactionv2::BLOCK_SIZE/2);
        // free(x);

        // Copy block headers
        safeCUDACall(
            cudaMemcpyAsync(dst + header_size, block_headers_host, block_headers_mem_size, cudaMemcpyHostToDevice));
        header_size += block_headers_mem_size;

        // Copy file header
        file_header.compressed_size += file_header.table_size;
        safeCUDACall(cudaMemcpyAsync(dst, &file_header, sizeof(CompactionV3TFileHeader), cudaMemcpyHostToDevice));

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
        assert(
            file_header.compressed_size - sizeof(CompactionV3TFileHeader) - file_header.table_size -
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
