#include <compressors/compactionv5t/compaction-compressor.cuh>
#include <compressors/compactionv5t/compaction-defines.cuh>
#include <compressors/compactionv5t/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fstream>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gtsst::compressors {
    CompressionConfiguration CompactionV5TCompressor::configure_compression(const size_t buf_size) {
        return CompressionConfiguration{.input_buffer_size = buf_size,
                                        .compression_buffer_size = buf_size,
                                        .temp_buffer_size = buf_size,
                                        .min_alignment_input = compactionv5t::WORD_ALIGNMENT,
                                        .min_alignment_output = compactionv5t::WORD_ALIGNMENT,
                                        .min_alignment_temp = compactionv5t::TMP_WORD_ALIGNMENT,
                                        .must_pad_alignment = true,
                                        .block_size = compactionv5t::BLOCK_SIZE,
                                        .table_range = compactionv5t::BLOCK_SIZE * compactionv5t::SUPER_BLOCK_SIZE,
                                        .must_pad_block = true,

                                        .escape_symbol = fsst::Symbol::escape,
                                        .padding_symbol = fsst::Symbol::ignore,
                                        .padding_enabled = true,

                                        .device_buffers = true};
    }

    GTSSTStatus CompactionV5TCompressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                                      CompressionConfiguration& config) {
        if (config.input_buffer_size > compactionv5t::BLOCK_SIZE * (size_t)0xFFFFFFFF) {
            return gtsstErrorTooBig;
        }

        if (config.block_size != compactionv5t::BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        if (config.min_alignment_input != compactionv5t::WORD_ALIGNMENT ||
            config.min_alignment_output != compactionv5t::WORD_ALIGNMENT ||
            config.min_alignment_temp != compactionv5t::TMP_WORD_ALIGNMENT) {
            return gtsstErrorBadBlockSize;
        }

        if ((uintptr_t)src % compactionv5t::WORD_ALIGNMENT != 0 || (uintptr_t)dst % compactionv5t::WORD_ALIGNMENT != 0
            ||
            (uintptr_t)tmp % compactionv5t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorBadAlignment;
        }

        if (config.input_buffer_size % compactionv5t::BLOCK_SIZE != 0 ||
            config.temp_buffer_size % compactionv5t::TMP_OUT_BLOCK_SIZE != 0) {
            return gtsstErrorBlockAlignment;
        }

        if (config.input_buffer_size % compactionv5t::WORD_ALIGNMENT != 0 ||
            config.temp_buffer_size % compactionv5t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorWordAlignment;
        }

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV5TCompressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src,
                                                  uint8_t* tmp, CompressionConfiguration& config, size_t* out_size,
                                                  CompressionStatistics& stats) {
        if (const GTSSTStatus buffer_validation = validate_compression_buffers(src, dst, tmp, config);
            buffer_validation != gtsstSuccess) {
            return buffer_validation;
        }

        // TODO: remove this assertion
        assert(!data_contains(sample_src, 254, config.input_buffer_size));

        // Some bookkeeping
        const uint64_t number_of_blocks = config.input_buffer_size / compactionv5t::BLOCK_SIZE;
        const uint64_t number_of_tables = (number_of_blocks - 1) / compactionv5t::SUPER_BLOCK_SIZE + 1;
        const uint64_t metadata_mem_size = sizeof(compactionv5t::GCompactionMetadata) * number_of_tables;
        const uint64_t block_headers_mem_size = sizeof(CompactionV5TBlockHeader) *
            number_of_blocks;
        const uint64_t approx_header_mem_size = sizeof(CompactionV5TFileHeader) + number_of_tables * sizeof(GBaseHeader)
            + block_headers_mem_size;

        // Update the device queue for internal transpose launches
        cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, compactionv5t::CUDA_QUEUE_LEN);
        assert(number_of_blocks < compactionv5t::CUDA_QUEUE_LEN);
        // If there are too many blocks, we cannot compress this file in one go (and some margin)
        if (number_of_blocks > compactionv5t::CUDA_QUEUE_LEN - 10) {
            return gtsstErrorTooBig;
        }

        compactionv5t::GCompactionMetadata* metadata_host;
        GBaseHeader* table_headers_host;
        CompactionV5TBlockHeader* block_headers_host;
        safeCUDACall(cudaMallocHost(&metadata_host, metadata_mem_size));
        safeCUDACall(cudaMallocHost(&table_headers_host, sizeof(GBaseHeader) * number_of_tables));
        safeCUDACall(cudaMallocHost(&block_headers_host, block_headers_mem_size));

        // Some CUDA bookkeeping
        compactionv5t::GCompactionMetadata* metadata_gpu;
        CompactionV5TBlockHeader* block_headers_gpu;
        uint8_t* header_gpu;

        // Allocate some CUDA buffers
        safeCUDACall(cudaMalloc(&metadata_gpu, metadata_mem_size));
        safeCUDACall(cudaMalloc(&block_headers_gpu, block_headers_mem_size));
        safeCUDACall(cudaMalloc(&header_gpu, approx_header_mem_size));

        // Phase 1: Symbol generation (CPU for now)
        const auto symbol_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(number_of_tables);
        for (uint32_t i = 0; i < number_of_tables; i++) {
            threads.emplace_back(gpu_create_metadata<symbols::SmallSymbolMatchTableData>, i,
                                 compactionv5t::BLOCK_SIZE * compactionv5t::SUPER_BLOCK_SIZE, metadata_host,
                                 table_headers_host, sample_src, config.input_buffer_size);
        }
        for (std::thread& t : threads) {
            t.join();
        }

        // Phase 2: Precomputation
        const auto precomputation_start = std::chrono::high_resolution_clock::now();
        // Copy metadata to GPU memory
        safeCUDACall(cudaMemcpyAsync(metadata_gpu, metadata_host, metadata_mem_size, cudaMemcpyHostToDevice));

        safeCUDACall(cudaPeekAtLastError());
        safeCUDACall(cudaDeviceSynchronize());

        // Phase 3: Encoding (GPU)
        const auto encoding_start = std::chrono::high_resolution_clock::now();

        // Run all blocks
        compactionv5t::gpu_compaction<<<number_of_blocks, compactionv5t::THREAD_COUNT>>>(
            metadata_gpu, block_headers_gpu, src, tmp, dst);
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
        CompactionV5TFileHeader file_header{
            {
                total_data_size + block_headers_mem_size + sizeof(CompactionV5TFileHeader),
                config.input_buffer_size, // TODO: would have to change this to support input padding
                (uint32_t)number_of_tables,
                0,
                (uint32_t)number_of_blocks,
            },
            compactionv5t::SUB_BLOCKS,
        };
        size_t header_size = sizeof(CompactionV5TFileHeader);

        // Copy tables
        for (int table_id = 0; table_id < number_of_tables; table_id++) {
            safeCUDACall(cudaMemcpyAsync(header_gpu + header_size, &table_headers_host[table_id],
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
            cudaMemcpyAsync(header_gpu + header_size, block_headers_host, block_headers_mem_size, cudaMemcpyHostToDevice
            ));
        header_size += block_headers_mem_size;

        // Copy file header
        file_header.compressed_size += file_header.table_size;
        safeCUDACall(
            cudaMemcpyAsync(header_gpu, &file_header, sizeof(CompactionV5TFileHeader), cudaMemcpyHostToDevice));

        // Then gather data
        uint64_t running_length = 0;
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            uint8_t* running_dst = tmp + running_length;
            size_t block_size = block_headers_host[block_id].flushes * compactionv5t::THREAD_COUNT * sizeof(uint32_t);
            safeCUDACall(
                cudaMemcpyAsync(running_dst, dst + compactionv5t::TMP_OUT_BLOCK_SIZE * block_id, block_size,
                    cudaMemcpyDeviceToDevice));
            running_length += block_size;
        }

        // Then do stream compaction on the actual data
        const thrust::device_ptr<uint8_t> thrust_gpu_in = thrust::device_pointer_cast(tmp);
        const thrust::device_ptr<uint8_t> thrust_gpu_out = thrust::device_pointer_cast(dst + header_size);
        const thrust::device_ptr<uint8_t> thrust_new_end = copy_if(thrust::device, thrust_gpu_in,
                                                                   thrust_gpu_in + running_length, thrust_gpu_out,
                                                                   is_not_ignore());
        const size_t thrust_out_size = thrust_new_end - thrust_gpu_out;
        const size_t out = thrust_out_size + header_size;

        // Copy header to dst
        safeCUDACall(cudaMemcpy(dst, header_gpu, header_size, cudaMemcpyDeviceToDevice));

        // Finally, free buffers
        safeCUDACall(cudaFreeHost(metadata_host));
        safeCUDACall(cudaFreeHost(table_headers_host));
        safeCUDACall(cudaFreeHost(block_headers_host));

        // And free cuda buffers
        safeCUDACall(cudaFree(metadata_gpu));
        safeCUDACall(cudaFree(block_headers_gpu));
        safeCUDACall(cudaFree(header_gpu));

        // Check and update output size
        assert(
            file_header.compressed_size - sizeof(CompactionV5TFileHeader) - file_header.table_size -
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
