#include <compressors/compactionv2/compaction-transpose.cuh>
#include <compressors/compactionv4t/compaction-compressor.cuh>
#include <compressors/compactionv4t/compaction-defines.cuh>
#include <compressors/compactionv4t/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fstream>
#include <thread>

namespace gtsst::compressors {
    CompressionConfiguration CompactionV4TCompressor::configure_compression(const size_t buf_size) {
        return CompressionConfiguration{.input_buffer_size = buf_size,
                                        .compression_buffer_size = buf_size,
                                        .temp_buffer_size = buf_size * 2,
                                        // TODO: if I'm creative with how I do final writes, temp buffer is not needed for this one. Make dst *2 and directly use that. No perf/comp, only mem usage
                                        .min_alignment_input = compactionv4t::WORD_ALIGNMENT,
                                        .min_alignment_output = compactionv4t::WORD_ALIGNMENT,
                                        .min_alignment_temp = compactionv4t::TMP_WORD_ALIGNMENT,
                                        .must_pad_alignment = true,
                                        .block_size = compactionv4t::BLOCK_SIZE,
                                        .table_range = compactionv4t::BLOCK_SIZE * compactionv4t::SUPER_BLOCK_SIZE,
                                        .must_pad_block = true,

                                        .escape_symbol = fsst::Symbol::escape,
                                        .padding_symbol = fsst::Symbol::ignore,
                                        .padding_enabled = true,

                                        .device_buffers = true};
    }

    GTSSTStatus CompactionV4TCompressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                                      CompressionConfiguration& config) {
        if (config.input_buffer_size > compactionv4t::BLOCK_SIZE * (size_t)0xFFFFFFFF) {
            return gtsstErrorTooBig;
        }

        if (config.block_size != compactionv4t::BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        if (config.min_alignment_input != compactionv4t::WORD_ALIGNMENT ||
            config.min_alignment_output != compactionv4t::WORD_ALIGNMENT ||
            config.min_alignment_temp != compactionv4t::TMP_WORD_ALIGNMENT) {
            return gtsstErrorBadBlockSize;
        }

        if ((uintptr_t)src % compactionv4t::WORD_ALIGNMENT != 0 || (uintptr_t)dst % compactionv4t::WORD_ALIGNMENT != 0
            ||
            (uintptr_t)tmp % compactionv4t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorAlignment;
        }

        if (config.input_buffer_size % compactionv4t::BLOCK_SIZE != 0 ||
            config.temp_buffer_size % compactionv4t::TMP_OUT_BLOCK_SIZE != 0) {
            return gtsstErrorBlockAlignment;
        }

        if (config.input_buffer_size % compactionv4t::WORD_ALIGNMENT != 0 ||
            config.temp_buffer_size % compactionv4t::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorWordAlignment;
        }

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV4TCompressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src,
                                                  uint8_t* tmp, CompressionConfiguration& config, size_t* out_size,
                                                  CompressionStatistics& stats) {
        if (const GTSSTStatus buffer_validation = validate_compression_buffers(src, dst, tmp, config);
            buffer_validation != gtsstSuccess) {
            return buffer_validation;
        }

        // TODO: remove this assertion
        assert(!data_contains(sample_src, 254, config.input_buffer_size));

        // Some bookkeeping
        const uint64_t number_of_blocks = config.input_buffer_size / compactionv4t::BLOCK_SIZE;
        const uint64_t number_of_tables = (number_of_blocks - 1) / compactionv4t::SUPER_BLOCK_SIZE + 1;
        const uint64_t metadata_mem_size = sizeof(compactionv4t::GCompactionMetadata) * number_of_tables;
        const uint64_t block_headers_mem_size = sizeof(CompactionV4TBlockHeader) *
            number_of_blocks;

        compactionv4t::GCompactionMetadata* metadata_host;
        GBaseHeader* table_headers_host;
        CompactionV4TBlockHeader* block_headers_host;
        safeCUDACall(cudaMallocHost(&metadata_host, metadata_mem_size));
        safeCUDACall(cudaMallocHost(&table_headers_host, sizeof(GBaseHeader) * number_of_tables));
        safeCUDACall(cudaMallocHost(&block_headers_host, block_headers_mem_size));

        // Some CUDA bookkeeping
        compactionv4t::GCompactionMetadata* metadata_gpu;
        CompactionV4TBlockHeader* block_headers_gpu;

        // Allocate some CUDA buffers
        safeCUDACall(cudaMalloc(&metadata_gpu, metadata_mem_size));
        safeCUDACall(cudaMalloc(&block_headers_gpu, block_headers_mem_size));

        // Start transpose while table is being generated
        if (compactionv4t::USE_TRANSPOSED_INPUT) {
            const auto aligned_transpose_src = (uint32_t*)src;
            const auto aligned_transpose_dst = (uint32_t*)dst;
            dim3 dimGrid(compactionv4t::n_words_per_tile / 32, compactionv4t::THREAD_COUNT * number_of_blocks / 32, 1);
            dim3 dimBlock(32, 8, 1);
            transpose_no_bank_conflicts<32, 8><<<dimGrid, dimBlock>>>(aligned_transpose_dst, aligned_transpose_src);
        }

        // Phase 1: Symbol generation (CPU for now)
        const auto symbol_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(number_of_tables);
        for (uint32_t i = 0; i < number_of_tables; i++) {
            threads.emplace_back(gpu_create_metadata<symbols::SmallSymbolMatchTableData>, i,
                                 compactionv4t::BLOCK_SIZE * compactionv4t::SUPER_BLOCK_SIZE, metadata_host,
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
        if (compactionv4t::USE_TRANSPOSED_INPUT) {
            compactionv4t::gpu_compaction<<<number_of_blocks, compactionv4t::THREAD_COUNT>>>(
                metadata_gpu, block_headers_gpu, dst, tmp);
        } else {
            compactionv4t::gpu_compaction<<<number_of_blocks, compactionv4t::THREAD_COUNT>>>(
                metadata_gpu, block_headers_gpu, src, tmp);
        }
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
        CompactionV4TFileHeader file_header{
            {
                total_data_size + block_headers_mem_size + sizeof(CompactionV4TFileHeader),
                config.input_buffer_size, // TODO: would have to change this to support input padding
                (uint32_t)number_of_tables,
                0,
                (uint32_t)number_of_blocks,
            },
            compactionv4t::SUB_BLOCKS,
        };
        size_t header_size = sizeof(CompactionV4TFileHeader);

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
        safeCUDACall(cudaMemcpyAsync(dst, &file_header, sizeof(CompactionV4TFileHeader), cudaMemcpyHostToDevice));

        // Then gather data
        uint64_t running_length = 0;
        for (uint32_t block_id = 0; block_id < number_of_blocks; block_id++) {
            uint8_t* running_dst = dst + header_size + running_length;
            safeCUDACall(
                cudaMemcpyAsync(running_dst, tmp + compactionv4t::TMP_OUT_BLOCK_SIZE * block_id, block_headers_host[
                    block_id].compressed_size, cudaMemcpyDeviceToDevice));
            running_length += block_headers_host[block_id].compressed_size;
        }
        const size_t out = running_length + header_size;

        // Finally, free buffers
        safeCUDACall(cudaFreeHost(metadata_host));
        safeCUDACall(cudaFreeHost(table_headers_host));
        safeCUDACall(cudaFreeHost(block_headers_host));

        // And free cuda buffers
        safeCUDACall(cudaFree(metadata_gpu));
        safeCUDACall(cudaFree(block_headers_gpu));

        // Check and update output size
        assert(
            file_header.compressed_size - sizeof(CompactionV4TFileHeader) - file_header.table_size -
            block_headers_mem_size ==
            total_data_size);
        assert(running_length == total_data_size);
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
