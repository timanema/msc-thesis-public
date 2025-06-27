#include <compressors/compactionv2/compaction-transpose.cuh>
#include <compressors/compactionv3/compaction-compressor.cuh>
#include <compressors/compactionv3/compaction-defines.cuh>
#include <compressors/compactionv3/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fstream>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace gtsst::compressors {
    CompressionConfiguration CompactionV3Compressor::configure_compression(const size_t buf_size) {
        return CompressionConfiguration{
            .input_buffer_size = buf_size,
            .compression_buffer_size = buf_size,
            .temp_buffer_size = buf_size * 5 / 2 * 2,
            // TODO: might be able to reduce this if transpose can be synced grid-wide
            .min_alignment_input = compactionv3::WORD_ALIGNMENT,
            .min_alignment_output = compactionv3::WORD_ALIGNMENT,
            .min_alignment_temp = compactionv3::TMP_WORD_ALIGNMENT,
            .must_pad_alignment = true,
            .block_size = compactionv3::BLOCK_SIZE,
            .table_range = compactionv3::BLOCK_SIZE * compactionv3::SUPER_BLOCK_SIZE,
            .must_pad_block = true,

            .escape_symbol = fsst::Symbol::escape,
            .padding_symbol = fsst::Symbol::ignore,
            .padding_enabled = true,

            .device_buffers = true};
    }

    GTSSTStatus CompactionV3Compressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                                     CompressionConfiguration& config) {
        if (config.input_buffer_size > compactionv3::BLOCK_SIZE * (size_t)0xFFFFFFFF) {
            return gtsstErrorTooBig;
        }

        if (config.block_size != compactionv3::BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        if (config.min_alignment_input != compactionv3::WORD_ALIGNMENT ||
            config.min_alignment_output != compactionv3::WORD_ALIGNMENT ||
            config.min_alignment_temp != compactionv3::TMP_WORD_ALIGNMENT) {
            return gtsstErrorBadBlockSize;
        }

        if ((uintptr_t)src % compactionv3::WORD_ALIGNMENT != 0 || (uintptr_t)dst % compactionv3::WORD_ALIGNMENT != 0 ||
            (uintptr_t)tmp % compactionv3::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorAlignment;
        }

        if (config.input_buffer_size % compactionv3::BLOCK_SIZE != 0 ||
            config.temp_buffer_size % compactionv3::TMP_OUT_BLOCK_SIZE != 0) {
            return gtsstErrorBlockAlignment;
        }

        if (config.input_buffer_size % compactionv3::WORD_ALIGNMENT != 0 ||
            config.temp_buffer_size % compactionv3::TMP_WORD_ALIGNMENT != 0) {
            return gtsstErrorWordAlignment;
        }

        return gtsstSuccess;
    }

    GTSSTStatus CompactionV3Compressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src,
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
        const uint64_t number_of_blocks = config.input_buffer_size / compactionv3::BLOCK_SIZE;
        const uint64_t working_block_size = number_of_blocks * compactionv3::TMP_OUT_BLOCK_SIZE;
        const uint64_t number_of_tables = (number_of_blocks - 1) / compactionv3::SUPER_BLOCK_SIZE + 1;
        const uint64_t metadata_mem_size = sizeof(compactionv3::GCompactionMetadata) * number_of_tables;
        const uint64_t block_headers_mem_size = sizeof(BlockHeader) * number_of_blocks;

        compactionv3::GCompactionMetadata* metadata_host;
        GBaseHeader* table_headers_host;
        BlockHeader* block_headers_host;
        safeCUDACall(cudaMallocHost(&metadata_host, metadata_mem_size));
        safeCUDACall(cudaMallocHost(&table_headers_host, sizeof(GBaseHeader) * number_of_tables));
        safeCUDACall(cudaMallocHost(&block_headers_host, block_headers_mem_size));

        // Update the device queue for internal transpose launches
        cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, compactionv3::CUDA_QUEUE_LEN);
        assert(number_of_blocks < compactionv3::CUDA_QUEUE_LEN);
        // If there are too many blocks, we cannot compress this file in one go (and some margin)
        if (number_of_blocks > compactionv3::CUDA_QUEUE_LEN - 10) {
            return gtsstErrorTooBig;
        }

        // Some CUDA bookkeeping
        compactionv3::GCompactionMetadata* metadata_gpu;
        BlockHeader* block_headers_gpu;
        cudaStream_t mem_stream;
        cudaStream_t compute_stream;
        safeCUDACall(cudaStreamCreate(&mem_stream));
        safeCUDACall(cudaStreamCreate(&compute_stream));

        // Allocate some CUDA buffers
        safeCUDACall(cudaMallocAsync(&metadata_gpu, metadata_mem_size, mem_stream));
        safeCUDACall(cudaMallocAsync(&block_headers_gpu, block_headers_mem_size, mem_stream));

        // Set temp_dst to all ignores, so all unused data is filtered in a later stage
        // safeCUDACall(cudaMemsetAsync(tmp, 254, working_block_size*2, mem_stream)); // TODO: figure out why this was blocking the tranpose kernel

        // Create cuda timing events
        cudaEvent_t pre_start, pre_done, encode_start, encode_done, post_done;
        safeCUDACall(cudaEventCreate(&pre_start));
        safeCUDACall(cudaEventCreate(&pre_done));
        safeCUDACall(cudaEventCreate(&encode_start));
        safeCUDACall(cudaEventCreate(&encode_done));
        safeCUDACall(cudaEventCreate(&post_done));

        // Start transpose while table is being generated
        const auto aligned_transpose_src = (uint32_t*)src;
        const auto aligned_transpose_dst = (uint32_t*)dst;
        dim3 dimGrid(compactionv3::n_words_per_tile / 32, compactionv3::THREAD_COUNT * number_of_blocks / 32, 1);
        dim3 dimBlock(32, 8, 1);
        safeCUDACall(cudaEventRecord(pre_start, compute_stream));
        transpose_no_bank_conflicts<32, 8><<<dimGrid, dimBlock, 0, compute_stream>>>(
            aligned_transpose_dst, aligned_transpose_src);
        safeCUDACall(cudaEventRecord(pre_done, compute_stream));
        safeCUDACall(cudaPeekAtLastError());

        // Phase 1: Symbol generation (CPU for now)
        const auto table_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(number_of_tables);
        for (uint32_t i = 0; i < number_of_tables; i++) {
            threads.emplace_back(gpu_create_metadata<symbols::SmallSymbolMatchTableData>, i,
                                 compactionv3::BLOCK_SIZE * compactionv3::SUPER_BLOCK_SIZE, metadata_host,
                                 table_headers_host, sample_src, config.input_buffer_size);
        }
        for (std::thread& t : threads) {
            t.join();
        }

        // Copy metadata to GPU memory
        safeCUDACall(
            cudaMemcpyAsync(metadata_gpu, metadata_host, metadata_mem_size, cudaMemcpyHostToDevice, mem_stream));
        const auto table_end = std::chrono::high_resolution_clock::now();

        // Phase 2: Precomputation
        // safeCUDACall(cudaEventRecord(pre_start, compute_stream));
        // compactionv3::transpose_no_bank_conflicts<32, 8><<<dimGrid, dimBlock, 0, compute_stream>>>(
        //     aligned_transpose_dst, aligned_transpose_src);
        // safeCUDACall(cudaEventRecord(pre_done, compute_stream));
        // safeCUDACall(cudaPeekAtLastError());

        // Phase 3: Encoding (GPU)
        // Run all blocks
        safeCUDACall(cudaStreamSynchronize(mem_stream));
        safeCUDACall(cudaEventRecord(encode_start, compute_stream));
        compactionv3::gpu_compaction<<<number_of_blocks, compactionv3::THREAD_COUNT, 0, compute_stream>>>(
            metadata_gpu, block_headers_gpu, dst, tmp + working_block_size);
        safeCUDACall(cudaEventRecord(encode_done, compute_stream));
        safeCUDACall(cudaPeekAtLastError());

        // Phase 4: Postprocessing (Partial CPU for now)
        // Copy comp headers & temp_dst to CPU
        safeCUDACall(
            cudaMemcpyAsync(block_headers_host, block_headers_gpu, block_headers_mem_size, cudaMemcpyDeviceToHost,
                compute_stream));
        safeCUDACall(cudaStreamSynchronize(compute_stream));

        // Transpose output data
        const auto aligned_transpose_output_src = (uint32_t*)(tmp + working_block_size);
        const auto aligned_transpose_output_dst = (uint32_t*)tmp;
        dim3 dimGridOutput(compactionv3::THREAD_COUNT * number_of_blocks / 32, compactionv3::tile_out_len_words / 32, 1);
        transpose_no_bank_conflicts<32, 8><<<dimGridOutput, dimBlock, 0, compute_stream>>>(
            aligned_transpose_output_dst, aligned_transpose_output_src);

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
                metadata_host[table_id].header_offset, cudaMemcpyHostToDevice, mem_stream));

            header_size += metadata_host[table_id].header_offset;
            file_header.table_size += metadata_host[table_id].header_offset;
        }

        // Copy block headers
        safeCUDACall(
            cudaMemcpyAsync(dst + header_size, block_headers_host, block_headers_mem_size, cudaMemcpyHostToDevice,
                mem_stream));
        header_size += block_headers_mem_size;

        // Copy file header
        file_header.compressed_size += file_header.table_size;
        safeCUDACall(cudaMemcpyAsync(dst, &file_header, sizeof(FileHeader), cudaMemcpyHostToDevice, mem_stream));

        // Then do stream compaction on the actual data
        const thrust::device_ptr<uint8_t> thrust_gpu_in = thrust::device_pointer_cast(tmp);
        const thrust::device_ptr<uint8_t> thrust_gpu_out = thrust::device_pointer_cast(dst + header_size);
        const thrust::device_ptr<uint8_t> thrust_new_end =
            copy_if(thrust::cuda::par.on(compute_stream), thrust_gpu_in, thrust_gpu_in + working_block_size,
                    thrust_gpu_out, is_not_ignore());
        const size_t thrust_out_size = thrust_new_end - thrust_gpu_out;
        const size_t out = thrust_out_size + header_size;
        safeCUDACall(cudaEventRecord(post_done, compute_stream));

        // Finally, free buffers
        safeCUDACall(cudaFreeHost(metadata_host));
        safeCUDACall(cudaFreeHost(table_headers_host));
        safeCUDACall(cudaFreeHost(block_headers_host));

        // And free cuda buffers
        safeCUDACall(cudaFreeAsync(metadata_gpu, mem_stream));
        safeCUDACall(cudaFreeAsync(block_headers_gpu, mem_stream));

        // Check and update output size
        assert(file_header.compressed_size - sizeof(FileHeader) - file_header.table_size - block_headers_mem_size ==
            total_data_size);
        assert(thrust_out_size == total_data_size);
        *out_size = out;

        // Sync on finish
        safeCUDACall(cudaDeviceSynchronize());

        // Update statistics
        float pre_duration_ms = 0;
        float encode_duration_ms = 0;
        float post_duration_ms = 0;
        safeCUDACall(cudaEventElapsedTime(&pre_duration_ms, pre_start, pre_done));
        safeCUDACall(cudaEventElapsedTime(&encode_duration_ms, encode_start, encode_done));
        safeCUDACall(cudaEventElapsedTime(&post_duration_ms, encode_done, post_done));
        safeCUDACall(cudaEventDestroy(pre_start));
        safeCUDACall(cudaEventDestroy(pre_done));
        safeCUDACall(cudaEventDestroy(encode_start));
        safeCUDACall(cudaEventDestroy(encode_done));
        safeCUDACall(cudaEventDestroy(post_done));
        safeCUDACall(cudaStreamDestroy(compute_stream));
        safeCUDACall(cudaStreamDestroy(mem_stream));


        stats.table_generation =
            std::chrono::duration_cast<std::chrono::microseconds>(table_end - table_start);
        stats.precomputation =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::duration<float, std::milli>(pre_duration_ms));
        stats.encoding =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::duration<float, std::milli>(encode_duration_ms));
        stats.postprocessing =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::duration<float, std::milli>(post_duration_ms));

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
