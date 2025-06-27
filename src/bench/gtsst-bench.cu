#include <bench/gtsst-bench.cuh>
#include <bench/gtsst-prepare.cuh>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <gtsst/gtsst.cuh>
#include <iostream>
#include <numeric>

namespace gtsst::bench {
    size_t read_file_size(const char* filename) {
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return 0;
        }

        // Get the size of the file
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        file.close();

        return file_size;
    }

    uint8_t* read_file(const char* filename, const size_t data_to_read, const size_t buffer_size, const bool silent) {
        assert(buffer_size >= data_to_read);
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return nullptr;
        }

        // Create a buffer to hold the data
        uint8_t* buffer = (uint8_t*)malloc(buffer_size);

        file.seekg(0, std::ios::beg);

        // Read the file into the buffer
        if (!file.read(reinterpret_cast<char*>(buffer), data_to_read)) {
            std::cerr << "Error: Failed to read file " << filename << std::endl;
            free(buffer);
            return nullptr;
        }

        file.close();

        if (!silent) {
            std::cout << "File read successfully. Size: " << data_to_read << " bytes." << std::endl;
        }

        return buffer;
    }

    void write_file(const char* filename, const char* data, const size_t len) {
        std::ofstream out_file(filename, std::ios::binary);
        out_file.write(data, len);
    }

    CompressionStats compress_single(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src, uint8_t* tmp,
                                     CompressionConfiguration& compression_configuration,
                                     CompressionManager& compression_manager) {
        CompressionStats stats = {.original_len = compression_configuration.input_buffer_size};

        const auto start = std::chrono::high_resolution_clock::now();
        const auto status = compression_manager.compress(src, dst, sample_src, tmp, compression_configuration,
                                                         &stats.compress_len, stats.internal_stats);
        const auto end = std::chrono::high_resolution_clock::now();

        if (status != gtsstSuccess) {
            std::cerr << "Compression error: " << status << std::endl;
            exit(1);
        }

        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.compression_duration = duration;
        stats.compression_throughput = (double)stats.original_len / (1.0e9 * (duration.count() * 1.0e-6));

        return stats;
    }

    AggregatedCompressionStats compress_repeat(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src,
                                               uint8_t* tmp, CompressionConfiguration& compression_configuration,
                                               CompressionManager& compression_manager, int iterations) {
        AggregatedCompressionStats aggregated_stats = {};

        if (iterations == 0) {
            return aggregated_stats;
        }

        std::vector<std::chrono::microseconds> compression_duration;
        std::vector<double> compression_throughput;

        std::vector<std::chrono::microseconds> table_generation;
        std::vector<std::chrono::microseconds> precomputation;
        std::vector<std::chrono::microseconds> encoding;
        std::vector<std::chrono::microseconds> postprocessing;

        for (int i = 0; i < iterations; i++) {
            CompressionStats stats =
                compress_single(src, dst, sample_src, tmp, compression_configuration, compression_manager);
            aggregated_stats.stats.push_back(stats);

            compression_duration.push_back(stats.compression_duration);
            compression_throughput.push_back(stats.compression_throughput);

            table_generation.push_back(stats.internal_stats.table_generation);
            precomputation.push_back(stats.internal_stats.precomputation);
            encoding.push_back(stats.internal_stats.encoding);
            postprocessing.push_back(stats.internal_stats.postprocessing);

            printf("encoding: %.3f\n",
                   (double)stats.original_len / (1.0e9 * (stats.internal_stats.encoding.count() * 1.0e-6)));
        }

        aggregated_stats.compression_duration =
            std::accumulate(compression_duration.begin(), compression_duration.end(), std::chrono::microseconds(0)) /
            compression_duration.size();
        aggregated_stats.compression_throughput =
            std::accumulate(compression_throughput.begin(), compression_throughput.end(), 0.) /
            static_cast<double>(compression_throughput.size());

        aggregated_stats.internal_stats.table_generation =
            std::accumulate(table_generation.begin(), table_generation.end(), std::chrono::microseconds(0)) /
            table_generation.size();
        aggregated_stats.internal_stats.precomputation =
            std::accumulate(precomputation.begin(), precomputation.end(), std::chrono::microseconds(0)) /
            precomputation.size();
        aggregated_stats.internal_stats.encoding =
            std::accumulate(encoding.begin(), encoding.end(), std::chrono::microseconds(0)) / encoding.size();
        aggregated_stats.internal_stats.postprocessing =
            std::accumulate(postprocessing.begin(), postprocessing.end(), std::chrono::microseconds(0)) /
            postprocessing.size();

        aggregated_stats.internal_throughputs.table_generation_throughput =
            (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.table_generation.count() * 1.0e-6));
        aggregated_stats.internal_throughputs.precomputation_throughput =
            (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.precomputation.count() * 1.0e-6));
        aggregated_stats.internal_throughputs.encoding_throughput = (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.encoding.count() * 1.0e-6));
        aggregated_stats.internal_throughputs.postprocessing_throughput =
            (double)aggregated_stats.stats[0].original_len /
            (1.0e9 * (aggregated_stats.internal_stats.postprocessing.count() * 1.0e-6));

        return aggregated_stats;
    }

    bool data_equal(const uint8_t* src, const uint8_t* src_other, const size_t size, const bool strict) {
        if (src == nullptr || src_other == nullptr) {
            return false;
        }

        for (size_t i = 0; i < size; i++) {
            if (src[i] != src_other[i]) {
                printf("error: %zu -> %d != %d\n", i, src[i], src_other[i]);

                // You don't want this, but GSST decompression leaves me no choice..
                if (strict) {
                    return false;
                }
            }
        }

        return true;
    }

    bool decompress_single(const char* filename, CompressionStats& stats, const uint8_t* src, uint8_t* dst,
                           const uint8_t* original_data, DecompressionConfiguration& decompression_configuration,
                           CompressionManager& compression_manager, const bool strict_checking, const int iterations) {
        std::vector<std::chrono::microseconds> decompression_times;

        for (int i = 0; i < iterations; i++) {
            const auto start = std::chrono::high_resolution_clock::now();
            const auto status =
                compression_manager.decompress(src, dst, decompression_configuration, &stats.decompress_len);
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            if (status != gtsstSuccess) {
                std::cerr << "Decompression error (" << filename << "): " << status << std::endl;
                exit(1);
            }

            printf("decomp: %.3f\n", (double)stats.original_len / (1.0e9 * (duration.count() * 1.0e-6)));
            decompression_times.push_back(duration);
        }

        stats.decompression_duration =
            std::accumulate(decompression_times.begin(), decompression_times.end(), std::chrono::microseconds(0)) /
            decompression_times.size();
        stats.decompression_throughput =
            (double)stats.decompress_len / (1.0e9 * (stats.decompression_duration.count() * 1.0e-6));
        stats.ratio = static_cast<float>(stats.original_len) / static_cast<float>(stats.compress_len);

        // Equality check, if available
        bool length_match = stats.decompress_len == stats.original_len;
        bool match = length_match;

        if (!length_match) {
            printf("error: decompression length mismatch. Expected %lu, was %lu\n", stats.original_len, stats.decompress_len);
        }

        if (original_data != nullptr && length_match) {
            uint8_t* dst_buf = dst;
            if (decompression_configuration.device_buffers) {
                dst_buf = (uint8_t*)malloc(stats.decompress_len);
                checkedCUDACall(cudaMemcpy(dst_buf, dst, stats.decompress_len, cudaMemcpyDeviceToHost));
            }

            match = data_equal(original_data, dst_buf, stats.original_len, strict_checking);

            if (decompression_configuration.device_buffers) {
                free(dst_buf);
            }
        }

        return match;
    }

    bool decompress_all(const char* filename, AggregatedCompressionStats& aggregated_stats, const uint8_t* src,
                        uint8_t* tmp, const uint8_t* original_data,
                        DecompressionConfiguration& decompression_configuration,
                        CompressionManager& compression_manager, const bool strict_checking, const int iterations) {
        bool all_match = true;

        if (aggregated_stats.stats.empty()) {
            return all_match;
        }

        std::vector<std::chrono::microseconds> decompression_duration;
        std::vector<double> decompression_throughput;
        std::vector<float> ratio;

        for (CompressionStats& stats : aggregated_stats.stats) {
            bool match = decompress_single(filename, stats, src, tmp, original_data, decompression_configuration,
                                           compression_manager, strict_checking, iterations);
            all_match &= match;

            decompression_duration.push_back(stats.decompression_duration);
            decompression_throughput.push_back(stats.decompression_throughput);
            ratio.push_back(stats.ratio);

            break;
        }

        aggregated_stats.decompression_duration =
            std::accumulate(decompression_duration.begin(), decompression_duration.end(),
                            std::chrono::microseconds(0)) /
            decompression_duration.size();
        aggregated_stats.decompression_throughput =
            std::accumulate(decompression_throughput.begin(), decompression_throughput.end(), 0.) /
            static_cast<double>(decompression_throughput.size());
        aggregated_stats.ratio = std::accumulate(ratio.begin(), ratio.end(), 0.f) / static_cast<float>(ratio.size());

        return all_match;
    }

    bool full_cycle(const char* filename, const int compression_iterations, const int decompression_iterations,
                    CompressionManager& compression_manager, const bool print_csv, const bool strict_checking) {
        size_t file_size = read_file_size(filename);
        const auto file_compression_config = compression_manager.configure_compression(file_size);

        // TODO: this is a beunfix, replace later (allow for buffer to contain padding)
        const size_t buffer_size = check_smaller_buffer(file_size, file_compression_config);
        file_size = buffer_size;
        const auto src = read_file(filename, file_size, buffer_size);

        // const size_t buffer_size = check_buffer_required_length(file_size, file_compression_config);
        // const auto src = read_file(filename, file_size, buffer_size);

        // Fix buffer if needed
        size_t data_len = file_size;
        if (const bool valid_buffer = fix_buffer(src, file_size, buffer_size, &data_len, file_compression_config);
            !valid_buffer) {
            std::cerr << "Unable to fix data from file " << filename << std::endl;
            return false;
        }

        // Print warning
        if (!strict_checking) {
            std::cerr << "Strict output checking is disabled, decompressed data might not match original data!"
                      << std::endl;
        }

        auto compression_configuration = compression_manager.configure_compression(data_len);
        const bool dev_buf = compression_configuration.device_buffers;

        // Allocate buffers
        const uint8_t* sample_src = src;
        auto* dst = (uint8_t*)malloc(compression_configuration.compression_buffer_size);
        uint8_t* tmp;
        uint8_t* func_src;
        uint8_t* func_dst;

        cudaStream_t mem_stream;
        checkedCUDACall(cudaStreamCreate(&mem_stream));

        if (dev_buf) {
            checkedCUDACall(cudaMallocAsync(&tmp, compression_configuration.temp_buffer_size, mem_stream));
            checkedCUDACall(cudaMallocAsync(&func_src, compression_configuration.input_buffer_size, mem_stream));
            checkedCUDACall(cudaMallocAsync(&func_dst, compression_configuration.compression_buffer_size, mem_stream));
            checkedCUDACall(cudaMemcpyAsync(func_src, src, data_len, cudaMemcpyHostToDevice, mem_stream));
            checkedCUDACall(cudaStreamSynchronize(mem_stream));
        } else {
            tmp = (uint8_t*)malloc(compression_configuration.temp_buffer_size);
            func_src = src;
            func_dst = dst;
        }

        // Run compression
        AggregatedCompressionStats compression_stats =
            compress_repeat(func_src, func_dst, sample_src, tmp, compression_configuration, compression_manager,
                            compression_iterations);

        // Run decompression
        DecompressionConfiguration decompression_configuration =
            compression_manager.configure_decompression_from_compress(compression_stats.stats[0].compress_len,
                                                                      compression_configuration);
        bool dev_buf_decomp = decompression_configuration.device_buffers;

        if (dev_buf_decomp && !dev_buf) {
            printf("Either both comp & decomp need to use device buffers, or only comp. Not only decomp!");
            return false;
        }

        // Move dst to host if needed
        if (dev_buf && !dev_buf_decomp) {
            checkedCUDACall(cudaMemcpyAsync(dst, func_dst, compression_stats.stats[0].compress_len,
                                            cudaMemcpyDeviceToHost, mem_stream));
            checkedCUDACall(cudaStreamSynchronize(mem_stream));
        }

        uint8_t* decomp_tmp;

        if (dev_buf_decomp) {
            decomp_tmp = func_src;
        } else {
            decomp_tmp = (uint8_t*)malloc(decompression_configuration.decompression_buffer_size);
        }

        const bool matching_decompression =
            decompress_all(filename, compression_stats, dev_buf_decomp ? func_dst : dst, decomp_tmp, src,
                           decompression_configuration, compression_manager, strict_checking, decompression_iterations);

        // Free buffers
        if (dev_buf) {
            checkedCUDACall(cudaFreeAsync(tmp, mem_stream));
            checkedCUDACall(cudaFreeAsync(func_src, mem_stream));
            checkedCUDACall(cudaFreeAsync(func_dst, mem_stream));
        } else {
            free(tmp);
            free(decomp_tmp);
        }

        checkedCUDACall(cudaStreamSynchronize(mem_stream));
        checkedCUDACall(cudaStreamDestroy(mem_stream));
        checkedCUDACall(cudaDeviceReset());

        free(dst);
        free(src);

        // Print results
        if (print_csv) {
            printf("%lu,%lu,%lu,%lu,%.3f,%lu,%lu,%.3f,%.4f,%.3f,%lu,%.3f,%lu,%.3f,%lu,%.3f,%lu\n",
                   compression_configuration.block_size, data_len, compression_configuration.table_range,
                   compression_stats.compression_duration.count(), compression_stats.compression_throughput,
                   compression_stats.stats[0].compress_len, compression_stats.decompression_duration.count(),
                   compression_stats.decompression_throughput, compression_stats.ratio,
                   compression_stats.internal_throughputs.table_generation_throughput,
                   compression_stats.internal_stats.table_generation.count(),
                   compression_stats.internal_throughputs.precomputation_throughput,
                   compression_stats.internal_stats.precomputation.count(),
                   compression_stats.internal_throughputs.encoding_throughput,
                   compression_stats.internal_stats.encoding.count(),
                   compression_stats.internal_throughputs.postprocessing_throughput,
                   compression_stats.internal_stats.postprocessing.count());
        } else {
            printf("Cycles (%d, %d) completed. Stats:\n"
                   "\tParameters:\n"
                   "\t\tBlock size: %lu\n"
                   "\t\tInput size: %lu\n"
                   "\t\tEffective table size: %lu\n"
                   "\t\tFile name: %s\n"
                   "\tCompression:\n"
                   "\t\tDuration (us): %lu \n"
                   "\t\tThroughput (GB/s): %.3f\n"
                   "\t\tCompressed size: %lu\n"
                   "\tDecompression:\n"
                   "\t\tDuration (us): %lu\n"
                   "\t\tThroughput (GB/s): %.3f\n"
                   "\t\tRatio: %.4f\n"
                   "\tCompression phases:\n"
                   "\t\tTable generation (GB/s, us): %.3f (%lu)\n"
                   "\t\tPrecomputation (GB/s, us): %.3f (%lu)\n"
                   "\t\tEncoding (GB/s, us): %.3f (%lu)\n"
                   "\t\tPostprocessing (GB/s, us): %.3f (%lu)\n",
                   compression_iterations, decompression_iterations, compression_configuration.block_size, data_len,
                   compression_configuration.table_range, filename, compression_stats.compression_duration.count(),
                   compression_stats.compression_throughput, compression_stats.stats[0].compress_len,
                   compression_stats.decompression_duration.count(), compression_stats.decompression_throughput,
                   compression_stats.ratio, compression_stats.internal_throughputs.table_generation_throughput,
                   compression_stats.internal_stats.table_generation.count(),
                   compression_stats.internal_throughputs.precomputation_throughput,
                   compression_stats.internal_stats.precomputation.count(),
                   compression_stats.internal_throughputs.encoding_throughput,
                   compression_stats.internal_stats.encoding.count(),
                   compression_stats.internal_throughputs.postprocessing_throughput,
                   compression_stats.internal_stats.postprocessing.count());
        }

        return matching_decompression;
    }

    bool full_cycle_directory(const std::vector<std::string>& directories, const bool use_dir,
                              const int compression_iterations, const int decompression_iterations,
                              CompressionManager& compression_manager, const bool print_csv,
                              const bool strict_checking) {
        for (auto& file : directories) {
            try {
                for (const auto& entry : std::filesystem::directory_iterator(file)) {
                    const bool match =
                        full_cycle(entry.path().c_str(), compression_iterations, decompression_iterations,
                                   compression_manager, print_csv, strict_checking);

                    // If any of the matches failed, return false to indicate cycle mismatch
                    if (!match) {
                        return false;
                    }

                    // If not using directory mode, just break out of loop after first iteration
                    if (!use_dir) {
                        break;
                    }
                }
            } catch (const std::filesystem::filesystem_error& err) {
                std::cerr << "Error: " << err.what() << "\n";
            }
        }

        return true;
    }
} // namespace gtsst::bench
