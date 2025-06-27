#include <algorithm>
#include <compressors/stats/stats-compressor.cuh>
#include <compressors/shared.cuh>
#include <compressors/plain/plain-compressor.cuh>
#include <fsst/fsst-lib.cuh>
#include <fsst/gtsst-fsst.cuh>
#include <gtsst/gtsst-tables.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
    constexpr size_t BLOCK_SIZE = 1ULL<<22; // Default FSST size

    CompressionConfiguration StatsCompressor::configure_compression(const size_t buf_size) {
        return CompressionConfiguration{.input_buffer_size = buf_size,
                                        .compression_buffer_size = buf_size,
                                        .temp_buffer_size = 0,
                                        .min_alignment_input = 1,
                                        .min_alignment_output = 1,
                                        .min_alignment_temp = 1,
                                        .must_pad_alignment = false,
                                        .block_size = BLOCK_SIZE,
                                        .table_range = BLOCK_SIZE,
                                        .must_pad_block = false,

                                        .escape_symbol = 255,
                                        .padding_symbol = 0,
                                        .padding_enabled = false,

                                        .device_buffers = false};
    }

    GTSSTStatus StatsCompressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src, uint8_t* tmp,
                                          CompressionConfiguration& config, size_t* out_size,
                                          CompressionStatistics& stats) {
        if (const GTSSTStatus buffer_validation = validate_compression_buffers(src, dst, tmp, config);
            buffer_validation != gtsstSuccess) {
            return buffer_validation;
        }

        size_t out = 0;
        size_t in = 0;

        auto symbol_table_generation_time = std::chrono::microseconds(0);
        auto encoding_time = std::chrono::microseconds(0);

        int symbol_2len_start_char[255] = {};
        int symbol_len[FSST_CODE_BITS] = {};
        int symbol_cnt = 0;
        int max_2len_start_chars = 0;
        int avg_2len_start_chars = 0;
        int max_2len_combinations = 0;
        float avg_2len_combinations = 0;
        int number_of_tables = 0;

        while (in < config.input_buffer_size) {
            const size_t block_len = std::min(config.input_buffer_size - in, (size_t)BLOCK_SIZE);

            auto symbol_start = std::chrono::high_resolution_clock::now();

            auto* sample_buf = new uint8_t[FSST_SAMPLEMAXSZ];
            const size_t sample_len = fsst::simple_make_sample(sample_buf, src+in, block_len);
            // TODO: use normal symbol table
            const auto enc_table = fsst::build_symbol_table<symbols::PlainSymbolTableData>(sample_buf, sample_len);
            size_t sLen = sizeof(uint32_t);
            size_t stLen = enc_table->export_table(dst + out + sLen);
            const auto symbol_end = std::chrono::high_resolution_clock::now();

            bool seen_2len_start_chars[255] = {};
            int seen_2len_combinations_chars[255] = {};

            for (auto sym : enc_table->symbols) {
                if (sym.length() == 0) continue;

                if (sym.length() == 2) {
                    symbol_2len_start_char[sym.first()] += 1;
                    seen_2len_start_chars[sym.first()] = true;
                    seen_2len_combinations_chars[sym.first()] += 1;
                }

                symbol_len[sym.length() - 1] += 1;
                symbol_cnt++;
            }

            int avg_total_seen_2len_combinations_chars = 0;
            int total_seen_2len_combinations_chars = 0;
            for (const int seen_2len_combinations_char : seen_2len_combinations_chars) {
                if (seen_2len_combinations_char > total_seen_2len_combinations_chars) total_seen_2len_combinations_chars = seen_2len_combinations_char;
                avg_total_seen_2len_combinations_chars += seen_2len_combinations_char;
            }
            if (total_seen_2len_combinations_chars > max_2len_combinations) {
                max_2len_combinations = total_seen_2len_combinations_chars;
            }
            int total_seen_2len_start_chars = 0;
            for (const bool seen_2len_start_char : seen_2len_start_chars) {
                if (seen_2len_start_char) total_seen_2len_start_chars += 1;
            }
            if (total_seen_2len_start_chars > max_2len_start_chars) {
                max_2len_start_chars = total_seen_2len_start_chars;
            }

            avg_2len_start_chars += total_seen_2len_start_chars;
            avg_2len_combinations += (float) avg_total_seen_2len_combinations_chars / (float) total_seen_2len_start_chars;

            size_t output = plain::small_compress(enc_table->encoding_data, src+in, dst+out + sLen + stLen, block_len);
            uint32_t total_out = stLen + output;

            TSST_WRITE_LEN(total_out, dst+out);

            auto encoding_end = std::chrono::high_resolution_clock::now();
            out += sLen + total_out;

            delete[] sample_buf;
            delete enc_table;

            in += block_len;

            // Do some timekeeping
            symbol_table_generation_time += std::chrono::duration_cast<std::chrono::microseconds>(symbol_end - symbol_start);
            encoding_time += std::chrono::duration_cast<std::chrono::microseconds>(encoding_end - symbol_end);
            number_of_tables += 1;
        }

        printf("avg_symbol_2len_start_char: ");
        for (int i = 0; i < 255; i++) {
            printf("%f,", (float) symbol_2len_start_char[i] / (float) number_of_tables);
        }

        printf("\navg_symbol_len: ");
        for (int i = 0; i < 8; i++) {
            printf("%f,", (float) symbol_len[i] / (float) number_of_tables);
        }
        printf("\navg_symbol_cnt: %f\n", (float) symbol_cnt / (float) number_of_tables);
        printf("avg_hashtable_usage: %f\n", (symbol_len[2] + symbol_len[3] + symbol_len[4] + symbol_len[5] + symbol_len[6] + symbol_len[7]) / (float) number_of_tables);
        printf("avg_shortcodes_usage: %f\n", (symbol_len[0] + symbol_len[1]) / (float) number_of_tables);
        printf("max_2len_start_chars: %d\n", max_2len_start_chars);
        printf("avg_2len_start_chars: %f\n", avg_2len_start_chars / (float) number_of_tables);
        printf("max_2len_combinations: %d\n", max_2len_combinations);
        printf("avg_2len_combinations: %f\n", avg_2len_combinations / (float) number_of_tables);

        *out_size = out;
        stats.table_generation = symbol_table_generation_time;
        stats.encoding = encoding_time;
        return gtsstSuccess;
    }

    DecompressionConfiguration StatsCompressor::configure_decompression(const size_t buf_size) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = buf_size * 3,
        };
    }

    GTSSTStatus StatsCompressor::decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                                            size_t* out_size) {
        size_t out = 0;
        size_t in = 0;

        while (in < config.input_buffer_size) {
            size_t sLen = sizeof(uint32_t);
            uint32_t block_len = TSST_READ_LEN(src + in);

            if (block_len > config.input_buffer_size - in) {
                return gtsstErrorCorruptHeader;
            }

            fsst::DecodingTable dec{};
            const size_t stLen = dec.import_table(src + in + sLen);
            const uint32_t block_out = seq_decompress(dec, src + in + sLen + stLen, dst + out, block_len - stLen);

            in += sLen + block_len;
            out += block_out;
        }

        *out_size = out;

        return gtsstSuccess;
    }

    GTSSTStatus StatsCompressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                              CompressionConfiguration& config) {
        if (config.block_size != BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
