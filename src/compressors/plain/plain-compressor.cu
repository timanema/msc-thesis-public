#include <algorithm>
#include <compressors/plain/plain-compressor.cuh>
#include <compressors/shared.cuh>
#include <fsst/fsst-lib.cuh>
#include <fsst/gtsst-fsst.cuh>
#include <gtsst/gtsst-tables.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
    constexpr uint32_t BLOCK_SIZE = 5 * 1024 * 1024;

    CompressionConfiguration PlainCompressor::configure_compression(const size_t buf_size) {
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

    namespace plain {
        size_t small_compress(const symbols::PlainSymbolTableData& symbol_table, const uint8_t* src, uint8_t* dst,
                              uint32_t len) {
            size_t out = 0;
            size_t idx = 0;

            for (size_t i = 0; i < len; i++) {
                auto s = fsst::Symbol(src + i, std::min((size_t)fsst::Symbol::maxLength, len - i));
                uint16_t code = symbol_table.findLongestSymbol(s); // Returns code:8, len: 4, code=255=escape
                auto sym = (uint8_t)code;
                auto sym_len = (uint8_t)(code >> 8);
                uint8_t escape = sym == 255;
                bool new_info = i == idx;

                // Premptive write
                dst[out + 1] = src[i];

                // If i == idx, then we have new information, so output
                if (new_info) {
                    // Write output
                    dst[out] = sym;

                    // Update pointers
                    out += 1 + escape;
                    idx += sym_len;

                    i += sym_len - 1; // PERF NOTE: add this to skip over already processed data, not nice on GPU though
                }
            }

            return out;
        }
    }

    CompressionStatistics simple_block_compress_step(const uint8_t* src, uint8_t* dst, uint32_t len,
                                                     uint32_t* block_out) {
        auto symbol_start = std::chrono::high_resolution_clock::now();

        auto* sample_buf = new uint8_t[FSST_SAMPLEMAXSZ];
        size_t sample_len = fsst::simple_make_sample(sample_buf, src, len);
        auto enc_table = fsst::build_symbol_table<symbols::PlainSymbolTableData>(sample_buf, sample_len);
        size_t sLen = sizeof(uint32_t);
        size_t stLen = enc_table->export_table(dst + sLen);
        auto symbol_end = std::chrono::high_resolution_clock::now();

        size_t output = plain::small_compress(enc_table->encoding_data, src, dst + sLen + stLen, len);
        uint32_t total_out = stLen + output;

        TSST_WRITE_LEN(total_out, dst);

        auto encoding_end = std::chrono::high_resolution_clock::now();
        *block_out = sLen + total_out;

        delete[] sample_buf;
        delete enc_table;

        return CompressionStatistics{
            .table_generation = std::chrono::duration_cast<std::chrono::microseconds>(symbol_end - symbol_start),
            .encoding = std::chrono::duration_cast<std::chrono::microseconds>(encoding_end - symbol_end)};
    }

    GTSSTStatus PlainCompressor::compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src, uint8_t* tmp,
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

        while (in < config.input_buffer_size) {
            uint32_t block_len = std::min(config.input_buffer_size - in, (size_t)BLOCK_SIZE);

            uint32_t block_out = 0;
            CompressionStatistics block_stats = simple_block_compress_step(src + in, dst + out, block_len, &block_out);

            in += block_len;
            out += block_out;

            // Do some timekeeping
            symbol_table_generation_time += block_stats.table_generation;
            encoding_time += block_stats.encoding;
        }

        *out_size = out;
        stats.table_generation = symbol_table_generation_time;
        stats.encoding = encoding_time;
        return gtsstSuccess;
    }

    DecompressionConfiguration PlainCompressor::configure_decompression(const size_t buf_size) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = buf_size * 3,
        };
    }

    GTSSTStatus PlainCompressor::decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
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

    GTSSTStatus PlainCompressor::validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                              CompressionConfiguration& config) {
        if (config.block_size != BLOCK_SIZE) {
            return gtsstErrorBadBlockSize;
        }

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
