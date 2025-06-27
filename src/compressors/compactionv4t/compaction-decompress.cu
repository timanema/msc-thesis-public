#include <compressors/compactionv4t/compaction-compressor.cuh>
#include <compressors/compactionv4t/compaction-defines.cuh>
#include <compressors/compactionv4t/compaction-encode.cuh>
#include <compressors/shared.cuh>

namespace gtsst::compressors {
    DecompressionConfiguration CompactionV4TCompressor::configure_decompression(size_t buf_size) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = buf_size * 3, // TODO: could be better if in separate header..
        };
    }

    size_t transpose_memcpy_v4t(uint32_t* dst, const uint32_t* src, const uint16_t flushes,
                        const uint16_t n_sub_blocks) {
        if (dst == nullptr || src == nullptr || n_sub_blocks == 0)
            return 0;

        for (int flush = 0; flush < flushes; flush++) {
            for (int thread = 0; thread < compactionv4t::THREAD_COUNT; thread++) {
                dst[thread * flushes + flush] = src[flush * compactionv4t::THREAD_COUNT + thread];
            }
        }

        return flushes * 4 * compactionv4t::THREAD_COUNT;
    }

    GTSSTStatus CompactionV4TCompressor::decompress(const uint8_t* src, uint8_t* dst,
                                                    DecompressionConfiguration& config,
                                                    size_t* out_size) {
        // Read file header
        CompactionV4TFileHeader file_header;
        memcpy(&file_header, src, sizeof(CompactionV4TFileHeader));

        std::vector<fsst::DecodingTable> decoders;
        size_t in = sizeof(CompactionV4TFileHeader);

        // First read all tables
        for (int table_id = 0; table_id < file_header.num_tables; table_id++) {
            fsst::DecodingTable dec{};
            const size_t table_len = dec.import_table(src + in);

            decoders.emplace_back(dec);
            in += table_len;
        }

        // Return error if table reading went wrong
        if (in - sizeof(CompactionV4TFileHeader) != file_header.table_size) {
            return gtsstErrorCorruptHeader;
        }

        // Also return corrupt header if number of subblocks doesn't match with the compiled constants
        if (file_header.num_sub_blocks != compactionv4t::SUB_BLOCKS) {
            return gtsstErrorCorruptHeader;
        }

        // Then read all blocks
        const size_t block_header_size =
            file_header.num_blocks * sizeof(CompactionV4TBlockHeader);
        auto* block_headers = (CompactionV4TBlockHeader*)malloc(block_header_size);
        memcpy(block_headers, src + in, block_header_size);
        in += block_header_size;

        const auto block_mem = (uint8_t*)malloc(compactionv4t::BLOCK_SIZE);

        // Then decode all blocks
        uint64_t out = 0;
        for (int block_id = 0; block_id < file_header.num_blocks; block_id++) {
            fsst::DecodingTable decoder = decoders[block_id / compactionv4t::SUPER_BLOCK_SIZE];
            const uint32_t block_size = block_headers[block_id].compressed_size;
            // memcpy(block_mem, src + in, block_size);
            const size_t transpose_size = transpose_memcpy_v4t(reinterpret_cast<uint32_t*>(block_mem),
                                                           reinterpret_cast<const uint32_t*>(src + in),
                                                           block_headers[block_id].flushes, compactionv4t::SUB_BLOCKS);
            // If transpose size doesn't match block size, the transpose went wrong
            if (transpose_size != block_size) {
                free(block_headers);
                free(block_mem);
                return gtsstErrorCorruptBlock;
            }

            const uint32_t block_out = seq_decompress(decoder, block_mem, dst + out, block_size);

            // If output size doesn't match, the block is corrupt
            if (block_out != block_headers[block_id].uncompressed_size) {
                free(block_headers);
                free(block_mem);
                return gtsstErrorCorruptBlock;
            }

            out += block_out;
            in += block_size;
        }

        // Free header buffer
        free(block_headers);
        free(block_mem);

        // Do final check if we consumed all data, and produced the expected amount of data
        if (in != file_header.compressed_size || out != file_header.uncompressed_size) {
            return gtsstErrorCorrupt;
        }

        // Update output
        *out_size = out;

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
