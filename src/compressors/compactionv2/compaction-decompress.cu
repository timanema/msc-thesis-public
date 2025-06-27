#include <compressors/compactionv2/compaction-compressor.cuh>
#include <compressors/compactionv2/compaction-defines.cuh>
#include <compressors/compactionv2/compaction-encode.cuh>
#include <compressors/shared.cuh>

namespace gtsst::compressors {
    DecompressionConfiguration CompactionV2Compressor::configure_decompression(size_t buf_size) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = buf_size * 3, // TODO: could be better if in separate header..
        };
    }

    GTSSTStatus CompactionV2Compressor::decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                                                   size_t* out_size) {
        if (config.input_buffer_size == 0) {
            return gtsstSuccess;
        }

        // Read file header
        FileHeader file_header;
        memcpy(&file_header, src, sizeof(FileHeader));

        std::vector<fsst::DecodingTable> decoders;
        size_t in = sizeof(FileHeader);

        // First read all tables
        for (int table_id = 0; table_id < file_header.num_tables; table_id++) {
            fsst::DecodingTable dec{};
            const size_t table_len = dec.import_table(src + in);

            decoders.emplace_back(dec);
            in += table_len;
        }

        // Return error if table reading went wrong
        if (in - sizeof(FileHeader) != file_header.table_size) {
            return gtsstErrorCorruptHeader;
        }

        // Then read all blocks
        const size_t block_header_size = file_header.num_blocks * sizeof(BlockHeader);
        auto* block_headers = (BlockHeader*)malloc(block_header_size);
        memcpy(block_headers, src + in, block_header_size);
        in += block_header_size;

        // Then decode all blocks
        uint64_t out = 0;
        for (int block_id = 0; block_id < file_header.num_blocks; block_id++) {
            fsst::DecodingTable decoder = decoders[block_id / compactionv2::SUPER_BLOCK_SIZE];
            const uint32_t block_size = block_headers[block_id].compressed_size;

            const uint32_t block_out = seq_decompress(decoder, src + in, dst + out, block_size);

            // If output size doesn't match, the block is corrupt
            if (block_out != block_headers[block_id].uncompressed_size) {
                free(block_headers);
                return gtsstErrorCorruptBlock;
            }

            out += block_out;
            in += block_size;
        }

        // Free header buffer
        free(block_headers);

        // Do final check if we consumed all data, and produced the expected amount of data
        if (in != file_header.compressed_size || out != file_header.uncompressed_size) {
            return gtsstErrorCorrupt;
        }

        // Update output
        *out_size = out;

        return gtsstSuccess;
    }
} // namespace gtsst::compressors
