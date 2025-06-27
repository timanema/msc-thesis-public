#include <compressors/compactionv3t/compaction-compressor.cuh>
#include <compressors/compactionv3t/compaction-defines.cuh>
#include <compressors/compactionv3t/compaction-encode.cuh>
#include <compressors/shared.cuh>

namespace gtsst::compressors {
    DecompressionConfiguration CompactionV3TCompressor::configure_decompression(size_t buf_size) {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = buf_size * 3, // TODO: could be better if in separate header..
        };
    }

    size_t transpose_memcpy_v3t(uint32_t* dst, const uint32_t* src, const uint16_t* n_flushes,
                            const uint16_t n_sub_blocks) {
        size_t sub_out = 0;

        if (dst == nullptr || src == nullptr || n_flushes == nullptr || n_sub_blocks == 0)
            return 0;

        for (int sub_id = 0; sub_id < n_sub_blocks; sub_id++) {
            const int flushes = n_flushes[sub_id];

            for (int flush = 0; flush < flushes; flush++) {
                for (int thread = 0; thread < 32; thread++) {
                    dst[sub_out + thread * flushes + flush] = src[sub_out + flush * 32 + thread];
                }
            }

            sub_out += flushes * 32;
        }

        return sub_out * 4;
    }

    GTSSTStatus CompactionV3TCompressor::decompress(const uint8_t* src, uint8_t* dst,
                                                    DecompressionConfiguration& config,
                                                    size_t* out_size) {
        // Read file header
        CompactionV3TFileHeader file_header;
        memcpy(&file_header, src, sizeof(CompactionV3TFileHeader));

        std::vector<fsst::DecodingTable> decoders;
        size_t in = sizeof(CompactionV3TFileHeader);

        // First read all tables
        for (int table_id = 0; table_id < file_header.num_tables; table_id++) {
            fsst::DecodingTable dec{};
            const size_t table_len = dec.import_table(src + in);

            decoders.emplace_back(dec);
            in += table_len;
        }

        // Return error if table reading went wrong
        if (in - sizeof(CompactionV3TFileHeader) != file_header.table_size) {
            return gtsstErrorCorruptHeader;
        }

        // Also return corrupt header if number of subblocks doesn't match with the compiled constants
        if (file_header.num_sub_blocks != compactionv3t::SUB_BLOCKS) {
            return gtsstErrorCorruptHeader;
        }

        // Then read all blocks
        const size_t block_header_size =
            file_header.num_blocks * sizeof(CompactionV3TBlockHeader<compactionv3t::SUB_BLOCKS>);
        auto* block_headers = (CompactionV3TBlockHeader<compactionv3t::SUB_BLOCKS>*)malloc(block_header_size);
        memcpy(block_headers, src + in, block_header_size);
        in += block_header_size;

        const auto block_mem = (uint8_t*)malloc(compactionv3t::BLOCK_SIZE);

        // Then decode all blocks
        uint64_t out = 0;
        for (int block_id = 0; block_id < file_header.num_blocks; block_id++) {
            fsst::DecodingTable decoder = decoders[block_id / compactionv3t::SUPER_BLOCK_SIZE];
            const uint32_t block_size = block_headers[block_id].compressed_size;
            // memcpy(block_mem, src + in, block_size);
            const size_t transpose_size = transpose_memcpy_v3t(reinterpret_cast<uint32_t*>(block_mem),
                                                           reinterpret_cast<const uint32_t*>(src + in),
                                                           block_headers[block_id].flushes, compactionv3t::SUB_BLOCKS);
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
