#ifndef COMPACTION_DEFINES2_CUH
#define COMPACTION_DEFINES2_CUH
#include <cstdint>

namespace gtsst::compressors::compactionv2 {
    constexpr uint32_t THREAD_COUNT = 96;
    static_assert(THREAD_COUNT % 32 == 0, "Thread count must fully utilize warp size");

    constexpr uint32_t tile_len = 10240; // Amount of input data per thread / tile
    constexpr uint32_t tile_word_size = sizeof(uint64_t); // Amount of symbols per word
    constexpr uint32_t n_words_per_tile = tile_len / tile_word_size; // Amount of symbols per word
    constexpr uint32_t n_words_per_chunk = 2; // Amount of load cycles per chunk
    constexpr uint32_t n_regs_per_chunk =
        n_words_per_chunk * tile_word_size / sizeof(uint32_t); // Amount of loads per chunk
    constexpr uint32_t n_chunks = n_words_per_tile / n_words_per_chunk; // Amount of chunks per tile
    static_assert(tile_len % tile_word_size == 0, "n_words_per_tile must be an integer");
    static_assert(n_words_per_tile % n_words_per_chunk == 0, "n_chunks must be an integer");

    constexpr uint32_t tile_out_len = tile_len * 3; // Amount of output data per thread / tile
    constexpr uint32_t tile_out_word_size = sizeof(uint32_t); // Amount of symbols per word
    constexpr uint32_t tile_out_len_words = tile_out_len / tile_out_word_size; // Amount of output words per iteration
    constexpr uint32_t n_stores_per_chunk = 12; // Amount of stored per chunk
    constexpr uint32_t n_symbols_per_chunk =
        n_stores_per_chunk * tile_out_word_size; // Amount of symbols (incl padding) per chunk
    static_assert(tile_out_len % n_symbols_per_chunk == 0, "tile_out_len must be a multiple of n_symbols_per_chunk");
    static_assert(tile_out_len / n_symbols_per_chunk == n_chunks,
                  "tile_out_len must be fully written to with n_symbols_per_chunk");

    constexpr uint32_t BLOCK_SIZE = tile_len * THREAD_COUNT;
    constexpr uint32_t WORD_ALIGNMENT = tile_word_size;
    constexpr uint32_t TMP_OUT_BLOCK_SIZE = BLOCK_SIZE * 3; // Intermediate buffer of 1.5x input
    constexpr uint32_t TMP_WORD_ALIGNMENT = tile_out_word_size;

    constexpr uint32_t SUPER_BLOCK_SIZE = 48;

    static_assert(BLOCK_SIZE % WORD_ALIGNMENT == 0, "Block size must be a multiple of WORD_ALIGNMENT");
    static_assert(tile_len % WORD_ALIGNMENT == 0, "Tile len must be a multiple of WORD_ALIGNMENT");

    static_assert(TMP_OUT_BLOCK_SIZE % TMP_WORD_ALIGNMENT == 0,
                  "Block output must be a multiple of TMP_WORD_ALIGNMENT");
    static_assert(tile_out_len % TMP_WORD_ALIGNMENT == 0, "Tile out len must be a multiple of TMP_WORD_ALIGNMENT");
    static_assert(TMP_OUT_BLOCK_SIZE % tile_out_len == 0, "TMP_BLOCK_SIZE must be a multiple of tile_out_len");
} // namespace gtsst::compressors::compactionv2

#endif // COMPACTION_DEFINES2_CUH
