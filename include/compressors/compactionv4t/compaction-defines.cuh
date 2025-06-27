#ifndef COMPACTION_DEFINES4t_CUH
#define COMPACTION_DEFINES4t_CUH
#include <cstdint>

namespace gtsst::compressors::compactionv4t {
    constexpr uint32_t THREAD_COUNT = 64;
    constexpr uint16_t SUB_BLOCKS = THREAD_COUNT / 32;
    static_assert(THREAD_COUNT % 32 == 0, "Thread count must fully utilize warp size");

    constexpr uint32_t tile_len = 10240; // Amount of input data per thread / tile
    constexpr uint32_t tile_word_size = sizeof(uint32_t); // Amount of symbols per input word
    constexpr uint32_t tile_word_buf_size = 8; // Amount of input words that are buffered (so also loaded in)
    static_assert(tile_len < std::numeric_limits<uint32_t>::max(), "uint32_t is used to specify input size");

    constexpr uint32_t tile_out_word_size = sizeof(uint32_t); // Amount of symbols per output word
    constexpr uint32_t tile_out_word_buf_size = 16; // Amount of output words that are buffered

    constexpr uint32_t n_words_per_tile = tile_len / tile_word_size; // Amount of symbols per word
    constexpr uint32_t n_words_per_block = tile_len * THREAD_COUNT / tile_word_size; // Amount of symbols per word
    constexpr uint32_t n_symbols_per_chunk = tile_word_buf_size * tile_word_size; // Amount of symbols read per chunk
    constexpr uint32_t n_regs_per_chunk = n_symbols_per_chunk / sizeof(uint32_t); // Amount of loads per chunk
    constexpr uint32_t n_chunks = n_words_per_tile / tile_word_buf_size; // Amount of chunks per tile
    static_assert(tile_len % tile_word_size == 0, "n_words_per_tile must be an integer");
    static_assert(n_words_per_tile % tile_word_buf_size == 0, "n_chunks must be an integer");
    static_assert(tile_len * THREAD_COUNT % tile_word_size == 0, "block must consist of perfect tiles");
    static_assert(n_symbols_per_chunk < std::numeric_limits<uint8_t>::max(),
                  "uint8_t is used to track input buffer location, either fix that or reduce max output buffer size");

    constexpr uint32_t tile_out_len = tile_len * 2; // Amount of output data per thread / tile
    constexpr uint32_t tile_out_len_words = tile_out_len / tile_out_word_size; // Amount of output words per tile
    constexpr uint32_t n_out_symbols_per_chunk =
        tile_out_word_buf_size * tile_out_word_size; // Amount of symbols (incl padding) per chunk
    static_assert(tile_out_len % n_out_symbols_per_chunk == 0,
                  "tile_out_len must be a multiple of n_symbols_per_chunk");
    static_assert(n_out_symbols_per_chunk * n_chunks <= tile_out_len,
                  "a tile must not be able to overwrite its own buffer");
    static_assert(n_out_symbols_per_chunk < std::numeric_limits<uint8_t>::max(),
                  "uint8_t is used to track output buffer location, either fix that or reduce max output buffer size");
    static_assert(tile_out_len_words < std::numeric_limits<uint32_t>::max(),
                  "uint32_t is used to track number of word flushes, either fix that or reduce max output buffer size");
    static_assert(tile_out_len < std::numeric_limits<uint32_t>::max(), "uint32_t is used to specify output size");
    static_assert(tile_out_len < std::numeric_limits<uint16_t>::max() * tile_out_word_size,
                  "uint16_t is used to specify the number of flushes, so there cannot be more data than "
                  "number_of_flushes * flush_word_size");

    constexpr uint32_t BLOCK_SIZE = tile_len * THREAD_COUNT;
    constexpr uint32_t WORD_ALIGNMENT = tile_word_size;
    constexpr uint32_t TMP_OUT_BLOCK_SIZE = tile_out_len * THREAD_COUNT; // Intermediate buffer of 1.5x input
    constexpr uint32_t TMP_OUT_WARP_SIZE = tile_out_len * 32;
    constexpr uint32_t TMP_WORD_ALIGNMENT = tile_out_word_size;
    static_assert((uint64_t)tile_len * THREAD_COUNT < std::numeric_limits<uint32_t>::max(),
                  "uint32_t is used to specify block size");
    static_assert((uint64_t)tile_out_len * THREAD_COUNT < std::numeric_limits<uint32_t>::max(),
                  "uint32_t is used to specify block size");

    constexpr uint32_t SUPER_BLOCK_SIZE = 48 * 2;

    static_assert(BLOCK_SIZE % WORD_ALIGNMENT == 0, "Block size must be a multiple of WORD_ALIGNMENT");
    static_assert(tile_len % WORD_ALIGNMENT == 0, "Tile len must be a multiple of WORD_ALIGNMENT");

    static_assert(TMP_OUT_BLOCK_SIZE % TMP_WORD_ALIGNMENT == 0,
                  "Block output must be a multiple of TMP_WORD_ALIGNMENT");
    static_assert(tile_out_len % TMP_WORD_ALIGNMENT == 0, "Tile out len must be a multiple of TMP_WORD_ALIGNMENT");
    static_assert(TMP_OUT_BLOCK_SIZE % tile_out_len == 0, "TMP_BLOCK_SIZE must be a multiple of tile_out_len");

    constexpr bool USE_TRANSPOSED_INPUT = false;
} // namespace gtsst::compressors::compactionv4t

#endif // COMPACTION_DEFINES4t_CUH
