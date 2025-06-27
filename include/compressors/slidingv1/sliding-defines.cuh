#ifndef SLIDING_DEFINES_CUH
#define SLIDING_DEFINES_CUH
#include <cstdint>

namespace gtsst::compressors::slidingv1 {
    constexpr uint32_t TILE_SIZE = 32; // One tile == one warp
    constexpr uint32_t THREAD_COUNT = 96;
    constexpr uint32_t TILE_COUNT = THREAD_COUNT / TILE_SIZE; // Number of tiles == warps
    static_assert(THREAD_COUNT % TILE_SIZE == 0, "Thread count must fully utilize warp/tile size");

    constexpr uint32_t tile_len = 4096 * 8; // 10240 * 8 * 4; // Amount of input data per thread / tile
    constexpr uint32_t tile_word_size = sizeof(uint32_t); // Amount of bytes per word load
    constexpr uint32_t tile_chunk_size =
        TILE_SIZE * tile_word_size; // Amount of bytes per load cycle that are loaded per warp
    constexpr uint32_t n_chunks = tile_len / tile_chunk_size; // Amount of load cycles
    static_assert(tile_len % tile_chunk_size == 0, "data within a tile must perfectly split in chunks");

    constexpr uint32_t tile_out_word_size = sizeof(uint32_t); // Amount of bytes per word store
    constexpr uint32_t tile_out_chunk_size =
        TILE_SIZE * tile_out_word_size; // Amount of bytes per store cycle that are stored per warp
    constexpr uint32_t max_out_chunks = tile_len / tile_out_chunk_size; // Maximum amount of store cycles allowed
    static_assert(max_out_chunks * tile_out_chunk_size <= tile_len, "cannot store more data than what was loaded");

    constexpr uint32_t BLOCK_SIZE = tile_len * TILE_COUNT; // 983040
    constexpr uint32_t WORD_ALIGNMENT = tile_word_size;
    static_assert(BLOCK_SIZE % WORD_ALIGNMENT == 0, "Block size must be a multiple of WORD_ALIGNMENT");
    static_assert(tile_len % WORD_ALIGNMENT == 0, "Tile len must be a multiple of WORD_ALIGNMENT");

    constexpr uint32_t TMP_WORD_ALIGNMENT = tile_out_word_size;
    static_assert(WORD_ALIGNMENT % TMP_WORD_ALIGNMENT == 0, "output must be a smaller or equal alignment than input");

    constexpr uint32_t SUPER_BLOCK_SIZE = 24;
} // namespace gtsst::compressors::slidingv1

#endif // SLIDING_DEFINES_CUH
