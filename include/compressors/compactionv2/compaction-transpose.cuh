#ifndef COMPACTION_TRANSPOSE_CUH
#define COMPACTION_TRANSPOSE_CUH
#include <cassert>
#include <cstdint>

namespace gtsst::compressors::compactionv2 {
    template <typename word_type, uint32_t block_size, uint32_t n_words_per_tile>
    __global__ void basic_transpose(const uint8_t* src, uint8_t* dst) {
        assert((uintptr_t)src % sizeof(word_type) == 0); // Ensure 8-byte alignment of source
        assert((uintptr_t)dst % sizeof(word_type) == 0); // Ensure 8-byte alignment of destination

        // Transpose block while taking into account the chunks
        const word_type* aligned_src = (word_type*)(src + blockIdx.x * (uint64_t)block_size);
        word_type* aligned_dst = (word_type*)(dst + blockIdx.x * (uint64_t)block_size);

        for (int i = 0; i < n_words_per_tile; i++) {
            const word_type word = aligned_src[i + threadIdx.x * n_words_per_tile];
            aligned_dst[i * blockDim.x + threadIdx.x] = word;
        }
    }

    template <typename word_type, uint32_t block_size, uint32_t n_words_per_tile, uint32_t thread_count>
    __global__ void shared_transpose(const uint8_t* src, uint8_t* dst) {
        assert((uintptr_t)src % sizeof(word_type) == 0); // Ensure 8-byte alignment of source
        assert((uintptr_t)dst % sizeof(word_type) == 0); // Ensure 8-byte alignment of destination

        // This transpose doesn't really work when there is less than 32 words per thread/tile
        assert(n_words_per_tile >= 32);

        // Transpose block while taking into account the chunks
        const word_type* aligned_src = (word_type*)(src + blockIdx.x * (uint64_t)block_size);
        word_type* aligned_dst = (word_type*)(dst + blockIdx.x * (uint64_t)block_size);

        __shared__ word_type shared_data[32][32 + 1];

        for (int i = 0; i < n_words_per_tile; i += 32) {
            for (int warp = 0; warp < thread_count / 32; warp++) {
                // First load in rows
                for (int j = 0; j < 32; j++) {
                    const int read_thread = warp * 32 + j;
                    word_type c = aligned_src[read_thread * n_words_per_tile + i + threadIdx.x];
                    shared_data[j][threadIdx.x] = c;
                }

                // Then write output
                for (int j = 0; j < 32; j++) {
                    aligned_dst[(i + j) * thread_count + warp * 32 + threadIdx.x] = shared_data[threadIdx.x][j];
                }
            }
        }
    }
} // namespace gtsst::compressors::compactionv2

#endif // COMPACTION_TRANSPOSE_CUH
