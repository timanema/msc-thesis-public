#ifndef COMPACTION_ENCODE_CUH
#define COMPACTION_ENCODE_CUH
#include <compressors/shared.cuh>
#include <cstdint>
#include <gtsst/gtsst-match-table.cuh>

namespace gtsst::compressors::compactionv3 {
    typedef Metadata<symbols::SmallSymbolMatchTableData> GCompactionMetadata;
    static_assert(std::is_assignable_v<GCompactionMetadata, Metadata<symbols::SmallSymbolMatchTableData>>);

    __global__ inline void trans(const uint8_t* src, uint8_t* dst) {
        if (threadIdx.x == 0) {
            const auto aligned_transpose_src = (uint32_t*)(src + (uint64_t)blockIdx.x * BLOCK_SIZE);
            const auto aligned_transpose_dst = (uint32_t*)(dst + (uint64_t)blockIdx.x * BLOCK_SIZE);

            dim3 dimGrid(n_words_per_tile / 32, 64 / 32, 1);
            dim3 dimBlock(32, 8, 1);
            transpose_no_bank_conflicts<32, 8><<<dimGrid, dimBlock>>>(aligned_transpose_dst, aligned_transpose_src);

            // Check that there were no launch errors
            assert(cudaSuccess == cudaGetLastError());
        }
    }

    __global__ void gpu_compaction(GCompactionMetadata* metadata, BlockHeader* headers, const uint8_t* src,
                                   uint8_t* dst);
} // namespace gtsst::compressors::compactionv3

#endif // COMPACTION_ENCODE_CUH
