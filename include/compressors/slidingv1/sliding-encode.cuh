#ifndef SLIDING_ENCODE_CUH
#define SLIDING_ENCODE_CUH
#include <compressors/shared.cuh>
#include <gtsst/gtsst-sliding-table.cuh>

namespace gtsst::compressors::slidingv1 {
    typedef Metadata<symbols::SymbolSlidingTableData> SlidingMetadata;

    __global__ void gpu_sliding_encode(SlidingMetadata* metadata, BlockHeader* headers, const uint8_t* src,
                                       uint8_t* dst);
} // namespace gtsst::compressors::slidingv1

#endif // SLIDING_ENCODE_CUH
