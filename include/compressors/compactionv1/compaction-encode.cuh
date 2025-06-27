#ifndef COMPACTION_ENCODE_CUH
#define COMPACTION_ENCODE_CUH
#include <compressors/shared.cuh>
#include <cstdint>
#include <gtsst/gtsst-match-table.cuh>

namespace gtsst::compressors::compactionv1 {
    typedef Metadata<symbols::SmallSymbolMatchTableData> GCompactionMetadata;
    static_assert(std::is_assignable_v<GCompactionMetadata, Metadata<symbols::SmallSymbolMatchTableData>>);

    __global__ void gpu_compaction(GCompactionMetadata* metadata, BlockHeader* headers, const uint8_t* src,
                                   uint8_t* dst);
} // namespace gtsst::compressors::compactionv1

#endif // COMPACTION_ENCODE_CUH
