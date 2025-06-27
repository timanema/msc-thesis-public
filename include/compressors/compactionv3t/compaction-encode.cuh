#ifndef COMPACTION_ENCODE_CUH
#define COMPACTION_ENCODE_CUH
#include <compressors/shared.cuh>
#include <cstdint>
#include <gtsst/gtsst-match-table.cuh>
#include "compaction-compressor.cuh"

namespace gtsst::compressors::compactionv3t {
    typedef Metadata<symbols::SmallSymbolMatchTableData> GCompactionMetadata;
    static_assert(std::is_assignable_v<GCompactionMetadata, Metadata<symbols::SmallSymbolMatchTableData>>);

    __global__ void gpu_compaction(GCompactionMetadata* metadata, CompactionV3TBlockHeader<SUB_BLOCKS>* headers,
                                   const uint8_t* src,
                                   uint8_t* dst);
} // namespace gtsst::compressors::compactionvt

#endif // COMPACTION_ENCODE_CUH
