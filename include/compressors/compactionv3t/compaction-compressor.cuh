#ifndef COMPACTION_COMPRESSOR3t_CUH
#define COMPACTION_COMPRESSOR3t_CUH
#include <compressors/shared.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
  template <uint16_t n_sub_blocks>
  struct CompactionV3TBlockHeader : BlockHeader {
    uint16_t flushes[n_sub_blocks];
  };

  struct CompactionV3TFileHeader : FileHeader {
    uint16_t num_sub_blocks;
  };

  struct CompactionV3TCompressor : CompressionManager {
    CompressionConfiguration configure_compression(size_t buf_size) override;
    GTSSTStatus compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src, uint8_t* tmp,
                         CompressionConfiguration& config, size_t* out_size, CompressionStatistics& stats) override;

    DecompressionConfiguration configure_decompression(size_t buf_size) override;
    GTSSTStatus decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                           size_t* out_size) override;

    GTSSTStatus validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                             CompressionConfiguration& config) override;
  };
} // namespace gtsst::compressors

#endif // COMPACTION_COMPRESSOR3t_CUH
