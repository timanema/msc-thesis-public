#ifndef COMPACTION_COMPRESSOR5t_CUH
#define COMPACTION_COMPRESSOR5t_CUH
#include <compressors/shared.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
  struct CompactionV5TBlockHeader : BlockHeader {
    uint16_t flushes;
  };

  struct CompactionV5TFileHeader : FileHeader {
    uint16_t num_sub_blocks;
  };

  struct CompactionV5TCompressor : CompressionManager {
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

#endif // COMPACTION_COMPRESSOR5t_CUH
