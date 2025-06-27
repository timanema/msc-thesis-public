#ifndef STATS_ENCODER_HPP
#define STATS_ENCODER_HPP
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
    struct StatsCompressor : CompressionManager {
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

#endif // STATS_ENCODER_HPP
