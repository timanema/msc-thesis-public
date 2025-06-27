#ifndef PLAIN_ENCODER_HPP
#define PLAIN_ENCODER_HPP
#include <gtsst/gtsst-ell-table.cuh>
#include <gtsst/gtsst-tables.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
    struct PlainCompressor : CompressionManager {
        CompressionConfiguration configure_compression(size_t buf_size) override;
        GTSSTStatus compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src, uint8_t* tmp,
                             CompressionConfiguration& config, size_t* out_size, CompressionStatistics& stats) override;

        DecompressionConfiguration configure_decompression(size_t buf_size) override;
        GTSSTStatus decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                               size_t* out_size) override;

        GTSSTStatus validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                 CompressionConfiguration& config) override;
    };

    namespace plain {
#define TSST_WRITE_LEN(l, buf8)                                                                                \
{                                                                                                              \
uint32_t* buf32 = (uint32_t*)(buf8);                                                                           \
buf32[0] = l;                                                                                                  \
}
#define TSST_READ_LEN(buf8) (((uint32_t*)(buf8))[0])

        size_t small_compress(const symbols::PlainSymbolTableData& symbol_table, const uint8_t* src, uint8_t* dst,
                              uint32_t len);
    }
} // namespace gtsst::compressors

#endif // PLAIN_ENCODER_HPP
