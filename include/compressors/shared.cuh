#ifndef SHARED_CUH
#define SHARED_CUH
#include <fsst/gtsst-fsst.cuh>
#include <gtsst/gtsst-symbols.cuh>
#include <gtsst/gtsst.cuh>

namespace gtsst::compressors {
#define safeCUDACall(ans)                                                                                              \
    if (auto res = gtsst::compressors::gpuAssert((ans), __FILE__, __LINE__); res != gtsstSuccess) {                    \
        return res;                                                                                                    \
    }

    inline GTSSTStatus gpuAssert(const cudaError_t code, const char* file, const int line) {
        if (code != cudaSuccess) {
            fprintf(stdout, "CUDA error: %s (%s) %s %d\n", cudaGetErrorString(code), cudaGetErrorName(code), file,
                    line);
            return gtsstErrorCudaError;
        }

        return gtsstSuccess;
    }

    struct BlockHeader {
        uint32_t uncompressed_size;
        uint32_t compressed_size;
    };

    struct FileHeader {
        uint64_t compressed_size; // Size of compressed data + tables + block headers + sizeof(FileHeader)
        uint64_t uncompressed_size; // Size of uncompressed data
        uint32_t num_tables;
        uint32_t table_size; // Size of tables
        uint32_t num_blocks;
    };

    /*
     * <FileHeader><Tables[num_tables]><BlockHeader[num_blocks]><Data[num_blocks]>
     *
     * assert(table_size == sizeof(Tables[num_tables]))
     * assert(compressed_size == sizeof(FileHeader) + table_size + sum(sizeof(BlockHeader)) +
     *                              sum(BlockHeader.compressed_size))
     */

    struct GBaseHeader {
        uint8_t decoding_table[fsst::SymbolTable::maxSize * 8 + 8 + 8 + 1]{}; // Size = maxSize * 8 + 8 + 8 + 1
        //      = tokens + histogram + version + zero termination
    };

    template <typename Encoding>
        requires std::constructible_from<Encoding, fsst::SymbolTable&> && (alignof(Encoding) == 4)
    struct alignas(4) Metadata {
        Encoding symbol_table;
        uint16_t header_offset;
    };

    template <typename Encoding>
        requires std::constructible_from<Encoding, fsst::SymbolTable&>
    void gpu_create_metadata(uint32_t block_id, const uint32_t block_size, Metadata<Encoding>* metadata,
                             GBaseHeader* headers, const uint8_t* src, const uint64_t max_len) {
        // Create symbol table
        Metadata<Encoding>& m = metadata[block_id];

        // Reading location is src + i * block_size
        const uint8_t* src_buf = src + block_id * (uint64_t)block_size;

        // Create sample
        auto* sample_buf = new uint8_t[FSST_SAMPLEMAXSZ];
        const size_t sample_len = fsst::simple_make_sample(
            sample_buf, src_buf, min((uint64_t)block_size, max_len - block_id * (uint64_t)block_size));

        // Create symbol table
        const auto enc_table = fsst::build_symbol_table<Encoding>(sample_buf, sample_len);
        m.symbol_table = enc_table->encoding_data;

        // Export symbol table
        const size_t stLen = enc_table->export_table(headers[block_id].decoding_table);
        m.header_offset = stLen;

        delete[] sample_buf;
        delete enc_table;
    }

    struct is_not_ignore {
        __host__ __device__

        bool
        operator()(const uint8_t x) const {
            return x != fsst::Symbol::ignore;
        }
    };

    __device__ inline uint64_t get_first_n_bytes_local(const uint64_t val, const uint8_t n) {
        return val * (n > 0) & 0xFFFFFFFFFFFFFFFF >> (8 - n) * 8;
    }

    __device__ inline uint64_t get_first_byte_local(const uint64_t val) {
        return val & 0xFF; // Special case of get_first_n_bytes (n=1)
    }

    __device__ inline uint64_t get_last_n_bytes_local(const uint64_t val, const uint8_t n) {
        return val * (n > 0) >> (4 - n) * 8 & 0xFFFFFFFFFFFFFFFF >> (8 - n) * 8;
    }

    template <typename T>
        requires(alignof(T) == alignof(uint32_t))
    __device__ void load_metadata_local(T* metadata, uint32_t* smem_target, const uint32_t super_block_size) {
        const uint32_t* m = (uint32_t*)&metadata[blockIdx.x / super_block_size];

        for (uint i = threadIdx.x; i < sizeof(T) / sizeof(uint32_t); i += blockDim.x) {
            smem_target[i] = m[i];
        }

        // Symbol table needs to be in shared memory before we can actually start with encoding
        __syncthreads();
    }

    size_t seq_decompress(const fsst::DecodingTable& dec, const uint8_t* src, uint8_t* dst, uint32_t len);

    inline bool data_contains(const uint8_t* src, const uint8_t b, const size_t size) {
        for (size_t i = 0; i < size; i++) {
            if (src[i] == b) {
                return true;
            }
        }

        return false;
    }

    __device__ inline uint8_t get_block_shift(const uint8_t bytes_used, const uint8_t offset) {
        return (offset + bytes_used) / 4;
    }

    __device__ inline symbols::GPUSymbol shift_symbol_once_local(const symbols::GPUSymbol previous,
                                                                 const uint32_t next_block, const uint8_t offset) {
        assert(offset <= 3);
        assert(previous.length() == fsst::Symbol::maxLength);

        const uint64_t next_data = get_first_byte_local(next_block >> offset * 8);

        return symbols::GPUSymbol(previous.val.num >> 8 | next_data << 7 * 8, previous.length());
    }

    __device__ inline symbols::GPUSymbol shift_symbol_once_partial_local(const symbols::GPUSymbol previous) {
        return symbols::GPUSymbol(previous.val.num >> 8, previous.length() - 1); // Take one symbol
    }

    __device__ inline symbols::GPUSymbol
    create_symbol_with_spillover_local(const uint64_t spillover, const uint8_t spillover_len,
                                       const uint32_t first_block, const uint32_t second_block, const uint8_t len) {
        assert(len <= 8);
        assert(spillover_len <= 8);
        // assert(spillover_len > 0);
        // Spillover cannot be not used, otherwise we shift an uint64_t 64 bits which is undefined behaviour

        const uint8_t bytes_from_spillover = ::min(spillover_len, len);
        const uint8_t bytes_from_first = ::min(len - bytes_from_spillover, 4);
        const uint8_t bytes_from_second = ::min(len - (bytes_from_spillover + bytes_from_first), 4);

        const uint64_t spillover_data = get_first_n_bytes_local(spillover, bytes_from_spillover);
        const uint64_t first_data = get_first_n_bytes_local(first_block, bytes_from_first);
        const uint64_t second_data = get_first_n_bytes_local(second_block, bytes_from_second);

        const uint64_t data = spillover_data | first_data << bytes_from_spillover * 8 |
            second_data << (bytes_from_spillover + bytes_from_first) * 8;

        return symbols::GPUSymbol(data, len);
    }

    __device__ inline symbols::GPUSymbol create_symbol_no_spillover_local(const uint32_t first_block,
                                                                          const uint32_t second_block,
                                                                          const uint32_t third_block,
                                                                          const uint8_t offset, const uint8_t len) {
        assert(len <= 8);
        assert(offset <= 3);

        const uint8_t bytes_from_first = ::min((int)len, 4 - offset);
        const uint8_t bytes_from_second = ::min(len - bytes_from_first, 4);
        const uint8_t bytes_from_third = ::min(len - (bytes_from_first + bytes_from_second), (int)offset);

        const uint64_t first_data = get_first_n_bytes_local(first_block >> offset * 8, bytes_from_first);
        const uint64_t second_data = get_first_n_bytes_local(second_block, bytes_from_second);
        const uint64_t third_data = get_first_n_bytes_local(third_block, bytes_from_third);

        const uint64_t data =
            first_data | second_data << bytes_from_first * 8 | third_data << (bytes_from_first + bytes_from_second) * 8;

        return symbols::GPUSymbol(data, len);
    }

    template <uint64_t tile_dim, uint64_t block_rows>
    __global__ void transpose_no_bank_conflicts(uint32_t* odata, const uint32_t* idata) {
        __shared__ uint32_t tile[tile_dim][tile_dim + 1];

        uint64_t x = blockIdx.x * tile_dim + threadIdx.x;
        uint64_t y = blockIdx.y * tile_dim + threadIdx.y;
        const uint64_t widthIn = gridDim.x * tile_dim;
        const uint64_t widthOut = gridDim.y * tile_dim;

        for (int j = 0; j < tile_dim; j += block_rows)
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * widthIn + x];

        __syncthreads();

        x = blockIdx.y * tile_dim + threadIdx.x; // transpose block offset
        y = blockIdx.x * tile_dim + threadIdx.y;

        for (int j = 0; j < tile_dim; j += block_rows)
            odata[(y + j) * widthOut + x] = tile[threadIdx.x][threadIdx.y + j];
    }
} // namespace gtsst::compressors

#endif // SHARED_CUH
