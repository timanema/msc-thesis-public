#include <cassert>
#include <compressors/compactionv3/compaction-defines.cuh>
#include <compressors/compactionv3/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fsst/fsst-lib.cuh>
#include <gtsst/gtsst-symbols.cuh>

namespace gtsst::compressors::compactionv3 {
    struct EncodeResult {
        uint32_t bytes_written;
        uint32_t bytes_consumed;
        uint64_t spillover;
        uint8_t spillover_len;
    };

    __device__ void pack_results_local(uint32_t result[tile_out_word_buf_size][THREAD_COUNT], const uint32_t offset,
                                       const uint32_t val) {
        assert(offset < n_symbols_per_chunk);

        const uint32_t shift = (offset & 3) * 8;
        const uint8_t res = offset / 4;
        const uint32_t val_mask = val << shift;
        const uint32_t clean_mask = ~(0xFFU << shift);

        const uint32_t current = result[res][threadIdx.x];

        result[res][threadIdx.x] = current & clean_mask | val_mask;
    }

    __device__ void shift_registers_with_base(uint32_t load[tile_out_word_buf_size][THREAD_COUNT],
                                              const uint8_t bytes_used, uint8_t* first_block_offset,
                                              uint8_t* block_offset, uint32_t* first, uint32_t* second,
                                              uint32_t* third) {
        // Shift all blocks if needed
        const uint8_t shift = (*first_block_offset + bytes_used) / sizeof(uint32_t);
        for (int i = 0; i < shift; i++) {
            *block_offset = min(n_regs_per_chunk - 1, *block_offset + 1);
            *first = *second;
            *second = *third;
            *third = load[*block_offset][threadIdx.x];
        }

        *first_block_offset = (*first_block_offset + bytes_used) % sizeof(uint32_t);
    }

    __device__ EncodeResult compaction_encode_chunked_local(const symbols::SmallSymbolMatchTableData& symbol_table,
                                                            uint32_t result[tile_out_word_buf_size][THREAD_COUNT],
                                                            uint32_t load[tile_word_buf_size][THREAD_COUNT],
                                                            uint64_t spillover, const uint8_t spillover_len,
                                                            const bool last_chunk) {
        uint8_t out = 0;
        uint8_t idx = 0;

        uint32_t first_block = load[0][threadIdx.x];
        uint32_t second_block = load[1][threadIdx.x];
        uint32_t third_block = load[2][threadIdx.x];
        uint8_t first_block_offset = 0;
        uint8_t block_offset = 2;

        constexpr uint16_t search_len =
            n_regs_per_chunk * sizeof(uint32_t); // can look ahead at all available bytes (capped at 8 anyways)
        constexpr uint16_t encode_len = search_len - 7;
        // cannot encode when there are less than 8 bytes available, unless in last chunk

        // First handle spillover
        while (idx < spillover_len) {
            const symbols::GPUSymbol s = create_symbol_with_spillover_local(spillover, spillover_len - idx, first_block,
                                                                            second_block, fsst::Symbol::maxLength);
            uint16_t code = symbol_table.findLongestSymbol(s);
            uint8_t sym = (uint8_t)code;
            uint8_t sym_len = (uint8_t)(code >> 8);
            uint8_t escape = sym == 255;

            assert(sym != fsst::Symbol::ignore);
            pack_results_local(result, out, sym);
            if (escape)
                pack_results_local(result, out + 1, (uint8_t)get_first_byte_local(s.val.num));

            // Update pointers
            out += 1 + escape;
            idx += sym_len;

            // Bookkeeping of spillover data
            spillover >>= 8 * sym_len;
        }

        // Update registers for possible shift after overusage of spillover
        shift_registers_with_base(load, idx - spillover_len, &first_block_offset, &block_offset, &first_block,
                                  &second_block, &third_block);

        // Then handle regular block
        const uint8_t encode_range = last_chunk ? search_len : encode_len;
        while (idx < spillover_len + encode_range) {
            const symbols::GPUSymbol s =
                create_symbol_no_spillover_local(first_block, second_block, third_block, first_block_offset,
                                                 min((int)fsst::Symbol::maxLength, search_len + spillover_len - idx));
            uint16_t code = symbol_table.findLongestSymbol(s);
            uint8_t sym = (uint8_t)code;
            uint8_t sym_len = (uint8_t)(code >> 8);
            uint8_t escape = sym == 255;

            assert(sym != fsst::Symbol::ignore);
            pack_results_local(result, out, sym);
            if (escape)
                pack_results_local(result, out + 1, (uint8_t)get_first_byte_local(s.val.num));

            // Update pointers
            out += 1 + escape;
            idx += sym_len;

            shift_registers_with_base(load, sym_len, &first_block_offset, &block_offset, &first_block, &second_block,
                                      &third_block);
        }

        // Then create new spillover
        const uint8_t spilled_bytes = search_len + spillover_len - idx;
        const symbols::GPUSymbol s =
            create_symbol_no_spillover_local(first_block, second_block, third_block, first_block_offset, spilled_bytes);

        return EncodeResult{out, idx, s.val.num, spilled_bytes};
    }

    __global__ void gpu_compaction(GCompactionMetadata* metadata, BlockHeader* headers, const uint8_t* src,
                                   uint8_t* dst) {
        __shared__ uint32_t global_size[THREAD_COUNT];
        __shared__ uint32_t load[tile_word_buf_size][THREAD_COUNT];
        __shared__ uint32_t result[tile_out_word_buf_size][THREAD_COUNT];

        // Load metadata into shared mem
        __shared__ GCompactionMetadata m;
        load_metadata_local<GCompactionMetadata>(metadata, (uint32_t*)&m, SUPER_BLOCK_SIZE);

        // Checks
        assert((uintptr_t)src % WORD_ALIGNMENT == 0); // Ensure 8-byte alignment of source
        assert((uintptr_t)dst % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment of dest
        assert(m.symbol_table.singleCodes[fsst::Symbol::ignore].code() ==
            fsst::Symbol::ignore); // Ensure that ignore symbol has been properly mapped

        // Active data
        uint64_t spillover = 0;
        uint8_t spillover_len = 0;
        global_size[threadIdx.x] = 0;

        const auto aligned_src = (uint32_t*)src;
        const auto dst_aligned = (uint32_t*)dst;

        for (uint32_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
            // Reset result
            for (int i = 0; i < tile_out_word_buf_size; i++) {
                result[i][threadIdx.x] = 0xFEFEFEFE;
            }

            // Step 1: Coalesced load into working memory
            for (int i = 0; i < tile_word_buf_size; i++) {
                load[i][threadIdx.x] = aligned_src[(chunk_id * tile_word_buf_size + i) * (uint64_t)blockDim.x * gridDim.
                    x + threadIdx.x + blockIdx.x * blockDim.x];
            }

            // Step 2: Run chunked compression on spillover and loaded window
            const auto encode_result = compaction_encode_chunked_local(m.symbol_table, result, load, spillover,
                                                                       spillover_len, chunk_id == n_chunks - 1);

            // Step 3: Update spillover
            spillover = encode_result.spillover;
            spillover_len = encode_result.spillover_len;
            global_size[threadIdx.x] += encode_result.bytes_written;

            assert(encode_result.bytes_written < n_symbols_per_chunk + 1); // Check for buffer overrun

            // Step 4: Coordinated output
            for (int i = 0; i < tile_out_word_buf_size; i++) {
                // dst_aligned[(chunk_id * tile_out_word_buf_size + i) * 64 + threadIdx.x] = result[i][threadIdx.x];
                dst_aligned[(chunk_id * tile_out_word_buf_size + i) * (uint64_t)blockDim.x * gridDim.
                    x + threadIdx.x + blockIdx.x * blockDim.x] = result[i][threadIdx.x];
            }

            // Check for overruns
            assert(threadIdx.x * tile_out_len + chunk_id * n_symbols_per_chunk + n_symbols_per_chunk - 1 <
                (threadIdx.x + 1) * tile_out_len); // tile overruns
            assert(threadIdx.x * tile_out_len + chunk_id * n_symbols_per_chunk + n_symbols_per_chunk - 1 <
                TMP_OUT_BLOCK_SIZE); // block overruns
        }

        assert(global_size[threadIdx.x] > 0);

        __syncthreads();

        // Sum block output
        if (threadIdx.x == THREAD_COUNT - 1) {
            uint32_t block_out = 0;

            for (int i = 0; i < THREAD_COUNT; i++) {
                block_out += global_size[i];
            }

            const BlockHeader header = {.uncompressed_size =
                                        BLOCK_SIZE, // TODO: change this later to support padding in input
                                        .compressed_size = block_out};
            headers[blockIdx.x] = header;
        }
    }
} // namespace gtsst::compressors::compactionv3
