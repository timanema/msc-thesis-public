#include <cassert>
#include <compressors/compactionv2/compaction-defines.cuh>
#include <compressors/compactionv2/compaction-encode.cuh>
#include <compressors/shared.cuh>
#include <fsst/fsst-lib.cuh>
#include <gtsst/gtsst-symbols.cuh>

namespace gtsst::compressors::compactionv2 {
    struct EncodeResult {
        uint32_t bytes_written;
        uint32_t bytes_consumed;
        uint64_t spillover;
        uint8_t spillover_len;
    };

    __device__ void pack_results_local(uint32_t result[n_stores_per_chunk][THREAD_COUNT], const uint32_t offset,
                                       const uint32_t val) {
        assert(offset < n_stores_per_chunk * tile_out_word_size);
        // Ensure the offset is within the limits of n_stores_per_chunk

        const uint32_t shift = (offset & 3) * 8;
        const uint8_t res = offset / 4;
        const uint32_t val_mask = val << shift;
        const uint32_t clean_mask = ~(0xFFU << shift);

        const uint32_t current = result[res][threadIdx.x];

        result[res][threadIdx.x] = current & clean_mask | val_mask;
    }

    __device__ void shift_registers(uint32_t load[n_stores_per_chunk][THREAD_COUNT], const uint8_t bytes_used,
                                    uint8_t* first_block_offset, uint32_t* first, uint32_t* second, uint32_t* third) {
        // Shift all blocks if needed
        const uint8_t shift = get_block_shift(*first_block_offset, bytes_used);
        *first = load[shift][threadIdx.x];
        *second = load[shift + 1][threadIdx.x];
        *third = load[::min(n_regs_per_chunk - 1, shift + 2)][threadIdx.x];

        *first_block_offset = (*first_block_offset + bytes_used) % sizeof(uint32_t);
    }

    __device__ void shift_registers_with_base(uint32_t load[n_stores_per_chunk][THREAD_COUNT], const uint8_t bytes_used,
                                              uint8_t* first_block_offset, uint8_t* block_offset, uint32_t* first,
                                              uint32_t* second, uint32_t* third) {
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

    __device__ EncodeResult compaction_encode_chunked_shift_local(
        const symbols::SmallSymbolMatchTableData& symbol_table, uint32_t result[n_stores_per_chunk][THREAD_COUNT],
        uint32_t load[n_stores_per_chunk][THREAD_COUNT], const uint64_t spillover, const uint8_t spillover_len,
        const bool last_chunk) {
        uint8_t out = 0;
        uint8_t idx = 0;

        uint32_t first_block = load[0][threadIdx.x];
        uint32_t second_block = load[1][threadIdx.x];
        uint32_t third_block = 0;
        uint8_t first_block_offset = 0;

        constexpr uint16_t search_len =
            n_regs_per_chunk * sizeof(uint32_t); // can look ahead at all available bytes (capped at 8 anyways)
        constexpr uint16_t encode_len = search_len - 7;
        // cannot encode when there are less than 8 bytes available, unless in last chunk

        // Create base symbol
        symbols::GPUSymbol base_symbol = create_symbol_with_spillover_local(spillover, spillover_len, first_block,
                                                                            second_block, fsst::Symbol::maxLength);

        // Account for the block shift (if any) when spillover is less than 8 (aka we need 1 or more bytes from the data
        // blocks)
        shift_registers(load, fsst::Symbol::maxLength - spillover_len, &first_block_offset, &first_block, &second_block,
                        &third_block);

        // First handle spillover
        for (uint16_t i = 0; i < spillover_len; i++) {
            // If idx == i, there is new info to be gained! So do a lookup
            if (idx == i) {
                const uint16_t code = symbol_table.findLongestSymbol(base_symbol);
                const auto sym = (uint8_t)code;
                const auto sym_len = (uint8_t)(code >> 8);
                const uint8_t escape = sym == fsst::Symbol::escape;
                const uint8_t ignore = sym == fsst::Symbol::ignore;

                assert(sym != fsst::Symbol::ignore);
                pack_results_local(result, out, sym);
                if (escape)
                    pack_results_local(result, out + 1, (uint8_t)get_first_byte_local(base_symbol.val.num));

                // Update pointers
                out += 1 + escape - ignore;
                idx += sym_len;
            }

            // Bookkeeping of data
            base_symbol = shift_symbol_once_local(base_symbol, first_block, first_block_offset);
            first_block_offset += 1;
            first_block_offset %= n_regs_per_chunk;

            // If first block is used, shift blocks
            if (first_block_offset == 0) {
                first_block = second_block;
                second_block = third_block;
                third_block = load[n_regs_per_chunk - 1][threadIdx.x];
            }
        }

        // Then handle regular block
#pragma unroll
        for (uint16_t i = 0; i < encode_len; i++) {
            // If idx == i, there is new info to be gained! So do a lookup
            if (idx == i + spillover_len) {
                const uint16_t code = symbol_table.findLongestSymbol(base_symbol);
                const auto sym = (uint8_t)code;
                const auto sym_len = (uint8_t)(code >> 8);
                const uint8_t escape = sym == fsst::Symbol::escape;
                const uint8_t ignore = sym == fsst::Symbol::ignore;

                assert(sym != fsst::Symbol::ignore);
                pack_results_local(result, out, sym);
                if (escape)
                    pack_results_local(result, out + 1, (uint8_t)get_first_byte_local(base_symbol.val.num));

                // Update pointers
                out += 1 + escape - ignore;
                idx += sym_len;
            }

            // Bookkeeping of data
            if (i < encode_len - 1) {
                base_symbol = shift_symbol_once_local(base_symbol, first_block, first_block_offset);
            } else if (last_chunk) {
                base_symbol = shift_symbol_once_partial_local(base_symbol);
            }

            first_block_offset += 1;
            first_block_offset %= n_regs_per_chunk;

            // If first block is used, shift blocks
            if (first_block_offset == 0) {
                first_block = second_block;
                second_block = third_block;
                third_block = load[n_regs_per_chunk - 1][threadIdx.x];
            }
        }

        // Then handle final block if last chunk
        if (last_chunk) {
            for (uint16_t i = 0; i < search_len - encode_len; i++) {
                // If idx == i, there is new info to be gained! So do a lookup
                if (idx == i + spillover_len + encode_len) {
                    const uint16_t code = symbol_table.findLongestSymbol(base_symbol);
                    const auto sym = (uint8_t)code;
                    const auto sym_len = (uint8_t)(code >> 8);
                    const uint8_t escape = sym == fsst::Symbol::escape;
                    const uint8_t ignore = sym == fsst::Symbol::ignore;

                    assert(sym != fsst::Symbol::ignore);
                    pack_results_local(result, out, sym);
                    if (escape)
                        pack_results_local(result, out + 1, (uint8_t)get_first_byte_local(base_symbol.val.num));

                    // Update pointers
                    out += 1 + escape - ignore;
                    idx += sym_len;
                }

                // Bookkeeping of data
                base_symbol = shift_symbol_once_partial_local(base_symbol);
            }
        }

        // Create new spillover
        const uint8_t spilled_bytes = search_len + spillover_len - idx;
        const uint64_t spilled_data = base_symbol.val.num >> ((8 - spilled_bytes) * 8);
        assert(spilled_bytes == 0 || !last_chunk);

        return EncodeResult{out, idx, spilled_data, spilled_bytes};
    }

    __device__ EncodeResult compaction_encode_chunked_local(const symbols::SmallSymbolMatchTableData& symbol_table,
                                                            uint32_t result[n_stores_per_chunk][THREAD_COUNT],
                                                            uint32_t load[4][THREAD_COUNT],
                                                            uint64_t spillover, const uint8_t spillover_len,
                                                            const bool last_chunk) {
        uint8_t out = 0;
        uint8_t idx = 0;

        uint32_t first_block = load[0][threadIdx.x];
        uint32_t second_block = load[1][threadIdx.x];
        uint32_t third_block = load[2][threadIdx.x];
        uint8_t first_block_offset = 0;
        uint8_t block_offset = n_regs_per_chunk - 1;

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

#define USE_SCALAR_PREFETCH 0
#define USE_SHIFT_ENCODE 0

    __global__ void gpu_compaction(GCompactionMetadata* metadata, BlockHeader* headers, const uint8_t* src,
                                   uint8_t* dst) {
        __shared__ uint32_t global_size[THREAD_COUNT];
        __shared__ uint32_t result[n_stores_per_chunk][THREAD_COUNT];
        __shared__ uint32_t input[4][THREAD_COUNT];

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

        const auto aligned_src = (uint64_t*)(src + BLOCK_SIZE * ((uint64_t)blockIdx.x));

#if USE_SCALAR_PREFETCH == 1
        uint64_t load1 = aligned_src[(0 * 2 + 0) * THREAD_COUNT + threadIdx.x];
        uint64_t load2 = aligned_src[(0 * 2 + 1) * THREAD_COUNT + threadIdx.x];
        uint64_t load3 = 0;
        uint64_t load4 = 0;
#endif

        for (uint32_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
            // Step 1: Coalesced load into working memory
#if USE_SCALAR_PREFETCH == 1
            if (chunk_id < n_chunks - 1) {
                load3 = aligned_src[(chunk_id * 2 + 0 + 2) * THREAD_COUNT + threadIdx.x];
                load4 = aligned_src[(chunk_id * 2 + 1 + 2) * THREAD_COUNT + threadIdx.x];
            }
#else
            uint64_t load1 = aligned_src[(chunk_id * 2 + 0) * THREAD_COUNT + threadIdx.x];
            uint64_t load2 = aligned_src[(chunk_id * 2 + 1) * THREAD_COUNT + threadIdx.x];
#endif
            input[0][threadIdx.x] = (uint32_t)load1;
            input[1][threadIdx.x] = (uint32_t)(load1 >> 32);
            input[2][threadIdx.x] = (uint32_t)load2;
            input[3][threadIdx.x] = (uint32_t)(load2 >> 32);

            // Clear result
            for (int i = 0; i < n_stores_per_chunk; i++) {
                result[i][threadIdx.x] = 0xFEFEFEFE;
            }

            // Step 2: Run chunked compression on spillover and loaded window
#if USE_SHIFT_ENCODE == 1
            const auto encode_result = compaction_encode_chunked_shift_local(m.symbol_table, result, result, spillover,
                                                                             spillover_len, chunk_id == n_chunks - 1);
#else
            const auto encode_result = compaction_encode_chunked_local(m.symbol_table, result, input, spillover,
                                                                       spillover_len, chunk_id == n_chunks - 1);
#endif

            // Step 3: Update spillover
            spillover = encode_result.spillover;
            spillover_len = encode_result.spillover_len;
            global_size[threadIdx.x] += encode_result.bytes_written;

            assert(encode_result.bytes_written < n_symbols_per_chunk + 1); // Check for buffer overrun

            // Step 4: Fix output buffer
            // const uint8_t fix_offset = encode_result.bytes_written % tile_out_word_size;
            // const uint8_t fix_block = encode_result.bytes_written / tile_out_word_size + 1;
            //
            // // This ensures that the current output block (that is partially done) is modified so that all bytes that
            // // are not written to in this iteration will be equal to 0xFE
            // result[fix_block - 1][threadIdx.x] =
            //     result[fix_block - 1][threadIdx.x] & ~(0xFFFFFFFFU << fix_offset * 8) | 0xFEFEFEFE << fix_offset * 8;
            //
            // // This ensures all blocks that have not been touched are reset to 0xFEFEFEFE
            // for (uint8_t i = fix_block; i < 6; i++) {
            //     result[i][threadIdx.x] = 0xFEFEFEFE;
            // }

            // Step 5: Coordinated output
            uint8_t* dst_loc = dst + blockIdx.x * (uint64_t)TMP_OUT_BLOCK_SIZE;
            uint32_t* dst_aligned = (uint32_t*)dst_loc;
            assert((uintptr_t)dst_loc % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment

            for (int i = 0; i < n_stores_per_chunk; i++) {
                dst_aligned[threadIdx.x * tile_out_len_words + chunk_id * n_stores_per_chunk + i] = result[i][threadIdx.
                    x];
            }

            // Check for overruns
            assert(threadIdx.x * tile_out_len + chunk_id * n_symbols_per_chunk + n_symbols_per_chunk - 1 <
                (threadIdx.x + 1) * tile_out_len); // tile overruns
            assert(threadIdx.x * tile_out_len + chunk_id * n_symbols_per_chunk + n_symbols_per_chunk - 1 <
                TMP_OUT_BLOCK_SIZE); // block overruns

#if USE_SCALAR_PREFETCH == 1
            load1 = load3;
            load2 = load4;
#endif
        }

        assert(global_size[threadIdx.x] > 0);

        __syncthreads();

        // Sum block output
        if (threadIdx.x == THREAD_COUNT - 1) {
            uint32_t block_out = 0;

            for (int i = 0; i < THREAD_COUNT; i++) {
                block_out += global_size[i];
            }

            const BlockHeader header = {
                .uncompressed_size = BLOCK_SIZE, // TODO: change this later to support padding in input
                .compressed_size = block_out
            };
            headers[blockIdx.x] = header;
        }
    }
} // namespace gtsst::compressors::compactionv2
