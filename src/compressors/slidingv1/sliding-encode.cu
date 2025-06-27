#include <compressors/slidingv1/sliding-defines.cuh>
#include <compressors/slidingv1/sliding-encode.cuh>

namespace gtsst::compressors::slidingv1 {
    __device__ void pack_results_local(uint32_t result[TILE_SIZE][TILE_COUNT], const uint32_t offset,
                                       const uint32_t val) {
        assert(offset < tile_out_chunk_size); // Ensure the offset is within the limits

        const uint8_t tile_id = threadIdx.x / 32;
        const uint32_t shift = (offset & 3) * 8;
        const uint8_t res = offset / 4;
        const uint32_t val_mask = val << shift;
        const uint32_t clean_mask = ~(0xFFU << shift);

        const uint32_t current = result[res][tile_id];

        result[res][tile_id] = current & clean_mask | val_mask;
    }

    struct BallotResult {
        uint8_t sym_len;
        uint8_t output_size;
    };

    __device__ BallotResult ballot_cycle(const symbols::SymbolSlidingTableData& symbol_table,
                                         const symbols::GPUSymbol symbol, uint32_t result[TILE_SIZE][TILE_COUNT],
                                         const uint16_t output_offset) {
        uint8_t code;
        uint8_t len;
        const uint8_t lane_id = threadIdx.x % 32;

        // TODO: add ignore logic (probably check, and separate counter for valid input bytes or something

        for (int i = 0; i < symbols::SymbolSlidingTableData::iterations; i++) {
            // Check if this symbol matches with our current guess
            const bool match = symbol_table.attemptMatch(symbol, i, &code, &len);

            // Do a ballot to see if any thread found a match
            const uint32_t mask = __ballot_sync(0xFFFFFFFF, match);

            // If nobody found a match, attempt next cycle
            if (mask == 0) {
                continue;
            }

            // If this thread has the best match, it will have to do the output processing
            const uint8_t best_match_lane = __ffs(mask) - 1;
            uint8_t s_len = len;
            if (lane_id == best_match_lane) {
                pack_results_local(result, output_offset, code);
            }

            // Synchronize input and output offsets between threads
            s_len = __shfl_sync(0xFFFFFFFF, s_len, best_match_lane);

            return BallotResult{
                .sym_len = s_len,
                .output_size = 1,
            };
        }

        // Escape sequence
        if (lane_id == 0) {
            // Two packing calls, should be fine since there is always room for escape in output buffer (since we
            // flush early)
            pack_results_local(result, output_offset, fsst::Symbol::escape);
            pack_results_local(result, output_offset + 1, (uint8_t)get_first_byte_local(symbol.val.num));
        }

        return BallotResult{
            .sym_len = 1,
            .output_size = 2,
        };
    }

    __device__ void flush_result(uint8_t* dst, const uint16_t flushes_before, uint32_t result[TILE_SIZE][TILE_COUNT]) {
        assert(__activemask() ==
            0xFFFFFFFF); // It should be impossible to have a divergent flush (then balloting went wrong)

        const uint8_t lane_id = threadIdx.x % 32;
        const uint8_t tile_id = threadIdx.x / 32;
        uint8_t* dst_loc = dst + blockIdx.x * BLOCK_SIZE + tile_len * tile_id;
        uint32_t* dst_aligned = (uint32_t*)dst_loc;
        assert((uintptr_t)dst_loc % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment

        // Write result and reset it
        dst_aligned[flushes_before * TILE_SIZE + lane_id] = result[lane_id][tile_id];
        result[lane_id][tile_id] = 0xFEFEFEFE;
    }

    __device__ uint8_t shift_registers_with_base(uint32_t load[TILE_SIZE + 2][TILE_COUNT], const uint32_t offset,
                                                 uint32_t* first, uint32_t* second, uint32_t* third) {
        const uint8_t tile_id = threadIdx.x / 32;

        const uint32_t block_offset = offset / sizeof(uint32_t);
        *first = load[block_offset][tile_id];
        *second = load[block_offset + 1][tile_id];
        *third = load[block_offset + 2][tile_id];

        return offset % sizeof(uint32_t);
    }

    struct EncodeResult {
        uint8_t times_flushed;
        uint16_t new_output_offset;
        uint16_t bytes_written; // Does not include padding!
        uint8_t spillover_len;
        uint64_t spillover;
    };

    __device__ bool should_flush(const uint32_t output_offset) {
        return output_offset > tile_out_chunk_size - 2;
    }

    __device__ EncodeResult encode_cycle(const symbols::SymbolSlidingTableData& symbol_table,
                                         uint32_t load[TILE_SIZE + 2][TILE_COUNT],
                                         uint32_t result[TILE_SIZE][TILE_COUNT], uint64_t spillover,
                                         const uint8_t spillover_len, const bool last_chunk, uint8_t* dst,
                                         const uint16_t flushes_before, const uint16_t current_output_offset) {
        const uint8_t warp_id = threadIdx.x / 32;
        uint16_t output_offset = current_output_offset;
        uint16_t input_offset = 0;

        uint32_t first_block = load[0][warp_id];
        uint32_t second_block = load[1][warp_id];
        uint32_t third_block = load[2][warp_id];
        uint8_t first_block_offset = 0;

        uint8_t flush_count = 0;
        uint16_t bytes_flushed = 0;

        constexpr uint16_t search_len =
            TILE_SIZE * sizeof(uint32_t); // can look ahead at all available bytes (capped at 8 anyways)
        constexpr uint16_t encode_len = search_len - 7;

        assert(__activemask() == 0xFFFFFFFF);

        // First handle spillover
        while (input_offset < spillover_len) {
            assert(__activemask() == 0xFFFFFFFF);

            const symbols::GPUSymbol s = create_symbol_with_spillover_local(
                spillover, spillover_len - input_offset, first_block, second_block, fsst::Symbol::maxLength);
            const auto [sym_len, output_size] = ballot_cycle(symbol_table, s, result, output_offset);

            // Bookkeeping of spillover data
            spillover >>= 8 * sym_len;
            output_offset += output_size;
            input_offset += sym_len;

            // If the ballot cycle indicated a flush is required, flush it and update counters
            if (should_flush(output_offset)) {
                flush_result(dst, flushes_before + flush_count, result);

                flush_count += 1;
                bytes_flushed += output_offset;
                output_offset = 0;
            }
        }

        assert(__activemask() == 0xFFFFFFFF);
        // Update registers for possible shift after overusage of spillover
        first_block_offset =
            shift_registers_with_base(load, input_offset - spillover_len, &first_block, &second_block, &third_block);
        assert(__activemask() == 0xFFFFFFFF);

        // Then handle regular block
        const uint8_t encode_range = last_chunk ? search_len : encode_len;
        while (input_offset < spillover_len + encode_range) {
            assert(__activemask() == 0xFFFFFFFF);
            const symbols::GPUSymbol s = create_symbol_no_spillover_local(
                first_block, second_block, third_block, first_block_offset,
                min((int)fsst::Symbol::maxLength, search_len + spillover_len - input_offset));
            const auto [sym_len, output_size] = ballot_cycle(symbol_table, s, result, output_offset);

            input_offset += sym_len;
            output_offset += output_size;

            // Shift registers based on sym_len
            first_block_offset = shift_registers_with_base(load, input_offset - spillover_len, &first_block,
                                                           &second_block, &third_block);

            // If the ballot cycle indicated a flush is required, flush it and update counters
            if (should_flush(output_offset)) {
                flush_result(dst, flushes_before + flush_count, result);

                flush_count += 1;
                bytes_flushed += output_offset;
                output_offset = 0;
            }
        }

        // If this is the last chunk, make sure to flush any remaining output
        if (last_chunk && output_offset > 0) {
            flush_result(dst, flushes_before + flush_count, result);

            flush_count += 1;
            bytes_flushed += output_offset;
            output_offset = 0;
        }

        // Then create new spillover
        const uint8_t spilled_bytes = search_len + spillover_len - input_offset;
        const symbols::GPUSymbol s =
            create_symbol_no_spillover_local(first_block, second_block, third_block, first_block_offset, spilled_bytes);

        return EncodeResult{flush_count, output_offset, bytes_flushed, spilled_bytes, s.val.num};
    }

    __global__ void gpu_sliding_encode(SlidingMetadata* metadata, BlockHeader* headers, const uint8_t* src,
                                       uint8_t* dst) {
        __shared__ uint32_t load[TILE_SIZE + 2][TILE_COUNT];
        __shared__ uint32_t result[TILE_SIZE][TILE_COUNT];
        __shared__ uint32_t global_size[TILE_COUNT];

        // Load metadata into shared mem
        __shared__ SlidingMetadata m;
        load_metadata_local<SlidingMetadata>(metadata, (uint32_t*)&m, SUPER_BLOCK_SIZE);

        // Checks
        assert((uintptr_t)src % WORD_ALIGNMENT == 0); // Ensure 8-byte alignment of source
        assert((uintptr_t)dst % TMP_WORD_ALIGNMENT == 0); // Ensure 4-byte alignment of dest

        // Active data
        uint32_t tile_output_size = 0;
        uint64_t spillover = 0;
        uint8_t spillover_len = 0;
        uint16_t flushed_completed = 0;
        uint16_t output_offset = 0;

        // Get source for this tile that is aligned for this block and tile
        const uint32_t tile_id = threadIdx.x / 32;
        const uint32_t lane_id = threadIdx.x % 32;
        const auto aligned_src = (uint32_t*)(src + BLOCK_SIZE * ((uint64_t)blockIdx.x) + tile_len * tile_id);

        // Reset result for first iteration
        result[lane_id][tile_id] = 0xFEFEFEFE;

        /*
         * Start encoding. So for every chunk:
         *  - Load data
         *  - Encode using balloting
         *      - For every cycle:
         *          - Update output size
         *          - If result left < 2 -> write results
         *      - End of encode cycle:
         *          - Update spillover
         *          - Update spillover length
         */
        assert(__activemask() == 0xFFFFFFFF);

        for (uint32_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
            // Step 1: Load data
            load[lane_id][tile_id] = aligned_src[chunk_id * 32 + lane_id];

            // Additional space to not worry about limits when shifting
            load[TILE_SIZE][tile_id] = load[TILE_SIZE - 1][tile_id];
            load[TILE_SIZE + 1][tile_id] = load[TILE_SIZE - 1][tile_id];

            // Step 2: Encode using balloting
            const bool last_chunk = chunk_id == n_chunks - 1;
            auto const encode_result = encode_cycle(m.symbol_table, load, result, spillover, spillover_len, last_chunk,
                                                    dst, flushed_completed, output_offset);

            // Copy some variables
            spillover = encode_result.spillover;
            spillover_len = encode_result.spillover_len;
            flushed_completed += encode_result.times_flushed;
            output_offset = encode_result.new_output_offset;
            tile_output_size += encode_result.bytes_written;

            // Some checks for debugging purposes
            assert(!last_chunk || spillover_len == 0); // There should be no spillover after the last encode cycle
            assert(!last_chunk || output_offset == 0); // All output should be flushed after the last cycle
            assert(flushed_completed <= max_out_chunks); // Check that no buffers were overrun
        }

        // The first thread of every warp writes its output
        if (lane_id == 0) {
            assert(tile_output_size > 0); // Ensure we did something
            global_size[tile_id] = tile_output_size;
        }

        __syncthreads();

        // And the first thread of the block writes the block output
        if (lane_id == 0) {
            uint32_t block_out = 0;

            for (int i = 0; i < TILE_COUNT; i++) {
                block_out += global_size[i];
            }

            // TODO: will have to do something with uncompressed_size to make padding work. Either keep track of
            // bytes that are not ignored, or pad out decompression side.
            const BlockHeader header = {.uncompressed_size = BLOCK_SIZE, .compressed_size = block_out};
            headers[blockIdx.x] = header;
        }
    }
} // namespace gtsst::compressors::slidingv1
