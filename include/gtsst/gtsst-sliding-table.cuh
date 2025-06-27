#ifndef GTSST_SLIDING_TABLE_CUH
#define GTSST_SLIDING_TABLE_CUH
#include "gtsst-symbols.cuh"

namespace gtsst::symbols {
    struct SymbolSlidingTableData {
        constexpr static int size = 256;
        constexpr static int warp_size = 32;
        constexpr static int iterations = size / warp_size;
        static_assert(size % warp_size == 0, "size must be a multiple of warp size");

        ComparableSmallSymbol symbols[size];

        SymbolSlidingTableData() = default;

        explicit SymbolSlidingTableData(const fsst::SymbolTable& st) {
            constexpr uint8_t ignore = fsst::Symbol::ignore;
            assert(st.nSymbols < ignore);

            // Fill symbols
            for (int i = 0; i < size; ++i) {
                if (i < st.nSymbols) {
                    symbols[i] = ComparableSmallSymbol(st.symbols[i]);
                } else {
                    symbols[i] = ComparableSmallSymbol();
                }
            }

            // Add ignore symbol to padding on the input side is properly matched
            symbols[st.nSymbols] = ComparableSmallSymbol(fsst::Symbol(ignore, ignore));
        }

        __device__ bool attemptMatch(const GPUSymbol& sym, const int iter, uint8_t* code, uint8_t* len) const {
            const ComparableSmallSymbol s = symbols[warp_size * iter + threadIdx.x % warp_size];

            *code = s.code();
            *len = s.length();

            return s.match(sym);
        }
    };

    static_assert(std::is_constructible_v<SymbolSlidingTableData, fsst::SymbolTable&>);
} // namespace gtsst::symbols

#endif // GTSST_SLIDING_TABLE_CUH
