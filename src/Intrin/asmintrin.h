#pragma once
#include "LeibnizCompiler.h"
#include "Leibniz.h"

namespace Leibniz::Intrin {
    namespace Internal {
#if LEIBNIZ_COMPILER_MSVC
        extern "C" void asmInitInterleave();
        extern "C" uint64_t asmMulxU64(uint64_t a, uint64_t b, uint64_t* hi);
        extern "C" void asmAdcxU64(uint64_t* a, uint64_t b);
        extern "C" void asmAdoxU64(uint64_t* a, uint64_t b);
#endif
    }

#if LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
    LEIBNIZ_FORCEINLINE uint64_t mulx_u64(uint64_t a, uint64_t b, uint64_t& hi) {
        uint64_t lo;
        asm volatile(
            "mulx %[b], %[lo], %[hi]"
            : [lo] "=&r"(lo), [hi] "=&r"(hi)
            : "d"(a), [b] "r"(b)
            );
        return lo;
    }
    LEIBNIZ_FORCEINLINE void adcx_u64(uint64_t* a, uint64_t b) {
        asm volatile("adcx %1, %0" : "+r"(*a) : "r"(b) : "cc");
    }
    LEIBNIZ_FORCEINLINE void adox_u64(uint64_t* a, uint64_t b) {
        asm volatile("adox %1, %0" : "+r"(*a) : "r"(b) : "cc");
    }

    LEIBNIZ_FORCEINLINE void init_interleave() {
        asm volatile(
            "xor %%rax, %%rax\n\t"
            "adox %%rax, %%rax\n\t"
            "adcx %%rax, %%rax\n\t"
            :
            :
            : "rax", "cc"
            );
    }

#elif LEIBNIZ_COMPILER_MSVC
    // Non-inline MSVC fallback — correct but not optimal
    LEIBNIZ_RUNTIME_API uint64_t mulx_u64(uint64_t a, uint64_t b, uint64_t& hi);
    LEIBNIZ_RUNTIME_API void adcx_u64(uint64_t* a, uint64_t b);
    LEIBNIZ_RUNTIME_API void adox_u64(uint64_t* a, uint64_t b);
    LEIBNIZ_RUNTIME_API void init_interleave();
#endif
}