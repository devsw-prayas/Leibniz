#pragma once

// Just don't mess with this header, exposes low level shit,
// Don't want system gettin' fucked when a supposedly impossible linux port occurs
// This header is from Spectra Render Engine

#if defined(__AVX512F__)
#define LEIBNIZ_SIMD_AVX512 1
#elif defined(__AVX2__)
#define LEIBNIZ_SIMD_AVX2 1
#elif defined(__SSE2__)
#define LEIBNIZ_SIMD_SSE2 1
#else
#define LEIBNIZ_SIMD_SCALAR 1
#endif

namespace Leibniz {
#if defined(_MSC_VER)
#define LEIBNIZ_COMPILER_MSVC 1
#else
#define LEIBNIZ_COMPILER_MSVC 0
#endif

#if defined(__clang__)
#define LEIBNIZ_COMPILER_CLANG 1
#else
#define LEIBNIZ_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define LEIBNIZ_COMPILER_GCC 1
#else
#define LEIBNIZ_COMPILER_GCC 0
#endif
}

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_FORCEINLINE __forceinline
#define LEIBNIZ_NOINLINE    __declspec(noinline)
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_FORCEINLINE inline __attribute__((always_inline))
#define LEIBNIZ_NOINLINE    __attribute__((noinline))
#else
#define LEIBNIZ_FORCEINLINE inline
#define LEIBNIZ_NOINLINE
#endif

#define LEIBNIZ_INLINE inline

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_COMPILER_BARRIER() _ReadWriteBarrier()
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_COMPILER_BARRIER() asm volatile("" ::: "memory")
#else
#define LEIBNIZ_COMPILER_BARRIER()
#endif

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_OPTIMIZE_OFF __pragma(optimize("", off))
#define LEIBNIZ_OPTIMIZE_ON  __pragma(optimize("", on))
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_OPTIMIZE_OFF _Pragma("clang optimize off")
#define LEIBNIZ_OPTIMIZE_ON  _Pragma("clang optimize on")
#else
#define LEIBNIZ_OPTIMIZE_OFF
#define LEIBNIZ_OPTIMIZE_ON
#endif

#if LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_LIKELY(x)   __builtin_expect(!!(x), 1)
#define LEIBNIZ_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LEIBNIZ_LIKELY(x)   (x)
#define LEIBNIZ_UNLIKELY(x) (x)
#endif

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_DEBUG_BREAK() __debugbreak()
#define LEIBNIZ_TRAP()        __debugbreak()
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_DEBUG_BREAK() __builtin_trap()
#define LEIBNIZ_TRAP()        __builtin_trap()
#else
#include <cstdlib>
#define LEIBNIZ_DEBUG_BREAK() std::abort()
#define LEIBNIZ_TRAP()        std::abort()
#endif

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_UNREACHABLE() __assume(0)
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_UNREACHABLE() __builtin_unreachable()
#else
#define LEIBNIZ_UNREACHABLE() LEIBNIZ_TRAP()
#endif

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_PRAGMA(x) __pragma(x)
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_PRAGMA(x) _Pragma(#x)
#else
#define LEIBNIZ_PRAGMA(x)
#endif

#define LEIBNIZ_DIAGNOSTIC_PUSH LEIBNIZ_PRAGMA(diagnostic push)
#define LEIBNIZ_DIAGNOSTIC_POP  LEIBNIZ_PRAGMA(diagnostic pop)

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_DISABLE_WARNING(w) LEIBNIZ_PRAGMA(warning(disable : w))
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_DISABLE_WARNING(w) LEIBNIZ_PRAGMA(clang diagnostic ignored w)
#else
#define LEIBNIZ_DISABLE_WARNING(w)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(fallthrough)
#define LEIBNIZ_FALLTHROUGH [[fallthrough]]
#else
#define LEIBNIZ_FALLTHROUGH
#endif
#else
#define LEIBNIZ_FALLTHROUGH
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#define LEIBNIZ_NODISCARD [[nodiscard]]
#if __cplusplus >= 202002L
#define LEIBNIZ_NODISCARD_MSG(msg) [[nodiscard(msg)]]
#else
#define LEIBNIZ_NODISCARD_MSG(msg) [[nodiscard]]
#endif
#else
#define LEIBNIZ_NODISCARD
#define LEIBNIZ_NODISCARD_MSG(msg)
#endif
#else
#define LEIBNIZ_NODISCARD
#define LEIBNIZ_NODISCARD_MSG(msg)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(maybe_unused)
#define LEIBNIZ_MAYBE_UNUSED [[maybe_unused]]
#else
#define LEIBNIZ_MAYBE_UNUSED
#endif
#else
#define LEIBNIZ_MAYBE_UNUSED
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define LEIBNIZ_DEPRECATED [[deprecated]]
#define LEIBNIZ_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#else
#define LEIBNIZ_DEPRECATED
#define LEIBNIZ_DEPRECATED_MSG(msg)
#endif
#else
#define LEIBNIZ_DEPRECATED
#define LEIBNIZ_DEPRECATED_MSG(msg)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(noreturn)
#define LEIBNIZ_NORETURN [[noreturn]]
#else
#define LEIBNIZ_NORETURN
#endif
#else
#define LEIBNIZ_NORETURN
#endif

#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_RESTRICT __restrict
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_RESTRICT __restrict__
#else
#define LEIBNIZ_RESTRICT
#endif

#define LEIBNIZ_ALIGNAS(n) alignas(n)

#if LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_ASSUME_ALIGNED(ptr, n) __builtin_assume_aligned((ptr), (n))
#else
#define LEIBNIZ_ASSUME_ALIGNED(ptr, n) __assume(reinterpret_cast<uintptr_t>(ptr) % n == 0)
#endif

#if LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_HOT  __attribute__((hot))
#define LEIBNIZ_COLD __attribute__((cold))
#else
#define LEIBNIZ_HOT
#define LEIBNIZ_COLD
#endif
