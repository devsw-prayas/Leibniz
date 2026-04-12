#pragma once
#include <LeibnizCompiler.h>

#if defined(_DEBUG) || defined(DEBUG)
#define LEIBNIZ_BUILD_DEBUG 1
#define LEIBNIZ_BUILD_RELEASE 0
#else
#define LEIBNIZ_BUILD_DEBUG 0
#define LEIBNIZ_BUILD_RELEASE 1
#endif

#if LEIBNIZ_BUILD_DEBUG
#define LEIBNIZ_ASSERT(expr)                        \
        do {                                            \
            if (!(expr)) {                             \
                LEIBNIZ_DEBUG_BREAK();                 \
                LEIBNIZ_TRAP();                        \
            }                                           \
        } while (0)
#else
#define LEIBNIZ_ASSERT(expr) do { (void)sizeof(expr); } while (0)
#endif

#if LEIBNIZ_BUILD_DEBUG
#define LEIBNIZ_ASSUME(expr) LEIBNIZ_ASSERT(expr)
#else
#if LEIBNIZ_COMPILER_MSVC
#define LEIBNIZ_ASSUME(expr) __assume(expr)
#else
#define LEIBNIZ_ASSUME(expr) do { if (!(expr)) LEIBNIZ_UNREACHABLE(); } while (0)
#endif
#endif

#if LEIBNIZ_BUILD_DEBUG
#define LEIBNIZ_DEBUG_ASSERT(expr) LEIBNIZ_ASSERT(expr)
#define LEIBNIZ_DEBUG_ASSUME(expr) LEIBNIZ_ASSUME(expr)
#else
#define LEIBNIZ_DEBUG_ASSERT(expr) do {} while (0)
#define LEIBNIZ_DEBUG_ASSUME(expr) do {} while (0)
#endif

#define LEIBNIZ_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#define LEIBNIZ_UNUSED(x) (void)(x)
