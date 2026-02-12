#pragma once

namespace Leibniz::Intrinsic {

#ifndef BitScanForward32
#if LEIBNIZ_COMPILER_MSVC
#define BitScanForward32(indexPtr, value) \
    _BitScanForward((indexPtr), (value))
#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG
#if defined(__has_builtin)
#if __has_builtin(__builtin_ctz)
#define BitScanForward32(indexPtr, value) \
        (*(indexPtr) = __builtin_ctz(value), 1)
#else
#error "BitScanForward32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for BitScanForward32"
#endif
#endif

#ifndef BitScanReverse32
#if LEIBNIZ_COMPILER_MSVC

#define BitScanReverse32(indexPtr, value) \
    _BitScanReverse((indexPtr), (value))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_clz)

#define BitScanReverse32(indexPtr, value) \
        (*(indexPtr) = 31 - __builtin_clz(value), 1)

#else
#error "BitScanReverse32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for BitScanReverse32"
#endif
#endif


#ifndef BitScanForward64
#if LEIBNIZ_COMPILER_MSVC

#define BitScanForward64(indexPtr, value) \
    _BitScanForward64((indexPtr), (value))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_ctzll)

#define BitScanForward64(indexPtr, value) \
        (*(indexPtr) = __builtin_ctzll(value), 1)

#else
#error "BitScanForward64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for BitScanForward64"
#endif
#endif

#ifndef BitScanReverse64
#if LEIBNIZ_COMPILER_MSVC

#define BitScanReverse64(indexPtr, value) \
    _BitScanReverse64((indexPtr), (value))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_clzll)

#define BitScanReverse64(indexPtr, value) \
        (*(indexPtr) = 63 - __builtin_clzll(value), 1)

#else
#error "BitScanReverse64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for BitScanReverse64"
#endif
#endif

#ifndef BitTest
#if LEIBNIZ_COMPILER_MSVC

#define BitTest(valuePtr, bit) \
    _bittest((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTest(valuePtr, bit) \
    (((*(valuePtr)) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTest"
#endif
#endif

#ifndef BitTest64
#if LEIBNIZ_COMPILER_MSVC

#define BitTest64(valuePtr, bit) \
    _bittest64((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTest64(valuePtr, bit) \
    (((*(valuePtr)) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTest64"
#endif
#endif

#ifndef BitTestAndSet
#if LEIBNIZ_COMPILER_MSVC

#define BitTestAndSet(valuePtr, bit) \
    _bittestandset((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTestAndSet(valuePtr, bit) \
    ((__sync_fetch_and_or((valuePtr), (1u << (bit))) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTestAndSet"
#endif
#endif

#ifndef BitTestAndSet64
#if LEIBNIZ_COMPILER_MSVC

#define BitTestAndSet64(valuePtr, bit) \
    _bittestandset64((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTestAndSet64(valuePtr, bit) \
    ((__sync_fetch_and_or((valuePtr), (1ull << (bit))) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTestAndSet64"
#endif
#endif

#ifndef BitTestAndReset
#if LEIBNIZ_COMPILER_MSVC

#define BitTestAndReset(valuePtr, bit) \
    _bittestandreset((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTestAndReset(valuePtr, bit) \
    ((__sync_fetch_and_and((valuePtr), ~(1u << (bit))) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTestAndReset"
#endif
#endif

#ifndef BitTestAndReset64
#if LEIBNIZ_COMPILER_MSVC

#define BitTestAndReset64(valuePtr, bit) \
    _bittestandreset64((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTestAndReset64(valuePtr, bit) \
    ((__sync_fetch_and_and((valuePtr), ~(1ull << (bit))) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTestAndReset64"
#endif
#endif

#ifndef BitTestAndComplement
#if LEIBNIZ_COMPILER_MSVC

#define BitTestAndComplement(valuePtr, bit) \
    _bittestandcomplement((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTestAndComplement(valuePtr, bit) \
    ((__sync_fetch_and_xor((valuePtr), (1u << (bit))) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTestAndComplement"
#endif
#endif


#ifndef BitTestAndComplement64
#if LEIBNIZ_COMPILER_MSVC

#define BitTestAndComplement64(valuePtr, bit) \
    _bittestandcomplement64((valuePtr), (bit))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define BitTestAndComplement64(valuePtr, bit) \
    ((__sync_fetch_and_xor((valuePtr), (1ull << (bit))) >> (bit)) & 1)

#else
#error "Unsupported compiler for BitTestAndComplement64"
#endif
#endif

#ifndef AddCarry8
#if LEIBNIZ_COMPILER_MSVC

#define AddCarry8(carryIn, a, b, resultPtr) \
    _addcarry_u8((carryIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_addc)

#define AddCarry8(carryIn, a, b, resultPtr) \
        __builtin_addc((carryIn), (a), (b), (resultPtr))

#else
#error "AddCarry8 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddCarry8"
#endif
#endif

#ifndef AddCarry16
#if LEIBNIZ_COMPILER_MSVC

#define AddCarry16(carryIn, a, b, resultPtr) \
    _addcarry_u16((carryIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_addc)

#define AddCarry16(carryIn, a, b, resultPtr) \
        __builtin_addc((carryIn), (a), (b), (resultPtr))

#else
#error "AddCarry16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddCarry16"
#endif
#endif

#ifndef AddCarry32
#if LEIBNIZ_COMPILER_MSVC

#define AddCarry32(carryIn, a, b, resultPtr) \
    _addcarry_u32((carryIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_addc)

#define AddCarry32(carryIn, a, b, resultPtr) \
        __builtin_addc((carryIn), (a), (b), (resultPtr))

#else
#error "AddCarry32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddCarry32"
#endif
#endif

#ifndef AddCarry64
#if LEIBNIZ_COMPILER_MSVC

#define AddCarry64(carryIn, a, b, resultPtr) \
    _addcarry_u64((carryIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_addcll)

#define AddCarry64(carryIn, a, b, resultPtr) \
        __builtin_addcll((carryIn), (a), (b), (resultPtr))

#else
#error "AddCarry64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddCarry64"
#endif
#endif

#ifndef SubBorrow8
#if LEIBNIZ_COMPILER_MSVC

#define SubBorrow8(borrowIn, a, b, resultPtr) \
    _subborrow_u8((borrowIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_subc)

#define SubBorrow8(borrowIn, a, b, resultPtr) \
        __builtin_subc((borrowIn), (a), (b), (resultPtr))

#else
#error "SubBorrow8 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubBorrow8"
#endif
#endif

#ifndef SubBorrow16
#if LEIBNIZ_COMPILER_MSVC

#define SubBorrow16(borrowIn, a, b, resultPtr) \
    _subborrow_u16((borrowIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_subc)

#define SubBorrow16(borrowIn, a, b, resultPtr) \
        __builtin_subc((borrowIn), (a), (b), (resultPtr))

#else
#error "SubBorrow16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubBorrow16"
#endif
#endif

#ifndef SubBorrow32
#if LEIBNIZ_COMPILER_MSVC

#define SubBorrow32(borrowIn, a, b, resultPtr) \
    _subborrow_u32((borrowIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_subc)

#define SubBorrow32(borrowIn, a, b, resultPtr) \
        __builtin_subc((borrowIn), (a), (b), (resultPtr))

#else
#error "SubBorrow32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubBorrow32"
#endif
#endif

#ifndef SubBorrow64
#if LEIBNIZ_COMPILER_MSVC

#define SubBorrow64(borrowIn, a, b, resultPtr) \
    _subborrow_u64((borrowIn), (a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_subcll)

#define SubBorrow64(borrowIn, a, b, resultPtr) \
        __builtin_subcll((borrowIn), (a), (b), (resultPtr))

#else
#error "SubBorrow64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubBorrow64"
#endif
#endif

#ifndef Mul128
#if LEIBNIZ_COMPILER_MSVC

#define Mul128(a, b, highPtr) \
    _mul128((a), (b), (highPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__SIZEOF_INT128__)

#define Mul128(a, b, highPtr)                     \
    do {                                          \
        __int128 _tmp = (__int128)(a) * (b);      \
        *(highPtr) = (int64_t)(_tmp >> 64);       \
        (void)(_tmp);                             \
    } while (0)

#else
#error "Mul128 intrinsic missing (__int128 not supported)"
#endif

#else
#error "Unsupported compiler for Mul128"
#endif
#endif

#ifndef Emul
#if LEIBNIZ_COMPILER_MSVC

#define Emul(a, b) \
    __emul((a), (b))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define Emul(a, b) \
    ((int64_t)(int32_t)(a) * (int64_t)(int32_t)(b))

#else
#error "Unsupported compiler for Emul"
#endif
#endif

#ifndef Emulu
#if LEIBNIZ_COMPILER_MSVC

#define Emulu(a, b) \
    __emulu((a), (b))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define Emulu(a, b) \
    ((uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b))

#else
#error "Unsupported compiler for Emulu"
#endif
#endif

#ifndef AddOverflow8
#if LEIBNIZ_COMPILER_MSVC

#define AddOverflow8(a, b, resultPtr) \
    _add_overflow_i8((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)

#define AddOverflow8(a, b, resultPtr) \
        __builtin_add_overflow((a), (b), (resultPtr))

#else
#error "AddOverflow8 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddOverflow8"
#endif
#endif

#ifndef AddOverflow16
#if LEIBNIZ_COMPILER_MSVC

#define AddOverflow16(a, b, resultPtr) \
    _add_overflow_i16((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)

#define AddOverflow16(a, b, resultPtr) \
        __builtin_add_overflow((a), (b), (resultPtr))

#else
#error "AddOverflow16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddOverflow16"
#endif
#endif

#ifndef AddOverflow32
#if LEIBNIZ_COMPILER_MSVC

#define AddOverflow32(a, b, resultPtr) \
    _add_overflow_i32((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)

#define AddOverflow32(a, b, resultPtr) \
        __builtin_add_overflow((a), (b), (resultPtr))

#else
#error "AddOverflow32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddOverflow32"
#endif
#endif

#ifndef AddOverflow64
#if LEIBNIZ_COMPILER_MSVC

#define AddOverflow64(a, b, resultPtr) \
    _add_overflow_i64((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)

#define AddOverflow64(a, b, resultPtr) \
        __builtin_add_overflow((a), (b), (resultPtr))

#else
#error "AddOverflow64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for AddOverflow64"
#endif
#endif

#ifndef SubOverflow8
#if LEIBNIZ_COMPILER_MSVC

#define SubOverflow8(a, b, resultPtr) \
    _sub_overflow_i8((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_sub_overflow)

#define SubOverflow8(a, b, resultPtr) \
        __builtin_sub_overflow((a), (b), (resultPtr))

#else
#error "SubOverflow8 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubOverflow8"
#endif
#endif

#ifndef SubOverflow16
#if LEIBNIZ_COMPILER_MSVC

#define SubOverflow16(a, b, resultPtr) \
    _sub_overflow_i16((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_sub_overflow)

#define SubOverflow16(a, b, resultPtr) \
        __builtin_sub_overflow((a), (b), (resultPtr))

#else
#error "SubOverflow16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubOverflow16"
#endif
#endif

#ifndef SubOverflow32
#if LEIBNIZ_COMPILER_MSVC

#define SubOverflow32(a, b, resultPtr) \
    _sub_overflow_i32((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_sub_overflow)

#define SubOverflow32(a, b, resultPtr) \
        __builtin_sub_overflow((a), (b), (resultPtr))

#else
#error "SubOverflow32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubOverflow32"
#endif
#endif

#ifndef SubOverflow64
#if LEIBNIZ_COMPILER_MSVC

#define SubOverflow64(a, b, resultPtr) \
    _sub_overflow_i64((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_sub_overflow)

#define SubOverflow64(a, b, resultPtr) \
        __builtin_sub_overflow((a), (b), (resultPtr))

#else
#error "SubOverflow64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for SubOverflow64"
#endif
#endif

#ifndef MulOverflow16
#if LEIBNIZ_COMPILER_MSVC

#define MulOverflow16(a, b, resultPtr) \
    _mul_overflow_i16((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulOverflow16(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulOverflow16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulOverflow16"
#endif
#endif

#ifndef MulOverflow32
#if LEIBNIZ_COMPILER_MSVC

#define MulOverflow32(a, b, resultPtr) \
    _mul_overflow_i32((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulOverflow32(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulOverflow32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulOverflow32"
#endif
#endif

#ifndef MulOverflow64
#if LEIBNIZ_COMPILER_MSVC

#define MulOverflow64(a, b, resultPtr) \
    _mul_overflow_i64((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulOverflow64(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulOverflow64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulOverflow64"
#endif
#endif

#ifndef MulFullOverflow8
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflow8(a, b, resultPtr) \
    _mul_full_overflow_i8((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflow8(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflow8 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflow8"
#endif
#endif

#ifndef MulFullOverflow16
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflow16(a, b, resultPtr) \
    _mul_full_overflow_i16((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflow16(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflow16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflow16"
#endif
#endif

#ifndef MulFullOverflow32
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflow32(a, b, resultPtr) \
    _mul_full_overflow_i32((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflow32(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflow32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflow32"
#endif
#endif

#ifndef MulFullOverflow64
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflow64(a, b, resultPtr) \
    _mul_full_overflow_i64((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflow64(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflow64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflow64"
#endif
#endif

#ifndef MulFullOverflowU8
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflowU8(a, b, resultPtr) \
    _mul_full_overflow_u8((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflowU8(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflowU8 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflowU8"
#endif
#endif

#ifndef MulFullOverflowU16
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflowU16(a, b, resultPtr) \
    _mul_full_overflow_u16((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflowU16(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflowU16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflowU16"
#endif
#endif

#ifndef MulFullOverflowU32
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflowU32(a, b, resultPtr) \
    _mul_full_overflow_u32((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflowU32(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflowU32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflowU32"
#endif
#endif

#ifndef MulFullOverflowU64
#if LEIBNIZ_COMPILER_MSVC

#define MulFullOverflowU64(a, b, resultPtr) \
    _mul_full_overflow_u64((a), (b), (resultPtr))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)

#define MulFullOverflowU64(a, b, resultPtr) \
        __builtin_mul_overflow((a), (b), (resultPtr))

#else
#error "MulFullOverflowU64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for MulFullOverflowU64"
#endif
#endif

#ifndef PopCount32
#if LEIBNIZ_COMPILER_MSVC

#define PopCount32(x) \
    __popcnt(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_popcount)

#define PopCount32(x) \
        __builtin_popcount(x)

#else
#error "PopCount32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for PopCount32"
#endif
#endif

#ifndef PopCount64
#if LEIBNIZ_COMPILER_MSVC

#define PopCount64(x) \
    __popcnt64(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_popcountll)

#define PopCount64(x) \
        __builtin_popcountll(x)

#else
#error "PopCount64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for PopCount64"
#endif
#endif

#ifndef RotateLeft32
#if LEIBNIZ_COMPILER_MSVC

#define RotateLeft32(x, n) \
    _lrotl((x), (n))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define RotateLeft32(x, n) \
    (((x) << ((n) & 31)) | ((x) >> (32 - ((n) & 31))))

#else
#error "Unsupported compiler for RotateLeft32"
#endif
#endif

#ifndef RotateRight32
#if LEIBNIZ_COMPILER_MSVC

#define RotateRight32(x, n) \
    _lrotr((x), (n))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define RotateRight32(x, n) \
    (((x) >> ((n) & 31)) | ((x) << (32 - ((n) & 31))))

#else
#error "Unsupported compiler for RotateRight32"
#endif
#endif

#ifndef RotateLeft64
#if LEIBNIZ_COMPILER_MSVC

#define RotateLeft64(x, n) \
    _rotl64((x), (n))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define RotateLeft64(x, n) \
    (((x) << ((n) & 63)) | ((x) >> (64 - ((n) & 63))))

#else
#error "Unsupported compiler for RotateLeft64"
#endif
#endif

#ifndef RotateRight64
#if LEIBNIZ_COMPILER_MSVC

#define RotateRight64(x, n) \
    _rotr64((x), (n))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define RotateRight64(x, n) \
    (((x) >> ((n) & 63)) | ((x) << (64 - ((n) & 63))))

#else
#error "Unsupported compiler for RotateRight64"
#endif
#endif
	
#ifndef ByteSwap16
#if LEIBNIZ_COMPILER_MSVC

#define ByteSwap16(x) \
    _byteswap_ushort(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_bswap16)

#define ByteSwap16(x) \
        __builtin_bswap16(x)

#else
#error "ByteSwap16 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for ByteSwap16"
#endif
#endif

#ifndef ByteSwap32
#if LEIBNIZ_COMPILER_MSVC

#define ByteSwap32(x) \
    _byteswap_ulong(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_bswap32)

#define ByteSwap32(x) \
        __builtin_bswap32(x)

#else
#error "ByteSwap32 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for ByteSwap32"
#endif
#endif

#ifndef ByteSwap64
#if LEIBNIZ_COMPILER_MSVC

#define ByteSwap64(x) \
    _byteswap_uint64(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_bswap64)

#define ByteSwap64(x) \
        __builtin_bswap64(x)

#else
#error "ByteSwap64 intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for ByteSwap64"
#endif
#endif

#ifndef IsNaN
#if LEIBNIZ_COMPILER_MSVC

#define IsNaN(x) \
    __is_nan(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#if defined(__has_builtin)
#if __has_builtin(__builtin_isnan)

#define IsNaN(x) \
        __builtin_isnan(x)

#else
#error "IsNaN intrinsic missing"
#endif
#else
#error "__has_builtin not supported by this compiler"
#endif

#else
#error "Unsupported compiler for IsNaN"
#endif
#endif

#ifndef IsNaNF
#if LEIBNIZ_COMPILER_MSVC

#define IsNaNF(x) \
    __is_nanf(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsNaNF(x) \
    __builtin_isnan(x)

#else
#error "Unsupported compiler for IsNaNF"
#endif
#endif

#ifndef IsNaNL
#if LEIBNIZ_COMPILER_MSVC

#define IsNaNL(x) \
    __is_nanl(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsNaNL(x) \
    __builtin_isnan(x)

#else
#error "Unsupported compiler for IsNaNL"
#endif
#endif

#ifndef IsFinite
#if LEIBNIZ_COMPILER_MSVC

#define IsFinite(x) \
    __is_finite(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsFinite(x) \
    __builtin_isfinite(x)

#else
#error "Unsupported compiler for IsFinite"
#endif
#endif

#ifndef IsInfinity
#if LEIBNIZ_COMPILER_MSVC

#define IsInfinity(x) \
    __is_infinity(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsInfinity(x) \
    __builtin_isinf(x)

#else
#error "Unsupported compiler for IsInfinity"
#endif
#endif


#ifndef IsSubnormal
#if LEIBNIZ_COMPILER_MSVC

#define IsSubnormal(x) \
    __is_subnormal(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsSubnormal(x) \
    (__builtin_fpclassify(x) == FP_SUBNORMAL)

#else
#error "Unsupported compiler for IsSubnormal"
#endif
#endif

#ifndef IsNormal
#if LEIBNIZ_COMPILER_MSVC

#define IsNormal(x) \
    __is_normal(x)

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsNormal(x) \
    __builtin_isnormal(x)

#else
#error "Unsupported compiler for IsNormal"
#endif
#endif

#ifndef IsUnordered
#if LEIBNIZ_COMPILER_MSVC

#define IsUnordered(a, b) \
    _isunordered((a), (b))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsUnordered(a, b) \
    __builtin_isunordered((a), (b))

#else
#error "Unsupported compiler for IsUnordered"
#endif
#endif

#ifndef IsUnorderedF
#if LEIBNIZ_COMPILER_MSVC

#define IsUnorderedF(a, b) \
    _isunorderedf((a), (b))

#elif LEIBNIZ_COMPILER_GCC || LEIBNIZ_COMPILER_CLANG

#define IsUnorderedF(a, b) \
    __builtin_isunordered((a), (b))

#else
#error "Unsupported compiler for IsUnorderedF"
#endif
#endif

}

