#pragma once
#include "Leibniz.h"
#include "StaticPipelines.h"
#include "LeibnizCompiler.h"
#include "LebnizDiagnostics.h"
#include "LeibnizInt.h"
#include "LeibnizFloat.h"

namespace Leibniz::LinAlg::Internal::Kernels::Scalar {
	// LEIBNIZ - BASE ELEMENTWISE OPERATIONS (CANONICAL LIST)

	// TIER 0 - CORE ARITHMETIC (MANDATORY)

	// add(a, b)        // a + b
	// sub(a, b)        // a - b
	// mul(a, b)        // a * b
	// div(a, b)        // a / b
	// neg(a)           // -a

	// TIER 0 - COMPARISON (MANDATORY)

	// eq(a, b)         // a == b
	// neq(a, b)        // a != b
	// lt(a, b)         // a < b
	// le(a, b)         // a <= b
	// gt(a, b)         // a > b
	// ge(a, b)         // a >= b

	// TIER 1 - MASK / SELECTION (CRITICAL)

	// select(mask, a, b)     // mask ? a : b (branchless)
	// where(mask, a, b)      // alias for select

	// TIER 1 - BASIC NUMERIC UTILITIES

	// min(a, b)
	// max(a, b)
	// clamp(x, lo, hi)

	// abs(x)
	// sign(x)

	// TIER 2 - PERFORMANCE / FUSED OPS (HPC CRITICAL)

	// fma(a, b, c)     // a*b + c (single rounding)
	// square(x)        // x * x
	// reciprocal(x)    // 1 / x

	// TIER 3 - ROUNDING / FRACTIONAL

	// floor(x)
	// ceil(x)
	// round(x)
	// trunc(x)
	// frac(x)          // x - floor(x)

	// TIER 3 - CLASSIFICATION / SAFETY

	// isnan(x)
	// isinf(x)
	// isfinite(x)

	// replace_nan(x, value)
	// replace_inf(x, value)

	// TIER 4 - EXPONENTIAL / LOGARITHMIC

	// exp(x)
	// exp2(x)
	// log(x)
	// log2(x)
	// pow(x, y)

	// TIER 4 - ROOTS

	// sqrt(x)
	// rsqrt(x)
	// cbrt(x)

	// TIER 4 - TRIGONOMETRIC

	// sin(x)
	// cos(x)
	// tan(x)

	// TIER 4 - INVERSE TRIGONOMETRIC

	// asin(x)
	// acos(x)
	// atan(x)
	// atan2(y, x)

	// TIER 4 - HYPERBOLIC

	// sinh(x)
	// cosh(x)
	// tanh(x)

	// TIER 4 - ML / SPECIAL

	// sigmoid(x)       // 1 / (1 + exp(-x))

	// TIER 5 - GENERIC ELEMENTWISE ABSTRACTION (TENSOR CORE)

	// map(op, x)               // unary elementwise
	// map2(op, a, b)           // binary elementwise
	// map3(op, a, b, c)        // ternary elementwise

	// OPTIONAL (BUT STRONGLY RECOMMENDED FOR COMPLETENESS)

	// expm1(x)        // exp(x) - 1 (stable small x)
	// log1p(x)        // log(1 + x)
	// erf(x)
	// erfc(x)

	// nearly_equal(a, b, eps)
	// next_after(a, b)
	// next_up(x)
	// next_down(x)

	// Elementwise add

	using cst_u32_ptr = const Numerics::UInt32*;
	using cst_u64_ptr = const Numerics::UInt64*;
	using u32_ptr = Numerics::UInt32*;
	using u64_ptr = Numerics::UInt64*;

	using cst_i32_ptr = const Numerics::Int32*;
	using cst_i64_ptr = const Numerics::Int64*;
	using i32_ptr = Numerics::Int32*;
	using i64_ptr = Numerics::Int64*;

	using cst_f32_ptr = const Numerics::Float32*;
	using cst_f64_ptr = const Numerics::Float64*;
	using f32_ptr = Numerics::Float32*;
	using f64_ptr = Numerics::Float64*;

	// Unrolled Scalar addition operations for upto N <= 8

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarAddElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] + inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] + inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarAddElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] + inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] + inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarAddElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] + inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] + inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarAddElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] + inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] + inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarAddElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] + inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] + inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarAddElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] + inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] + inB[idx];
										 });
#endif
	}

	// Unrolled Scalar subtraction operations for upto N <= 8

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarSubElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] - inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] - inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarSubElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] - inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] - inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarSubElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] - inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] - inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarSubElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] - inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] - inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarSubElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] - inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] - inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarSubElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] - inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] - inB[idx];
										 });
#endif
	}

	// Unrolled Scalar multiplication operations for upto N <= 8

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarMulElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] * inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] * inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarMulElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] * inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] * inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarMulElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] * inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] * inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarMulElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] * inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] * inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarMulElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] * inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] * inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarMulElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] * inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] * inB[idx];
										 });
#endif
	}

	// Unrolled Scalar division operations for upto N <= 8

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarDivideElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] / inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] / inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarDivideElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] / inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] / inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarDivideElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] / inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] / inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarDivideElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] / inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] / inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarDivideElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] / inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] / inB[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarDivideElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr);			LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = inA[i] / inB[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = inA[idx] / inB[idx];
										 });
#endif
	}

	// Unrolled Scalar Negate operations for upto N <= 8

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarNegateElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			i32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = -inA[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = -inA[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarNegateElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			i64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = -inA[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = -inA[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarNegateElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			f32_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = -inA[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = -inA[idx];
										 });
#endif
	}

	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarNegateElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			f64_ptr LEIBNIZ_RESTRICT out
		) {
		LEIBNIZ_ASSERT(inA != nullptr);	   		LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(out != nullptr);			LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++)
			out[i] = -inA[i];
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		Utils::Unrolled::staticFor<0, N>([inA, inB, out](auto idx) {
			out[idx] = -inA[idx];
										 });
#endif
	}

	// i32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpEqualElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] == inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] == inB[idx]; });
#endif
	}

	// i64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpEqualElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] == inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] == inB[idx]; });
#endif
	}

	// u32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpEqualElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] == inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] == inB[idx]; });
#endif
	}

	// u64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpEqualElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] == inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] == inB[idx]; });
#endif
	}

	// f32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpEqualElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] == inB[i];		 //TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] == inB[idx]; });
#endif
	}

	// f64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpEqualElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] == inB[i];		  // TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] == inB[idx]; });
#endif
	}

	// ===================== CmpNotEqual =====================
	// i32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpNotEqualElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] != inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] != inB[idx]; });
#endif
	}
	// i64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpNotEqualElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] != inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] != inB[idx]; });
#endif
	}
	// u32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpNotEqualElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] != inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] != inB[idx]; });
#endif
	}
	// u64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpNotEqualElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] != inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] != inB[idx]; });
#endif
	}
	// f32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpNotEqualElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] != inB[i];		 //TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] != inB[idx]; });
#endif
	}
	// f64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpNotEqualElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] != inB[i];		  // TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] != inB[idx]; });
#endif
	}

	// ===================== CmpLessThan =====================
	// i32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] < inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] < inB[idx]; });
#endif
	}
	// i64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] < inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] < inB[idx]; });
#endif
	}
	// u32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] < inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] < inB[idx]; });
#endif
	}
	// u64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] < inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] < inB[idx]; });
#endif
	}
	// f32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] < inB[i];		 //TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] < inB[idx]; });
#endif
	}
	// f64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] < inB[i];		  // TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] < inB[idx]; });
#endif
	}

	// ===================== CmpLessThanOrEqual =====================
	// i32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanOrEqualElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] <= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] <= inB[idx]; });
#endif
	}
	// i64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanOrEqualElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] <= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] <= inB[idx]; });
#endif
	}
	// u32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanOrEqualElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] <= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] <= inB[idx]; });
#endif
	}
	// u64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanOrEqualElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] <= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] <= inB[idx]; });
#endif
	}
	// f32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanOrEqualElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] <= inB[i];		 //TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] <= inB[idx]; });
#endif
	}
	// f64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpLessThanOrEqualElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] <= inB[i];		  // TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] <= inB[idx]; });
#endif
	}

	// ===================== CmpGreater =====================
	// i32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] > inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] > inB[idx]; });
#endif
	}
	// i64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] > inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] > inB[idx]; });
#endif
	}
	// u32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] > inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] > inB[idx]; });
#endif
	}
	// u64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] > inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] > inB[idx]; });
#endif
	}
	// f32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] > inB[i];		 //TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] > inB[idx]; });
#endif
	}
	// f64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] > inB[i];		  // TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] > inB[idx]; });
#endif
	}

	// ===================== CmpGreaterThanOrEqual =====================
	// i32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterThanOrEqualElements_i32(
			cst_i32_ptr LEIBNIZ_RESTRICT inA,
			cst_i32_ptr LEIBNIZ_RESTRICT inB,
			i32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] >= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] >= inB[idx]; });
#endif
	}
	// i64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterThanOrEqualElements_i64(
			cst_i64_ptr LEIBNIZ_RESTRICT inA,
			cst_i64_ptr LEIBNIZ_RESTRICT inB,
			i64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] >= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] >= inB[idx]; });
#endif
	}
	// u32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterThanOrEqualElements_u32(
			cst_u32_ptr LEIBNIZ_RESTRICT inA,
			cst_u32_ptr LEIBNIZ_RESTRICT inB,
			u32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] >= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] >= inB[idx]; });
#endif
	}
	// u64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterThanOrEqualElements_u64(
			cst_u64_ptr LEIBNIZ_RESTRICT inA,
			cst_u64_ptr LEIBNIZ_RESTRICT inB,
			u64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] >= inB[i];
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] >= inB[idx]; });
#endif
	}
	// f32
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterThanOrEqualElements_f32(
			cst_f32_ptr LEIBNIZ_RESTRICT inA,
			cst_f32_ptr LEIBNIZ_RESTRICT inB,
			f32_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] >= inB[i];		 //TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] >= inB[idx]; });
#endif
	}
	// f64
	template<size_t N> requires N >= 8
		LEIBNIZ_FORCEINLINE void unrolledScalarCmpGreaterThanOrEqualElements_f64(
			cst_f64_ptr LEIBNIZ_RESTRICT inA,
			cst_f64_ptr LEIBNIZ_RESTRICT inB,
			f64_ptr LEIBNIZ_RESTRICT out) {
		LEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
		LEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
		LEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);
#if LEIBNIZ_COMPILER_MSVC
		for (size_t i = 0; i < N; i++) out[i] = inA[i] >= inB[i];		  // TODO
#else
		Utils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = inA[idx] >= inB[idx]; });
#endif
	}
}
