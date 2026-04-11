#pragma once
//
// FloatIntrin.h
// Platform abstraction for FP classification and rounding.
// Wraps MSVC <float.h> and GCC/Clang __builtin_* uniformly.
// DO NOT call __builtin_* or _fpclass directly outside this header.
//

#include "LeibnizCompiler.h"
#include "LeibnizFloat.h"
#include <type_traits>

#if LEIBNIZ_COMPILER_MSVC
#include <cfloat>   // _isnan, _finite, _fpclass, _FPCLASS_*
#endif

namespace Leibniz::FloatIntrin {
	//
	// Classification
	//

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE bool isNan(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return _isnan(static_cast<double>(x)) != 0;
#else
		return __builtin_isnan(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE bool isInf(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return (_fpclass(static_cast<double>(x)) & (_FPCLASS_NINF | _FPCLASS_PINF)) != 0;
#else
		return __builtin_isinf(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE bool isFinite(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return _finite(static_cast<double>(x)) != 0;
#else
		return __builtin_isfinite(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE bool isPosInf(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return (_fpclass(static_cast<double>(x)) & _FPCLASS_PINF) != 0;
#else
		return __builtin_isinf(x) && x > T(0);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE bool isNegInf(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return (_fpclass(static_cast<double>(x)) & _FPCLASS_NINF) != 0;
#else
		return __builtin_isinf(x) && x < T(0);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE bool isSubnormal(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return (_fpclass(static_cast<double>(x)) & (_FPCLASS_ND | _FPCLASS_PD)) != 0;
#else
		return __builtin_fpclassify(0, 0, 0, 1, 0, x);
#endif
	}

	//
	// Rounding
	//

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE T floor(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		const T t = static_cast<T>(static_cast<long long>(x));
		return (t > x) ? t - T(1) : t;
#else
		if constexpr (std::is_same_v<T, Numerics::Float32>)
			return __builtin_floorf(x);
		else
			return __builtin_floor(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE T ceil(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		const T t = static_cast<T>(static_cast<long long>(x));
		return (t < x) ? t + T(1) : t;
#else
		if constexpr (std::is_same_v<T, Numerics::Float32>)
			return __builtin_ceilf(x);
		else
			return __builtin_ceil(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE T trunc(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return static_cast<T>(static_cast<long long>(x));
#else
		if constexpr (std::is_same_v<T, Numerics::Float32>)
			return __builtin_truncf(x);
		else
			return __builtin_trunc(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE T round(T x) noexcept {
#if LEIBNIZ_COMPILER_MSVC
		return static_cast<T>(static_cast<long long>(x + (x >= T(0) ? T(0.5) : T(-0.5))));
#else
		if constexpr (std::is_same_v<T, Numerics::Float32>)
			return __builtin_roundf(x);
		else
			return __builtin_round(x);
#endif
	}

	template<typename T>
	[[nodiscard]] LEIBNIZ_FORCEINLINE T frac(T x) noexcept {
		return x - Leibniz::FloatIntrin::floor(x);
	}
}
