#pragma once
#include "Lebnitz.h"

namespace Lebnitz::Traits {
	// ---- Numeric Properties ----

	template<typename T> struct LEBNITZ IsNumeric : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsNumericV = IsNumeric<T>::value;

	template<typename T> struct LEBNITZ IsSymbolic : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsSymbolicV = IsSymbolic<T>::value;

	template<typename T> struct LEBNITZ IsReal : std::false_type {};
	template<typename  T> inline constexpr LEBNITZ bool IsRealV = IsReal<T>::value;

	template<typename T> struct LEBNITZ IsComplex : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsComplexV = IsComplex<T>::value;

	template<typename T> struct LEBNITZ IsInt : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsIntV = IsInt<T>::value;

	template<typename T> struct LEBNITZ IsFloat : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsFloatV = IsFloat<T>::value;

	template<typename T> struct LEBNITZ IsSignedInt : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsSignedIntV = IsSignedInt<T>::value;

	template<typename T> struct LEBNITZ IsUnsignedInt : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsUnsignedIntV = IsUnsignedInt<T>::value;

	template<typename T> struct LEBNITZ IsRational : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsRationalV = IsRational<T>::value;

	template<typename T> struct LEBNITZ IsIrrational : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsIrrationalV = IsIrrational<T>::value;

	// ---- Precision and Width ----

	template<typename T> struct LEBNITZ IsFinitePrecision : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsFinitePrecisionV = IsFinitePrecision<T>::value;

	template<typename T> struct LEBNITZ IsArbitraryPrecision : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsArbitraryPrecisionV = IsArbitraryPrecision<T>::value;

	template<typename T> struct LEBNITZ IsFiniteWidth : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsFiniteWidthV = IsFiniteWidth<T>::value;

	template<typename T> struct LEBNITZ IsArbitraryWidth : std::false_type {};
	template<typename T> inline constexpr LEBNITZ bool IsArbitraryWidthV = IsArbitraryWidth<T>::value;
}