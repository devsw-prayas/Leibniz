#pragma once
#include "Leibniz.h"

namespace Leibniz::Traits {
	// ---- Numeric Properties ----

	template<typename T> struct LEIBNIZ IsNumeric : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsNumericV = IsNumeric<T>::value;

	template<typename T> struct LEIBNIZ IsSymbolic : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsSymbolicV = IsSymbolic<T>::value;

	template<typename T> struct LEIBNIZ IsReal : std::false_type {};
	template<typename  T> inline constexpr LEIBNIZ bool IsRealV = IsReal<T>::value;

	template<typename T> struct LEIBNIZ IsComplex : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsComplexV = IsComplex<T>::value;

	template<typename T> struct LEIBNIZ IsInt : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsIntV = IsInt<T>::value;

	template<typename T> struct LEIBNIZ IsFloat : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsFloatV = IsFloat<T>::value;

	template<typename T> struct LEIBNIZ IsSignedInt : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsSignedIntV = IsSignedInt<T>::value;

	template<typename T> struct LEIBNIZ IsUnsignedInt : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsUnsignedIntV = IsUnsignedInt<T>::value;

	template<typename T> struct LEIBNIZ IsRational : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsRationalV = IsRational<T>::value;

	template<typename T> struct LEIBNIZ IsIrrational : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsIrrationalV = IsIrrational<T>::value;

	// ---- Precision and Width ----

	template<typename T> struct LEIBNIZ IsFinitePrecision : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsFinitePrecisionV = IsFinitePrecision<T>::value;

	template<typename T> struct LEIBNIZ IsArbitraryPrecision : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsArbitraryPrecisionV = IsArbitraryPrecision<T>::value;

	template<typename T> struct LEIBNIZ IsFiniteWidth : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsFiniteWidthV = IsFiniteWidth<T>::value;

	template<typename T> struct LEIBNIZ IsArbitraryWidth : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsArbitraryWidthV = IsArbitraryWidth<T>::value;

	// ---- Exactness And Approximation ----

	template<typename T> struct LEIBNIZ IsExact final : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsExactV = IsExact<T>::value;

	template<typename T> struct LEIBNIZ IsApproximated final : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ bool IsApproximatedV = IsApproximated<T>::value;
}