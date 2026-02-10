#pragma once
#include "Leibniz.h"

namespace Leibniz::Traits {
	// ---- Numeric Properties ----

	template<typename T> struct LEIBNIZ_RUNTIME_API IsNumeric : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsNumericV = IsNumeric<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsSymbolic : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsSymbolicV = IsSymbolic<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsReal : std::false_type {};
	template<typename  T> inline constexpr LEIBNIZ_RUNTIME_API bool IsRealV = IsReal<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsComplex : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsComplexV = IsComplex<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsInt : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsIntV = IsInt<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsFloat : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsFloatV = IsFloat<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsSignedInt : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsSignedIntV = IsSignedInt<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsUnsignedInt : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsUnsignedIntV = IsUnsignedInt<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsRational : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsRationalV = IsRational<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsIrrational : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsIrrationalV = IsIrrational<T>::value;

	// ---- Precision and Width ----

	template<typename T> struct LEIBNIZ_RUNTIME_API IsFinitePrecision : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsFinitePrecisionV = IsFinitePrecision<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsArbitraryPrecision : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsArbitraryPrecisionV = IsArbitraryPrecision<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsFiniteWidth : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsFiniteWidthV = IsFiniteWidth<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsArbitraryWidth : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsArbitraryWidthV = IsArbitraryWidth<T>::value;

	// ---- Exactness And Approximation ----

	template<typename T> struct LEIBNIZ_RUNTIME_API IsExact final : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsExactV = IsExact<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsApproximated final : std::false_type {};
	template<typename T> inline constexpr LEIBNIZ_RUNTIME_API bool IsApproximatedV = IsApproximated<T>::value;
}