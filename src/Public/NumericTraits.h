#pragma once
#include "Leibniz.h"

namespace Leibniz::Traits {
	// ---- Numeric Properties ----

	template<typename T> struct  IsNumeric : std::false_type {};
	template<typename T> constexpr  bool IsNumericV = IsNumeric<T>::value;

	template<typename T> struct  IsSymbolic : std::false_type {};
	template<typename T> constexpr  bool IsSymbolicV = IsSymbolic<T>::value;

	template<typename T> struct  IsReal : std::false_type {};
	template<typename  T> constexpr  bool IsRealV = IsReal<T>::value;

	template<typename T> struct  IsComplex : std::false_type {};
	template<typename T> constexpr  bool IsComplexV = IsComplex<T>::value;

	template<typename T> struct  IsInt : std::false_type {};
	template<typename T> constexpr  bool IsIntV = IsInt<T>::value;

	template<typename T> struct  IsFloat : std::false_type {};
	template<typename T> constexpr  bool IsFloatV = IsFloat<T>::value;

	template<typename T> struct  IsSignedInt : std::false_type {};
	template<typename T> constexpr  bool IsSignedIntV = IsSignedInt<T>::value;

	template<typename T> struct  IsUnsignedInt : std::false_type {};
	template<typename T> constexpr  bool IsUnsignedIntV = IsUnsignedInt<T>::value;

	template<typename T> struct  IsRational : std::false_type {};
	template<typename T> constexpr  bool IsRationalV = IsRational<T>::value;

	template<typename T> struct  IsIrrational : std::false_type {};
	template<typename T> constexpr  bool IsIrrationalV = IsIrrational<T>::value;

	// ---- Precision and Width ----

	template<typename T> struct  IsFinitePrecision : std::false_type {};
	template<typename T> constexpr  bool IsFinitePrecisionV = IsFinitePrecision<T>::value;

	template<typename T> struct  IsArbitraryPrecision : std::false_type {};
	template<typename T> constexpr  bool IsArbitraryPrecisionV = IsArbitraryPrecision<T>::value;

	template<typename T> struct  IsFiniteWidth : std::false_type {};
	template<typename T> constexpr  bool IsFiniteWidthV = IsFiniteWidth<T>::value;

	template<typename T> struct  IsArbitraryWidth : std::false_type {};
	template<typename T> constexpr  bool IsArbitraryWidthV = IsArbitraryWidth<T>::value;

	// ---- Exactness And Approximation ----

	template<typename T> struct  IsExact final : std::false_type {};
	template<typename T> constexpr  bool IsExactV = IsExact<T>::value;

	template<typename T> struct  IsApproximated final : std::false_type {};
	template<typename T> constexpr  bool IsApproximatedV = IsApproximated<T>::value;

	// ---- Compiler Types ---
	template<typename T> struct IsCompilerType final : std::false_type{};
	template<typename T> constexpr bool IsCompilerTypeV = IsCompilerType<T>::value;
}