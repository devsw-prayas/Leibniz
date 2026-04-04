#pragma once
#include "Leibniz.h"
#include "NumericTraits.h"

namespace Leibniz::Numerics {
	using Float32 = float;
	using Float64 = double;
}

namespace Leibniz::Traits {
	template<> struct IsNumeric<Numerics::Float32> : std::true_type {};
	template<> struct IsFloat<Numerics::Float32> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::Float32> : std::true_type {};
	template<> struct IsApproximated<Numerics::Float32> : std::true_type {};
	template<> struct IsCompilerType<Numerics::Float32> : std::true_type {};

	template<> struct IsNumeric<Numerics::Float64> : std::true_type {};
	template<> struct IsFloat<Numerics::Float64> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::Float64> : std::true_type {};
	template<> struct IsApproximated<Numerics::Float64> : std::true_type {};
	template<> struct IsCompilerType<Numerics::Float64> : std::true_type {};
}
