#pragma once
#include "Leibniz.h"
#include "NumericTraits.h"

namespace Leibniz::Numerics {
	using Byte = std::int8_t;
	using Int16 = std::int16_t;
	using Int32 = std::int32_t;
	using Int64 = std::int64_t;

	using UByte = std::uint8_t;
	using UInt16 = std::uint16_t;
	using UInt32 = std::uint32_t;
	using UInt64 = std::uint64_t;
}

namespace Leibniz::Traits {
	// ---- Signed Integers ----

	template<> struct IsNumeric<Numerics::Byte> : std::true_type {};
	template<> struct IsInt<Numerics::Byte> : std::true_type {};
	template<> struct IsSignedInt<Numerics::Byte> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::Byte> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::Byte> : std::true_type {};
	template<> struct IsCompilerType<Numerics::Byte> : std::true_type {};

	template<> struct IsNumeric<Numerics::Int16> : std::true_type {};
	template<> struct IsInt<Numerics::Int16> : std::true_type {};
	template<> struct IsSignedInt<Numerics::Int16> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::Int16> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::Int16> : std::true_type {};
	template<> struct IsCompilerType<Numerics::Int16> : std::true_type {};

	template<> struct IsNumeric<Numerics::Int32> : std::true_type {};
	template<> struct IsInt<Numerics::Int32> : std::true_type {};
	template<> struct IsSignedInt<Numerics::Int32> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::Int32> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::Int32> : std::true_type {};
	template<> struct IsCompilerType<Numerics::Int32> : std::true_type {};

	template<> struct IsNumeric<Numerics::Int64> : std::true_type {};
	template<> struct IsInt<Numerics::Int64> : std::true_type {};
	template<> struct IsSignedInt<Numerics::Int64> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::Int64> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::Int64> : std::true_type {};
	template<> struct IsCompilerType<Numerics::Int64> : std::true_type {};

	// ---- Unsigned Integers ----

	template<> struct IsNumeric<Numerics::UByte> : std::true_type {};
	template<> struct IsInt<Numerics::UByte> : std::true_type {};
	template<> struct IsUnsignedInt<Numerics::UByte> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::UByte> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::UByte> : std::true_type {};
	template<> struct IsCompilerType<Numerics::UByte> : std::true_type {};

	template<> struct IsNumeric<Numerics::UInt16> : std::true_type {};
	template<> struct IsInt<Numerics::UInt16> : std::true_type {};
	template<> struct IsUnsignedInt<Numerics::UInt16> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::UInt16> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::UInt16> : std::true_type {};
	template<> struct IsCompilerType<Numerics::UInt16> : std::true_type {};

	template<> struct IsNumeric<Numerics::UInt32> : std::true_type {};
	template<> struct IsInt<Numerics::UInt32> : std::true_type {};
	template<> struct IsUnsignedInt<Numerics::UInt32> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::UInt32> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::UInt32> : std::true_type {};
	template<> struct IsCompilerType<Numerics::UInt32> : std::true_type {};

	template<> struct IsNumeric<Numerics::UInt64> : std::true_type {};
	template<> struct IsInt<Numerics::UInt64> : std::true_type {};
	template<> struct IsUnsignedInt<Numerics::UInt64> : std::true_type {};
	template<> struct IsFinitePrecision<Numerics::UInt64> : std::true_type {};
	template<> struct IsFiniteWidth<Numerics::UInt64> : std::true_type {};
	template<> struct IsCompilerType<Numerics::UInt64> : std::true_type {};
}