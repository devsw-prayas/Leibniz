#pragma once
#include <bit>

#include "LeibnizInt.h"
#include "FpLayout.h"
#include "LeibnizFloat.h"
#include "VectorizationTraits.h"

namespace Leibniz::Numerics::Support {
	template<typename T>
	struct BitMask final {
		using Masking = void;
	};

	template<>
	struct BitMask<Numbers::Fp::ieeeSinglePrecision> final {
		using Masking = UInt32;
	};

	template<>
	struct BitMask<Numbers::Fp::ieeeDoublePrecision> final {
		using Masking = UInt64;
	};

	using FloatingPoint32 = Numbers::Fp::ieeeSinglePrecision;
	using FloatingPoint64 = Numbers::Fp::ieeeDoublePrecision;

	template<typename T>
	class FpMasking final {
		using Format = T;
		using Mask = BitMask<T>::Masking;
	public:
		static constexpr Mask signMask() {
			return Mask{ 1 } << (Format::exponent + Format::mantissa);
		}

		static constexpr Mask expMask() {
			return ((Mask{ 1 } << Format::exponent) - 1) << Format::mantissa;
		}

		static constexpr Mask mantissaMask() {
			return	(Mask{ 1 } << Format::mantissa) - 1;
		}

		static constexpr Mask absMask() {
			return expMask() | mantissaMask();
		}

		static constexpr Mask infBits() {
			return expMask();
		}

		static constexpr Mask quietNanBit() {
			return Mask{ 1 } << (Format::mantissa - 1);
		}

		static constexpr Mask expBias() {
			return (Mask{ 1 } << (Format::exponent - 1)) - 1;
		}

		static constexpr Mask maxExpRaw() {
			return (Mask{ 1 } << Format::exponent) - 1;
		}
	};

	template<typename T>
	class FpUtils final {
		LEIBNIZ_STATIC_ASSERT(!std::is_same_v<typename BitMask<T>::Masking, void>, "FpMasking: unsupported format");
		using Mask = BitMask<T>::Masking;
	public:
		static constexpr Mask extractSign(Mask bits) {
			return bits & FpMasking<T>::signMask();
		}

		static constexpr Mask extractExponent(Mask bits) {
			return bits & FpMasking<T>::expMask();
		}

		static constexpr Mask extractMantissa(Mask bits) {
			return bits & FpMasking<T>::mantissaMask();
		}

		static constexpr Mask extractAbs(Mask bits) {
			return bits & FpMasking<T>::absMask();
		}
	};

	template<typename T>
	LEIBNIZ_FORCEINLINE constexpr T trueVal() noexcept {
		LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported ooperations");
		LEIBNIZ_UNREACHABLE();
	};

	template<typename T>
	LEIBNIZ_FORCEINLINE constexpr T falseVal() noexcept {
		LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported ooperations");
		LEIBNIZ_UNREACHABLE();
	}

	// --- Integer specializations ---

	template<> LEIBNIZ_FORCEINLINE constexpr Int32  trueVal<Int32>()  noexcept { return ~Int32(0); }
	template<> LEIBNIZ_FORCEINLINE constexpr Int64  trueVal<Int64>()  noexcept { return ~Int64(0); }
	template<> LEIBNIZ_FORCEINLINE constexpr UInt32 trueVal<UInt32>() noexcept { return ~UInt32(0); }
	template<> LEIBNIZ_FORCEINLINE constexpr UInt64 trueVal<UInt64>() noexcept { return ~UInt64(0); }

	template<> LEIBNIZ_FORCEINLINE constexpr Int32  falseVal<Int32>()  noexcept { return Int32(0); }
	template<> LEIBNIZ_FORCEINLINE constexpr Int64  falseVal<Int64>()  noexcept { return Int64(0); }
	template<> LEIBNIZ_FORCEINLINE constexpr UInt32 falseVal<UInt32>() noexcept { return UInt32(0); }
	template<> LEIBNIZ_FORCEINLINE constexpr UInt64 falseVal<UInt64>() noexcept { return UInt64(0); }

	// --- Float specializations (std::bit_cast) ---

	template<> LEIBNIZ_FORCEINLINE constexpr Float32 trueVal<Float32>() noexcept {
		return std::bit_cast<Float32>(0xFFFFFFFFu);
	}
	template<> LEIBNIZ_FORCEINLINE constexpr Float32 falseVal<Float32>() noexcept {
		return std::bit_cast<Float32>(0x00000000u);
	}
	template<> LEIBNIZ_FORCEINLINE constexpr Float64 trueVal<Float64>() noexcept {
		return std::bit_cast<Float64>(0xFFFFFFFFFFFFFFFFull);
	}
	template<> LEIBNIZ_FORCEINLINE constexpr Float64 falseVal<Float64>() noexcept {
		return std::bit_cast<Float64>(0x0000000000000000ull);
	}
}
