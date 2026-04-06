#pragma once
#include "LeibnizInt.h"
#include "FpLayout.h"

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
}
