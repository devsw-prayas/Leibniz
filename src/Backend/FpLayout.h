#pragma once
#include "LebnizDiagnostics.h"

namespace Leibniz::Numbers::Fp {
	template<size_t Bytes, size_t E, size_t M>
	struct FpFormat final {
		static constexpr size_t signBit = 1;
		static constexpr size_t exponent = E;
		static constexpr size_t mantissa = M;
		LEIBNIZ_STATIC_ASSERT(Bytes == (signBit + exponent + mantissa), "Invalid format");
	};

	using ieeeBrainPrecision = FpFormat<16, 8, 7>;
	using ieeeHalfPrecision = FpFormat<16, 5, 10>;
	using ieeeSinglePrecision = FpFormat<32, 8, 23>;
	using ieeeDoublePrecision = FpFormat<64, 11, 52>;
	using ieeeQuadPrecision = FpFormat<128, 15, 112>;
	using leibnizOctaPrecision = FpFormat<256, 23, 232>;
	using leibnizHalfRefWidthPrecision = FpFormat<512, 32, 479>;
	using leibnizFullRefWidthPrecision = FpFormat<1024, 32, 991>;
}
