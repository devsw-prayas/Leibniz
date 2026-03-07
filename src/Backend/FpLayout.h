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

	using ieeeQuadPrecision = FpFormat<128, 15, 12>;
	using leibnizOctaPrecision = FpFormat<256, 23, 232>;
	using leibnizHalfRefWidthPrecision = uint64_t; // TODO Format	    512bit
	using leibnizFullRefWidthPrecision = uint64_t; // TODO Format		1024bit
}
