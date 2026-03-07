#pragma once
#include "BackendInternalHelpers.h"

namespace Leibniz::Numbers::Backend::Internal::Kernels {
	LEIBNIZ_FORCEINLINE uint64_t reciprocalSeed(uint64_t v_Div) {
		int s = Clz64(static_cast<unsigned long long>(v_Div));
		uint64_t divN = v_Div << s;
		uint64_t x0 = ~v_Div;
		Types::U128 t = mul128(divN, x0);
		x0 = x0 * (2 - (t.high >> 63));
		return x0;
	}
}
