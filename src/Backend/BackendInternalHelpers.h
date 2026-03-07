#pragma once
#include "BackendInternalTypes.h"
#include "MathIntrin.h"

namespace Leibniz::Numbers::Backend::Internal {
	LEIBNIZ_FORCEINLINE Types::U128 mul128(uint64_t v_OpA, uint64_t v_OpB) {
#if LEIBNIZ_COMPILER_MSVC
		Types::U128 r;
		r.low = _umul128(v_OpA, v_OpB, &r.high);
		return r;
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
		__uint128_t r = (__uint128_t) v_OpA * v_OpB;

		return {
			(uint64_t) r,
			(uint64_t) (r >> 64)
		};
#endif
	}
}
