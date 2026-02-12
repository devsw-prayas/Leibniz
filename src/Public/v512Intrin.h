#pragma once
#include <Leibniz.h>
#define ALLOW_VECTOR_INTRIN_256
#include <v256Intrin.h>
#ifdef ALLOW_VECTOR_INTRIN_512
#if !defined(__AVX512__) && !defined(LEIBNIZ_EDITOR_MODE)
#error "Leibniz SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else
namespace Leibniz::Vectorization::Intrinsic::v512 {
	using r512i = __m512i;
	using r512f32 = __m512;
	using r512f64 = __m512d;
}
#endif
#endif
