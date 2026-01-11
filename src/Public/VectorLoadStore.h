#pragma once
#include <immintrin.h>

namespace Lebnitz::Vectorization::Utils {
	// -----------------------------------------------------------------------------
	// SIMD backend specialization hooks
	//
	// The following templates are intentionally forward-declared and left undefined
	// in the generic case.
	//
	// Each SIMD backend (e.g. AVX2) is expected to provide *full specializations*
	// of these templates for supported scalar types (float, double, int32, etc.).
	//
	// Responsibilities:
	// - These specializations define the *mechanical* mapping from memory to
	//   hardware SIMD registers (load/store, aligned/masked variants).
	// - They must contain backend-specific intrinsics and register types.
	// - They MUST NOT encode semantic meaning, arithmetic behavior, or algorithms.
	//
	// Higher-level abstractions (e.g. Stripe<T>) delegate to these specializations
	// to obtain raw SIMD registers, then define all numeric semantics on top.
	//
	// If no specialization exists for a type T, that type is considered
	// non-vectorizable on this backend.
	// -----------------------------------------------------------------------------																			

	template<typename T> struct VectorLoad;
	template<typename T> struct VectorStore;
}
