#pragma once
#include "Lebnitz.h"
#include "NumericTraits.h"
#include "VectorizationTraits.h"
#include "LebtnitzInt.h"
#include "immintrin.h"

namespace Lebnitz::Vectorization::v256 {
	struct StripeU8 final {
	private:
	public:
		StripeU8(Numerics::UByte(&ra_Vector)[32]);
	};
}

namespace Lebnitz::Vectorization::Traits {
	template<>
	struct VectorizationIntrospect<v256::StripeU8> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 32;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeU8 [Unsigned Integer 1 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};
}