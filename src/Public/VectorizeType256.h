#pragma once
#include "Leibniz.h"
#include "NumericTraits.h"
#include "VectorizationTraits.h"
#include "LeibnizInt.h"
#include "immintrin.h"

namespace Leibniz::Vectorization::v256 {
#if defined(__AVX2__)													
	using Reg256i = __m256i;

	struct LEBNITZ alignas(32) StripeU8 final {
		Reg256i m_VectorBin;

		StripeU8(const StripeU8&) = default;
		StripeU8& operator=(const StripeU8&) = default;

		StripeU8(StripeU8&&) noexcept = default;
		StripeU8& operator=(StripeU8&&) noexcept = default;
	};

	struct LEBNITZ alignas(32) StripeU16 final {
		Reg256i m_VectorBin;

		StripeU16(const StripeU16&) = default;
		StripeU16& operator=(const StripeU16&) = default;

		StripeU16(StripeU16&&) noexcept = default;
		StripeU16& operator=(StripeU16&&) noexcept = default;
	};

	struct LEBNITZ alignas(32) StripeU32 final {
		Reg256i m_VectorBin;

		StripeU32(const StripeU32&) = default;
		StripeU32& operator=(const StripeU32&) = default;

		StripeU32(StripeU32&&) noexcept = default;
		StripeU32& operator=(StripeU32&&) noexcept = default;
	};

	struct LEBNITZ alignas(32) StripeU64 final {
		Reg256i m_VectorBin;

		StripeU64(const StripeU64&) = default;
		StripeU64& operator=(const StripeU64&) = default;

		StripeU64(StripeU64&&) noexcept = default;
		StripeU64& operator=(StripeU64&&) noexcept = default;
	};

	static_assert()
#endif
}

namespace Leibniz::Vectorization::Traits {
	template<>
	struct VectorizationIntrospect<v256::StripeU8> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 32;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeU8 [Unsigned Integer 1 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};
}