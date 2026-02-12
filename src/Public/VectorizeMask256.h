#pragma once
#include "Leibniz.h"
#include "LeibnizInt.h"
#include "VectorizationTraits.h"
#include "VectorizeType256.h"

namespace Leibniz::Vectorization::msk256 {
	constexpr Numerics::UByte  EnabledLane1B = 0xFF;
	constexpr Numerics::UByte  DisabledLane1B = 0x00;

	constexpr Numerics::UInt16 EnabledLane2B = 0xFFFF;
	constexpr Numerics::UInt16 DisabledLane2B = 0x0000;

	constexpr Numerics::UInt32 EnabledLane4B = 0xFFFFFFFFu;
	constexpr Numerics::UInt32 DisabledLane4B = 0x00000000u;

	constexpr Numerics::UInt64 EnabledLane8B = 0xFFFFFFFFFFFFFFFFull;
	constexpr Numerics::UInt64 DisabledLane8B = 0x0000000000000000ull;

	using Mask256 = __m256i;

	struct LEIBNIZ_RUNTIME_API alignas(32) Mask4 final {
		Mask256	m_VectorMask;
		Mask4() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		Mask4(const Mask4&) = default;
		Mask4& operator=(const Mask4&) = default;

		Mask4(Mask4&&) noexcept = default;
		Mask4& operator=(Mask4&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Mask8 final {
		Mask256	m_VectorMask;

		Mask8() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		Mask8(const Mask8&) = default;
		Mask8& operator=(const Mask8&) = default;

		Mask8(Mask8&&) noexcept = default;
		Mask8& operator=(Mask8&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Mask16 final {
		Mask256	m_VectorMask;

		Mask16() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		Mask16(const Mask16&) = default;
		Mask16& operator=(const Mask16&) = default;

		Mask16(Mask16&&) noexcept = default;
		Mask16& operator=(Mask16&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Mask32 final {
		Mask256	m_VectorMask;

		Mask32() noexcept : m_VectorMask(_mm256_setzero_si256()) {}
		Mask32(const Mask32&) = default;
		Mask32& operator=(const Mask32&) = default;

		Mask32(Mask32&&) noexcept = default;
		Mask32& operator=(Mask32&&) noexcept = default;
	};

	static_assert(sizeof(Mask4) == 32, "Mask4 must occupy exactly one AVX2 register");
	static_assert(sizeof(Mask8) == 32, "Mask8 must occupy exactly one AVX2 register");
	static_assert(sizeof(Mask16) == 32, "Mask16 must occupy exactly one AVX2 register");
	static_assert(sizeof(Mask32) == 32, "Mask32 must occupy exactly one AVX2 register");

	static_assert(alignof(Mask4) == 32, "Mask4 must be 32-byte aligned");
	static_assert(alignof(Mask8) == 32, "Mask8 must be 32-byte aligned");
	static_assert(alignof(Mask16) == 32, "Mask16 must be 32-byte aligned");
	static_assert(alignof(Mask32) == 32, "Mask32 must be 32-byte aligned");

	static_assert(std::is_trivially_copyable_v<Mask4>, "Mask4 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Mask8>, "Mask8 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Mask16>, "Mask16 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Mask32>, "Mask32 must be trivially copyable");

	static_assert(std::is_standard_layout_v<Mask4>, "Mask4 has invalid memory layout");
	static_assert(std::is_standard_layout_v<Mask8>, "Mask8 has invalid memory layout");
	static_assert(std::is_standard_layout_v<Mask16>, "Mask16 has invalid memory layout");
	static_assert(std::is_standard_layout_v<Mask32>, "Mask32 has invalid memory layout");
}

namespace Leibniz::Vectorization::Traits {
	template<>
	struct VectorizationIntrospect<msk256::Mask4> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 4;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 4 Lane Mask (64 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe64> {
		using type = msk256::Mask4;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU64> {
		using type = msk256::Mask4;
	};

	template<>
	struct VectorizeMaskType<v256::StripeFP64> {
		using type = msk256::Mask4;
	};

	template<>
	struct VectorizationIntrospect<msk256::Mask8> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 8;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 8 Lane Mask (32 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe32> {
		using type = msk256::Mask8;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU32> {
		using type = msk256::Mask8;
	};

	template<>
	struct VectorizeMaskType<v256::StripeFP32> {
		using type = msk256::Mask8;
	};

	template<>
	struct VectorizationIntrospect<msk256::Mask16> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 16;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 16 Lane Mask (16 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe16> {
		using type = msk256::Mask16;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU16> {
		using type = msk256::Mask16;
	};

	template<>
	struct VectorizationIntrospect<msk256::Mask32> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 32;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit 32 Lane Mask (8 bit per lane)";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeMaskType<v256::Stripe8> {
		using type = msk256::Mask32;
	};

	template<>
	struct VectorizeMaskType<v256::StripeU8> {
		using type = msk256::Mask32;
	};

	template<>
	struct LEIBNIZ_RUNTIME_API IsVectorMask<msk256::Mask4> final : std::true_type {};

	template<>
	struct LEIBNIZ_RUNTIME_API IsVectorMask<msk256::Mask8> final : std::true_type {};	
	
	template<>
	struct LEIBNIZ_RUNTIME_API IsVectorMask<msk256::Mask16> final : std::true_type {};

	template<>
	struct LEIBNIZ_RUNTIME_API IsVectorMask<msk256::Mask32> final : std::true_type {};

}

namespace Leibniz::Vectorization::msk256 {
	static_assert(256 / Traits::VectorizationIntrospect<Mask4>::s_Lanes == 64,
		"Mask4 lane-width invariant violated (expected 64 bits per lane)");
	static_assert(256 / Traits::VectorizationIntrospect<Mask8>::s_Lanes == 32,
		"Mask8 lane-width invariant violated (expected 32 bits per lane)");
	static_assert(256 / Traits::VectorizationIntrospect<Mask16>::s_Lanes == 16, 
		"Mask16 lane-width invariant violated (expected 16 bits per lane)");
	static_assert(256 / Traits::VectorizationIntrospect<Mask32>::s_Lanes == 8, 
		"Mask32 lane-width invariant violated (expected 8 bits per lane)");
}