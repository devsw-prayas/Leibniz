#pragma once
#include "Leibniz.h"
#include "VectorizationTraits.h"
#include "LeibnizInt.h"
#include "LeibnizFloat.h"
#include "VectorNumericTraits.h"
#include "NumericTraits.h"

namespace Leibniz::Vectorization::v256 {
	// All the Stripe Forms for Integer Stripes

	using Reg256i = __m256i;
	using RegFP256 = __m256;
	using RegD256 = __m256d;

	struct LEIBNIZ_RUNTIME_API alignas(32) StripeU8 final {
		Reg256i m_VectorBin;

		StripeU8();
		explicit StripeU8(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		StripeU8(const StripeU8&) = default;
		StripeU8& operator=(const StripeU8&) = default;

		StripeU8(StripeU8&&) noexcept = default;
		StripeU8& operator=(StripeU8&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) StripeU16 final {
		Reg256i m_VectorBin;

		StripeU16();
		explicit StripeU16(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		StripeU16(const StripeU16&) = default;
		StripeU16& operator=(const StripeU16&) = default;

		StripeU16(StripeU16&&) noexcept = default;
		StripeU16& operator=(StripeU16&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) StripeU32 final {
		Reg256i m_VectorBin;

		StripeU32();
		explicit StripeU32(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		StripeU32(const StripeU32&) = default;
		StripeU32& operator=(const StripeU32&) = default;

		StripeU32(StripeU32&&) noexcept = default;
		StripeU32& operator=(StripeU32&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) StripeU64 final {
		Reg256i m_VectorBin;

		StripeU64();
		explicit StripeU64(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		StripeU64(const StripeU64&) = default;
		StripeU64& operator=(const StripeU64&) = default;

		StripeU64(StripeU64&&) noexcept = default;
		StripeU64& operator=(StripeU64&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Stripe8 final {
		Reg256i m_VectorBin;

		Stripe8();
		explicit Stripe8(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		Stripe8(const Stripe8&) = default;
		Stripe8& operator=(const Stripe8&) = default;

		Stripe8(Stripe8&&) noexcept = default;
		Stripe8& operator=(Stripe8&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Stripe16 final {
		Reg256i m_VectorBin;

		Stripe16();
		explicit Stripe16(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		Stripe16(const Stripe16&) = default;
		Stripe16& operator=(const Stripe16&) = default;

		Stripe16(Stripe16&&) noexcept = default;
		Stripe16& operator=(Stripe16&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Stripe32 final {
		Reg256i m_VectorBin;

		Stripe32();
		explicit Stripe32(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		Stripe32(const Stripe32&) = default;
		Stripe32& operator=(const Stripe32&) = default;

		Stripe32(Stripe32&&) noexcept = default;
		Stripe32& operator=(Stripe32&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) Stripe64 final {
		Reg256i m_VectorBin;

		Stripe64();
		explicit Stripe64(Reg256i v_Raw) noexcept : m_VectorBin(v_Raw) {}

		Stripe64(const Stripe64&) = default;
		Stripe64& operator=(const Stripe64&) = default;

		Stripe64(Stripe64&&) noexcept = default;
		Stripe64& operator=(Stripe64&&) noexcept = default;
	};

	// All the floating point Stripes

	struct LEIBNIZ_RUNTIME_API alignas(32) StripeFP32 final {
		RegFP256 m_VectorBin;

		StripeFP32();
		explicit StripeFP32(RegFP256 v_Raw) noexcept : m_VectorBin(v_Raw) {}

		StripeFP32(const StripeFP32&) = default;
		StripeFP32& operator=(const StripeFP32&) = default;

		StripeFP32(StripeFP32&&) noexcept = default;
		StripeFP32& operator=(StripeFP32&&) noexcept = default;
	};

	struct LEIBNIZ_RUNTIME_API alignas(32) StripeFP64 final {
		RegD256 m_VectorBin;

		StripeFP64();
		explicit StripeFP64(RegD256 v_Raw) noexcept : m_VectorBin(v_Raw) {}

		StripeFP64(const StripeFP64&) = default;
		StripeFP64& operator=(const StripeFP64&) = default;

		StripeFP64(StripeFP64&&) noexcept = default;
		StripeFP64& operator=(StripeFP64&&) noexcept = default;
	};

	static_assert(sizeof(StripeU8) == 32, "Invalid Stripe Size, StripeU8 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(StripeU16) == 32, "Invalid Stripe Size, StripeU16 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(StripeU32) == 32, "Invalid Stripe Size, StripeU32 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(StripeU64) == 32, "Invalid Stripe Size, StripeU64 must be exactly 32 bytes (one AVX2 register)");

	static_assert(alignof(StripeU8) == 32, "Invalid Stripe Alignment, StripeU8 has illegal alignment");
	static_assert(alignof(StripeU16) == 32, "Invalid Stripe Alignment, StripeU16 has illegal alignment");
	static_assert(alignof(StripeU32) == 32, "Invalid Stripe Alignment, StripeU32 has illegal alignment");
	static_assert(alignof(StripeU64) == 32, "Invalid Stripe Alignment, StripeU64 has illegal alignment");

	static_assert(std::is_standard_layout_v<StripeU8>, "StripeU8 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeU16>, "StripeU16 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeU32>, "StripeU32 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeU64>, "StripeU64 has invalid memory layout");

	static_assert(std::is_trivially_copyable_v<StripeU8>, "StripeU8 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeU16>, "StripeU16 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeU32>, "StripeU32 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeU64>, "StripeU64 must be trivially copyable");

	static_assert(sizeof(Stripe8) == 32, "Invalid Stripe Size, Stripe8 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(Stripe16) == 32, "Invalid Stripe Size, Stripe16 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(Stripe32) == 32, "Invalid Stripe Size, Stripe32 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(Stripe64) == 32, "Invalid Stripe Size, Stripe64 must be exactly 32 bytes (one AVX2 register)");

	static_assert(alignof(Stripe8) == 32, "Invalid Stripe Alignment, Stripe8 has illegal alignment");
	static_assert(alignof(Stripe16) == 32, "Invalid Stripe Alignment, Stripe16 has illegal alignment");
	static_assert(alignof(Stripe32) == 32, "Invalid Stripe Alignment, Stripe32 has illegal alignment");
	static_assert(alignof(Stripe64) == 32, "Invalid Stripe Alignment, Stripe64 has illegal alignment");

	static_assert(std::is_standard_layout_v<Stripe8>, "Stripe8 has invalid memory layout");
	static_assert(std::is_standard_layout_v<Stripe16>, "Stripe16 has invalid memory layout");
	static_assert(std::is_standard_layout_v<Stripe32>, "Stripe32 has invalid memory layout");
	static_assert(std::is_standard_layout_v<Stripe64>, "Stripe64 has invalid memory layout");

	static_assert(std::is_trivially_copyable_v<Stripe8>, "Stripe8 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Stripe16>, "Stripe16 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Stripe32>, "Stripe32 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Stripe64>, "Stripe64 must be trivially copyable");

	static_assert(sizeof(StripeFP32) == 32, "Invalid Stripe Size, StripeFP32 must be exactly 32 bytes (one AVX2 register)");
	static_assert(sizeof(StripeFP64) == 32, "Invalid Stripe Size, StripeFP64 must be exactly 32 bytes (one AVX2 register)");

	static_assert(alignof(StripeFP32) == 32, "Invalid Stripe Alignment, StripeFP32 has illegal alignment");
	static_assert(alignof(StripeFP64) == 32, "Invalid Stripe Alignment, StripeFP64 has illegal alignment");

	static_assert(std::is_standard_layout_v<StripeFP32>, "StripeFP32 has invalid memory layout");
	static_assert(std::is_standard_layout_v<StripeFP64>, "StripeFP64 has invalid memory layout");

	static_assert(std::is_trivially_copyable_v<StripeFP32>, "StripeFP32 must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<StripeFP64>, "StripeFP64 must be trivially copyable");
}

namespace Leibniz::Traits{
	template<>
	struct  IsCompilerType<Vectorization::v256::Stripe8> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::Stripe16> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::Stripe32> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::Stripe64> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::StripeU8> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::StripeU16> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::StripeU32> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::StripeU64> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::StripeFP32> final : std::true_type {};

	template<>
	struct  IsCompilerType<Vectorization::v256::StripeFP64> final : std::true_type {};
}

namespace Leibniz::Vectorization::Traits {
	template<>
	struct  IsVectorStripe<v256::Stripe8> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::Stripe16> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::Stripe32> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::Stripe64> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::StripeU8> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::StripeU16> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::StripeU32> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::StripeU64> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::StripeFP32> final : std::true_type {};
	template<>
	struct  IsVectorStripe<v256::StripeFP64> final : std::true_type {};

	template<>
	struct VectorizationIntrospect<v256::StripeU8> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 32;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeU8 [Unsigned Integer 1 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::UByte> {
		using type = v256::StripeU8;
	};

	template<>
	struct ScalarType<v256::StripeU8> {
		using type = Numerics::UByte;
	};

	template<>
	struct VectorizationIntrospect<v256::StripeU16> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 16;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeU16 [Unsigned Integer 2 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::UInt16> {
		using type = v256::StripeU16;
	};

	template<>
	struct ScalarType<v256::StripeU16> {
		using type = Numerics::UInt16;
	};

	template<>
	struct VectorizationIntrospect<v256::StripeU32> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 8;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeU32 [Unsigned Integer 4 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::UInt32> {
		using type = v256::StripeU32;
	};

	template<>
	struct ScalarType<v256::StripeU32> {
		using type = Numerics::UInt32;
	};

	template<>
	struct VectorizationIntrospect<v256::StripeU64> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 4;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeU64 [Unsigned Integer 8 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::UInt64> {
		using type = v256::StripeU64;
	};

	template<>
	struct ScalarType<v256::StripeU64> {
		using type = Numerics::UInt64;
	};

	template<>
	struct VectorizationIntrospect<v256::Stripe8> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 32;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit Stripe8 [Signed Integer 1 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::Byte> {
		using type = v256::Stripe8;
	};

	template<>
	struct ScalarType<v256::Stripe8> {
		using type = Numerics::Byte;
	};

	template<>
	struct VectorizationIntrospect<v256::Stripe16> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 16;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit Stripe16 [Signed Integer 2 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::Int16> {
		using type = v256::Stripe16;
	};

	template<>
	struct ScalarType<v256::Stripe16> {
		using type = Numerics::Int16;
	};

	template<>
	struct VectorizationIntrospect<v256::Stripe32> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 8;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit Stripe32 [Signed Integer 4 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::Int32> {
		using type = v256::Stripe32;
	};

	template<>
	struct ScalarType<v256::Stripe32> {
		using type = Numerics::Int32;
	};

	template<>
	struct VectorizationIntrospect<v256::Stripe64> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 4;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit Stripe64 [Signed Integer 8 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::Int64> {
		using type = v256::Stripe64;
	};

	template<>
	struct ScalarType<v256::Stripe64> {
		using type = Numerics::Int64;
	};

	template<>
	struct VectorizationIntrospect<v256::StripeFP32> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 8;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeFP32 [IEEE 754 FP32 (Single Precision) 4 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::Float32> {
		using type = v256::StripeFP32;
	};

	template<>
	struct ScalarType<v256::StripeFP32> {
		using type = Numerics::Float32;
	};

	template<>
	struct VectorizationIntrospect<v256::StripeFP64> {
		static constexpr size_t s_Alignment = 32;
		static constexpr size_t s_Lanes = 4;
		static constexpr size_t s_Width = 256;
		static constexpr const char* s_Name = "256 bit StripeFP64 [IEEE 754 FP64 (Double Precision) 8 byte]";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::AVX2;
	};

	template<>
	struct VectorizeType<Numerics::Float64> {
		using type = v256::StripeFP64;
	};

	template<>
	struct ScalarType<v256::StripeFP64> {
		using type = Numerics::Float64;
	};
}

namespace Leibniz::Vectorized::Traits {
	template<>
	struct  IsVectorizedInt<Vectorization::v256::Stripe8> : std::true_type {};
	template<>
	struct  IsVectorizedInt<Vectorization::v256::Stripe16> : std::true_type {};
	template<>
	struct  IsVectorizedInt<Vectorization::v256::Stripe32> : std::true_type {};
	template<>
	struct  IsVectorizedInt<Vectorization::v256::Stripe64> : std::true_type {};

	template<>
	struct  IsVectorizedInt<Vectorization::v256::StripeU8> : std::true_type {};
	template<>
	struct  IsVectorizedInt<Vectorization::v256::StripeU16> : std::true_type {};
	template<>
	struct  IsVectorizedInt<Vectorization::v256::StripeU32> : std::true_type {};
	template<>
	struct  IsVectorizedInt<Vectorization::v256::StripeU64> : std::true_type {};

	template<>
	struct  IsSignedVectorInt<Vectorization::v256::Stripe8> : std::true_type {};
	template<>
	struct  IsSignedVectorInt<Vectorization::v256::Stripe16> : std::true_type {};
	template<>
	struct  IsSignedVectorInt<Vectorization::v256::Stripe32> : std::true_type {};
	template<>
	struct  IsSignedVectorInt<Vectorization::v256::Stripe64> : std::true_type {};

	template<>
	struct  IsUnsignedVectorInt<Vectorization::v256::StripeU8> : std::true_type {};
	template<>
	struct  IsUnsignedVectorInt<Vectorization::v256::StripeU16> : std::true_type {};
	template<>
	struct  IsUnsignedVectorInt<Vectorization::v256::StripeU32> : std::true_type {};
	template<>
	struct  IsUnsignedVectorInt<Vectorization::v256::StripeU64> : std::true_type {};

	template<>
	struct  IsVectorizedFloat<Vectorization::v256::StripeFP32> : std::true_type {};
	template<>
	struct  IsVectorizedFloat<Vectorization::v256::StripeFP64> : std::true_type {};
}
