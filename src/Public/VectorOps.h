#pragma once
#include "Leibniz.h"
#include "LeibnizInt.h"
#include "VectorizationTraits.h"
#include "LeibnizFloat.h"
#include "VectorizeMask256.h"

#if !defined(__AVX2__) && !defined(LEIBNIZ_EDITOR_MODE)
#error "Leibniz SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else
namespace Leibniz::Vectorization::Ops {
	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] T negate(T a) noexcept {
		using Scalar = Traits::ScalarForm<T>;

		if constexpr (Leibniz::Traits::IsIntV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_sub_epi64(_mm256_setzero_si256(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_sub_epi32(_mm256_setzero_si256(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ _mm256_sub_epi16(_mm256_setzero_si256(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ _mm256_sub_epi8(_mm256_setzero_si256(), a.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_sub_ps(_mm256_setzero_ps(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_sub_pd(_mm256_setzero_pd(), a.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "negate() used with unsupported Stripe type");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T> requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
		[[nodiscard]] T operator+(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_add_epi64(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_add_epi32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ _mm256_add_epi16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ _mm256_add_epi8(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_add_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_add_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T> requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
		[[nodiscard]] T operator-(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_sub_epi64(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_sub_epi32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ _mm256_sub_epi16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ _mm256_sub_epi8(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_sub_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_sub_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>  requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
		[[nodiscard]] T operator*(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_mullo_epi32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ _mm256_mullo_epi16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_mul_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_mul_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>  requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
		[[nodiscard]] T operator/(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			static_assert(Traits::TemplateFalseV<T>, "Integer division not supported ");
			LEIBNIZ_UNREACHABLE();
		}
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_div_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_div_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	// Bitwise Ops

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator&(T a, T b) noexcept {
		return T{ _mm256_and_si256(a.m_VectorBin, b.m_VectorBin) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator|(T a, T b) noexcept {
		return T{ _mm256_or_si256(a.m_VectorBin, b.m_VectorBin) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator^(T a, T b) noexcept {
		return T{ _mm256_xor_si256(a.m_VectorBin, b.m_VectorBin) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator~(T a) noexcept {
		return T{ _mm256_xor_si256(a.m_VectorBin, _mm256_set1_epi32(-1)) };
	}

	// Mask Operations

	template<typename M>  requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator&(M a, M b) noexcept {
		return M{ _mm256_and_si256(a.m_VectorMask, b.m_VectorMask) };
	}

	template<typename M>requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator|(M a, M b) noexcept {
		return M{ _mm256_or_si256(a.m_VectorMask, b.m_VectorMask) };
	}

	template<typename M> requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator^(M a, M b) noexcept {
		return M{ _mm256_xor_si256(a.m_VectorMask, b.m_VectorMask) };
	}

	template<typename M> requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator~(M a) noexcept {
		return M{ _mm256_xor_si256(a.m_VectorMask, _mm256_set1_epi32(-1)) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] Traits::VectorizeMaskType<T> operator==(T a, T b) noexcept {
		using Scalar = Traits::ScalarForm<T>;
		using Mask = Traits::VectorizeMaskType<T>;

		if constexpr (Leibniz::Traits::IsSignedIntV<Scalar> ||
			Leibniz::Traits::IsUnsignedIntV<Scalar>) {

			if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ _mm256_cmpeq_epi64(a.m_VectorBin, b.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ _mm256_cmpeq_epi32(a.m_VectorBin, b.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return Mask{ _mm256_cmpeq_epi16(a.m_VectorBin, b.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return Mask{ _mm256_cmpeq_epi8(a.m_VectorBin, b.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid integer lane count");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {

			if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ _mm256_castps_si256(
					_mm256_cmp_ps(a.m_VectorBin, b.m_VectorBin, _CMP_EQ_OQ)) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ _mm256_castpd_si256(
					_mm256_cmp_pd(a.m_VectorBin, b.m_VectorBin, _CMP_EQ_OQ)) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float lane count");
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe type");
			LEIBNIZ_UNREACHABLE();
		}
	}


	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] Traits::VectorizeMaskType<T> operator!=(T a, T b) noexcept {
		return ~(a == b);
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] Traits::VectorizeMaskType<T> operator<(T a, T b) noexcept {
		using Scalar = Traits::ScalarForm<T>;
		using Mask = Traits::VectorizeMaskType<T>;

		if constexpr (Leibniz::Traits::IsSignedIntV<Scalar>) {

			if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ _mm256_cmpgt_epi64(b.m_VectorBin, a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ _mm256_cmpgt_epi32(b.m_VectorBin, a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return Mask{ _mm256_cmpgt_epi16(b.m_VectorBin, a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return Mask{ _mm256_cmpgt_epi8(b.m_VectorBin, a.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>);
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsUnsignedIntV<Scalar>) {

			if constexpr (Traits::IntrospectLanes<T> == 4) {
				auto bias = _mm256_set1_epi64x(0x8000000000000000ll);
				return Mask{ _mm256_cmpgt_epi64(
					_mm256_xor_si256(b.m_VectorBin, bias),
					_mm256_xor_si256(a.m_VectorBin, bias)) };
			}
			else if constexpr (Traits::IntrospectLanes<T> == 8) {
				auto bias = _mm256_set1_epi32(0x80000000);
				return Mask{ _mm256_cmpgt_epi32(
					_mm256_xor_si256(b.m_VectorBin, bias),
					_mm256_xor_si256(a.m_VectorBin, bias)) };
			}
			else if constexpr (Traits::IntrospectLanes<T> == 16) {
				auto bias = _mm256_set1_epi16(0x8000);
				return Mask{ _mm256_cmpgt_epi16(
					_mm256_xor_si256(b.m_VectorBin, bias),
					_mm256_xor_si256(a.m_VectorBin, bias)) };
			}
			else if constexpr (Traits::IntrospectLanes<T> == 32) {
				auto bias = _mm256_set1_epi8(char(0x80));
				return Mask{ _mm256_cmpgt_epi8(
					_mm256_xor_si256(b.m_VectorBin, bias),
					_mm256_xor_si256(a.m_VectorBin, bias)) };
			}
			else {
				static_assert(Traits::TemplateFalseV<T>);
				LEIBNIZ_UNREACHABLE();
			}
		}
		else if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {

			if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ _mm256_castps_si256(
					_mm256_cmp_ps(a.m_VectorBin, b.m_VectorBin, _CMP_LT_OQ)) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ _mm256_castpd_si256(
					_mm256_cmp_pd(a.m_VectorBin, b.m_VectorBin, _CMP_LT_OQ)) };
			else {
				static_assert(Traits::TemplateFalseV<T>);
				LEIBNIZ_UNREACHABLE();
			}
		}
		else {
			static_assert(Traits::TemplateFalseV<T>);
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] Traits::VectorizeMaskType<T> operator>(T a, T b) noexcept {
		return b < a;
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] Traits::VectorizeMaskType<T> operator<=(T a, T b) noexcept {
		return (a < b) | (a == b);
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] Traits::VectorizeMaskType<T> operator>=(T a, T b) noexcept {
		return (a > b) | (a == b);
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	[[nodiscard]] T select(Traits::VectorizeMaskType<T> mask, T a, T b) noexcept {
		using Scalar = Traits::ScalarForm<T>;

		if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {

			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_blendv_ps(b.m_VectorBin, a.m_VectorBin,
										   _mm256_castsi256_ps(mask.m_VectorMask)) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_blendv_pd(b.m_VectorBin, a.m_VectorBin,
										   _mm256_castsi256_pd(mask.m_VectorMask)) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}

		}
		else if constexpr (Leibniz::Traits::IsIntV<Scalar>) {

			// byte-wise blend, mask MSB decides
			return T{ _mm256_blendv_epi8(b.m_VectorBin, a.m_VectorBin,
										 mask.m_VectorMask) };
		}
		else {
			static_assert(Traits::TemplateFalseV<T>, "select() used with unsupported Stripe type");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator<<(T a, int shift) noexcept {

		if constexpr (Traits::IntrospectLanes<T> == 4)
			return T{ _mm256_slli_epi64(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 8)
			return T{ _mm256_slli_epi32(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 16)
			return T{ _mm256_slli_epi16(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 32)
			return T{ _mm256_slli_epi8(a.m_VectorBin, shift) };
		else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid integer Stripe lanes");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsUnsignedIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator>>(T a, int shift) noexcept {

		if constexpr (Traits::IntrospectLanes<T> == 4)
			return T{ _mm256_srli_epi64(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 8)
			return T{ _mm256_srli_epi32(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 16)
			return T{ _mm256_srli_epi16(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 32)
			return T{ _mm256_srli_epi8(a.m_VectorBin, shift) };
		else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid unsigned integer Stripe lanes");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> != Traits::VectorizationBackend::UNKNOWN
	&& Leibniz::Traits::IsSignedIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator>>(T a, int shift) noexcept {

		if constexpr (Traits::IntrospectLanes<T> == 8)
			return T{ _mm256_srai_epi32(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 16)
			return T{ _mm256_srai_epi16(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 32)
			return T{ _mm256_srai_epi8(a.m_VectorBin, shift) };
		else {
			static_assert(Traits::TemplateFalseV<T>,
				"Arithmetic right shift not supported for this integer Stripe width");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename M>
		requires Traits::IsVectorMaskV<M>
	[[nodiscard]] bool any(M m) noexcept {
		return _mm256_movemask_epi8(m.m_VectorMask) != 0;
	}

	template<typename M>
		requires Traits::IsVectorMaskV<M>
	[[nodiscard]] bool none(M m) noexcept {
		return _mm256_movemask_epi8(m.m_VectorMask) == 0;
	}

	template<typename M>
		requires Traits::IsVectorMaskV<M>
	[[nodiscard]] bool all(M m) noexcept {
		return _mm256_movemask_epi8(m.m_VectorMask) == 0xFFFFFFFF;
	}

}
#endif
