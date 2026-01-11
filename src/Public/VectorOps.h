#pragma once
#include "Leibniz.h"
#include "LeibnizInt.h"
#include "VectorizationTraits.h"
#include "LeibnizFloat.h"

#if !defined(__AVX2__) && !defined(LEIBNIZ_EDITOR_MODE)
#error "Leibniz SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else
namespace Leibniz::Vectorization::Ops {
	template<typename T>
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
				unreachable();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_add_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_add_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			unreachable();
		}
		unreachable();
	}

	template<typename T>
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
				unreachable();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_sub_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_sub_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			unreachable();
		}
		unreachable();
	}

	template<typename T>
	[[nodiscard]] T operator*(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_mullo_epi32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ _mm256_mullo_epi16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				unreachable();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_mul_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_mul_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			unreachable();
		}
		unreachable();
	}

	template<typename T>
	[[nodiscard]] T operator/(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			static_assert(Traits::TemplateFalseV<T>, "Integer division not supported ");
			unreachable();
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_div_ps(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_div_pd(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			unreachable();
		}
		unreachable();
	}
}
#endif
