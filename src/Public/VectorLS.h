#pragma once
#include "Leibniz.h"
#include "LeibnizInt.h"
#include "LeibnizFloat.h"
#include "VectorizeType256.h"

#if !defined(__AVX2__) && !defined(LEIBNIZ_EDITOR_MODE)
#error "Leibniz SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else

namespace Leibniz::Vectorization::mem256 {
	// Aligned Load
	template<typename T>
	[[nodiscard]] T load(const Traits::ScalarForm<T>* p_Memory) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>)
			return T{ _mm256_load_si256(reinterpret_cast<const v256::Reg256i*>(p_Memory)) };
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Leibniz::Vectorization::Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_load_pd(p_Memory) };
			else if constexpr (Leibniz::Vectorization::Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_load_ps(p_Memory) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid floating point load. Invariant violated");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid integer load. Invariants violated");
			unreachable();
		}
		unreachable();
	}

	// Unaligned load
	template<typename T>
	[[nodiscard]] T loadU(const Traits::ScalarForm<T>* p_Memory) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>)
			return T{ _mm256_loadu_si256(reinterpret_cast<const v256::Reg256i*>(p_Memory)) };
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Leibniz::Vectorization::Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_loadu_pd(p_Memory) };
			else if constexpr (Leibniz::Vectorization::Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_loadu_ps(p_Memory) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid floating point load. Invariant violated");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid integer load. Invariants violated");
			unreachable();
		}
		unreachable();
	}

	// Broadcasting

	template<typename T>
	[[nodiscard]] T broadcast(const Traits::ScalarForm<T>& v_Value) noexcept {
		using scalar = Traits::ScalarForm<T>;

		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_set1_epi64x(static_cast<long long>(v_Value)) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_set1_epi32(static_cast<int>(v_Value)) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ _mm256_set1_epi16(static_cast<short>(v_Value)) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ _mm256_set1_epi8(static_cast<char>(v_Value)) };
			else {
				static_assert(Traits::TemplateFalseV<T>,
					"Invalid integer Stripe lane count");
				unreachable();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_set1_ps(v_Value) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_set1_pd(v_Value) };
			else {
				static_assert(Traits::TemplateFalseV<T>,
					"Invalid float Stripe lane count");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>,
				"broadcast used with invalid Stripe scalar type");
			unreachable();
		}

		unreachable();
	}

	template<typename T>
	[[nodiscard]] T broadcast(const Traits::ScalarForm<T>* p_Memory) noexcept {
		using scalar = Traits::ScalarForm<T>;

		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			scalar val = *p_Memory;
			return broadcast<T>(val);
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_broadcast_ss(p_Memory) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_broadcast_sd(p_Memory) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float Stripe lane count");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid broadcast scalar type");
			unreachable();
		}

		unreachable();
	}

	template<typename T>
	void store(Traits::ScalarForm<T>* p_Memory, T v_Register) {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>)
			_mm256_store_si256(reinterpret_cast<__m256i*>(p_Memory), v_Register.m_VectorBin);
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				_mm256_store_ps(p_Memory, v_Register.m_VectorBin);
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				_mm256_store_pd(p_Memory, v_Register.m_VectorBin);
			else static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
		} else static_assert(Traits::TemplateFalseV<T>, "Invalid Stripe data type");
	}

	template<typename T>
	void storeU(Traits::ScalarForm<T>* p_Memory, T v_Register) {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>)
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(p_Memory), v_Register.m_VectorBin);
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				_mm256_storeu_ps(p_Memory, v_Register.m_VectorBin);
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				_mm256_storeu_pd(p_Memory, v_Register.m_VectorBin);
			else static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
		} else static_assert(Traits::TemplateFalseV<T>, "Invalid Stripe data type");
	}

	template<typename T>
	[[nodiscard]] T zero() {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>)
			return T{ _mm256_setzero_si256() };
		else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ _mm256_setzero_ps() };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ _mm256_setzero_pd() };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe lanes");
				unreachable();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid Stripe data type");
			unreachable();
		}
		unreachable();
	}
}

#endif			  