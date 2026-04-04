#pragma once
#include "Leibniz.h"
#include "VectorizationTraits.h"
#include "VectorizeMask256.h"

#define ALLOW_VECTOR_INTRIN_256
#include <v256Intrin.h>

namespace Leibniz::Vectorization::Ops256 {
	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] T negate(T a) noexcept {
		using Scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::sub64(Intrinsic::v256::zero(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::sub32(Intrinsic::v256::zero(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ Intrinsic::v256::sub16(Intrinsic::v256::zero(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ Intrinsic::v256::sub8(Intrinsic::v256::zero(), a.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::sub(Intrinsic::v256::zeroSinglePrecision(), a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::sub(Intrinsic::v256::zeroDoublePrecision(), a.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "negate() used with unsupported Stripe type");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T> requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
		[[nodiscard]] T operator+(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::add64(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::add32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ Intrinsic::v256::add16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ Intrinsic::v256::add8(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::add(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::add(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T> requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
		[[nodiscard]] T operator-(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::sub64(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::sub32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ Intrinsic::v256::sub16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return T{ Intrinsic::v256::sub8(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::sub(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::sub(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>  requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
		[[nodiscard]] T operator*(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::mulLow32(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return T{ Intrinsic::v256::mulLow16(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Integer Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::mul(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::mul(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>  requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
		[[nodiscard]] T operator/(T v_OperandA, T v_OperandB) noexcept {
		using scalar = Traits::ScalarForm<T>;
		if constexpr (Leibniz::Traits::IsIntV<scalar>) {
			static_assert(Traits::TemplateFalseV<T>, "Integer division not supported ");
			LEIBNIZ_UNREACHABLE();
		} else if constexpr (Leibniz::Traits::IsFloatV<scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::div(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::div(v_OperandA.m_VectorBin, v_OperandB.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid Float Stripe Lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe data type used");
			LEIBNIZ_UNREACHABLE();
		}
	}

	// Bitwise Ops

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator&(T a, T b) noexcept {
		return T{ Intrinsic::v256::bitAnd(a.m_VectorBin, b.m_VectorBin) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator|(T a, T b) noexcept {
		return T{ Intrinsic::v256::bitOr(a.m_VectorBin, b.m_VectorBin) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator^(T a, T b) noexcept {
		return T{ Intrinsic::v256::bitXor(a.m_VectorBin, b.m_VectorBin) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator~(T a) noexcept {
		return T{ Intrinsic::v256::bitXor(a.m_VectorBin, _mm256_set1_epi32(-1)) };
	}

	// Mask Operations

	template<typename M>  requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator&(M a, M b) noexcept {
		return M{ Intrinsic::v256::bitAnd(a.m_VectorMask, b.m_VectorMask) };
	}

	template<typename M>requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator|(M a, M b) noexcept {
		return M{ Intrinsic::v256::bitOr(a.m_VectorMask, b.m_VectorMask) };
	}

	template<typename M> requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator^(M a, M b) noexcept {
		return M{ Intrinsic::v256::bitXor(a.m_VectorMask, b.m_VectorMask) };
	}

	template<typename M> requires Traits::IsVectorMaskV<M>
	[[nodiscard]] M operator~(M a) noexcept {
		return M{ Intrinsic::v256::bitXor(a.m_VectorMask, _mm256_set1_epi32(-1)) };
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] Traits::VectorizedMask<T> operator==(T a, T b) noexcept {
		using Scalar = Traits::ScalarForm<T>;
		using Mask = Traits::VectorizeMaskType<T>;

		if constexpr (Leibniz::Traits::IsSignedIntV<Scalar> ||
					  Leibniz::Traits::IsUnsignedIntV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ Intrinsic::v256::cmpEqual64(a.m_VectorBin, b.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ Intrinsic::v256::cmpEqual32(a.m_VectorBin, b.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return Mask{ Intrinsic::v256::cmpEqual16(a.m_VectorBin, b.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return Mask{ Intrinsic::v256::cmpEqual8(a.m_VectorBin, b.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid integer lane count");
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ Intrinsic::v256::castFp32ToInt(
					Intrinsic::v256::cmp<_CMP_EQ_OQ>(a.m_VectorBin, b.m_VectorBin)) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ Intrinsic::v256::castFp64ToInt(
					Intrinsic::v256::cmp<_CMP_EQ_OQ>(a.m_VectorBin, b.m_VectorBin)) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float lane count");
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>, "Unsupported Stripe type");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] Traits::VectorizedMask<T> operator!=(T a, T b) noexcept {
		return ~(a == b);
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] Traits::VectorizedMask<T> operator>(T a, T b) noexcept {
		using Scalar = Traits::ScalarForm<T>;
		using Mask = Traits::VectorizeMaskType<T>;

		if constexpr (Leibniz::Traits::IsSignedIntV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ Intrinsic::v256::cmpGreater64(b.m_VectorBin, a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ Intrinsic::v256::cmpGreater32(b.m_VectorBin, a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 16)
				return Mask{ Intrinsic::v256::cmpGreater16(b.m_VectorBin, a.m_VectorBin) };
			else if constexpr (Traits::IntrospectLanes<T> == 32)
				return Mask{ Intrinsic::v256::cmpGreater8(b.m_VectorBin, a.m_VectorBin) };
			else {
				static_assert(Traits::TemplateFalseV<T>);
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsUnsignedIntV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 4) {
				auto bias = _mm256_set1_epi64x(0x8000000000000000ll);
				return Mask{ Intrinsic::v256::cmpGreater64(
					Intrinsic::v256::bitXor(b.m_VectorBin, bias),
					Intrinsic::v256::bitXor(a.m_VectorBin, bias)) };
			} else if constexpr (Traits::IntrospectLanes<T> == 8) {
				auto bias = _mm256_set1_epi32(0x80000000);
				return Mask{ Intrinsic::v256::cmpGreater32(
					Intrinsic::v256::bitXor(b.m_VectorBin, bias),
					Intrinsic::v256::bitXor(a.m_VectorBin, bias)) };
			} else if constexpr (Traits::IntrospectLanes<T> == 16) {
				auto bias = _mm256_set1_epi16(0x8000);
				return Mask{ Intrinsic::v256::cmpGreater16(
					Intrinsic::v256::bitXor(b.m_VectorBin, bias),
					Intrinsic::v256::bitXor(a.m_VectorBin, bias)) };
			} else if constexpr (Traits::IntrospectLanes<T> == 32) {
				auto bias = _mm256_set1_epi8(char(0x80));
				return Mask{ _mm256_cmpgt_epi8(
					Intrinsic::v256::bitXor(b.m_VectorBin, bias),
					Intrinsic::v256::bitXor(a.m_VectorBin, bias)) };
			} else {
				static_assert(Traits::TemplateFalseV<T>);
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return Mask{ Intrinsic::v256::castFp32ToInt(
					Intrinsic::v256::cmp< _CMP_LT_OQ>(a.m_VectorBin, b.m_VectorBin)) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return Mask{ Intrinsic::v256::castFp64ToInt(
					Intrinsic::v256::cmp< _CMP_LT_OQ>(a.m_VectorBin, b.m_VectorBin)) };
			else {
				static_assert(Traits::TemplateFalseV<T>);
				LEIBNIZ_UNREACHABLE();
			}
		} else {
			static_assert(Traits::TemplateFalseV<T>);
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] Traits::VectorizedMask<T> operator<(T a, T b) noexcept {
		return b > a;
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] Traits::VectorizedMask<T> operator<=(T a, T b) noexcept {
		return (a < b) | (a == b);
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] Traits::VectorizedMask<T> operator>=(T a, T b) noexcept {
		return (a > b) | (a == b);
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	[[nodiscard]] T select(Traits::VectorizedMask<T> mask, T a, T b) noexcept {
		using Scalar = Traits::ScalarForm<T>;

		if constexpr (Leibniz::Traits::IsFloatV<Scalar>) {
			if constexpr (Traits::IntrospectLanes<T> == 8)
				return T{ Intrinsic::v256::blendMasked(b.m_VectorBin, a.m_VectorBin,
										   Intrinsic::v256::castIntToFp32(mask.m_VectorMask)) };
			else if constexpr (Traits::IntrospectLanes<T> == 4)
				return T{ Intrinsic::v256::blendMasked(b.m_VectorBin, a.m_VectorBin,
										   Intrinsic::v256::castIntToFp64(mask.m_VectorMask)) };
			else {
				static_assert(Traits::TemplateFalseV<T>, "Invalid float Stripe lanes");
				LEIBNIZ_UNREACHABLE();
			}
		} else if constexpr (Leibniz::Traits::IsIntV<Scalar>) {
			// byte-wise blend, mask MSB decides
			return T{ Intrinsic::v256::blendMasked(b.m_VectorBin, a.m_VectorBin,
										 mask.m_VectorMask) };
		} else {
			static_assert(Traits::TemplateFalseV<T>, "select() used with unsupported Stripe type");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator<<(T a, int shift) noexcept {
		if constexpr (Traits::IntrospectLanes<T> == 4)
			return T{ Intrinsic::v256::shiftLeft64(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 8)
			return T{ Intrinsic::v256::shiftLeft32(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 16)
			return T{ Intrinsic::v256::shiftLeft16(a.m_VectorBin, shift) };
		else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid integer Stripe lanes");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsUnsignedIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator>>(T a, int shift) noexcept {
		if constexpr (Traits::IntrospectLanes<T> == 4)
			return T{ Intrinsic::v256::shiftRightLogical64(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 8)
			return T{ Intrinsic::v256::shiftRightLogical32(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 16)
			return T{ Intrinsic::v256::shiftRightLogical16(a.m_VectorBin, shift) };
		else {
			static_assert(Traits::TemplateFalseV<T>, "Invalid unsigned integer Stripe lanes");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename T>
		requires Traits::IntrospectBackend<T> == Traits::VectorizationBackend::AVX2
	&& Leibniz::Traits::IsSignedIntV<Traits::ScalarForm<T>>
		[[nodiscard]] T operator>>(T a, int shift) noexcept {
		if constexpr (Traits::IntrospectLanes<T> == 8)
			return T{ Intrinsic::v256::shiftRightArithmetic32(a.m_VectorBin, shift) };
		else if constexpr (Traits::IntrospectLanes<T> == 16)
			return T{ Intrinsic::v256::shiftRightArithmetic16(a.m_VectorBin, shift) };
		else {
			static_assert(Traits::TemplateFalseV<T>,
						  "Arithmetic right shift not supported for this integer Stripe width");
			LEIBNIZ_UNREACHABLE();
		}
	}

	template<typename M>
		requires Traits::IsVectorMaskV<M>
	[[nodiscard]] bool any(M m) noexcept {
		return Intrinsic::v256::movemask(m.m_VectorMask) != 0;
	}

	template<typename M>
		requires Traits::IsVectorMaskV<M>
	[[nodiscard]] bool none(M m) noexcept {
		return Intrinsic::v256::movemask(m.m_VectorMask) == 0;
	}

	template<typename M>
		requires Traits::IsVectorMaskV<M>
	[[nodiscard]] bool all(M m) noexcept {
		return Intrinsic::v256::movemask(m.m_VectorMask) == 0xFFFFFFFF;
	}

}