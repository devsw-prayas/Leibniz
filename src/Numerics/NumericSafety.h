#pragma once
#include <bit>

#include "LeibnizInt.h"
#include "LeibnizLimits.h"
#include "VectorizeType256.h"
#include "VectorNumericTraits.h"
#include "VectorOps256.h"
#include "VectorizeMask256.h"
#include "v256Intrin.h"

namespace Leibniz::Numerics::Safety {
	using namespace Vectorization::Ops256;
	template<typename T> requires Traits::IsCompilerTypeV<T> &&
		(Traits::IsFloatV<T> || Traits::IsFloatV<typename Vectorization::Traits::ScalarType<T>::type>)
		struct NumericSafety {
		private:
			using StFp32 = Vectorization::v256::StripeFP32;
			using StFp64 = Vectorization::v256::StripeFP64;
			using StU32 = Vectorization::v256::StripeU32;
			using StU64 = Vectorization::v256::StripeU64;
			using St32 = Vectorization::v256::Stripe32;
			using St64 = Vectorization::v256::Stripe64;

			using DiffType = std::conditional_t<std::is_same_v<T, Float32>, Int32,
				std::conditional_t<std::is_same_v<T, Float64>, Int64,
				std::conditional_t<std::is_same_v<T, StFp32>, Vectorization::v256::Stripe32,
				Vectorization::v256::Stripe64>>>;
		public:
			template<typename U = std::conditional_t<Vectorized::Traits::IsVectorizedFloatV<T>, Vectorization::Traits::VectorizedMask<T>, bool> >
			static U almostEquals(T a, T b, T eps = Constants::NumericLimits<T>::epsilon()) {
				if constexpr (std::is_same_v<T, Float32>) {
					UInt32 diff = std::bit_cast<UInt32>(a - b);
					diff &= 0x7fffffff;
					return std::bit_cast<Float32>(diff) < eps;
				} else if constexpr (std::is_same_v<T, Float64>) {
					UInt64 diff = std::bit_cast<UInt64>(a - b);
					diff &= 0x7fffffffffffffffULL;
					return std::bit_cast<Float64>(diff) < eps;
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP32>) {
					StU32 diff{ Vectorization::Intrinsic::v256::castFp32ToInt((a - b).m_VectorBin) };
					const StU32 mask{ Vectorization::Intrinsic::v256::setOne(0x7fffffff) };
					diff = diff & mask;
					return StFp32(Vectorization::Intrinsic::v256::castIntToFp32(diff.m_VectorBin)) < eps;
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP64>) {
					StU64 diff{ Vectorization::Intrinsic::v256::castFp64ToInt((a - b).m_VectorBin) };
					const StU64 mask{ Vectorization::Intrinsic::v256::setOne(0x7fffffffffffffffULL) };
					diff = diff & mask;
					return StFp64(Vectorization::Intrinsic::v256::castIntToFp64(diff.m_VectorBin)) < eps;
				} else {
					LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid Floating point type");
					LEIBNIZ_UNREACHABLE();
				}
			}

			static DiffType ulpDiff(T a, T b) {
				if constexpr (std::is_same_v<T, Float32>) {
					uint32_t ua = std::bit_cast<UInt32>(a);
					uint32_t ub = std::bit_cast<UInt32>(b);
					if (ua & 0x80000000u) ua = 0x80000000u - ua;
					if (ub & 0x80000000u) ub = 0x80000000u - ub;
					return std::bit_cast<Int32>(ua - ub);
				} else if constexpr (std::is_same_v<T, Float64>) {
					uint64_t ua = std::bit_cast<UInt64>(a);
					uint64_t ub = std::bit_cast<UInt64>(b);
					if (ua & 0x8000000000000000ull) ua = 0x8000000000000000ull - ua;
					if (ub & 0x8000000000000000ull) ub = 0x8000000000000000ull - ub;
					return std::bit_cast<Int64>(ua - ub);
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP32>) {
					StU32 ua{ Vectorization::Intrinsic::v256::castFp32ToInt(a.m_VectorBin) };
					StU32 ub{ Vectorization::Intrinsic::v256::castFp32ToInt(b.m_VectorBin) };
					StU32 signMask{ Vectorization::Intrinsic::v256::setOne(0x80000000u) };

					StU32 flippedA = signMask - ua;
					StU32 flippedB = signMask - ub;
					Vectorization::msk256::Mask8 maskA{ Vectorization::Intrinsic::v256::cmp<_CMP_LT_OQ>(a.m_VectorBin, Vectorization::Intrinsic::v256::zeroSinglePrecision()) };
					Vectorization::msk256::Mask8 maskB{ Vectorization::Intrinsic::v256::cmp<_CMP_LT_OQ>(b.m_VectorBin, Vectorization::Intrinsic::v256::zeroSinglePrecision()) };

					ua = StU32(Vectorization::Intrinsic::v256::blendMasked(ua.m_VectorBin, flippedA.m_VectorBin, maskA.m_VectorMask));
					ub = StU32(Vectorization::Intrinsic::v256::blendMasked(ub.m_VectorBin, flippedB.m_VectorBin, maskB.m_VectorMask));
					return St32{ (ua - ub).m_VectorBin };
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP64>) {
					StU64 ua{ Vectorization::Intrinsic::v256::castFp64ToInt(a.m_VectorBin) };
					StU64 ub{ Vectorization::Intrinsic::v256::castFp64ToInt(b.m_VectorBin) };
					StU64 signMask{ Vectorization::Intrinsic::v256::setOne(0x8000000000000000ull) };

					StU64 flippedA = signMask - ua;
					StU64 flippedB = signMask - ub;
					Vectorization::msk256::Mask4 maskA{ Vectorization::Intrinsic::v256::cmp<_CMP_LT_OQ>(a.m_VectorBin, Vectorization::Intrinsic::v256::zeroDoublePrecision()) };
					Vectorization::msk256::Mask4 maskB{ Vectorization::Intrinsic::v256::cmp<_CMP_LT_OQ>(b.m_VectorBin, Vectorization::Intrinsic::v256::zeroDoublePrecision()) };

					ua = StU64(Vectorization::Intrinsic::v256::blendMasked(ua.m_VectorBin, flippedA.m_VectorBin, maskA.m_VectorMask));
					ub = StU64(Vectorization::Intrinsic::v256::blendMasked(ub.m_VectorBin, flippedB.m_VectorBin, maskB.m_VectorMask));
					return St64{ (ua - ub).m_VectorBin };
				} else {
					LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid Floating point type");
					LEIBNIZ_UNREACHABLE();
				}
			}

			template<typename U = std::conditional_t<Vectorized::Traits::IsVectorizedFloatV<T>, Vectorization::Traits::VectorizedMask<T>, bool> >
			static U isNan(T a) {
				if constexpr (std::is_same_v<T, Float32>) {
					UInt32 ua = std::bit_cast<UInt32>(a);
					return ((ua & 0x7F800000u) == 0x7F800000u) && ((ua & 0x007FFFFFu) != 0u);
				} else if constexpr (std::is_same_v<T, Float64>) {
					UInt64 ua = std::bit_cast<UInt64>(a);
					return ((ua & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) &&
						((ua & 0x000FFFFFFFFFFFFFULL) != 0ull);
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP32>) {
					StU32 expMask{ Vectorization::Intrinsic::v256::setOne(0x7F800000u) };
					StU32 mantMask{ Vectorization::Intrinsic::v256::setOne(0x007FFFFFu) };
					StU32 ua{ Vectorization::Intrinsic::v256::castFp32ToInt(a.m_VectorBin) };
					return (ua & expMask) == expMask & (ua & mantMask) != StU32{ Vectorization::Intrinsic::v256::zero() };
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP64>) {
					StU64 expMask{ Vectorization::Intrinsic::v256::setOne(0x7FF0000000000000ULL) };
					StU64 mantMask{ Vectorization::Intrinsic::v256::setOne(0x000FFFFFFFFFFFFFull) };
					StU64 ua{ Vectorization::Intrinsic::v256::castFp64ToInt(a.m_VectorBin) };
					return (ua & expMask) == expMask & (ua & mantMask) != StU64{ Vectorization::Intrinsic::v256::zero() };
				} else {
					LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid Floating point type");
					LEIBNIZ_UNREACHABLE();
				}
			}

			template<typename U = std::conditional_t<Vectorized::Traits::IsVectorizedFloatV<T>, Vectorization::Traits::VectorizedMask<T>, bool> >
			static U isInfinite(T a) {
				if constexpr (std::is_same_v<T, Float32>) {
					UInt32 ua = std::bit_cast<UInt32>(a);
					return (ua & 0x7FFFFFFFu) == 0x7F800000u;
				} else if constexpr (std::is_same_v<T, Float64>) {
					UInt64 ua = std::bit_cast<UInt64>(a);
					return (ua & 0x7FFFFFFFFFFFFFFFULL) == 0x7FF0000000000000ULL;
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP32>) {
					StU32 mask{ Vectorization::Intrinsic::v256::setOne(0x7FFFFFFFu) };
					StU32 inf{ Vectorization::Intrinsic::v256::setOne(0x7F800000u) };
					StU32 ua{ Vectorization::Intrinsic::v256::castFp32ToInt(a.m_VectorBin) };
					return (ua & mask) == inf;
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP64>) {
					StU64 mask{ Vectorization::Intrinsic::v256::setOne(0x7FFFFFFFFFFFFFFFULL) };
					StU64 inf{ Vectorization::Intrinsic::v256::setOne(0x7FF0000000000000ULL) };
					StU64 ua{ Vectorization::Intrinsic::v256::castFp64ToInt(a.m_VectorBin) };
					return (ua & mask) == inf;
				} else {
					LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid Floating point type");
					LEIBNIZ_UNREACHABLE();
				}
			}

			template<typename U = std::conditional_t<Vectorized::Traits::IsVectorizedFloatV<T>, Vectorization::Traits::VectorizedMask<T>, bool> >
			static U isFinite(T a) {
				if constexpr (std::is_same_v<T, Float32>) {
					UInt32 ua = std::bit_cast<UInt32>(a);
					return (ua & 0x7F800000u) != 0x7F800000u;
				} else if constexpr (std::is_same_v<T, Float64>) {
					UInt64 ua = std::bit_cast<UInt64>(a);
					return (ua & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP32>) {
					StU32 mask{ Vectorization::Intrinsic::v256::setOne(0x7F800000u) };
					StU32 ua{ Vectorization::Intrinsic::v256::castFp32ToInt(a.m_VectorBin) };
					return (ua & mask) != mask;
				} else if constexpr (std::is_same_v<T, Vectorization::v256::StripeFP64>) {
					StU64 mask{ Vectorization::Intrinsic::v256::setOne(0x7FF0000000000000ULL) };
					StU64 ua{ Vectorization::Intrinsic::v256::castFp64ToInt(a.m_VectorBin) };
					return (ua & mask) != mask;
				} else {
					LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid Floating point type");
					LEIBNIZ_UNREACHABLE();
				}
			}
	};
}
