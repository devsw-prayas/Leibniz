#pragma once
#include "LebnizDiagnostics.h"
#include "NumericTraits.h"
#include "v256Intrin.h"
#include "VectorNumericTraits.h"

namespace Leibniz::Numbers::Backend {
	// IEEE 754 - 2019 extended rounding mode for operations where a terminating representation
	// is not possible due to precision limits or the nature of the number to be represented
	enum class LEIBNIZ_RUNTIME_API  RoundingMode : uint8_t {
		RNearestEven,				  // -> Rounded to nearest even
		RUp,						  // -> Round up towards +ve infinity
		RDown,						  // -> Round down towards -ve infinity
		RZero,						  // -> Round towards zero
		RAway,						  // -> Round away from zero
		Unbounded,		              // -> Don't perform rounding
		None						  // -> Invalid Rounding mode used
	};

	// RangeReduction Modes are accumulation behaviors that are only supported by fixed width backends
	enum class LEIBNIZ_RUNTIME_API RangeReduction : uint8_t {
		ReductionTruncate,			  //  -> Truncate higher order bits in accumulation to fixed witth
		ReductionWrapAround,		  //  -> WrapAround on overflow
		ReductionSaturate,			  //  -> Satruate accumulation, if exceeds max
		ReductionOverflow,			  //  -> Overfow causes an error			 
		ReductionSustain			  //  -> Sustains the full accumulation
	};

	struct InvalidContext {};

	template<typename T>
	struct ProvideContextLoader {
		using type_ = InvalidContext;
	};

	using Width = size_t;
	constexpr Width INFINITE_WIDTH = 0;	// Use when the actual backend is infinite width in nature

	template<typename D, typename L, Width V>
	struct ILimbContext {
		LEIBNIZ_STATIC_ASSERT(!Traits::IsArbitraryWidthV<L>, "Limb type must a fixed width type");
		LEIBNIZ_STATIC_ASSERT(Traits::IsIntV<L> || Vectorized::Traits::IsVectorizedIntV<L>, "Limb type must be an integer type");
		LEIBNIZ_STATIC_ASSERT(!Traits::IsSignedIntV<L> || Vectorized::Traits::IsSignedVectorIntV<L>, "Limb cannot be signed integer type");

		using limbT_ = L;
		using derived_ = D;

		static constexpr Width limbCount = V;
		static constexpr Width bitWidth = V * sizeof(L) * 8;

		~ILimbContext() = default;
		ILimbContext() = default;

		ILimbContext(const ILimbContext&) = default;
		ILimbContext& operator=(const ILimbContext&) = default;
		ILimbContext(ILimbContext&&) noexcept = default;
		ILimbContext& operator=(ILimbContext&&) noexcept = default;

		LEIBNIZ_MAYBE_UNUSED constexpr limbT_* data() noexcept {
			return static_cast<derived_*>(this)->dataImpl();
		}

		LEIBNIZ_MAYBE_UNUSED constexpr const limbT_* data() const noexcept {
			return static_cast<const derived_*>(this)->dataImpl();
		}

		LEIBNIZ_MAYBE_UNUSED constexpr limbT_& operator[](Width i) noexcept {
			return data()[i];
		}

		LEIBNIZ_MAYBE_UNUSED constexpr const limbT_& operator[](Width i) const noexcept {
			return data()[i];
		}

		LEIBNIZ_MAYBE_UNUSED constexpr limbT_* begin() noexcept { return data(); }
		LEIBNIZ_MAYBE_UNUSED constexpr limbT_* end() noexcept { return data() + limbCount; }

		LEIBNIZ_MAYBE_UNUSED constexpr const limbT_* begin() const noexcept { return data(); }
		LEIBNIZ_MAYBE_UNUSED constexpr const limbT_* end() const noexcept { return data() + limbCount; }
	};

	template<typename T>
	concept HasContext = !std::is_same_v<typename ProvideContextLoader<T>::type_, InvalidContext>;

	template<typename T>
	concept SupportedBackendImplementation = HasContext<T> && requires (
		typename ProvideContextLoader<T>::type_& ctx,
		const typename ProvideContextLoader<T>::type_& cCtx) {
			{ T::zeroImpl(ctx) };
			{ T::copyImpl(ctx, cCtx) };
			{ T::compareImpl(cCtx, cCtx) } -> std::same_as<int>;
			{ T::addImpl(ctx, cCtx, cCtx) } -> std::same_as<bool>;
			{ T::subImpl(ctx, cCtx, cCtx) } -> std::same_as<bool>;
			{ T::mulImpl(ctx, cCtx, cCtx) };
			{ T::divImpl(ctx, ctx, cCtx, cCtx) };
			{ T::bitWidthImpl(cCtx) } -> std::same_as<size_t>;
			{ T::isZeroImpl(cCtx) } -> std::same_as<bool>;
			{ T::shiftLeftImpl(ctx, cCtx, size_t{}) } -> std::same_as<bool>;
			{ T::shiftRightImpl(ctx, cCtx, size_t{}) } -> std::same_as<bool>;
			{ T::squareImpl(ctx, cCtx) };
			{ T::fusedMulAddImpl(ctx, cCtx, cCtx) };
			{ T::fusedMulSubImpl(ctx, cCtx, cCtx) };
	};

}
