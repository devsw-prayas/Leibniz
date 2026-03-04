#pragma once
#include "Leibniz.h"
#include "ComputeBackends.h"
#include "BackendUtils.h"

namespace Leibniz::Numbers::Backend::Implementations {
	struct FW78PerfContext;
	class FW78PerfBackend;
}

namespace Leibniz::Numbers::Backend {
	template<>
	struct ProvideContextLoader<Implementations::FW78PerfBackend> {
		using type_ = Implementations::FW78PerfContext;
	};
}

namespace Leibniz::Numbers::Backend::Implementations {
	namespace Intermediate {
		struct DoubleFW78PerfContext final : ILimbContext<DoubleFW78PerfContext, uint64_t, 8> {
			using DoubleWidth = void;
			alignas(64) limbT_ m_Stripes[8];
			constexpr limbT_* dataImpl() noexcept {
				return m_Stripes;
			}

			LEIBNIZ_NODISCARD_MSG("Cannot discard const ptr to storage")
				constexpr const limbT_* dataImpl() const noexcept {
				return m_Stripes;
			}
		};
	}

	struct FW78PerfContext final : ILimbContext<FW78PerfContext, uint64_t, 4> {
		using DoubleWidth = Intermediate::DoubleFW78PerfContext;
		alignas(32) limbT_ m_Stripes[4];
		constexpr limbT_* dataImpl() noexcept {
			return m_Stripes;
		}

		LEIBNIZ_NODISCARD_MSG("Cannot discard const ptr to storage")
			constexpr const limbT_* dataImpl() const noexcept {
			return m_Stripes;
		}
	};

	class LEIBNIZ_RUNTIME_API FW78PerfBackend : public IComputeBackend<FW78PerfBackend> {
		using context_ = ProvideContextLoader<FW78PerfBackend>::type_;
	public:
		static void zeroImpl(context_& ro_Out) {
			
		}

		static void copyImpl(context_& ro_Out, const context_& ro_In) {
			
		}

		LEIBNIZ_NODISCARD_MSG("Cannot discard comparison")
		static int compareImpl(const context_& ro_OpA, const context_& ro_OpB) {
			LEIBNIZ_UNREACHABLE(); // TODO dont call and kill urself
		}

		static bool addImpl(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			LEIBNIZ_UNREACHABLE();
		}

		static bool subImpl(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			LEIBNIZ_UNREACHABLE();
		}

		static void mulImpl(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			
		}

		static void divImpl(context_& ro_Quo, context_& ro_Rem, const context_& ro_OpA, const context_& ro_OpB) {

		}

		static size_t bitWidthImpl(const context_& ro_Op) {
			LEIBNIZ_UNREACHABLE();
		}

		static bool isZeroImpl(const context_& ro_Op) {
			LEIBNIZ_UNREACHABLE();
		}

		LEIBNIZ_NODISCARD_MSG("Cannot discard shift state")
		static bool shiftLeftImpl(context_& ro_Out, const context_& ro_Op, size_t v_Bits) {
			LEIBNIZ_UNREACHABLE();
		}

		LEIBNIZ_NODISCARD_MSG("Cannot discard shift state")
		static bool shiftRightImpl(context_& ro_Out, const context_& ro_Op, size_t v_Bits) {
			LEIBNIZ_UNREACHABLE();
		}

		static void squareImpl(context_& ro_Out, const context_& ro_Op) {

		}

		static void fusedMulAddImpl(context_& ro_Acc, const context_& ro_OpA, const context_& ro_OpB) {

		}

		static void fusedMulSubImpl(context_& ro_Acc, const context_& ro_OpA, const context_& ro_OpB) {

		}

	};
}
