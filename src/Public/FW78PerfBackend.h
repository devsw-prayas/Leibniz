#pragma once
#include "Leibniz.h"
#include "ComputeBackends.h"
#include "BackendUtils.h"
#include "StaticPipelines.h"

namespace Leibniz::Numbers::Backend::Implementations {
	struct FW78PerfContext;
	template<bool AllowFullWidth>
	class FW78PerfBackend;
}

namespace Leibniz::Numbers::Backend {
	template<>
	struct ProvideContextLoader<Implementations::FW78PerfBackend<true>> {
		using type_ = Implementations::FW78PerfContext;
	};

	template<>
	struct ProvideContextLoader<Implementations::FW78PerfBackend<false>> {
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

	template<bool AllowFullWidth>
	class LEIBNIZ_RUNTIME_API FW78PerfBackend : public IComputeBackend<FW78PerfBackend<AllowFullWidth>> {
		using context_ = ProvideContextLoader<FW78PerfBackend>::type_;
		using dcontext_ = context_::DoubleWidth;
	public:
		// Zeroes a 256-bit numeric context represented as 4x uint64_t limbs.
		// The implementation uses fixed unrolled stores so the optimizer can
		// recognize the memory region as a contiguous 32-byte block.
		//
		// Expected Assembly (GCC/Clang/MSVC, -O3 -mavx2):
		//     vxorps      xmm0, xmm0, xmm0
		//     vmovups     ymmword ptr [rdi], ymm0
		//     vzeroupper
		//     ret
		//
		// Compiler Flags Used for Inspection:
		//     -O3 -march=native -mavx2
		//
		// Behavior:
		// The compiler collapses the four scalar stores into a single 256-bit AVX
		// zero store. This replaces four 64-bit writes with one YMM store, making
		// the operation store-bound and close to the hardware throughput limit
		// (~2-3 cycles on modern x86 cores). No explicit intrinsics are required;
		// the unrolled scalar pattern is sufficient for auto-vectorization.
		LEIBNIZ_FORCEINLINE static void zeroImpl(context_& ro_Out) {
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticFor<0, 4>([&](auto idx) {
				ro_Out.m_Stripes[idx] = 0; }),
				Utils::Unrolled::staticFor<0, 2>([&](auto idx) {
				ro_Out.m_Stripes[idx] = 0; })
			);
		}

		// Copies a fixed-width numeric context (256 bits) represented as 4X uint64_t
		// limbs. The fields are stored contiguously in memory, allowing the optimizer
		// to treat the operation as a full 32-byte object copy.
		//
		// Expected Assembly (GCC/Clang/MSVC, -O3 -march=native -mavx2):
		//     vmovups     ymm0, ymmword ptr [rsi]
		//     vmovups     ymmword ptr [rdi], ymm0
		//     vzeroupper
		//     ret
		//
		// Compiler Flags Used for Inspection:
		//     -O3 -march=native -mavx2
		//
		// Behavior:
		// The optimizer recognizes the four scalar assignments as a contiguous
		// 32-byte transfer and lowers the operation to a single 256-bit AVX load
		// followed by a 256-bit store. This replaces four load/store pairs with a
		// single vector copy, making the operation memory-throughput bound
		// (~2-3 cycles on modern x86 cores).
		LEIBNIZ_FORCEINLINE static void copyImpl(
			context_& LEIBNIZ_RESTRICT ro_Out,
			const context_& LEIBNIZ_RESTRICT ro_In
		) {
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticFor<0, 4>([&](auto idx) {
				ro_Out.m_Stripes[idx] = ro_In.m_Stripes[idx]; }),
				Utils::Unrolled::staticFor<0, 2>([&](auto idx) {
				ro_Out.m_Stripes[idx] = ro_In.m_Stripes[idx]; }));
		}

		// Lexicographic comparison of a 256-bit integer stored as 4x uint64_t limbs.
		// Limbs are compared from most-significant to least-significant to match
		// standard multi-precision integer ordering semantics.
		//
		// Expected Assembly Shape (GCC/Clang/MSVC):
		//     mov   limb
		//     cmp   limb
		//     jne   mismatch
		//     ...
		//     setcc
		//     lea   result
		//
		// Compiler Flags Used for Inspection:
		//     -O3 -march=native
		//
		// Behavior:
		// The compiler lowers the reverse-unrolled comparison into a chain of
		// MSB->LSB limb comparisons with early exit on the first mismatch.
		// The final result is produced branchlessly via setcc/lea, returning
		// -1 or +1 depending on ordering, or 0 when all limbs match.
		LEIBNIZ_FORCEINLINE LEIBNIZ_NODISCARD_MSG("Cannot discard comparison")
			static int compareImpl(const context_& ro_OpA, const context_& ro_OpB) {
			int result = 0;
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticForReverse<0, 4>([&](auto index) {
				if (ro_OpA.m_Stripes[index] != ro_OpB.m_Stripes[index])
					result = ro_OpA.m_Stripes[index] > ro_OpB.m_Stripes[index] ? -1 : 1; }),
				Utils::Unrolled::staticForReverse<0, 2>([&](auto index) {
				if (ro_OpA.m_Stripes[index] != ro_OpB.m_Stripes[index])
					result = ro_OpA.m_Stripes[index] > ro_OpB.m_Stripes[index] ? -1 : 1; }));
			return result;
		}

		// Multi-precision addition of a 256-bit integer represented as 4x uint64_t limbs.
		// Operands are added from least-significant to most-significant limb to propagate
		// the carry flag through the chain. The final carry-out is returned as a boolean.
		//
		// Expected Assembly Shape (GCC/Clang/MSVC):
		//     mov   limb
		//     add   limb
		//     mov   result
		//     adc   limb
		//     adc   limb
		//     adc   limb
		//     setb  result
		//
		// Compiler Flags Used for Inspection:
		//     -O3 -march=native
		//
		// Behavior:
		// The compiler lowers the intrinsic-based implementation into a fully unrolled
		// add/adc carry chain. Each limb addition consumes the previous carry flag and
		// produces the next, forming the optimal serialized dependency chain for
		// multi-precision addition. The final carry flag is materialized via setb and
		// returned to the caller.
		LEIBNIZ_FORCEINLINE static bool addImpl(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			unsigned char carry = 0;
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticFor<0, 4>([&](auto index) {
				carry = _addcarryx_u64(carry, ro_OpA.m_Stripes[index], ro_OpB.m_Stripes[index], reinterpret_cast<unsigned long long*>(&ro_Out.m_Stripes[index])); }),
				Utils::Unrolled::staticFor<0, 2>([&](auto index) {
				carry = _addcarryx_u64(carry, ro_OpA.m_Stripes[index], ro_OpB.m_Stripes[index], reinterpret_cast<unsigned long long*>(&ro_Out.m_Stripes[index])); }));
			return carry;
		}

		// Multi-precision subtraction of a 256-bit integer represented as 4x uint64_t
		// limbs. Limbs are processed from least-significant to most-significant so
		// that the borrow flag propagates correctly through the chain. The final
		// borrow-out is returned as a boolean.
		//
		// Expected Assembly Shape (GCC/Clang/MSVC):
		//     mov   limb
		//     sub   limb
		//     mov   result
		//     sbb   limb
		//     sbb   limb
		//     sbb   limb
		//     setb  result
		//
		// Compiler Flags Used for Inspection:
		//     -O3 -march=native
		//
		// Behavior:
		// The compiler lowers the intrinsic-based implementation into a fully
		// unrolled sub/sbb borrow chain. Each limb consumes the previous borrow
		// flag and produces the next, forming the optimal serialized dependency
		// chain for multi-precision subtraction. The final borrow flag is
		// materialized via setb and returned to the caller.
		LEIBNIZ_FORCEINLINE static bool subImpl(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			unsigned char borrow = 0;
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticFor<0, 4>([&](auto index) {
				borrow = _subborrow_u64(borrow, ro_OpA.m_Stripes[index], ro_OpB.m_Stripes[index], reinterpret_cast<unsigned long long*>(&ro_Out.m_Stripes[index])); }),
				Utils::Unrolled::staticFor<0, 2>([&](auto index) {
				borrow = _subborrow_u64(borrow, ro_OpA.m_Stripes[index], ro_OpB.m_Stripes[index], reinterpret_cast<unsigned long long*>(&ro_Out.m_Stripes[index])); }));
			return borrow;
		}

		LEIBNIZ_FORCEINLINE static void mulImpl(dcontext_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			Utils::Unrolled::staticIf<AllowFullWidth>([&]() {
				uint64_t sum = 0, carry = 0;
				Utils::Unrolled::staticDiagonal<4>([&](auto d, auto i, auto j) {
					if constexpr (i == 0) {
						sum = carry;
						carry = 0;
					}
					uint64_t hi, lo;
					lo = _mulx_u64(ro_OpA.m_Stripes[i], ro_OpB.m_Stripes[j], reinterpret_cast<unsigned long long*>(&hi));
					uint64_t old = sum;
					sum += lo;
					carry += hi;
					carry += (sum < old);

					if constexpr (j == 0 || i == 3) {
						ro_Out.m_Stripes[d] = sum;
					}});
					ro_Out.m_Stripes[7] = carry; },
				[&]() {
				uint64_t sum = 0, carry = 0;
				Utils::Unrolled::staticDiagonal<2>([&](auto d, auto i, auto j) {
					if constexpr (i == 0) {
						sum = carry;
						carry = 0;
					}
					uint64_t hi, lo;
					lo = _mulx_u64(ro_OpA.m_Stripes[i], ro_OpB.m_Stripes[j], reinterpret_cast<unsigned long long*>(&hi));
					uint64_t old = sum;
					sum += lo;
					carry += hi;
					carry += (sum < old);

					if constexpr (j == 0 || i == 1) {
						ro_Out.m_Stripes[d] = sum;
					}});
					ro_Out.m_Stripes[3] = carry; }
			);
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
