#pragma once
#include <iso646.h>

#include "Leibniz.h"
#include "ComputeBackends.h"
#include "BackendUtils.h"
#include "StaticPipelines.h"
#include "asmintrin.h"
#include "MathIntrin.h"

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
		LEIBNIZ_FORCEINLINE static void zeroImpl(context_& ro_Out) {
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticFor<0, 4>([&](auto idx) {
				ro_Out.m_Stripes[idx] = 0; }),
				Utils::Unrolled::staticFor<0, 2>([&](auto idx) {
				ro_Out.m_Stripes[idx] = 0; })
			);
		}

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

		LEIBNIZ_FORCEINLINE static bool addImpl(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			unsigned char carry = 0;
			Utils::Unrolled::staticIf<AllowFullWidth>(
				Utils::Unrolled::staticFor<0, 4>([&](auto index) {
				carry = _addcarryx_u64(carry, ro_OpA.m_Stripes[index], ro_OpB.m_Stripes[index], reinterpret_cast<unsigned long long*>(&ro_Out.m_Stripes[index])); }),
				Utils::Unrolled::staticFor<0, 2>([&](auto index) {
				carry = _addcarryx_u64(carry, ro_OpA.m_Stripes[index], ro_OpB.m_Stripes[index], reinterpret_cast<unsigned long long*>(&ro_Out.m_Stripes[index])); }));
			return carry;
		}

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
			Utils::Unrolled::staticIf<AllowFullWidth>([&ro_OpA, &ro_OpB, &ro_Out]() {
				uint64_t sum = 0, carry = 0;
				Intrin::init_interleave();
				Utils::Unrolled::staticDiagonal<4>([&ro_OpA, &ro_OpB, &ro_Out, &sum, &carry](auto d, auto i, auto j) {
					if constexpr (i == 0) {
						sum = carry;
						carry = 0;
					}
					uint64_t hi, lo = Intrin::mulx_u64(ro_OpA.m_Stripes[i], ro_OpB.m_Stripes[j], hi);
					Intrin::adcx_u64(&sum, lo);
					Intrin::adox_u64(&carry, hi);

					if constexpr (j == 0 || i == 3) {
						ro_Out.m_Stripes[d] = sum;
					}});
					ro_Out.m_Stripes[7] = carry; },
				[&]() {
				uint64_t sum = 0, carry = 0;
				Utils::Unrolled::staticDiagonal<2>([&ro_OpA, &ro_OpB, &ro_Out, &sum, &carry](auto d, auto i, auto j) {
					if constexpr (i == 0) {
						sum = carry;
						carry = 0;
					}
					uint64_t hi, lo = Intrin::mulx_u64(ro_OpA.m_Stripes[i], ro_OpB.m_Stripes[j], hi);
					Intrin::adcx_u64(&sum, lo);
					Intrin::adox_u64(&carry, hi);

					if constexpr (j == 0 || i == 1) {
						ro_Out.m_Stripes[d] = sum;
					}});
					ro_Out.m_Stripes[3] = carry; }
			);
		}

		LEIBNIZ_RUNTIME_API static void divImpl(context_& ro_Quo, context_& ro_Rem, const context_& ro_OpA, const context_& ro_OpB) {
			Utils::Unrolled::staticIf<AllowFullWidth>(
				[&ro_Quo, &ro_Rem, &ro_OpA, &ro_OpB]() {
				// Generate reciprocal approximation
				// TODO
			},
			[&ro_Quo, &ro_Rem, &ro_OpA, &ro_OpB]() {
			}
			);
		}

		LEIBNIZ_FORCEINLINE static size_t bitWidthImpl(const context_& ro_Op) {
			size_t width = 0;

			Utils::Unrolled::staticIf<AllowFullWidth>(
				[&]() {
				if (ro_Op.m_Stripes[3])
					width = 256 - Clz64(ro_Op.m_Stripes[3]);
				else if (ro_Op.m_Stripes[2])
					width = 192 - Clz64(ro_Op.m_Stripes[2]);
				else if (ro_Op.m_Stripes[1])
					width = 128 - Clz64(ro_Op.m_Stripes[1]);
				else if (ro_Op.m_Stripes[0])
					width = 64 - Clz64(ro_Op.m_Stripes[0]);
				else
					width = 0;
			},
				[&]() {
				if (ro_Op.m_Stripes[1])
					width = 128 - Clz64(ro_Op.m_Stripes[1]);
				else if (ro_Op.m_Stripes[0])
					width = 64 - Clz64(ro_Op.m_Stripes[0]);
				else
					width = 0;
			}
			);

			return width;
		}

		static bool isZeroImpl(const context_& ro_Op) {
			uint64_t limbAcc = 0;
			Utils::Unrolled::staticIf<AllowFullWidth>(
				[&]() {
				Utils::Unrolled::staticFor<0, 4>([&](auto i) {
					limbAcc |= ro_Op.m_Stripes[i];
				});
			},
				[&]() {
				Utils::Unrolled::staticFor<0, 2>([&](auto i) {
					limbAcc |= ro_Op.m_Stripes[i];
				});
			}
			);
			return limbAcc == 0;
		}

		LEIBNIZ_NODISCARD_MSG("Cannot discard shift state")
			static bool shiftLeftImpl(context_& ro_Out, const context_& ro_Op, size_t v_Bits) {
			bool sticky = false;

			uint64_t s0 = ro_Op.m_Stripes[0];
			uint64_t s1 = ro_Op.m_Stripes[1];

			Utils::Unrolled::staticIf<AllowFullWidth>(
				[&]() {
				uint64_t s2 = ro_Op.m_Stripes[2];
				uint64_t s3 = ro_Op.m_Stripes[3];

				if (v_Bits == 0) {
					ro_Out.m_Stripes[0] = s0;
					ro_Out.m_Stripes[1] = s1;
					ro_Out.m_Stripes[2] = s2;
					ro_Out.m_Stripes[3] = s3;
					sticky = false;
					return;
				}

				if (v_Bits >= 256) {
					sticky = (s0 | s1 | s2 | s3) != 0;

					ro_Out.m_Stripes[0] = 0;
					ro_Out.m_Stripes[1] = 0;
					ro_Out.m_Stripes[2] = 0;
					ro_Out.m_Stripes[3] = 0;
					return;
				}

				size_t limbShift = v_Bits >> 6;
				size_t bitShift = v_Bits & 63;

				uint64_t lost = 0;

				if (limbShift) {
					if (limbShift >= 1) lost |= s3;
					if (limbShift >= 2) lost |= s2;
					if (limbShift >= 3) lost |= s1;

					if (limbShift == 1) { s3 = s2; s2 = s1; s1 = s0; s0 = 0; } else if (limbShift == 2) { s3 = s1; s2 = s0; s1 = 0; s0 = 0; } else if (limbShift == 3) { s3 = s0; s2 = s1 = s0 = 0; }
				}

				if (bitShift) {
					lost |= s3 >> (64 - bitShift);

					s3 = (s3 << bitShift) | (s2 >> (64 - bitShift));
					s2 = (s2 << bitShift) | (s1 >> (64 - bitShift));
					s1 = (s1 << bitShift) | (s0 >> (64 - bitShift));
					s0 <<= bitShift;
				}

				ro_Out.m_Stripes[0] = s0;
				ro_Out.m_Stripes[1] = s1;
				ro_Out.m_Stripes[2] = s2;
				ro_Out.m_Stripes[3] = s3;

				sticky = lost != 0;
			},
				[&]() {
				if (v_Bits == 0) {
					ro_Out.m_Stripes[0] = s0;
					ro_Out.m_Stripes[1] = s1;
					sticky = false;
					return;
				}

				if (v_Bits >= 128) {
					sticky = (s0 | s1) != 0;

					ro_Out.m_Stripes[0] = 0;
					ro_Out.m_Stripes[1] = 0;
					return;
				}

				size_t limbShift = v_Bits >> 6;
				size_t bitShift = v_Bits & 63;

				uint64_t lost = 0;

				if (limbShift) {
					lost |= s1;
					s1 = s0;
					s0 = 0;
				}

				if (bitShift) {
					lost |= s1 >> (64 - bitShift);

					s1 = (s1 << bitShift) | (s0 >> (64 - bitShift));
					s0 <<= bitShift;
				}

				ro_Out.m_Stripes[0] = s0;
				ro_Out.m_Stripes[1] = s1;

				sticky = lost != 0;
			});

			return sticky;
		}

		LEIBNIZ_NODISCARD_MSG("Cannot discard shift state")
			static bool shiftRightImpl(context_& ro_Out, const context_& ro_Op, size_t v_Bits) {
			bool sticky = false;

			uint64_t s0 = ro_Op.m_Stripes[0];
			uint64_t s1 = ro_Op.m_Stripes[1];

			Utils::Unrolled::staticIf<AllowFullWidth>(
				[&]() {
				uint64_t s2 = ro_Op.m_Stripes[2];
				uint64_t s3 = ro_Op.m_Stripes[3];

				if (v_Bits == 0) {
					ro_Out.m_Stripes[0] = s0;
					ro_Out.m_Stripes[1] = s1;
					ro_Out.m_Stripes[2] = s2;
					ro_Out.m_Stripes[3] = s3;
					sticky = false;
					return;
				}

				if (v_Bits >= 256) {
					sticky = (s0 | s1 | s2 | s3) != 0;

					ro_Out.m_Stripes[0] = 0;
					ro_Out.m_Stripes[1] = 0;
					ro_Out.m_Stripes[2] = 0;
					ro_Out.m_Stripes[3] = 0;
					return;
				}

				size_t limbShift = v_Bits >> 6;
				size_t bitShift = v_Bits & 63;

				uint64_t lost = 0;

				if (limbShift) {
					if (limbShift >= 1) lost |= s0;
					if (limbShift >= 2) lost |= s1;
					if (limbShift >= 3) lost |= s2;

					if (limbShift == 1) { s0 = s1; s1 = s2; s2 = s3; s3 = 0; } else if (limbShift == 2) { s0 = s2; s1 = s3; s2 = 0; s3 = 0; } else if (limbShift == 3) { s0 = s3; s1 = 0; s2 = 0; s3 = 0; }
				}

				if (bitShift) {
					lost |= s0 << (64 - bitShift);

					s0 = (s0 >> bitShift) | (s1 << (64 - bitShift));
					s1 = (s1 >> bitShift) | (s2 << (64 - bitShift));
					s2 = (s2 >> bitShift) | (s3 << (64 - bitShift));
					s3 >>= bitShift;
				}

				ro_Out.m_Stripes[0] = s0;
				ro_Out.m_Stripes[1] = s1;
				ro_Out.m_Stripes[2] = s2;
				ro_Out.m_Stripes[3] = s3;

				sticky = lost != 0;
			},
				[&]() {
				if (v_Bits == 0) {
					ro_Out.m_Stripes[0] = s0;
					ro_Out.m_Stripes[1] = s1;
					sticky = false;
					return;
				}

				if (v_Bits >= 128) {
					sticky = (s0 | s1) != 0;

					ro_Out.m_Stripes[0] = 0;
					ro_Out.m_Stripes[1] = 0;
					return;
				}

				size_t limbShift = v_Bits >> 6;
				size_t bitShift = v_Bits & 63;

				uint64_t lost = 0;

				if (limbShift) {
					lost |= s0;
					s0 = s1;
					s1 = 0;
				}

				if (bitShift) {
					lost |= s0 << (64 - bitShift);

					s0 = (s0 >> bitShift) | (s1 << (64 - bitShift));
					s1 >>= bitShift;
				}

				ro_Out.m_Stripes[0] = s0;
				ro_Out.m_Stripes[1] = s1;

				sticky = lost != 0;
			});

			return sticky;
		}

		LEIBNIZ_FORCEINLINE static void squareImpl(dcontext_& ro_Out, const context_& ro_Op) {
			mul(ro_Out, ro_Op, ro_Op);
		}

		static void fusedMulAddImpl(context_& ro_Acc, const context_& ro_OpA, const context_& ro_OpB) {
		}

		static void fusedMulSubImpl(context_& ro_Acc, const context_& ro_OpA, const context_& ro_OpB) {
		}
	};
}
