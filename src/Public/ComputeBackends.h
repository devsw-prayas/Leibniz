/*
* Copyright (c) 2026 StormWeaver
*
* This file is part of the Leibniz Math Compute Library
*
* Licensed under the MIT License. You may obtain a copy of the License at
* https://opensource.org/licenses/MIT
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
*/
#pragma once
#include <Leibniz.h>
#include <LebnizDiagnostics.h>

#include "BackendUtils.h"

namespace Leibniz::Numbers::Backend {
	template<typename T> 
	class LEIBNIZ_RUNTIME_API IComputeBackend {
		using derived_ = T;
		using context_ = ProvideContextLoader<derived_>::type_;
	public:
		IComputeBackend() = delete;
		~IComputeBackend() = delete; 

		IComputeBackend(const IComputeBackend&) = delete;
		IComputeBackend& operator=(const IComputeBackend&) = delete;

		IComputeBackend(IComputeBackend&&) noexcept = delete;
		IComputeBackend& operator=(IComputeBackend&&) noexcept = delete;

		// Guarantees that the implementation will always construct a full width multi-limb zero if fixed precision
		// or will use a single zero limb if the precision is arbitrary
		static void zero(context_& ro_Out) {
			derived_::zeroImpl(ro_Out);
		}

		// Guarantees that the implementation will always perform a perfect value copy, whether arbitrary or
		// fixed precision.
		static void copy(context_& ro_Out, const context_& ro_In) {
			derived_::copyImpl(ro_Out, ro_In);
		}

		// Guarantees that the implementation will always perform a true numeric bitwise compare to enforce
		// natural ordering of unsigned integers
		static int LEIBNIZ_NODISCARD_MSG("Cannot discard comparison") compare(const context_& ro_OpA, const context_& ro_OpB) {
			return derived_::compareImpl(ro_OpA, ro_OpB);
		}

		// Guarantees that the implementation will perform a limb based addition operation with full carry over
		// It is necessary that output context be provided and the carry-out should not be discarded
		static bool LEIBNIZ_NODISCARD_MSG("Cannot discard final carry") add(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			return derived_::addImpl(ro_Out, ro_OpA, ro_OpB);
		}

		// Guarantees that the implementation will perform a limb based subtraction operation with full borrow
		// It is necessary that output context be provided and the borrow-in should not be discarded.
		static bool LEIBNIZ_NODISCARD_MSG("Cannot discard final borrow") sub(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			return derived_::subImpl(ro_Out, ro_OpA, ro_OpB);
		}

		// Guarantees that the implementation will perform a full-width limb-based multiplication producing an exact unsigned product representation.
		// The output context must be pre-sized to hold the full result domain (e.g., 2n limbs for n-limb operands in arbitrary precision).
		// No allocation or growth is performed inside the backend.
		static void mul(context_& ro_Out, const context_& ro_OpA, const context_& ro_OpB) {
			derived_::mulImpl(ro_Out, ro_OpA, ro_OpB);
		}

		// Guarantees that the implementation will perform an unsigned integer division producing both quotient and remainder.
		// The quotient and remainder contexts must be pre-sized appropriately by the frontend. No resizing or allocation is performed internally.
		// Behavior is undefined if divisor represents zero.
		static void div(context_& ro_Quo, context_& ro_Rem, const context_& ro_OpA, const context_& ro_OpB) {
			derived_::divImpl(ro_Quo, ro_Rem, ro_OpA, ro_OpB);
		}

		// Guarantees that the implementation will perform a logical left shift by the specified bit count across the full limb domain.
		// The output context must be pre-sized. Returns the high carry-out bit (or limb overflow indicator).
		// Mo growth or canonicalization is performed.
		static bool LEIBNIZ_NODISCARD_MSG("Cannot discard shift state") shiftLeft(context_& ro_Out, const context_& ro_Op, size_t v_Bits) {
			return derived_::shiftLeftImpl(ro_Out, ro_Op, v_Bits);
		}

		// Guarantees that the implementation will perform a logical right shift by the specified bit count across the full limb domain.
		// The output context must be pre-sized. Returns an indicator representing whether any non-zero bits were shifted out.
		// No normalization or trimming is performed internally.
		static bool LEIBNIZ_NODISCARD_MSG("Cannot discard shift state") shiftRight(context_& ro_Out, const context_& ro_Op, size_t v_Bits) {
			return derived_::shiftRightImpl(ro_Out, ro_Op, v_Bits);
		}

		// Guarantees that the implementation will compute the exact unsigned bit-width of the operand (position of the most significant set bit).
		// Returns zero if and only if the operand is numerically zero. This operation must not mutate the context.
		static size_t bitWidth(const context_& ro_Op) {
			return derived_::bitWidthImpl(ro_Op);
		}

		// Guarantees that the implementation will determine whether the operand represents the numeric value zero across its full limb domain.
		// This operation must not mutate the context.
		static bool isZero(const context_& ro_Op) {
			return derived_::isZeroImpl(ro_Op);
		}

		// Guarantees that the implementation will compute the full-width square of the operand and store the result in the output context.
		// The output context must be pre-sized appropriately. This operation may use specialized squaring optimizations
		// distinct from generic multiplication.
		static void square(context_& ro_Out, const context_& ro_Op) {
			derived_::squareImpl(ro_Out, ro_Op);
		}

		// Guarantees that the implementation will perform a fused unsigned
		// multiply-accumulate operation:
		//
		//     ro_Acc += ro_OpA * ro_OpB
		//
		// The accumulator context must be pre-sized to hold the full result. No intermediate temporary allocation is performed.
		// This operation is intended for high-performance accumulation paths such as multi-limb multiplication or polynomial expansion.
		static void fusedMulAdd(context_& ro_Acc, const context_& ro_OpA, const context_& ro_OpB) {
			derived_::fusedMulAddImpl(ro_Acc, ro_OpA, ro_OpB);
		}

		// Guarantees that the implementation will perform a fused unsigned
		// multiply-subtract operation:
		//
		//     ro_Acc -= ro_OpA * ro_OpB
		//
		// The accumulator context must be pre-sized appropriately. No allocation or growth is performed internally.
		// Borrow propagation must be fully handled by the implementation.
		static void fusedMulSub(context_& ro_Acc, const context_& ro_OpA, const context_& ro_OpB) {
			derived_::fusedMulSubImpl(ro_Acc, ro_OpA, ro_OpB);
		}
	};
}
