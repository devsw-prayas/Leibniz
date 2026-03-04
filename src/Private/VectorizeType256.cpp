#include "Leibniz.h"
#include <VectorizeType256.h>

#define ALLOW_VECTOR_INTRIN_256
#include <v256Intrin.h>

namespace Leibniz::Vectorization::v256 {
	StripeU8::StripeU8() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	StripeU16::StripeU16() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	StripeU32::StripeU32() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	StripeU64::StripeU64() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	Stripe8::Stripe8() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	Stripe16::Stripe16() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	Stripe32::Stripe32() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	Stripe64::Stripe64() {
		m_VectorBin = Intrinsic::v256::zero();
	}

	StripeFP32::StripeFP32() {
		m_VectorBin = Intrinsic::v256::zeroSinglePrecision();
	}

	StripeFP64::StripeFP64() {
		m_VectorBin = Intrinsic::v256::zeroDoublePrecision();
	}

}