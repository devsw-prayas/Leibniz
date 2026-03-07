#pragma once

namespace Leibniz::Numbers::Backend::Internal::Types {
	// A portable dual limb representation of uint128

	struct U128 final {
		uint64_t low;
		uint64_t high;

		U128(uint64_t l = 0, uint64_t h = 0) : low(l), high(h) {}
	};
}
