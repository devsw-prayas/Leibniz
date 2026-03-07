#include "Leibniz.h"
#include "asmintrin.h"

#if LEIBNIZ_COMPILER_MSVC
namespace Leibniz::Intrin {
	uint64_t mulx_u64(uint64_t a, uint64_t b, uint64_t& hi) {
		return Internal::asmMulxU64(a, b, &hi);
	}

	void adcx_u64(uint64_t* a, uint64_t b) {
		Internal::asmAdcxU64(a, b);
	}

	void adox_u64(uint64_t* a, uint64_t b) {
		Internal::asmAdoxU64(a, b);
	}

	void init_interleave() {
		Internal::asmInitInterleave();
	}
}
#endif