#pragma once
#include <Leibniz.h>
#include <LebnizDiagnostics.h>

namespace Leibniz::Numbers::Frontend {
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

	template<typename T> 
	class LEIBNIZ_RUNTIME_API IComputeBackend final {
	public:

		IComputeBackend() {
			_mulx_u64()
		}
	};

}
