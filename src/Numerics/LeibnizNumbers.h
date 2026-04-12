#pragma once
#include "Leibniz.h"
#include "LeibnizFloat.h"
#include "VectorizationTraits.h"
#include "LeibnizDiagnostics.h"
#include "LeibnizLimits.h"
#include "NumericConstants.h"

namespace Leibniz::Numerics {
	// LeibnizConstants<T>
	// Unified compile-time constants and numeric limits for all float types.
	// Includes: math constants (pi, ln2, sqrt2, ...) and numeric limits
	// (epsilon, sqrtEpsilon, minNormal, minSubnormal, max).

	template<typename T> requires Traits::IsFloatV<T>
	struct LeibnizConstants final {
		// Numeric Limits

		static constexpr T epsilon() noexcept {
			if constexpr (std::is_same_v<T, Float32>)
				return Limits::NumericLimits<Float32>::epsilon();
			else if constexpr (std::is_same_v<T, Float64>)
				return Limits::NumericLimits<Float64>::epsilon();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T sqrtEpsilon() noexcept {
			if constexpr (std::is_same_v<T, Float32>)
				return Limits::NumericLimits<Float32>::sqrtEpsilon();
			else if constexpr (std::is_same_v<T, Float64>)
				return Limits::NumericLimits<Float64>::sqrtEpsilon();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T minNormal() noexcept {
			if constexpr (std::is_same_v<T, Float32>)
				return Limits::NumericLimits<Float32>::minNormal();
			else if constexpr (std::is_same_v<T, Float64>)
				return Limits::NumericLimits<Float64>::minNormal();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T minSubnormal() noexcept {
			if constexpr (std::is_same_v<T, Float32>)
				return Limits::NumericLimits<Float32>::minSubnormal();
			else if constexpr (std::is_same_v<T, Float64>)
				return Limits::NumericLimits<Float64>::minSubnormal();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T max() noexcept {
			if constexpr (std::is_same_v<T, Float32>)
				return Limits::NumericLimits<Float32>::max();
			else if constexpr (std::is_same_v<T, Float64>)
				return Limits::NumericLimits<Float64>::max();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		// Integral metadata-not type T, always int
		static constexpr int mantissaBits() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Limits::NumericLimits<Float32>::mantissaBits;
			else if constexpr (std::is_same_v<T, Float64>) return Limits::NumericLimits<Float64>::mantissaBits;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr int exponentBits() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Limits::NumericLimits<Float32>::exponentBits;
			else if constexpr (std::is_same_v<T, Float64>) return Limits::NumericLimits<Float64>::exponentBits;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr int exponentBias() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Limits::NumericLimits<Float32>::exponentBias;
			else if constexpr (std::is_same_v<T, Float64>) return Limits::NumericLimits<Float64>::exponentBias;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr int digits10() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Limits::NumericLimits<Float32>::digits10;
			else if constexpr (std::is_same_v<T, Float64>) return Limits::NumericLimits<Float64>::digits10;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		// Core Constants

		static constexpr T pi() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::pi();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::pi();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T tau() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::tau();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::tau();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T euler() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::euler();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::euler();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T sqrt2() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::sqrt2();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::sqrt2();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T sqrt3() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::sqrt3();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::sqrt3();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T sqrt5() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::sqrt5();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::sqrt5();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T goldenRatio() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::goldenRatio();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::goldenRatio();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		// Logarithmic

		static constexpr T ln2() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::ln2();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::ln2();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T ln10() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::ln10();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::ln10();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T log2e() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::log2e();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::log2e();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T log10e() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::log10e();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::log10e();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T log2_10() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::log2_10();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::log2_10();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T log10_2() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::log10_2();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::log10_2();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		// Inverses

		static constexpr T invPi() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::invPi();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::invPi();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T invTwoPi() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::invTwoPi();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::invTwoPi();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T invE() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::invE();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::invE();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T invLn2() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::invLn2();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::invLn2();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T invLn10() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::invLn10();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::invLn10();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		// Angle

		static constexpr T degToRad() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::degToRad();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::degToRad();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T radToDeg() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::radToDeg();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::radToDeg();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		// Special

		static constexpr T eulerMascheroni() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::eulerMascheroni();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::eulerMascheroni();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T catalan() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::catalan();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::catalan();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static constexpr T apery() noexcept {
			if constexpr (std::is_same_v<T, Float32>)      return Constants::MathConstants<Float32>::apery();
			else if constexpr (std::is_same_v<T, Float64>) return Constants::MathConstants<Float64>::apery();
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Unsupported float type");
				LEIBNIZ_UNREACHABLE();
			}
		}
	};
} // namespace Leibniz::Numerics