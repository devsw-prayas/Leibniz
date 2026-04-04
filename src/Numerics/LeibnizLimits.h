#pragma once
#include "Leibniz.h"
#include "LeibnizFloat.h"
#include "LeibnizCompiler.h"
#include "LebnizDiagnostics.h"
#include "VectorizationTraits.h"

namespace Leibniz::Numerics::Constants{
	template<typename T> requires Traits::IsFloatV<T>
	struct NumericLimits final {
		static T epsilon() {
			if constexpr (std::is_same_v<T, Float32>)
				return 1.19209290e-7f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 2.2204460492503131e-16;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T sqrtEpsilon() {
			if constexpr (std::is_same_v<T, Float32>)
				return 3.45266983e-4f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 1.4901161193847656e-8;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}


		static T pi() {
			if constexpr (std::is_same_v<T, Float32>)
				return 3.14159265f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 3.141592653589793;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T tau() {
			if constexpr (std::is_same_v<T, Float32>)
				return 6.28318530f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 6.283185307179586;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}
		
		static T euler() {
			if constexpr (std::is_same_v<T, Float32>)
				return 2.71828182f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 2.718281828459045;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T sqrt2() {
			if constexpr (std::is_same_v<T, Float32>)
				return 1.41421356f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 1.4142135623730951;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T goldenRatio() {
			if constexpr (std::is_same_v<T, Float32>)
				return 1.61803398f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 1.6180339887498949;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T ln2() {
			if constexpr (std::is_same_v<T, Float32>)
				return 0.69314718f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 0.6931471805599453;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T ln10() {
			if constexpr (std::is_same_v<T, Float32>)
				return 2.30258509f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 2.302585092994046;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T log2e() {
			if constexpr (std::is_same_v<T, Float32>)
				return 1.44269504f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 1.4426950408889634;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T log10e() {
			if constexpr (std::is_same_v<T, Float32>)
				return 0.43429448f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 0.4342944819032518;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T eulerMascheroni() {
			if constexpr (std::is_same_v<T, Float32>)
				return 0.57721566f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 0.5772156649015329;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T catalan() {
			if constexpr (std::is_same_v<T, Float32>)
				return 0.91596559f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 0.9159655941772190;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}

		static T apery() {
			if constexpr (std::is_same_v<T, Float32>)
				return 1.20205690f;
			else if constexpr (std::is_same_v<T, Float64>)
				return 1.2020569031595943;
			else {
				LEIBNIZ_STATIC_ASSERT(Vectorization::Traits::TemplateFalseV<T>, "Invalid floating point type");
				LEIBNIZ_UNREACHABLE();
			}
		}
		
	};
}
