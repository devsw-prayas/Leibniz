#pragma once 
#include <Leibniz.h>

namespace Leibniz::Vectorized::Traits {
	template<typename T> struct LEIBNIZ_RUNTIME_API IsVectorizedInt final : std::false_type{};
	template<typename T> constexpr bool LEIBNIZ_RUNTIME_API IsVectorizedIntV = IsVectorizedInt<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsVectorizedFloat final : std::false_type {};
	template<typename T> constexpr bool LEIBNIZ_RUNTIME_API IsVectorizedFloatV = IsVectorizedFloat<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsSignedVectorInt final : std::false_type {};
	template<typename T> constexpr bool LEIBNIZ_RUNTIME_API IsSignedVectorIntV = IsSignedVectorInt<T>::value;

	template<typename T> struct LEIBNIZ_RUNTIME_API IsUnsignedVectorInt final : std::false_type {};
	template<typename T> constexpr bool LEIBNIZ_RUNTIME_API IsUnsignedVectorIntV = IsUnsignedVectorInt<T>::value;
}
