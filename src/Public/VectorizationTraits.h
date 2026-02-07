#pragma once
#include "Leibniz.h"

namespace Leibniz::Vectorization::Traits {

	enum class LEIBNIZ VectorizationBackend : uint8_t {
		UNKNOWN,
		AVX2
	};

	template<VectorizationBackend Backend = VectorizationBackend::UNKNOWN>
	struct LEIBNIZ ImplementationType final {
		static constexpr size_t s_Width = 0;
		static constexpr size_t s_Alignment = 0;
	};

	template<VectorizationBackend Backend>
	inline constexpr LEIBNIZ size_t VectorizationWidth = ImplementationType<Backend>::s_Width;

	template<VectorizationBackend Backend>
	inline constexpr LEIBNIZ size_t VectorizationAlignment = ImplementationType<Backend>::s_Alignment;

	template<>
	struct LEIBNIZ ImplementationType<VectorizationBackend::AVX2> final{
		static constexpr size_t s_Width = 256;
		static constexpr size_t s_Alignment = 32;
	};

	template<typename T, typename = void>
	struct LEIBNIZ VectorizeType final {
		using type = void;
	};

	template<typename T>
	using VectorizedForm = VectorizeType<T>::type;

	template<typename T, typename = void>
	struct LEIBNIZ VectorizeMaskType final {
		using type = void;
	};

	template<typename T>
	using  VectorizedMask = VectorizeMaskType<T>::type;

	template<typename T, typename = void>
	struct LEIBNIZ ScalarType final {
		using type = void;
	};

	template<typename T>
	using ScalarForm = ScalarType<T>::type;

	struct Blank final {};

	template<typename T, typename = void>
	struct LEIBNIZ IsVectorStripe final : std::false_type {};

	template<typename T, typename = void>
	struct LEIBNIZ IsVectorMask final : std::false_type {};

	template<typename T>
	inline static constexpr LEIBNIZ bool IsVectorStripeV = IsVectorStripe<T>::value;

	template<typename T>
	inline static constexpr LEIBNIZ bool IsVectorMaskV = IsVectorMask<T>::value;


	template<typename T = Blank>
	struct LEIBNIZ VectorizationIntrospect final {
		static constexpr size_t s_Alignment = 0;
		static constexpr size_t s_Lanes = 0;
		static constexpr size_t s_Width = 0;
		static constexpr const char* s_Name = "Invalid Type";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::UNKNOWN;
	};

	template<typename T = Blank>
	inline static constexpr LEIBNIZ size_t IntrospectAlignment = VectorizationIntrospect<T>::s_Alignment;

	template<typename T = Blank>
	inline static constexpr LEIBNIZ size_t IntrospectLanes = VectorizationIntrospect<T>::s_Lanes;

	template<typename T = Blank>
	inline static constexpr LEIBNIZ size_t IntrospectWidth = VectorizationIntrospect<T>::s_Width;

	template<typename T = Blank>
	inline static constexpr LEIBNIZ const char* IntrospectTypeName = VectorizationIntrospect<T>::s_Name;

	template<typename T = Blank>
	inline static constexpr LEIBNIZ VectorizationBackend IntrospectBackend = VectorizationIntrospect<T>::s_Backend;

	template<typename T>
	inline static constexpr LEIBNIZ bool TemplateFalseV = false;
}
