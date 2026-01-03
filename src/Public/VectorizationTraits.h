#pragma once
#include "Lebnitz.h"

namespace Lebnitz::Vectorization::Traits {

	enum class LEBNITZ VectorizationBackend : uint8_t {
		UNKNOWN,
		AVX2
	};

	template<VectorizationBackend Backend = VectorizationBackend::UNKNOWN>
	struct LEBNITZ ImplementationType final {
		static constexpr size_t s_Width = 0;
		static constexpr size_t s_Alignment = 0;
	};

	template<VectorizationBackend Backend>
	inline constexpr LEBNITZ size_t VectorizationWidth = ImplementationType<Backend>::s_Width;

	template<VectorizationBackend Backend>
	inline constexpr LEBNITZ size_t VectorizationAlignment = ImplementationType<Backend>::s_Alignment;

	template<>
	struct LEBNITZ ImplementationType<VectorizationBackend::AVX2> final{
		static constexpr size_t s_Width = 256;
		static constexpr size_t s_Alignment = 32;
	};

	template<typename T, typename = void>
	struct LEBNITZ VectorizeType final {
		using type = void;
	};

	template<typename T>
	using VectorizedForm = VectorizeType<T>::type;

	template<typename T, typename = void>
	struct LEBNITZ VectorizeMaskType final {
		using type = void;
	};

	template<typename T>
	using  VectorizedMask = VectorizeMaskType<T>::type;


	struct Blank final {};

	template<typename T = Blank>
	struct LEBNITZ VectorizationIntrospect final {
		static constexpr size_t s_Alignment = 0;
		static constexpr size_t s_Lanes = 0;
		static constexpr size_t s_Width = 0;
		static constexpr const char* s_Name = "Invalid Type";
		static constexpr VectorizationBackend s_Backend = VectorizationBackend::UNKNOWN;
	};

	template<typename T = Blank>
	inline static constexpr LEBNITZ size_t IntrospectAlignment = VectorizationIntrospect<T>::s_Alignment;

	template<typename T = Blank>
	inline static constexpr LEBNITZ size_t IntrospectLanes = VectorizationIntrospect<T>::s_Lanes;

	template<typename T = Blank>
	inline static constexpr LEBNITZ size_t IntrospectWidth = VectorizationIntrospect<T>::s_Width;

	template<typename T = Blank>
	inline static constexpr LEBNITZ const char* IntrospectTypeName = VectorizationIntrospect<T>::s_Name;

	template<typename T = Blank>
	inline static constexpr LEBNITZ VectorizationBackend IntrospectBackend = VectorizationIntrospect<T>::s_Backend;

}
