#pragma once
#include <LeibnizCompiler.h>
#if defined(LEIBNIZ_SHARED)

#if LEIBNIZ_COMPILER_MSVC
#if defined(LEIBNIZ_BUILDING_RUNTIME)					  
#define LEIBNIZ_RUNTIME_API __declspec(dllexport)
#else
#define LEIBNIZ_RUNTIME_API __declspec(dllimport)
#endif
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
#define LEIBNIZ_RUNTIME_API __attribute__((visibility("default")))
#else
#define LEIBNIZ_RUNTIME_API
#endif

#else
// Static build -> no import/export
#define LEIBNIZ_RUNTIME_API
#endif

#include <type_traits>
#include <array>