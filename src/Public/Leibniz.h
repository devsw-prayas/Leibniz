#pragma once
#ifndef LEIBNIZ
#define LEIBNIZ __declspec(dllexport)
#endif
void LEIBNIZ Init();

#if defined(__INTELLISENSE__) || (defined(_MSC_VER) && !defined(__AVX2__))
#define LEIBNIZ_EDITOR_MODE 1
#endif

[[moreturn]] inline void unreachable() noexcept {
#if defined(_MSC_VER)
	__assume(false);
#endif
}
#include <type_traits>
#include <array>