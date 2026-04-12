#!/usr/bin/env python3
"""
gen_scalar_kernels.py
Reads LookUpTables.h, emits ElementwiseScalarKernels.h

Usage:
    python gen_scalar_kernels.py LookUpTables.h [output_dir]
"""

import sys
import re
import os

# ── Function ordering and LUT-name → Op-name mapping ─────────────────────────

FUNC_ORDER = [
    'Exp', 'Expm1', 'Exp2', 'Exp10',
    'Log1p', 'Log2', 'Log10',
    'Sin', 'Cos', 'Tan', 'Asin', 'Atan',
    'Sinh', 'Cosh', 'Asinh', 'Acosh', 'Atanh',
    'Erf', 'ErfcAsymptotic',
    'SqrtSeed', 'RsqrtSeed', 'CbrtSeed',
    'Lanczos',
]

LUT_NAME_TO_OP = {
    'Sine':   'Sin',
    'Cosine': 'Cos',
}

TIERS     = ['Balanced', 'Approximate']
TYPE_KEYS = ['f32', 'f64']

FUNC_DESCRIPTIONS = {
    'Exp':            'base-e exponential, Horner(x, c)',
    'Expm1':          'exp(x)-1, x * Horner(x, c)',
    'Exp2':           'base-2 exponential, Horner(x, c)',
    'Exp10':          'base-10 exponential, Horner(x, c)',
    'Log1p':          'log(1+x), Horner(x, c)',
    'Log2':           'base-2 logarithm, Horner(x, c)',
    'Log10':          'base-10 logarithm, Horner(x, c)',
    'Sin':            'sine, x * Horner(x^2, c)',
    'Cos':            'cosine, Horner(x^2, c)',
    'Tan':            'tangent, x * Horner(x^2, c)',
    'Asin':           'arcsine, x * Horner(x^2, c)',
    'Atan':           'arctangent, x * Horner(x^2, c)',
    'Sinh':           'hyperbolic sine, x * Horner(x^2, c)',
    'Cosh':           'hyperbolic cosine, Horner(x^2, c)',
    'Asinh':          'inverse hyperbolic sine, x * Horner(x^2, c)',
    'Acosh':          'acosh(1+u)=sqrt(2u)*Horner(u,c), u=x-1',
    'Atanh':          'inverse hyperbolic tangent, x * Horner(x^2, c)',
    'Erf':            'error function, x * Horner(x^2, c)',
    'ErfcAsymptotic': 'erfc asymptotic, (exp(-x^2)/(x*sqrt(pi)))*Horner(1/x^2,c)',
    'SqrtSeed':       'sqrt seed for Newton-Raphson',
    'RsqrtSeed':      'rsqrt seed for Newton-Raphson',
    'CbrtSeed':       'cbrt seed for Newton-Raphson',
    'Lanczos':        'Lanczos gamma, rational sum g=7 9-term',
}


# ── 1. Parse LookUpTables.h ───────────────────────────────────────────────────

def parse_lut(path):
    """
    Returns COEFFS[op_name][tier][type_key] = [coeff_str, ...]
    Skips Exact tier arrays entirely.
    Maps Sine→Sin, Cosine→Cos.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read()

    decl_re = re.compile(
        r'std::array<(?:float|double),\s*\d+>\s+'
        r's_([A-Za-z0-9]+)(Balanced|Approximate|Exact)Coefficients_(f32|f64)\s*=\s*\{([^}]*)\};',
        re.DOTALL
    )

    coeff_re = re.compile(
        r'-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+\-]\d+f?'
        r'|-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?f?'
    )

    COEFFS = {}

    for m in decl_re.finditer(text):
        lut_name = m.group(1)
        tier     = m.group(2)
        type_key = m.group(3)
        body     = m.group(4)

        if tier == 'Exact':
            continue

        op_name = LUT_NAME_TO_OP.get(lut_name, lut_name)

        if op_name not in FUNC_ORDER:
            continue

        coeffs = coeff_re.findall(body)
        if not coeffs:
            continue

        COEFFS.setdefault(op_name, {}).setdefault(tier, {})[type_key] = coeffs

    return COEFFS


# ── 2. Horner builder ─────────────────────────────────────────────────────────

def build_horner_scalar(coeffs, var):
    """
    coeffs : list of coeff strings, index 0 = constant term (lowest degree)
    var    : element variable string e.g. 'in[i]' or 'in[idx]'
    Returns a single C++ expression string.
    """
    if len(coeffs) == 1:
        return coeffs[0]
    expr = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        expr = f"({c} + {var} * {expr})"
    return expr


# ── 3. Function emitter ───────────────────────────────────────────────────────

def emit_function(op, tier, type_key, coeffs):
    """Returns the complete function string for one (op, tier, type_key) triple."""

    func_name = f"unrolledScalar{op}{tier}_{type_key}"

    if type_key == 'f32':
        ptr_type = 'Numerics::Float32'
    else:
        ptr_type = 'Numerics::Float64'

    horner_i   = build_horner_scalar(coeffs, 'in[i]')
    horner_idx = build_horner_scalar(coeffs, 'in[idx]')

    lines = []
    lines.append(f"// --- {op} {tier} {type_key} ---")
    lines.append(f"template<size_t N> requires (N <= 8)")
    lines.append(f"LEIBNIZ_FORCEINLINE void {func_name}(")
    lines.append(f"    const {ptr_type}* LEIBNIZ_RESTRICT in,")
    lines.append(f"    {ptr_type}*       LEIBNIZ_RESTRICT out")
    lines.append(f") {{")
    lines.append(f"    LEIBNIZ_ASSERT(in  != nullptr);")
    lines.append(f"    LEIBNIZ_ASSERT(out != nullptr);")
    lines.append(f"    LEIBNIZ_ASSUME_ALIGNED(in,  32);")
    lines.append(f"    LEIBNIZ_ASSUME_ALIGNED(out, 32);")
    lines.append(f"")
    lines.append(f"#if LEIBNIZ_COMPILER_MSVC")
    lines.append(f"    for (size_t i = 0; i < N; i++)")
    lines.append(f"        out[i] = {horner_i};")
    lines.append(f"#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC")
    lines.append(f"    Utils::Unrolled::staticFor<0, N>([=](auto idx) {{")
    lines.append(f"        out[idx] = {horner_idx};")
    lines.append(f"    }});")
    lines.append(f"#endif")
    lines.append(f"}}")

    return '\n'.join(lines)


# ── 4. Section emitter ────────────────────────────────────────────────────────

def emit_section(op, coeffs_map):
    """Returns section divider + all 4 functions for this op."""

    desc = FUNC_DESCRIPTIONS.get(op, op)

    def term_count(tier, tk):
        try:
            return len(coeffs_map[tier][tk])
        except KeyError:
            return '?'

    bal_f32 = term_count('Balanced',    'f32')
    bal_f64 = term_count('Balanced',    'f64')
    app_f32 = term_count('Approximate', 'f32')
    app_f64 = term_count('Approximate', 'f64')

    lines = []
    lines.append("// ============================================================")
    lines.append(f"// {op} — {desc}")
    lines.append(f"// Balanced:    {bal_f32} terms (f32)   {bal_f64} terms (f64)")
    lines.append(f"// Approximate: {app_f32} terms (f32)   {app_f64} terms (f64)")
    lines.append("// ============================================================")
    lines.append("")

    emit_order = [
        ('Balanced',    'f32'),
        ('Approximate', 'f32'),
        ('Balanced',    'f64'),
        ('Approximate', 'f64'),
    ]

    for tier, tk in emit_order:
        try:
            coeffs = coeffs_map[tier][tk]
        except KeyError:
            lines.append(f"// WARNING: {op} {tier} {tk} coefficients not found in LUT — skipped")
            lines.append("")
            continue
        lines.append(emit_function(op, tier, tk, coeffs))
        lines.append("")

    return '\n'.join(lines)


# ── 5. Header boilerplate ─────────────────────────────────────────────────────

HEADER_TOP = """\
// ============================================================
// ElementwiseScalarKernels.h
// AUTO-GENERATED by gen_scalar_kernels.py — DO NOT EDIT BY HAND
//
// Scalar unrolled transcendental kernels for Balanced and Approximate tiers.
// Used exclusively as tail handlers by ElementwiseVectorKernels.h.
// N is a compile-time constant — compiler fully unrolls staticFor.
//
// Naming: unrolledScalar{Op}{Tier}_{type}<N>
//   Op:   Exp, Expm1, Exp2, Exp10, Log1p, Log2, Log10,
//         Sin, Cos, Tan, Asin, Atan,
//         Sinh, Cosh, Asinh, Acosh, Atanh,
//         Erf, ErfcAsymptotic, SqrtSeed, RsqrtSeed, CbrtSeed, Lanczos
//   Tier: Balanced | Approximate
//   type: f32 | f64
//
// Usage:
//   unrolledScalarExpBalanced_f32<3>(in + offset, out + offset);
//   unrolledScalarExpApproximate_f64<5>(in + offset, out + offset);
//
// Section map:
//   Exp family ...................................... ~line 60
//   Log family ...................................... ~line 260
//   Trig family ..................................... ~line 460
//   Hyp family ...................................... ~line 700
//   Special family .................................. ~line 900
//   Root family ..................................... ~line 1060
// ============================================================

#pragma once
#include <bit>
#include "StaticPipelines.h"
#include "LeibnizCompiler.h"
#include "LeibnizInt.h"
#include "LeibnizFloat.h"
#include "LeibnizDiagnostics.h"

namespace Leibniz::LinAlg::Internal::Kernels::Scalar {

"""

HEADER_BOTTOM = """\
} // namespace Leibniz::LinAlg::Internal::Kernels::Scalar
"""


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_scalar_kernels.py LookUpTables.h [output_dir]", file=sys.stderr)
        sys.exit(1)

    lut_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else '.'

    print(f"[gen] Parsing {lut_path} ...", file=sys.stderr)
    COEFFS = parse_lut(lut_path)

    missing = [op for op in FUNC_ORDER if op not in COEFFS]
    if missing:
        print(f"[gen] WARNING: No coefficients found for: {', '.join(missing)}", file=sys.stderr)

    found = [op for op in FUNC_ORDER if op in COEFFS]
    print(f"[gen] Found coefficients for {len(found)}/{len(FUNC_ORDER)} functions: {', '.join(found)}",
          file=sys.stderr)

    out_parts = [HEADER_TOP]

    for op in FUNC_ORDER:
        if op not in COEFFS:
            out_parts.append(
                f"// ============================================================\n"
                f"// {op} — SKIPPED (no coefficients found in LUT)\n"
                f"// ============================================================\n\n"
            )
            continue
        out_parts.append(emit_section(op, COEFFS[op]))

    out_parts.append(HEADER_BOTTOM)

    out_path = os.path.join(out_dir, 'ElementwiseScalarKernels.h')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_parts))

    print(f"[gen] Written: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
