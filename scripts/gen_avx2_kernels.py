#!/usr/bin/env python3
"""
gen_avx2_kernels.py
Reads LookUpTables.h, emits ElementwiseVectorKernels.h

Usage:
    python gen_avx2_kernels.py LookUpTables.h [output_dir]
"""

import sys
import re
import os

# ── Function ordering and LUT-name → Op-name mapping ─────────────────────────

# Canonical emission order per spec
FUNC_ORDER = [
    'Exp', 'Expm1', 'Exp2', 'Exp10',
    'Log1p', 'Log2', 'Log10',
    'Sin', 'Cos', 'Tan', 'Asin', 'Atan',
    'Sinh', 'Cosh', 'Asinh', 'Acosh', 'Atanh',
    'Erf', 'ErfcAsymptotic',
    'SqrtSeed', 'RsqrtSeed', 'CbrtSeed',
    'Lanczos',
]

# LUT file may spell Sin/Cos as Sine/Cosine — map both directions
LUT_NAME_TO_OP = {
    'Sine':   'Sin',
    'Cosine': 'Cos',
}

# Reverse: op name → lut name (for lookup during parse)
OP_TO_LUT_NAME = {v: k for k, v in LUT_NAME_TO_OP.items()}

TIERS    = ['Balanced', 'Approximate']
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
    Maps Sine→Sin, Cosine→Cos in the output keys.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read()

    # Pattern: optional 'inline static '
    # std::array<float|double, N> s_{Name}{Tier}Coefficients_{f32|f64} = { ... };
    # The body may span multiple lines and each value is on its own line.
    decl_re = re.compile(
        r'std::array<(?:float|double),\s*\d+>\s+'
        r's_([A-Za-z0-9]+)(Balanced|Approximate|Exact)Coefficients_(f32|f64)\s*=\s*\{([^}]*)\};',
        re.DOTALL
    )

    # Individual coefficient: optional leading minus, hex-float or decimal, optional f suffix
    coeff_re = re.compile(r'-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+\-]\d+f?|-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?f?')

    COEFFS = {}

    for m in decl_re.finditer(text):
        lut_name = m.group(1)   # e.g. 'Exp', 'Sine', 'Cosine'
        tier     = m.group(2)   # 'Balanced' | 'Approximate' | 'Exact'
        type_key = m.group(3)   # 'f32' | 'f64'
        body     = m.group(4)

        # Skip Exact tier
        if tier == 'Exact':
            continue

        # Normalise LUT name → Op name
        op_name = LUT_NAME_TO_OP.get(lut_name, lut_name)

        # Only keep functions we care about
        if op_name not in FUNC_ORDER:
            continue

        coeffs = coeff_re.findall(body)
        if not coeffs:
            continue

        COEFFS.setdefault(op_name, {}).setdefault(tier, {})[type_key] = coeffs

    return COEFFS


# ── 2. Horner builder ─────────────────────────────────────────────────────────

def build_horner_avx2(coeffs, var, set1_fn, fmadd_fn):
    """
    coeffs   : list of coeff strings, index 0 = constant term (lowest degree)
    var      : AVX2 register variable name, e.g. 'x' or 'xd'
    set1_fn  : '_mm256_set1_ps' or '_mm256_set1_pd'
    fmadd_fn : '_mm256_fmadd_ps' or '_mm256_fmadd_pd'
    Returns a single C++ expression string (Horner from highest to lowest degree).
    """
    if len(coeffs) == 1:
        return f"{set1_fn}({coeffs[0]})"
    # Start from innermost (highest degree coefficient)
    expr = f"{set1_fn}({coeffs[-1]})"
    for c in reversed(coeffs[:-1]):
        expr = f"{fmadd_fn}({expr}, {var}, {set1_fn}({c}))"
    return expr


# ── 3. Struct emitter ─────────────────────────────────────────────────────────

def emit_struct(op, tier, type_key, coeffs):
    """Returns the complete struct string for one (op, tier, type_key) triple."""

    struct_name = f"Vectorize{op}{tier}_{type_key}"
    scalar_tail = (
        f"Leibniz::LinAlg::Internal::Kernels::Scalar::\n"
        f"                Unrolled{op}{tier}_{type_key}"
    )

    if type_key == 'f32':
        lane_count  = 8
        reg_type    = '__m256'
        load_fn     = '_mm256_load_ps'
        store_fn    = '_mm256_store_ps'
        set1_fn     = '_mm256_set1_ps'
        fmadd_fn    = '_mm256_fmadd_ps'
        var         = 'x'
        ptr_type    = 'float'
        lane_note   = 'AVX2 FP32 lane count (8)'
    else:
        lane_count  = 4
        reg_type    = '__m256d'
        load_fn     = '_mm256_load_pd'
        store_fn    = '_mm256_store_pd'
        set1_fn     = '_mm256_set1_pd'
        fmadd_fn    = '_mm256_fmadd_pd'
        var         = 'xd'
        ptr_type    = 'double'
        lane_note   = 'AVX2 FP64 lane count (4)'

    horner_expr = build_horner_avx2(coeffs, var, set1_fn, fmadd_fn)

    # Indent the horner expression to align inside the assignment
    horner_stmt = f"{var} = {horner_expr};"

    lines = []
    lines.append(f"// --- {struct_name} ---")
    lines.append(f"")
    lines.append(f"template<size_t Size, size_t UnrollWidth = 4>")
    lines.append(f"struct {struct_name} {{")
    lines.append(f"    static constexpr size_t lane_count = {lane_count};")
    lines.append(f"")
    lines.append(f'    static_assert(Size > 0, "Size must be greater than zero.");')
    lines.append(f'    static_assert(Size % lane_count == 0,')
    lines.append(f'        "Size must be a multiple of {lane_note}.");')
    lines.append(f'    static_assert(UnrollWidth > 0, "UnrollWidth must be greater than zero.");')
    lines.append(f'    static_assert((Size / lane_count) % UnrollWidth == 0,')
    lines.append(f'        "Size / lane_count must be divisible by UnrollWidth.");')
    lines.append(f"")
    lines.append(f"    static LEIBNIZ_FORCEINLINE void compute(")
    lines.append(f"        const {ptr_type}* LEIBNIZ_RESTRICT in,")
    lines.append(f"        {ptr_type}*       LEIBNIZ_RESTRICT out")
    lines.append(f"    ) {{")
    lines.append(f"        LEIBNIZ_ASSERT(in  != nullptr);")
    lines.append(f"        LEIBNIZ_ASSERT(out != nullptr);")
    lines.append(f"        LEIBNIZ_ASSUME_ALIGNED(in,  32);")
    lines.append(f"        LEIBNIZ_ASSUME_ALIGNED(out, 32);")
    lines.append(f"")
    lines.append(f"        constexpr size_t num_blocks = Size / (UnrollWidth * lane_count);")
    lines.append(f"        constexpr size_t remainder  = Size % (UnrollWidth * lane_count);")
    lines.append(f"")
    lines.append(f"        for (size_t block = 0; block < num_blocks; ++block) {{")
    lines.append(f"            const size_t base = block * UnrollWidth * lane_count;")
    lines.append(f"")
    lines.append(f"#if LEIBNIZ_COMPILER_MSVC")
    lines.append(f"            for (size_t i = 0; i < UnrollWidth; i++) {{")
    lines.append(f"                {reg_type} {var} = {load_fn}(in + base + i * lane_count);")
    lines.append(f"                {horner_stmt}")
    lines.append(f"                {store_fn}(out + base + i * lane_count, {var});")
    lines.append(f"            }}")
    lines.append(f"#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC")
    lines.append(f"            Utils::Unrolled::staticFor<0, UnrollWidth>([&](auto idx) {{")
    lines.append(f"                {reg_type} {var} = {load_fn}(in + base + idx * lane_count);")
    lines.append(f"                {horner_stmt}")
    lines.append(f"                {store_fn}(out + base + idx * lane_count, {var});")
    lines.append(f"            }});")
    lines.append(f"#endif")
    lines.append(f"        }}")
    lines.append(f"")
    lines.append(f"        if constexpr (remainder > 0) {{")
    lines.append(f"            constexpr size_t tail_offset = num_blocks * UnrollWidth * lane_count;")
    lines.append(f"            {scalar_tail}<remainder>::compute(in + tail_offset, out + tail_offset);")
    lines.append(f"        }}")
    lines.append(f"    }}")
    lines.append(f"}};")

    return '\n'.join(lines)


# ── 4. Section emitter ────────────────────────────────────────────────────────

def emit_section(op, coeffs_map):
    """
    Returns section divider + all 4 structs for this function.
    coeffs_map = COEFFS[op]
    """
    desc = FUNC_DESCRIPTIONS.get(op, op)

    # Gather term counts for the header comment
    def term_count(tier, tk):
        try:
            return len(coeffs_map[tier][tk])
        except KeyError:
            return '?'

    bal_f32  = term_count('Balanced',    'f32')
    bal_f64  = term_count('Balanced',    'f64')
    app_f32  = term_count('Approximate', 'f32')
    app_f64  = term_count('Approximate', 'f64')

    lines = []
    lines.append("// ============================================================")
    lines.append(f"// {op} — {desc}")
    lines.append(f"// Balanced:    {bal_f32} terms (f32)   {bal_f64} terms (f64)")
    lines.append(f"// Approximate: {app_f32} terms (f32)   {app_f64} terms (f64)")
    lines.append("// ============================================================")
    lines.append("")

    # Emit in order: Balanced_f32, Approximate_f32, Balanced_f64, Approximate_f64
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
        lines.append(emit_struct(op, tier, tk, coeffs))
        lines.append("")

    return '\n'.join(lines)


# ── 5. Header boilerplate ─────────────────────────────────────────────────────

HEADER_TOP = """\
// ============================================================
// ElementwiseVectorKernels.h
// AUTO-GENERATED by gen_avx2_kernels.py — DO NOT EDIT BY HAND
//
// AVX2 vectorized elementwise kernels for Leibniz.
// Arbitrary compile-time Size, tier-specialized, UnrollWidth templated.
//
// Tail remainder delegates to ElementwiseKernels.h scalar fallbacks.
//
// Naming: Vectorize{Op}{Tier}_{type}<Size, UnrollWidth>
//   Op:    Exp, Expm1, Exp2, Exp10, Log1p, Log2, Log10,
//          Sin, Cos, Tan, Asin, Atan,
//          Sinh, Cosh, Asinh, Acosh, Atanh,
//          Erf, ErfcAsymptotic, Lanczos,
//          SqrtSeed, RsqrtSeed, CbrtSeed
//   Tier:  Balanced | Approximate
//   type:  f32 | f64
//
// Usage:
//   VectorizeExpBalanced_f32<64>::compute(in, out);
//   VectorizeExpBalanced_f32<128, 8>::compute(in, out);
//
// Section map:
//   Exp family ...................................... ~line 60
//   Log family ...................................... ~line 280
//   Trig family ..................................... ~line 500
//   Hyp family ...................................... ~line 740
//   Special family .................................. ~line 980
//   Root family ..................................... ~line 1180
// ============================================================

#pragma once
#include <immintrin.h>
#include "LeibnizCompiler.h"
#include "LeibnizDiagnostics.h"
#include "ElementwiseKernels.h"

namespace Leibniz::LinAlg::Internal::Kernels {

"""

HEADER_BOTTOM = """\
} // namespace Leibniz::LinAlg::Internal::Kernels
"""


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_avx2_kernels.py LookUpTables.h [output_dir]", file=sys.stderr)
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

    out_path = os.path.join(out_dir, 'ElementwiseVectorKernels.h')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_parts))

    print(f"[gen] Written: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
