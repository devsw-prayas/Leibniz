"""
Leibniz HPC Math Library — Unified LUT Generator
Produces LookUpTables.h with all three tiers in a single pass:

  Exact       — Taylor/Maclaurin coefficients (full precision per format)
  Balanced    — Chebyshev minimax, ~N/2 terms
  Approximate — Chebyshev minimax, ~N/3 terms

Formats:
  f32   (ieeeSinglePrecision,            23-bit mantissa) — live float arrays
  f64   (ieeeDoublePrecision,            52-bit mantissa) — live double arrays
  f128  (ieeeQuadPrecision,             112-bit mantissa) — comment stubs
  f256  (leibnizOctaPrecision,          232-bit mantissa) — comment stubs
  f512  (leibnizHalfRefWidthPrecision,  479-bit mantissa) — comment stubs
  f1024 (leibnizFullRefWidthPrecision,  991-bit mantissa) — comment stubs

Usage:
    python3 lut_gen_unified.py                         # all formats, all series
    python3 lut_gen_unified.py --fmt f32 f64           # live formats only
    python3 lut_gen_unified.py --fmt f64 --series exp sin cos
    python3 lut_gen_unified.py --out MyLUT.h
"""

import mpmath
import math
import sys
import time
import argparse
from typing import Callable, List, Tuple

# ---------------------------------------------------------------------------
# Shared hex-float formatter (inline — no external dependency)
# ---------------------------------------------------------------------------

def _set_prec(mbits: int):
    mpmath.mp.prec = mbits + 64

def to_hex_float(v, mbits: int, suffix: str) -> str:
    """Format mpmath value as exact C hex float literal: 0x1.XXXp+N{suffix}."""
    if v == 0:
        return f"0x0p+0{suffix}"
    _set_prec(mbits)
    v = mpmath.mpf(v)
    sign = ""
    if v < 0:
        sign = "-"
        v = -v
    m, exp = mpmath.frexp(v)
    m   = m * 2
    exp = exp - 1
    frac_int = int(mpmath.nint((m - 1) * mpmath.mpf(2) ** mbits))
    shift = (-mbits) % 4
    frac_aligned = frac_int << shift
    hex_digits = (mbits + 3) // 4
    frac_hex = format(frac_aligned, f"0{hex_digits}x").rstrip("0") or "0"
    return f"{sign}0x1.{frac_hex}p{exp:+d}{suffix}"

# ---------------------------------------------------------------------------
# Format table
# ---------------------------------------------------------------------------

FORMATS = [
    # name    mbits  ebits  is_live  suffix  ctype       native_ctype
    ("f32",    23,    8,    True,    "f",   "float",    "float"),
    ("f64",    52,    11,   True,    "",    "double",   "double"),
    ("f128",   112,   15,   False,   "",    "double",   "Float128"),
    ("f256",   232,   23,   False,   "",    "double",   "Float256"),
    ("f512",   479,   32,   False,   "",    "double",   "Float512"),
    ("f1024",  991,   32,   False,   "",    "double",   "Float1024"),
]

FMT_BY_NAME = {f[0]: f for f in FORMATS}
FMT_ORDER   = [f[0] for f in FORMATS]

# ---------------------------------------------------------------------------
# Term budget calculation (exact tier sizes, computed dynamically)
# ---------------------------------------------------------------------------

def _terms_needed(mbits: int, x_bound: float, guard: int = 4) -> int:
    target = mpmath.mpf(2) ** (-(mbits + guard))
    term   = mpmath.mpf(1)
    x      = mpmath.mpf(x_bound)
    for n in range(1, 2000):
        term *= x / n
        if term < target:
            return n + 1
    return 200

def exact_budget(key: str, mbits: int) -> int:
    ln2h = math.log(2) / 2
    pi4  = math.pi / 4
    budgets = {
        "exp":    _terms_needed(mbits, ln2h),
        "expm1":  _terms_needed(mbits, ln2h),
        "exp2":   _terms_needed(mbits, ln2h),
        "exp10":  _terms_needed(mbits, ln2h),
        "log1p":  _terms_needed(mbits, 0.5),
        "log2":   _terms_needed(mbits, 0.5),
        "log10":  _terms_needed(mbits, 0.5),
        "sin":    _terms_needed(mbits, pi4),
        "cos":    _terms_needed(mbits, pi4),
        "tan":    _terms_needed(mbits, pi4),
        "asin":   _terms_needed(mbits, 0.5),
        "atan":   _terms_needed(mbits, 1.0),
        "sinh":   _terms_needed(mbits, ln2h),
        "cosh":   _terms_needed(mbits, ln2h),
        "asinh":  _terms_needed(mbits, math.log(2)),
        "acosh":  _terms_needed(mbits, 0.5),
        "atanh":  _terms_needed(mbits, 0.5),
        "erf":    _terms_needed(mbits, 1.0),
        "erfc":   min(10, max(4, mbits // 20)),
        "sqrt":   5,
        "rsqrt":  5,
        "cbrt":   5,
        "lanczos": 9,
    }
    return budgets[key]

def balanced_budget(key: str, n_exact: int) -> int:
    if key in ("sqrt", "rsqrt", "cbrt", "lanczos"):
        return n_exact
    if key == "erfc":
        return max(4, n_exact - 2)
    return max(4, n_exact // 2)

def approx_budget(key: str, n_exact: int) -> int:
    if key in ("sqrt", "rsqrt", "cbrt", "lanczos"):
        return max(3, n_exact - 1)
    if key == "erfc":
        return max(3, n_exact // 2)
    return max(3, n_exact // 3)

# ---------------------------------------------------------------------------
# Exact Taylor coefficient generators
# ---------------------------------------------------------------------------

def _prec(mbits):
    _set_prec(mbits)

def coeffs_exp(n, mbits):
    _prec(mbits)
    return [1 / mpmath.factorial(k) for k in range(n)]

def coeffs_expm1(n, mbits):
    _prec(mbits)
    return [1 / mpmath.factorial(k + 1) for k in range(n)]

def coeffs_exp2(n, mbits):
    _prec(mbits)
    ln2 = mpmath.log(2)
    return [ln2**k / mpmath.factorial(k) for k in range(n)]

def coeffs_exp10(n, mbits):
    _prec(mbits)
    ln10 = mpmath.log(10)
    return [ln10**k / mpmath.factorial(k) for k in range(n)]

def coeffs_log1p(n, mbits):
    _prec(mbits)
    return [mpmath.mpf((-1)**k) / (k + 1) for k in range(n)]

def coeffs_log2(n, mbits):
    _prec(mbits)
    ln2 = mpmath.log(2)
    return [mpmath.mpf((-1)**k) / ((k + 1) * ln2) for k in range(n)]

def coeffs_log10(n, mbits):
    _prec(mbits)
    ln10 = mpmath.log(10)
    return [mpmath.mpf((-1)**k) / ((k + 1) * ln10) for k in range(n)]

def coeffs_sin(n, mbits):
    _prec(mbits)
    return [mpmath.mpf((-1)**k) / mpmath.factorial(2*k + 1) for k in range(n)]

def coeffs_cos(n, mbits):
    _prec(mbits)
    return [mpmath.mpf((-1)**k) / mpmath.factorial(2*k) for k in range(n)]

def coeffs_tan(n, mbits):
    _prec(mbits)
    coeffs = []
    for k in range(n):
        if k == 0:
            coeffs.append(mpmath.mpf(1))
        else:
            B = mpmath.bernoulli(2*k)
            c = (mpmath.mpf(-1)**(k-1)) * mpmath.mpf(4)**k * (mpmath.mpf(4)**k - 1) * B / mpmath.factorial(2*k)
            coeffs.append(c)
    return coeffs

def coeffs_asin(n, mbits):
    _prec(mbits)
    coeffs = []
    for k in range(n):
        num = mpmath.factorial(2*k)
        den = mpmath.mpf(4)**k * mpmath.factorial(k)**2 * (2*k + 1)
        coeffs.append(num / den)
    return coeffs

def coeffs_atan(n, mbits):
    _prec(mbits)
    return [mpmath.mpf((-1)**k) / (2*k + 1) for k in range(n)]

def coeffs_sinh(n, mbits):
    _prec(mbits)
    return [1 / mpmath.factorial(2*k + 1) for k in range(n)]

def coeffs_cosh(n, mbits):
    _prec(mbits)
    return [1 / mpmath.factorial(2*k) for k in range(n)]

def coeffs_asinh(n, mbits):
    _prec(mbits)
    coeffs = []
    for k in range(n):
        num = mpmath.factorial(2*k)
        den = mpmath.mpf(4)**k * mpmath.factorial(k)**2 * (2*k + 1)
        coeffs.append(mpmath.mpf((-1)**k) * num / den)
    return coeffs

def coeffs_acosh(n, mbits):
    _prec(mbits)
    # acosh(1+u) = sqrt(2u) * Horner(u, c)
    # c[k] = (-1)^k * (2k)! / (4^k * (k!)^2 * (2k+1) * 2^k)
    coeffs = []
    for k in range(n):
        num = mpmath.factorial(2*k)
        den = mpmath.mpf(4)**k * mpmath.factorial(k)**2 * (2*k + 1) * mpmath.mpf(2)**k
        coeffs.append(mpmath.mpf((-1)**k) * num / den)
    return coeffs

def coeffs_atanh(n, mbits):
    _prec(mbits)
    return [mpmath.mpf(1) / (2*k + 1) for k in range(n)]

def coeffs_erf(n, mbits):
    _prec(mbits)
    two_over_sqrtpi = 2 / mpmath.sqrt(mpmath.pi)
    return [two_over_sqrtpi * mpmath.mpf((-1)**k) / (mpmath.factorial(k) * (2*k + 1))
            for k in range(n)]

def coeffs_erfc(n, mbits):
    _prec(mbits)
    def dfact(k):
        if k == 0: return mpmath.mpf(1)
        r = mpmath.mpf(1)
        for i in range(1, k + 1):
            r *= (2*i - 1)
        return r
    return [mpmath.mpf((-1)**k) * dfact(k) / mpmath.mpf(2)**k for k in range(n)]

def coeffs_sqrt(n, mbits):
    return [
        mpmath.mpf("0.17507985753745012766"),
        mpmath.mpf("1.26268988687865788626"),
        mpmath.mpf("-0.83699202458812122527"),
        mpmath.mpf("0.46802808918308036404"),
        mpmath.mpf("-0.07879957940947228718"),
    ]

def coeffs_rsqrt(n, mbits):
    return [
        mpmath.mpf("1.77627265330128064765"),
        mpmath.mpf("-1.40548406985049399516"),
        mpmath.mpf("0.97535974700047022183"),
        mpmath.mpf("-0.37208153342741210767"),
        mpmath.mpf("0.05629370826748617694"),
    ]

def coeffs_cbrt(n, mbits):
    return [
        mpmath.mpf("0.61202001372078029694"),
        mpmath.mpf("0.86389031032291174775"),
        mpmath.mpf("-0.56980263724990773765"),
        mpmath.mpf("0.27430536394745040498"),
        mpmath.mpf("-0.05895143985793026823"),
    ]

def coeffs_lanczos(n, mbits):
    return [
        mpmath.mpf("0.99999999999980993227"),
        mpmath.mpf("676.52036812188509856"),
        mpmath.mpf("-1259.13921672240287085"),
        mpmath.mpf("771.32342877765307879"),
        mpmath.mpf("-176.61502916214059157"),
        mpmath.mpf("12.50734327868690481"),
        mpmath.mpf("-0.13857109526572011689"),
        mpmath.mpf("0.0000098650755802898471"),
        mpmath.mpf("0.0000001505143016942768"),
    ]

# ---------------------------------------------------------------------------
# Chebyshev / Remez minimax fitter
# ---------------------------------------------------------------------------

def _make_odd(f: Callable) -> Callable:
    def g(t):
        if t == 0: return mpmath.mpf(1)
        sq = mpmath.sqrt(t)
        return f(sq) / sq
    return g

def _make_even(f: Callable) -> Callable:
    def g(t):
        return f(mpmath.sqrt(t))
    return g

def _chebyfit(f, lo, hi, n) -> Tuple[List, mpmath.mpf]:
    poly, err = mpmath.chebyfit(f, [lo, hi], n, error=True)
    return list(reversed(poly)), err

def _remez_fit(key: str, mbits: int, n_terms: int) -> Tuple[List[mpmath.mpf], mpmath.mpf]:
    """Fit series key for mbits precision with n_terms Chebyshev terms."""
    _set_prec(mbits)
    ln2h = mpmath.log(2) / 2
    pi4  = mpmath.pi / 4
    EPS  = mpmath.mpf("1e-20")

    if key == "erfc":
        # Divergent asymptotic — truncate exact series, don't Chebyshev-fit
        def dfact(k):
            if k == 0: return mpmath.mpf(1)
            r = mpmath.mpf(1)
            for i in range(1, k+1): r *= (2*i - 1)
            return r
        c = [mpmath.mpf((-1)**k) * dfact(k) / mpmath.mpf(2)**k for k in range(n_terms)]
        return c, abs(c[-1])

    if key in ("sqrt", "rsqrt", "cbrt", "lanczos"):
        # Seed polys: reuse exact values (Newton-Raphson handles precision)
        exact_fn = {
            "sqrt":    coeffs_sqrt,
            "rsqrt":   coeffs_rsqrt,
            "cbrt":    coeffs_cbrt,
            "lanczos": coeffs_lanczos,
        }[key]
        c = exact_fn(n_terms, mbits)[:n_terms]
        # Estimate error via chebyfit on the actual function
        domains = {
            "sqrt":    (mpmath.sqrt,              mpmath.mpf("0.5"), mpmath.mpf(2)),
            "rsqrt":   (lambda x: 1/mpmath.sqrt(x), mpmath.mpf("0.5"), mpmath.mpf(2)),
            "cbrt":    (mpmath.cbrt,              mpmath.mpf("0.5"), mpmath.mpf(1)),
            "lanczos": (None, None, None),
        }
        fn, lo, hi = domains[key]
        if fn is not None:
            _, err = _chebyfit(fn, lo, hi, n_terms)
        else:
            err = mpmath.mpf(0)
        return c, err

    # All other series
    straight_specs = {
        "exp":    (mpmath.exp,                                               -ln2h, ln2h),
        "expm1":  (lambda x: mpmath.expm1(x)/x if x != 0 else mpmath.mpf(1), -ln2h, ln2h),
        "exp2":   (lambda x: mpmath.power(2, x),                             -ln2h, ln2h),
        "exp10":  (lambda x: mpmath.power(10, x),                            -ln2h, ln2h),
        "log1p":  (mpmath.log1p,                                             -mpmath.mpf("0.5"), mpmath.mpf("0.5")),
        "log2":   (lambda x: mpmath.log1p(x)/mpmath.log(2),                 -mpmath.mpf("0.5"), mpmath.mpf("0.5")),
        "log10":  (lambda x: mpmath.log1p(x)/mpmath.log(10),                -mpmath.mpf("0.5"), mpmath.mpf("0.5")),
        # acosh: fit g(u)=acosh(1+u)/sqrt(u) straight on [eps, 0.5]
        "acosh":  (lambda u: mpmath.acosh(1+u)/mpmath.sqrt(u) if u > 0 else mpmath.sqrt(2),
                   EPS, mpmath.mpf("0.5")),
    }
    odd_specs = {
        "sin":   (mpmath.sin,   mpmath.mpf(0), pi4),
        "tan":   (mpmath.tan,   mpmath.mpf(0), pi4),
        "asin":  (mpmath.asin,  mpmath.mpf(0), mpmath.mpf("0.5")),
        "atan":  (mpmath.atan,  mpmath.mpf(0), mpmath.mpf(1)),
        "sinh":  (mpmath.sinh,  mpmath.mpf(0), ln2h),
        "asinh": (mpmath.asinh, mpmath.mpf(0), mpmath.log(2)),
        "atanh": (mpmath.atanh, mpmath.mpf(0), mpmath.mpf("0.5")),
        "erf":   (mpmath.erf,   mpmath.mpf(0), mpmath.mpf(1)),
    }
    even_specs = {
        "cos":   (mpmath.cos,  mpmath.mpf(0), pi4),
        "cosh":  (mpmath.cosh, mpmath.mpf(0), ln2h),
    }

    if key in straight_specs:
        fn, lo, hi = straight_specs[key]
        return _chebyfit(fn, lo, hi, n_terms)

    if key in odd_specs:
        fn, lo, hi = odd_specs[key]
        g    = _make_odd(fn)
        lo_t = EPS
        hi_t = hi ** 2
        return _chebyfit(g, lo_t, hi_t, n_terms)

    if key in even_specs:
        fn, lo, hi = even_specs[key]
        g    = _make_even(fn)
        lo_t = EPS
        hi_t = hi ** 2
        return _chebyfit(g, lo_t, hi_t, n_terms)

    raise ValueError(f"Unknown series key: {key}")

# ---------------------------------------------------------------------------
# Series metadata
# ---------------------------------------------------------------------------

SERIES = [
    # key        exact_fn         display         prefix            eval_note
    ("exp",      coeffs_exp,      "EXP",          "s_Exp",          "Horner(x, c)"),
    ("expm1",    coeffs_expm1,    "EXPM1",        "s_Expm1",        "x * Horner(x, c)"),
    ("exp2",     coeffs_exp2,     "EXP2",         "s_Exp2",         "Horner(x, c)"),
    ("exp10",    coeffs_exp10,    "EXP10",        "s_Exp10",        "Horner(x, c)"),
    ("log1p",    coeffs_log1p,    "LOG1P",        "s_Log1p",        "Horner(u, c), u=x-1"),
    ("log2",     coeffs_log2,     "LOG2",         "s_Log2",         "Horner(u, c), u=x-1"),
    ("log10",    coeffs_log10,    "LOG10",        "s_Log10",        "Horner(u, c), u=x-1"),
    ("sin",      coeffs_sin,      "SIN",          "s_Sine",         "x * Horner(x^2, c)"),
    ("cos",      coeffs_cos,      "COS",          "s_Cosine",       "Horner(x^2, c)"),
    ("tan",      coeffs_tan,      "TAN",          "s_Tan",          "x * Horner(x^2, c)"),
    ("asin",     coeffs_asin,     "ASIN",         "s_Asin",         "x * Horner(x^2, c)"),
    ("atan",     coeffs_atan,     "ATAN",         "s_Atan",         "x * Horner(x^2, c)"),
    ("sinh",     coeffs_sinh,     "SINH",         "s_Sinh",         "x * Horner(x^2, c)"),
    ("cosh",     coeffs_cosh,     "COSH",         "s_Cosh",         "Horner(x^2, c)"),
    ("asinh",    coeffs_asinh,    "ASINH",        "s_Asinh",        "x * Horner(x^2, c)"),
    ("acosh",    coeffs_acosh,    "ACOSH",        "s_Acosh",        "sqrt(2u) * Horner(u, c), u=x-1"),
    ("atanh",    coeffs_atanh,    "ATANH",        "s_Atanh",        "x * Horner(x^2, c)"),
    ("erf",      coeffs_erf,      "ERF",          "s_Erf",          "x * Horner(x^2, c)"),
    ("erfc",     coeffs_erfc,     "ERFC_ASYMPTOTIC","s_ErfcAsymptotic","(exp(-x^2)/(x*sqrt(pi))) * Horner(1/x^2, c)"),
    ("sqrt",     coeffs_sqrt,     "SQRT_SEED",    "s_SqrtSeed",     "seed + Newton-Raphson"),
    ("rsqrt",    coeffs_rsqrt,    "RSQRT_SEED",   "s_RsqrtSeed",    "seed + Newton-Raphson"),
    ("cbrt",     coeffs_cbrt,     "CBRT_SEED",    "s_CbrtSeed",     "seed + Newton-Raphson"),
    ("lanczos",  coeffs_lanczos,  "LANCZOS",      "s_Lanczos",      "rational sum, g=7, 9-term"),
]

# Functions intentionally without tables (computed from primitives at eval time)
NO_TABLE_NOTES = [
    ("ACOS",      "acos(x) = pi/2 - asin(x)  — reuses s_Asin*"),
    ("ATAN2",     "atan2(y,x) uses atan with quadrant correction — reuses s_Atan*"),
    ("TANH",      "tanh(x) = sinh(x)/cosh(x) — reuses s_Sinh* and s_Cosh*"),
    ("POW",       "pow(x,y) = exp(y*log(x))  — reuses s_Exp* and s_Log1p*"),
    ("HYPOT",     "hypot(a,b) = sqrt(a^2+b^2) — reuses s_SqrtSeed*"),
    ("SIGMOID",   "sigmoid(x) = 1/(1+exp(-x)) — reuses s_Exp*"),
    ("SOFTPLUS",  "softplus(x) = log(1+exp(x)) — reuses s_Exp* and s_Log1p*"),
    ("LGAMMA",    "lgamma/tgamma share s_Lanczos* coefficients"),
]

ALL_KEYS = [s[0] for s in SERIES]

# ---------------------------------------------------------------------------
# C++ array emitter
# ---------------------------------------------------------------------------

def _sig_digits(mbits: int) -> int:
    return int(math.ceil(mbits * math.log10(2))) + 6

def emit_array(
    lines: List[str],
    arr_name: str,
    ctype: str,
    coeffs: List,
    suffix: str,
    mbits: int,
    is_live: bool,
    tier_comment: str,
    fmt_name: str,
):
    n = len(coeffs)
    if is_live:
        lines.append(f"\t// {tier_comment}")
        lines.append(f"\tinline static std::array<{ctype}, {n}> {arr_name} = {{")
        for i, c in enumerate(coeffs):
            comma = "," if i < n - 1 else ""
            lines.append(f"\t\t{to_hex_float(c, mbits, suffix)}{comma}")
        lines.append(f"\t}};")
        lines.append("")
    else:
        lines.append(f"\t// {tier_comment}")
        lines.append(f"\t// Full-precision hex values ({_sig_digits(mbits)} significant digits):")
        lines.append(f"\t// clang-format off")
        for i, c in enumerate(coeffs):
            lines.append(f"\t//   [{i:3d}] = {to_hex_float(c, mbits, '')}")
        lines.append(f"\t// clang-format on")
        lines.append(f"\t// TODO: replace zeros once {fmt_name} type is wired up")
        lines.append(f"\tinline static std::array<double, {n}> {arr_name} = {{")
        for i in range(n):
            comma = "," if i < n - 1 else ""
            lines.append(f"\t\t/* TODO */ 0.0{comma}")
        lines.append(f"\t}};")
        lines.append("")

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(target_fmts: List[str], target_keys: List[str]) -> str:
    lines = []
    def L(s=""): lines.append(s)

    L("// AUTO-GENERATED by lut_gen_unified.py — DO NOT EDIT BY HAND")
    L("// Leibniz HPC Math Library — LUT Coefficient Tables (all three tiers)")
    L("//")
    L("// Tiers per series per format:")
    L("//   Exact       — Taylor/Maclaurin at full format precision")
    L("//   Balanced    — Chebyshev minimax, ~N/2 terms")
    L("//   Approximate — Chebyshev minimax, ~N/3 terms")
    L("//")
    L("// Live formats  (f32, f64):      float/double hex-float literals")
    L("// Stub formats  (f128 and wider): zeroed arrays, values in block comments")
    L("//")
    L("// Evaluation conventions:")
    L("//   Straight  (exp, log*, expm1, exp2, exp10, acosh): Horner(x, c)")
    L("//   Odd       (sin, tan, asin, atan, sinh, asinh, atanh, erf): x * Horner(x^2, c)")
    L("//   Even      (cos, cosh):                             Horner(x^2, c)")
    L("//   Seeds     (sqrt, rsqrt, cbrt):                     seed + Newton-Raphson")
    L("//   Lanczos   (lgamma/tgamma):                         rational sum g=7, 9-term")
    L("//   acosh     : acosh(1+u) = sqrt(2u) * Horner(u, c), u = x - 1")
    L("//   erfc      : asymptotic only — (exp(-x^2)/(x*sqrt(pi))) * Horner(1/x^2, c)")
    L("//")
    L("// Functions without tables (computed from primitives):")
    for name, note in NO_TABLE_NOTES:
        L(f"//   {name:<12} {note}")
    L()
    L("#pragma once")
    L('#include "Leibniz.h"')
    L()
    L("namespace Leibniz::Numerics::Lookup {")
    L()

    total = len(target_fmts) * len(target_keys) * 2  # balanced + approx fits
    fit_n = 0
    t0_total = time.time()

    for fmt_name in target_fmts:
        fmt_name, mbits, ebits, is_live, suffix, ctype, native_ctype = FMT_BY_NAME[fmt_name]

        L(f"\t// {'='*70}")
        L(f"\t// FORMAT: {fmt_name}  ({mbits}-bit mantissa)")
        if is_live:
            L(f"\t// Status: LIVE — {ctype} hex-float literals")
        else:
            L(f"\t// Status: STUB — hex values in comments, zeros in arrays")
        L(f"\t// {'='*70}")
        L()

        target_series = [(k, fn, dn, px, en) for k, fn, dn, px, en in SERIES if k in target_keys]

        for key, exact_fn, display, prefix, eval_note in target_series:
            n_exact = exact_budget(key, mbits)
            n_bal   = balanced_budget(key, n_exact)
            n_app   = approx_budget(key, n_exact)

            L(f"\t// {'-'*68}")
            L(f"\t// {display} ({fmt_name})  —  {eval_note}")
            L(f"\t// Exact: {n_exact} terms  |  Balanced: {n_bal} terms  |  Approximate: {n_app} terms")
            L(f"\t// {'-'*68}")
            L()

            # --- Exact tier ---
            exact_coeffs = exact_fn(n_exact, mbits)
            emit_array(lines, f"{prefix}ExactCoefficients_{fmt_name}",
                       ctype, exact_coeffs, suffix, mbits, is_live,
                       f"Exact — {n_exact}-term Taylor/Maclaurin", fmt_name)

            # --- Balanced tier ---
            fit_n += 1
            print(f"  [{fit_n:3d}/{total}] {fmt_name:6s} {key:8s} balanced  ({n_bal:3d} terms) ... ",
                  end="", flush=True)
            t0 = time.time()
            try:
                bal_coeffs, bal_err = _remez_fit(key, mbits, n_bal)
                print(f"err={mpmath.nstr(bal_err, 4)}  ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                bal_coeffs = [mpmath.mpf(0)] * n_bal
                bal_err    = mpmath.mpf(-1)

            emit_array(lines, f"{prefix}BalancedCoefficients_{fmt_name}",
                       ctype, bal_coeffs, suffix, mbits, is_live,
                       f"Balanced — {n_bal}-term Chebyshev minimax  err={mpmath.nstr(bal_err, 4)}",
                       fmt_name)

            # --- Approximate tier ---
            fit_n += 1
            print(f"  [{fit_n:3d}/{total}] {fmt_name:6s} {key:8s} approx    ({n_app:3d} terms) ... ",
                  end="", flush=True)
            t0 = time.time()
            try:
                app_coeffs, app_err = _remez_fit(key, mbits, n_app)
                print(f"err={mpmath.nstr(app_err, 4)}  ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                app_coeffs = [mpmath.mpf(0)] * n_app
                app_err    = mpmath.mpf(-1)

            emit_array(lines, f"{prefix}ApproximateCoefficients_{fmt_name}",
                       ctype, app_coeffs, suffix, mbits, is_live,
                       f"Approximate — {n_app}-term Chebyshev minimax  err={mpmath.nstr(app_err, 4)}",
                       fmt_name)

        L()  # blank line between formats

    total_time = time.time() - t0_total
    print(f"\nDone in {total_time:.1f}s — {len(lines)} lines generated")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leibniz unified LUT generator")
    parser.add_argument("--fmt",    nargs="*", default=FMT_ORDER,
                        help=f"Formats (default: all). Choices: {FMT_ORDER}")
    parser.add_argument("--series", nargs="*", default=ALL_KEYS,
                        help=f"Series (default: all). Choices: {ALL_KEYS}")
    parser.add_argument("--out",    default="LookUpTables.h",
                        help="Output file (default: LookUpTables.h)")
    args = parser.parse_args()

    bad_fmts   = [f for f in args.fmt    if f not in FMT_BY_NAME]
    bad_series = [s for s in args.series if s not in ALL_KEYS]
    if bad_fmts:
        print(f"Unknown formats: {bad_fmts}. Valid: {FMT_ORDER}")
        sys.exit(1)
    if bad_series:
        print(f"Unknown series: {bad_series}. Valid: {ALL_KEYS}")
        sys.exit(1)

    print(f"Leibniz LUT Generator")
    print(f"  Formats : {args.fmt}")
    print(f"  Series  : {args.series}")
    print(f"  Output  : {args.out}")
    print()

    output = generate(args.fmt, args.series)

    with open(args.out, "w") as f:
        f.write(output)
    print(f"Written to {args.out}")
