#!/usr/bin/env python3
"""
gen_unrolled_kernels.py
Generates UnrolledScalarKernels.h for Leibniz::LinAlg::Internal::Kernels::Scalar

Covers all elementwise ops from the canonical tier list.
Tier 0-3: direct operator expression
Tier 4+:  inlined Horner chain using LUT Balanced coefficients

Fixes applied vs original:
  - ISSUE 1/5: ~T(0) replaced with Leibniz::Mask::trueVal/falseVal<T>() for all comparison ops
  - ISSUE 2:   ternary_kind field; fma/clamp use "fused" (inA/inB/inC), select/where use "select" (mask/inA/inB)
  - ISSUE 3:   raw _fpclass/_finite replaced with Leibniz::FloatIntrin::* calls
  - ISSUE 4:   staticFor index access standardized via IDX_VAL constant (idx.value)
  - ISSUE 5:   #include <bit> added to HEADER_TOP
  - NEW OPS:   log10, exp10, tanh, atan2, pow, acos added
  - COMMENTS:  full tier banners, per-op section comments, per-function doc comments
"""

# ============================================================
# staticFor index form
# Utils::Unrolled::staticFor passes std::integral_constant<size_t,I>
# Array subscript works with bare idx (implicit conversion), but
# expressions that need a plain size_t must use idx.value.
# We use IDX_VAL everywhere to be explicit and consistent.
# ============================================================
IDX_VAL = "idx.value"

# ============================================================
# COEFFICIENT TABLES  (Balanced tier, f32 and f64)
# Horner order: c[0] + x*(c[1] + x*(c[2] + ...))
# ============================================================

COEFFS = {
    # --- EXP ---
    "exp": {
        "f32": ["0x1.fff61cp-1f", "0x1.fffe06p-1f", "0x1.029248p-1f", "0x1.57636ep-3f"],
        "f64": ["0x1.0p+0", "0x1.000000a1fd6a7p+0", "0x1.00000028794dfp-1",
                "0x1.5554043e289bcp-3", "0x1.5554ace120b86p-5",
                "0x1.126fa6fd6e815p-7", "0x1.6d7531eae5468p-10"],
    },
    # --- EXPM1 ---
    "expm1": {
        "f32": ["0x1.fffe06p-1f", "0x1.ffff58p-2f", "0x1.57636ep-3f", "0x1.56b3d8p-5f"],
        "f64": ["0x1.0p+0", "0x1.00000028794dfp-1", "0x1.5555556750672p-3",
                "0x1.5554ace120b86p-5", "0x1.1110c63a4eedp-7",
                "0x1.6d7531eae5468p-10", "0x1.a15169e096556p-13"],
    },
    # --- EXP2 ---
    "exp2": {
        "f32": ["0x1.fffdb8p-1f", "0x1.62e3ep-1f", "0x1.ee5a96p-3f", "0x1.c800d2p-5f"],
        "f64": ["0x1.0p+0", "0x1.62e4300881c7cp-1", "0x1.ebfbe000ca124p-3",
                "0x1.c6b025ee4a366p-5", "0x1.3b2a931e876e3p-7",
                "0x1.5e5f5e0421cbp-10", "0x1.439e545c3ea41p-13"],
    },
    # --- EXP10 ---
    "exp10": {
        "f32": ["0x1.fee536p-1f", "0x1.269acp+1f", "0x1.65aaacp+1f", "0x1.0cd7acp+1f"],
        "f64": ["0x1.0p+0", "0x1.26bb89ad792a1p+1", "0x1.53526bffeca71p+1",
                "0x1.0453c65b5f11ep+1", "0x1.2bc6f9fb54aa7p+0",
                "0x1.1b92cf22ad3dcp-1", "0x1.b070655a64ecep-3"],
    },
    # --- LOG ---
    "log": {
        "f32": ["-0x1.5ap-109f", "0x1.fd7234p-1f", "-0x1.fbb0fp-2f",
                "0x1.a35456p-2f", "-0x1.41a74ap-2f"],
        "f64": ["0x1.8d7991535d116p-18", "0x1.ffff4d94c6107p-1", "-0x1.0062bc78e07f5p-1",
                "0x1.560698110ba1bp-2", "-0x1.e23434a7ebd19p-3",
                "0x1.7edf5b3ff3342p-3", "-0x1.01ea76bbc4ecap-2", "0x1.c0cfad98c53b4p-3"],
    },
    # --- LOG2 ---
    "log2": {
        "f32": ["-0x1.0p-108f", "0x1.6f7cd8p+0f", "-0x1.6e38c6p-1f",
                "0x1.2e7b74p-1f", "-0x1.d00c42p-2f"],
        "f64": ["0x1.1eb7af09a833ap-17", "0x1.7153f59f021bbp+0", "-0x1.71e2e8906e383p-1",
                "0x1.ed70597005a53p-2", "-0x1.5bd62bf398f84p-2",
                "0x1.142f2263e4ce6p-2", "-0x1.74180d4905dbp-2", "0x1.43bfb66774ef3p-2"],
    },
    # --- LOG10 ---
    "log10": {
        "f32": ["-0x1.76p-110f", "0x1.ba7fdp-2f", "-0x1.b8f996p-3f",
                "0x1.6c3994p-3f", "-0x1.17627ap-3f"],
        "f64": ["0x1.593e06a240da5p-19", "0x1.bcb716596ef52p-2", "-0x1.bd63370d3230bp-3",
                "0x1.29146dae7a7fap-3", "-0x1.a2d646531efa7p-4",
                "0x1.4c8f0e906d925p-4", "-0x1.c00bb72eb21a3p-4", "0x1.85d51e35149d8p-4"],
    },
    # --- LOG1P ---
    "log1p": {
        "f32": ["-0x1.5ap-109f", "0x1.fd7234p-1f", "-0x1.fbb0fp-2f",
                "0x1.a35456p-2f", "-0x1.41a74ap-2f"],
        "f64": ["0x1.8d7991535d116p-18", "0x1.ffff4d94c6107p-1", "-0x1.0062bc78e07f5p-1",
                "0x1.560698110ba1bp-2", "-0x1.e23434a7ebd19p-3",
                "0x1.7edf5b3ff3342p-3", "-0x1.01ea76bbc4ecap-2", "0x1.c0cfad98c53b4p-3"],
    },
    # --- SIN ---
    "sin": {
        "f32": ["0x1.1p-1f", "-0x1.555556p-3f", "0x1.111112p-7f",
                "-0x1.a019f8p-13f", "0x1.71d732p-19f", "-0x1.a9508p-26f"],
        "f64": ["0x1.1p-1", "-0x1.5555555555555p-3", "0x1.1111111111111p-7",
                "-0x1.a01a01a01a014p-13", "0x1.71de3a556bb7ep-19",
                "-0x1.ae64567d9d7a9p-26", "0x1.61245edeb306ap-33",
                "-0x1.ae7b75b7f3ea8p-41", "0x1.91e5d3ff5106bp-49"],
    },
    # --- COS ---
    "cos": {
        "f32": ["0x1.1p-1f", "-0x1.0p-1f", "0x1.555556p-5f",
                "-0x1.6c16c2p-10f", "0x1.a01a02p-16f", "-0x1.27e4fcp-22f"],
        "f64": ["0x1.1p-1", "-0x1.1p-2", "0x1.5555555555555p-5",
                "-0x1.6c16c16c16c09p-10", "0x1.a01a01a01844fp-16",
                "-0x1.27e4fb7581302p-22", "0x1.1eed8c32f1021p-29",
                "-0x1.9392cccc6be36p-37", "0x1.aa9bc439ae3a9p-45"],
    },
    # --- TAN ---
    "tan": {
        "f32": ["0x1.0p+0f", "0x1.5555aep-2f", "0x1.1121b2p-3f",
                "0x1.bb64dep-5f", "0x1.6da21ep-6f", "0x1.b9f73cp-7f"],
        "f64": ["0x1.000000005a5b8p+0", "0x1.555553e31f706p-2", "0x1.11118d1f228a7p-3",
                "0x1.b9fbd9de55dfcp-5", "0x1.68548d37e29e3p-6",
                "0x1.10332abe2f058p-7", "0x1.49f42392f1bf8p-8",
                "-0x1.4c5d8d7f2d263p-11", "0x1.0d67c14888fd1p-9"],
    },
    # --- ASIN ---
    "asin": {
        "f32": ["0x1.0p+0f", "0x1.5555eep-3f", "0x1.337c4ep-4f",
                "0x1.6e1bbep-5f", "0x1.b7de58p-6f"],
        "f64": ["0x1.ffffffffe1fddp-1", "0x1.55555644ffd27p-3", "0x1.3332974333413p-4",
                "0x1.6ddd3297735ebp-5", "0x1.ed3310aa58a51p-6",
                "0x1.93e1363c82c76p-6", "0x1.e5084c0c9e239p-8", "0x1.1ca8ba846c6cbp-5"],
    },
    # --- ATAN ---
    "atan": {
        "f32": ["0x1.0p+0f", "-0x1.5553dep-2f", "0x1.99a0b2p-3f",
                "-0x1.2317e2p-3f", "0x1.c0de44p-4f", "-0x1.5fd07ep-4f"],
        "f64": ["0x1.ffffffe79bf93p-1", "-0x1.55552f0db432p-2", "0x1.998f82493d066p-3",
                "-0x1.240b8d994abf9p-3", "0x1.bfc9e8c2b0243p-4",
                "-0x1.5601364b1cac3p-4", "0x1.d3ad79f53e142p-5",
                "-0x1.f148f2a829af1p-6", "0x1.57a8c747ed314p-7", "-0x1.bdf39729124b6p-10"],
    },
    # --- SINH ---
    "sinh": {
        "f32": ["0x1.0p+0f", "0x1.5555d6p-3f", "0x1.11110cp-7f", "0x1.a02de8p-13f"],
        "f64": ["0x1.0p+0", "0x1.5555555555555p-3", "0x1.1111111111113p-7",
                "0x1.a01a01a0185b8p-13", "0x1.71de3a5f3a85fp-19",
                "0x1.ae641aa4b3bc4p-26", "0x1.61d987751d063p-33"],
    },
    # --- COSH ---
    "cosh": {
        "f32": ["0x1.0p+0f", "0x1.0p-1f", "0x1.5555dep-5f", "0x1.6c1636p-10f"],
        "f64": ["0x1.0p+0", "0x1.1p-2", "0x1.555555555555dp-5",
                "0x1.6c16c16c13a9dp-10", "0x1.a01a01b27d442p-16",
                "0x1.27e4c358312d7p-22", "0x1.1f97684759d2cp-29"],
    },
    # --- ASINH ---
    "asinh": {
        "f32": ["0x1.0p+0f", "-0x1.55556ap-3f", "0x1.33307ep-4f",
                "-0x1.6d8ca4p-5f", "0x1.f0c458p-6f"],
        "f64": ["0x1.ffffffffd31a3p-1", "-0x1.5555546833a3bp-3", "0x1.3332cb1306bap-4",
                "-0x1.6da50cdef9bd8p-5", "0x1.f03a9698a9a1ap-6",
                "-0x1.64a1d4e884e18p-6", "0x1.eb17f1de08ap-7",
                "-0x1.0afc2605b475ap-7", "0x1.38db71d294bacp-9"],
    },
    # --- ACOSH ---
    "acosh": {
        "f32": ["0x1.6a09e6p+0f", "-0x1.e2b74ep-4f", "0x1.b26b72p-6f",
                "-0x1.025c36p-7f", "0x1.5f89fap-9f"],
        "f64": ["0x1.6a09e667efe68p+0", "-0x1.e2b7dda271da1p-4", "0x1.b2721effa4bf4p-6",
                "-0x1.0294304858003p-7", "0x1.5f548154f92bfp-9",
                "-0x1.fabbd22ca2814p-11", "0x1.5677d4f887a75p-12", "-0x1.2dd8e606c72cbp-14"],
    },
    # --- ATANH ---
    "atanh": {
        "f32": ["0x1.0p+0f", "0x1.55558ep-2f", "0x1.99888ap-3f",
                "0x1.241b84p-3f", "0x1.c5d50ap-4f"],
        "f64": ["0x1.ffffffff5e368p-1", "0x1.555557db56886p-2", "0x1.9997f57975c75p-3",
                "0x1.24c5e33e88f82p-3", "0x1.c0f55975c35e6p-4",
                "0x1.a66a48ce8cef7p-4", "0x1.859eb5c8eb941p-6", "0x1.69d411adf045dp-3"],
    },
    # --- ERF ---
    "erf": {
        "f32": ["0x1.20dd76p+0f", "-0x1.812748p-2f", "0x1.ce2f3p-4f",
                "-0x1.b82e16p-6f", "0x1.565bc6p-8f", "-0x1.c02dbcp-11f"],
        "f64": ["0x1.20dd750429b19p+0", "-0x1.812746b0271ecp-2", "0x1.ce2f2197bc2efp-4",
                "-0x1.b82ce15c3b7f8p-6", "0x1.565ba0406c001p-8",
                "-0x1.c0286cad68e02p-11", "0x1.f941bed7e7847p-14",
                "-0x1.f0705a1dde9aap-17", "0x1.9ae0986f9de4fp-20", "-0x1.c1a515de8637ap-24"],
    },
    # --- ERFC (asymptotic series) ---
    "erfc": {
        "f32": ["0x1.0p+0f", "-0x1.0p-1f", "0x1.8p-1f", "-0x1.ep+0f"],
        "f64": ["0x1.0p+0", "-0x1.0p-1", "0x1.8p-1", "-0x1.ep+0"],
    },
    # --- SQRT (seed polynomial, then Newton-Raphson) ---
    "sqrt": {
        "f32": ["0x1.669046p-3f", "0x1.433fa4p+0f", "-0x1.ac8a38p-1f",
                "0x1.df42c2p-2f", "-0x1.42c35ap-4f"],
        "f64": ["0x1.669044b27e55fp-3", "0x1.433fa4f9223f1p+0", "-0x1.ac8a37f93cd48p-1",
                "0x1.df42c1629a518p-2", "-0x1.42c35908092b3p-4"],
    },
    # --- RSQRT (seed polynomial, then Newton-Raphson) ---
    "rsqrt": {
        "f32": ["0x1.c6b9cep+0f", "-0x1.67cdcep+0f", "0x1.f36258p-1f",
                "-0x1.7d02f4p-2f", "0x1.cd2878p-5f"],
        "f64": ["0x1.c6b9cdfab5492p+0", "-0x1.67cdcdd30e8ddp+0", "0x1.f3625a4e67596p-1",
                "-0x1.7d02f10610a8p-2", "0x1.cd28767f57645p-5"],
    },
    # --- CBRT (seed polynomial, then Newton-Raphson) ---
    "cbrt": {
        "f32": ["0x1.395abp-1f", "0x1.ba4fd4p-1f", "-0x1.23bd2cp-1f",
                "0x1.18e382p-2f", "-0x1.e2ee22p-5f"],
        "f64": ["0x1.395aafeedb4p-1", "0x1.ba4fd4ac561dfp-1", "-0x1.23bd2bd853685p-1",
                "0x1.18e3815d16328p-2", "-0x1.e2ee2147bdd1fp-5"],
    },
    # --- SIGMOID (1/(1+exp(-x)), derived via exp) ---
    "sigmoid": {
        "f32": ["0x1.fff61cp-1f", "0x1.fffe06p-1f", "0x1.029248p-1f", "0x1.57636ep-3f"],
        "f64": ["0x1.0p+0", "0x1.000000a1fd6a7p+0", "0x1.00000028794dfp-1",
                "0x1.5554043e289bcp-3", "0x1.5554ace120b86p-5",
                "0x1.126fa6fd6e815p-7", "0x1.6d7531eae5468p-10"],
    },
}

# pi constants for atan2
PI_F32   = "0x1.921fb6p+1f"
PI_2_F32 = "0x1.921fb6p+0f"
PI_F64   = "0x1.921fb54442d18p+1"
PI_2_F64 = "0x1.921fb54442d18p+0"

# ============================================================
# OP DEFINITIONS
# ============================================================

# ternary_kind:
#   "select" -> params are (mask, inA, inB)  — mask is boolean selector
#   "fused"  -> params are (inA, inB, inC)   — all three are data inputs

OPS = [
    # ---- TIER 0: CORE ARITHMETIC ----
    {"name": "Add",    "fn": "add", "arity": "binary", "types": "all6", "expr": "inA[{i}] + inB[{i}]"},
    {"name": "Sub",    "fn": "sub", "arity": "binary", "types": "all6", "expr": "inA[{i}] - inB[{i}]"},
    {"name": "Mul",    "fn": "mul", "arity": "binary", "types": "all6", "expr": "inA[{i}] * inB[{i}]"},
    {"name": "Divide", "fn": "div", "arity": "binary", "types": "all6", "expr": "inA[{i}] / inB[{i}]"},
    {"name": "Negate", "fn": "neg", "arity": "unary",  "types": "signed", "expr": "-inA[{i}]"},

    # ---- TIER 0: COMPARISON ----
    # Integer eq/neq: exact bitwise
    {"name": "CmpEqual",    "fn": "eq",  "arity": "binary", "types": "int4",
     "expr": "(inA[{i}] == inB[{i}]) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "CmpNotEqual", "fn": "neq", "arity": "binary", "types": "int4",
     "expr": "(inA[{i}] != inB[{i}]) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    # Float eq/neq: machine epsilon comparison
    {"name": "CmpEqual",    "fn": "eq",  "arity": "binary", "types": "fp",
     "expr": "((inA[{i}] > inB[{i}] ? inA[{i}] - inB[{i}] : inB[{i}] - inA[{i}]) <= Leibniz::Numerics::Limits::NumericLimits<{T}>::epsilon()) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "CmpNotEqual", "fn": "neq", "arity": "binary", "types": "fp",
     "expr": "((inA[{i}] > inB[{i}] ? inA[{i}] - inB[{i}] : inB[{i}] - inA[{i}]) > Leibniz::Numerics::Limits::NumericLimits<{T}>::epsilon()) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    # Ordered comparisons: all types, bitmask output
    {"name": "CmpLessThan",           "fn": "lt", "arity": "binary", "types": "all6",
     "expr": "(inA[{i}] < inB[{i}])  ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "CmpLessThanOrEqual",    "fn": "le", "arity": "binary", "types": "all6",
     "expr": "(inA[{i}] <= inB[{i}]) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "CmpGreater",            "fn": "gt", "arity": "binary", "types": "all6",
     "expr": "(inA[{i}] > inB[{i}])  ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "CmpGreaterThanOrEqual", "fn": "ge", "arity": "binary", "types": "all6",
     "expr": "(inA[{i}] >= inB[{i}]) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},

    # ---- TIER 1: MASK / SELECTION ----
    {"name": "Select", "fn": "select", "arity": "ternary", "ternary_kind": "select", "types": "all6",
     "expr": "mask[{i}] ? inA[{i}] : inB[{i}]"},
    {"name": "Where",  "fn": "where",  "arity": "ternary", "ternary_kind": "select", "types": "all6",
     "expr": "mask[{i}] ? inA[{i}] : inB[{i}]"},

    # ---- TIER 1: BASIC NUMERIC ----
    {"name": "Min",   "fn": "min",   "arity": "binary",  "types": "all6",
     "expr": "inA[{i}] < inB[{i}] ? inA[{i}] : inB[{i}]"},
    {"name": "Max",   "fn": "max",   "arity": "binary",  "types": "all6",
     "expr": "inA[{i}] > inB[{i}] ? inA[{i}] : inB[{i}]"},
    # Clamp: fused ternary — inA=value, inB=lo, inC=hi
    {"name": "Clamp", "fn": "clamp", "arity": "ternary", "ternary_kind": "fused",  "types": "all6",
     "expr": "inA[{i}] < inB[{i}] ? inB[{i}] : (inA[{i}] > inC[{i}] ? inC[{i}] : inA[{i}])"},
    {"name": "Abs",   "fn": "abs",   "arity": "unary",   "types": "signed",
     "expr": "inA[{i}] < 0 ? -inA[{i}] : inA[{i}]"},
    {"name": "Sign",  "fn": "sign",  "arity": "unary",   "types": "signed",
     "expr": "({T})((inA[{i}] > 0) - (inA[{i}] < 0))"},

    # ---- TIER 2: FUSED / PERFORMANCE ----
    # Fma: fused ternary — inA*inB + inC (NOT a mask select)
    {"name": "Fma",        "fn": "fma",        "arity": "ternary", "ternary_kind": "fused", "types": "fp",
     "expr": "inA[{i}] * inB[{i}] + inC[{i}]"},
    {"name": "Square",     "fn": "square",     "arity": "unary",   "types": "all6",
     "expr": "inA[{i}] * inA[{i}]"},
    {"name": "Reciprocal", "fn": "reciprocal", "arity": "unary",   "types": "fp",
     "expr": "static_cast<{T}>(1) / inA[{i}]"},

    # ---- TIER 3: ROUNDING ----
    # Platform-dispatched via Leibniz::FloatIntrin::*
    {"name": "Floor", "fn": "floor", "arity": "unary", "types": "fp",
     "expr": "Leibniz::FloatIntrin::floor(inA[{i}])"},
    {"name": "Ceil",  "fn": "ceil",  "arity": "unary", "types": "fp",
     "expr": "Leibniz::FloatIntrin::ceil(inA[{i}])"},
    {"name": "Round", "fn": "round", "arity": "unary", "types": "fp",
     "expr": "Leibniz::FloatIntrin::round(inA[{i}])"},
    {"name": "Trunc", "fn": "trunc", "arity": "unary", "types": "fp",
     "expr": "Leibniz::FloatIntrin::trunc(inA[{i}])"},
    {"name": "Frac",  "fn": "frac",  "arity": "unary", "types": "fp",
     "expr": "Leibniz::FloatIntrin::frac(inA[{i}])"},

    # ---- TIER 3: CLASSIFICATION ----
    # IsNan: x != x is portable, no platform wrapper needed
    {"name": "IsNan",    "fn": "isnan",    "arity": "unary",  "types": "fp",
     "expr": "(inA[{i}] != inA[{i}]) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    # IsInf/IsFinite: dispatched via FloatIntrin
    {"name": "IsInf",    "fn": "isinf",    "arity": "unary",  "types": "fp",
     "expr": "Leibniz::FloatIntrin::isInf(inA[{i}])    ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "IsFinite", "fn": "isfinite", "arity": "unary",  "types": "fp",
     "expr": "Leibniz::FloatIntrin::isFinite(inA[{i}]) ? Leibniz::Mask::trueVal<{T}>() : Leibniz::Mask::falseVal<{T}>()"},
    {"name": "ReplaceNan", "fn": "replace_nan", "arity": "binary", "types": "fp",
     "expr": "(inA[{i}] != inA[{i}]) ? inB[{i}] : inA[{i}]"},
    {"name": "ReplaceInf", "fn": "replace_inf", "arity": "binary", "types": "fp",
     "expr": "Leibniz::FloatIntrin::isInf(inA[{i}]) ? inB[{i}] : inA[{i}]"},

    # ---- TIER 4: TRANSCENDENTAL (Horner-inlined) ----
    {"name": "Exp",   "fn": "exp",   "arity": "unary", "types": "fp", "horner": "exp"},
    {"name": "Expm1", "fn": "expm1", "arity": "unary", "types": "fp", "horner": "expm1"},
    {"name": "Exp2",  "fn": "exp2",  "arity": "unary", "types": "fp", "horner": "exp2"},
    {"name": "Exp10", "fn": "exp10", "arity": "unary", "types": "fp", "horner": "exp10"},
    {"name": "Log",   "fn": "log",   "arity": "unary", "types": "fp", "horner": "log"},
    {"name": "Log2",  "fn": "log2",  "arity": "unary", "types": "fp", "horner": "log2"},
    {"name": "Log10", "fn": "log10", "arity": "unary", "types": "fp", "horner": "log10"},
    {"name": "Log1p", "fn": "log1p", "arity": "unary", "types": "fp", "horner": "log1p"},
    {"name": "Sin",   "fn": "sin",   "arity": "unary", "types": "fp", "horner": "sin"},
    {"name": "Cos",   "fn": "cos",   "arity": "unary", "types": "fp", "horner": "cos"},
    {"name": "Tan",   "fn": "tan",   "arity": "unary", "types": "fp", "horner": "tan"},
    {"name": "Asin",  "fn": "asin",  "arity": "unary", "types": "fp", "horner": "asin"},
    # Acos derived from asin: acos(x) = pi/2 - asin(x)
    {"name": "Acos",  "fn": "acos",  "arity": "unary", "types": "fp", "horner": "acos_derived"},
    {"name": "Atan",  "fn": "atan",  "arity": "unary", "types": "fp", "horner": "atan"},
    # Atan2: binary, special emitter — inA=y (numerator), inB=x (denominator)
    {"name": "Atan2", "fn": "atan2", "arity": "binary", "types": "fp"},
    {"name": "Sinh",  "fn": "sinh",  "arity": "unary", "types": "fp", "horner": "sinh"},
    {"name": "Cosh",  "fn": "cosh",  "arity": "unary", "types": "fp", "horner": "cosh"},
    # Tanh derived: sinh(x)/cosh(x)
    {"name": "Tanh",  "fn": "tanh",  "arity": "unary", "types": "fp", "horner": "tanh_derived"},
    {"name": "Asinh", "fn": "asinh", "arity": "unary", "types": "fp", "horner": "asinh"},
    {"name": "Acosh", "fn": "acosh", "arity": "unary", "types": "fp", "horner": "acosh"},
    {"name": "Atanh", "fn": "atanh", "arity": "unary", "types": "fp", "horner": "atanh"},
    {"name": "Erf",   "fn": "erf",   "arity": "unary", "types": "fp", "horner": "erf"},
    {"name": "Erfc",  "fn": "erfc",  "arity": "unary", "types": "fp", "horner": "erfc"},
    {"name": "Sqrt",  "fn": "sqrt",  "arity": "unary", "types": "fp", "horner": "sqrt"},
    {"name": "Rsqrt", "fn": "rsqrt", "arity": "unary", "types": "fp", "horner": "rsqrt"},
    {"name": "Cbrt",  "fn": "cbrt",  "arity": "unary", "types": "fp", "horner": "cbrt"},
    # Pow: binary, special emitter — exp(inB * log(inA)), valid for inA > 0
    {"name": "Pow",     "fn": "pow",     "arity": "binary", "types": "fp"},
    {"name": "Sigmoid", "fn": "sigmoid", "arity": "unary",  "types": "fp", "horner": "sigmoid"},
]

# ============================================================
# TYPE SYSTEM
# ============================================================

TYPE_SETS = {
    "all6":   ["i32", "i64", "u32", "u64", "f32", "f64"],
    "signed": ["i32", "i64", "f32", "f64"],
    "fp":     ["f32", "f64"],
    "int4":   ["i32", "i64", "u32", "u64"],
}

TYPE_INFO = {
    "i32": {"ctype": "Numerics::Int32",   "ptr_const": "cst_i32_ptr", "ptr": "i32_ptr", "T": "Numerics::Int32"},
    "i64": {"ctype": "Numerics::Int64",   "ptr_const": "cst_i64_ptr", "ptr": "i64_ptr", "T": "Numerics::Int64"},
    "u32": {"ctype": "Numerics::UInt32",  "ptr_const": "cst_u32_ptr", "ptr": "u32_ptr", "T": "Numerics::UInt32"},
    "u64": {"ctype": "Numerics::UInt64",  "ptr_const": "cst_u64_ptr", "ptr": "u64_ptr", "T": "Numerics::UInt64"},
    "f32": {"ctype": "Numerics::Float32", "ptr_const": "cst_f32_ptr", "ptr": "f32_ptr", "T": "Numerics::Float32"},
    "f64": {"ctype": "Numerics::Float64", "ptr_const": "cst_f64_ptr", "ptr": "f64_ptr", "T": "Numerics::Float64"},
}

FP_LITERAL = {"f32": "f", "f64": ""}

# ============================================================
# COMMENT / TAG TABLES
# ============================================================

TIER_TAGS = {
    "Add": "Tier0", "Sub": "Tier0", "Mul": "Tier0", "Divide": "Tier0", "Negate": "Tier0",
    "CmpEqual": "Tier0", "CmpNotEqual": "Tier0", "CmpLessThan": "Tier0",
    "CmpLessThanOrEqual": "Tier0", "CmpGreater": "Tier0", "CmpGreaterThanOrEqual": "Tier0",
    "Select": "Tier1", "Where": "Tier1",
    "Min": "Tier1", "Max": "Tier1", "Clamp": "Tier1", "Abs": "Tier1", "Sign": "Tier1",
    "Fma": "Tier2", "Square": "Tier2", "Reciprocal": "Tier2",
    "Floor": "Tier3", "Ceil": "Tier3", "Round": "Tier3", "Trunc": "Tier3", "Frac": "Tier3",
    "IsNan": "Tier3", "IsInf": "Tier3", "IsFinite": "Tier3",
    "ReplaceNan": "Tier3", "ReplaceInf": "Tier3",
    "Exp": "Tier4", "Expm1": "Tier4", "Exp2": "Tier4", "Exp10": "Tier4",
    "Log": "Tier4", "Log2": "Tier4", "Log10": "Tier4", "Log1p": "Tier4",
    "Sin": "Tier4", "Cos": "Tier4", "Tan": "Tier4",
    "Asin": "Tier4", "Acos": "Tier4", "Atan": "Tier4", "Atan2": "Tier4",
    "Sinh": "Tier4", "Cosh": "Tier4", "Tanh": "Tier4",
    "Asinh": "Tier4", "Acosh": "Tier4", "Atanh": "Tier4",
    "Erf": "Tier4", "Erfc": "Tier4", "Sigmoid": "Tier4",
    "Sqrt": "Tier4", "Rsqrt": "Tier4", "Cbrt": "Tier4", "Pow": "Tier4",
}

OP_DESCS = {
    "Add":      "elementwise addition",
    "Sub":      "elementwise subtraction",
    "Mul":      "elementwise multiplication",
    "Divide":   "elementwise division",
    "Negate":   "elementwise negation",
    "CmpEqual":              "equality comparison, bitmask output",
    "CmpNotEqual":           "inequality comparison, bitmask output",
    "CmpLessThan":           "less-than comparison, bitmask output",
    "CmpLessThanOrEqual":    "less-than-or-equal comparison, bitmask output",
    "CmpGreater":            "greater-than comparison, bitmask output",
    "CmpGreaterThanOrEqual": "greater-than-or-equal comparison, bitmask output",
    "Select":    "branchless ternary select: mask ? a : b",
    "Where":     "alias for select: mask ? a : b",
    "Min":       "elementwise minimum",
    "Max":       "elementwise maximum",
    "Clamp":     "elementwise clamp to [lo, hi]",
    "Abs":       "elementwise absolute value",
    "Sign":      "elementwise sign: -1, 0, or +1",
    "Fma":       "fused multiply-add: inA*inB + inC, single rounding",
    "Square":    "elementwise square: x*x, exact",
    "Reciprocal":"elementwise reciprocal: 1/x",
    "Floor":     "round toward -inf, platform-dispatched",
    "Ceil":      "round toward +inf, platform-dispatched",
    "Round":     "round to nearest (ties away from zero)",
    "Trunc":     "round toward zero",
    "Frac":      "fractional part: x - floor(x)",
    "IsNan":     "true if NaN (x != x), bitmask output",
    "IsInf":     "true if +/-inf, bitmask output",
    "IsFinite":  "true if finite (not NaN, not inf), bitmask output",
    "ReplaceNan":"replace NaN with fallback value",
    "ReplaceInf":"replace +/-inf with fallback value",
    "Exp":       "e^x, Balanced Horner",
    "Expm1":     "e^x - 1, accurate near x=0, Balanced Horner",
    "Exp2":      "2^x, Balanced Horner",
    "Exp10":     "10^x, Balanced Horner",
    "Log":       "ln(x), Balanced Horner",
    "Log2":      "log2(x), Balanced Horner",
    "Log10":     "log10(x), Balanced Horner",
    "Log1p":     "ln(1+x), accurate near x=0, Balanced Horner",
    "Sin":       "sine, range-reduced, Balanced Horner",
    "Cos":       "cosine, range-reduced, Balanced Horner",
    "Tan":       "tangent, Balanced Horner",
    "Asin":      "arcsin, domain [-1,1], Balanced Horner",
    "Acos":      "arccos, domain [-1,1], derived: pi/2 - asin(x)",
    "Atan":      "arctan, Balanced Horner",
    "Atan2":     "arctan2(y,x), quadrant-corrected, inA=y inB=x",
    "Sinh":      "hyperbolic sine, Balanced Horner",
    "Cosh":      "hyperbolic cosine, Balanced Horner",
    "Tanh":      "hyperbolic tangent, derived sinh/cosh",
    "Asinh":     "inverse hyperbolic sine, Balanced Horner",
    "Acosh":     "inverse hyperbolic cosine, domain [1,inf), Balanced Horner",
    "Atanh":     "inverse hyperbolic tangent, domain (-1,1), Balanced Horner",
    "Erf":       "error function, Balanced Horner",
    "Erfc":      "complementary error function, asymptotic series",
    "Sigmoid":   "1/(1+e^-x), derived from exp",
    "Sqrt":      "square root, seed polynomial + 1x Newton-Raphson",
    "Rsqrt":     "reciprocal square root, seed polynomial + 1x Newton-Raphson",
    "Cbrt":      "cube root, seed polynomial + 1x Newton-Raphson",
    "Pow":       "x^y = exp(y*log(x)), valid for x > 0",
}

def get_tier_tag(op_name):
    return TIER_TAGS.get(op_name, "Tier?")

def get_op_desc(op, type_key):
    base = OP_DESCS.get(op["name"], op["fn"])
    horner_key = op.get("horner")
    if horner_key and horner_key in COEFFS and type_key in COEFFS.get(horner_key, {}):
        n_terms = len(COEFFS[horner_key][type_key])
        return f"{base} ({n_terms} terms)"
    return base

# ============================================================
# TIER / OP SECTION COMMENTS
# ============================================================

TIER_COMMENTS = {
    "Add": """\
\t// ============================================================
\t// TIER 0 — CORE ARITHMETIC
\t// Fundamental elementwise arithmetic: add, sub, mul, div, neg.
\t// All 6 types (i32, i64, u32, u64, f32, f64) for binary ops.
\t// Signed types only (i32, i64, f32, f64) for neg.
\t// ============================================================
""",
    "CmpEqual": """\
\t// ============================================================
\t// TIER 0 — COMPARISON
\t// Elementwise comparison ops. Output is a SIMD-style bitmask:
\t//   true  -> Leibniz::Mask::trueVal<T>()   (all bits set)
\t//   false -> Leibniz::Mask::falseVal<T>()  (all bits clear)
\t// eq / neq for float types use machine epsilon comparison.
\t// eq / neq for integer types use exact bitwise equality.
\t// lt, le, gt, ge apply to all 6 types.
\t// ============================================================
""",
    "Select": """\
\t// ============================================================
\t// TIER 1 — MASK / SELECTION
\t// Branchless elementwise selection via a bitmask condition.
\t// select and where are aliases: out[i] = mask[i] ? a[i] : b[i]
\t// mask is interpreted as boolean (nonzero = true).
\t// All 6 types supported.
\t// ============================================================
""",
    "Min": """\
\t// ============================================================
\t// TIER 1 — BASIC NUMERIC UTILITIES
\t// min, max, clamp, abs, sign.
\t// min/max/clamp: all 6 types.
\t// abs/sign: signed types only (i32, i64, f32, f64).
\t// ============================================================
""",
    "Fma": """\
\t// ============================================================
\t// TIER 2 — FUSED / PERFORMANCE OPS
\t// fma:        out = a*b + c  (single rounding, fused params inA/inB/inC)
\t// square:     out = a*a      (exact, all 6 types)
\t// reciprocal: out = 1/a      (float only)
\t// ============================================================
""",
    "Floor": """\
\t// ============================================================
\t// TIER 3 — ROUNDING / FRACTIONAL
\t// floor, ceil, round, trunc, frac.
\t// Float types only (f32, f64).
\t// Platform-dispatched via Leibniz::FloatIntrin::* wrappers.
\t// ============================================================
""",
    "IsNan": """\
\t// ============================================================
\t// TIER 3 — CLASSIFICATION / SAFETY
\t// isNan, isInf, isFinite: float types, bitmask output.
\t// replaceNan, replaceInf: replace invalid values with a fallback.
\t// isNan uses x != x (portable, no platform wrapper needed).
\t// isInf, isFinite dispatched via Leibniz::FloatIntrin::*.
\t// ============================================================
""",
    "Exp": """\
\t// ============================================================
\t// TIER 4 — EXPONENTIAL / LOGARITHMIC
\t// exp, exp2, exp10, expm1, log, log2, log10, log1p.
\t// Float types only (f32, f64).
\t// Inlined Horner chain using LUT Balanced-tier coefficients.
\t// ============================================================
""",
    "Sin": """\
\t// ============================================================
\t// TIER 4 — TRIGONOMETRIC
\t// sin, cos, tan.
\t// Float types only. Inlined Horner on range-reduced argument.
\t// ============================================================
""",
    "Asin": """\
\t// ============================================================
\t// TIER 4 — INVERSE TRIGONOMETRIC
\t// asin, acos, atan, atan2.
\t// Float types only.
\t// atan2: inA=y (numerator), inB=x (denominator). Quadrant-corrected.
\t// ============================================================
""",
    "Sinh": """\
\t// ============================================================
\t// TIER 4 — HYPERBOLIC
\t// sinh, cosh, tanh.
\t// Float types only. Inlined Horner chains.
\t// tanh computed as sinh(x) / cosh(x) from existing LUT chains.
\t// ============================================================
""",
    "Asinh": """\
\t// ============================================================
\t// TIER 4 — INVERSE HYPERBOLIC
\t// asinh, acosh, atanh.
\t// Float types only.
\t// ============================================================
""",
    "Erf": """\
\t// ============================================================
\t// TIER 4 — SPECIAL FUNCTIONS
\t// erf, erfc, sigmoid.
\t// Float types only.
\t// ============================================================
""",
    "Sqrt": """\
\t// ============================================================
\t// TIER 4 — ROOTS / POWER
\t// sqrt, rsqrt, cbrt: seed polynomial + Newton-Raphson refinement.
\t// pow: derived as exp(y * log(x)). Valid for x > 0 only.
\t// Float types only.
\t// ============================================================
""",
}

OP_COMMENTS = {
    "Sub":                   "\t// --- Sub ---",
    "Mul":                   "\t// --- Mul ---",
    "Divide":                "\t// --- Divide ---",
    "Negate":                "\t// --- Negate ---",
    "CmpNotEqual":           "\t// --- CmpNotEqual (bitmask output: trueVal<T> / falseVal<T>) ---",
    "CmpLessThan":           "\t// --- CmpLessThan (bitmask output: trueVal<T> / falseVal<T>) ---",
    "CmpLessThanOrEqual":    "\t// --- CmpLessThanOrEqual (bitmask output: trueVal<T> / falseVal<T>) ---",
    "CmpGreater":            "\t// --- CmpGreater (bitmask output: trueVal<T> / falseVal<T>) ---",
    "CmpGreaterThanOrEqual": "\t// --- CmpGreaterThanOrEqual (bitmask output: trueVal<T> / falseVal<T>) ---",
    "Where":     "\t// --- Where (alias for Select) ---",
    "Max":       "\t// --- Max ---",
    "Clamp":     "\t// --- Clamp (fused ternary: inA=value, inB=lo, inC=hi) ---",
    "Abs":       "\t// --- Abs ---",
    "Sign":      "\t// --- Sign ---",
    "Square":    "\t// --- Square ---",
    "Reciprocal":"\t// --- Reciprocal ---",
    "Ceil":      "\t// --- Ceil ---",
    "Round":     "\t// --- Round ---",
    "Trunc":     "\t// --- Trunc ---",
    "Frac":      "\t// --- Frac ---",
    "IsInf":     "\t// --- IsInf ---",
    "IsFinite":  "\t// --- IsFinite ---",
    "ReplaceNan":"\t// --- ReplaceNan ---",
    "ReplaceInf":"\t// --- ReplaceInf ---",
    "Expm1":     "\t// --- Expm1 ---",
    "Exp2":      "\t// --- Exp2 ---",
    "Exp10":     "\t// --- Exp10 ---",
    "Log":       "\t// --- Log ---",
    "Log2":      "\t// --- Log2 ---",
    "Log10":     "\t// --- Log10 ---",
    "Log1p":     "\t// --- Log1p ---",
    "Cos":       "\t// --- Cos ---",
    "Tan":       "\t// --- Tan ---",
    "Acos":      "\t// --- Acos (derived: pi/2 - asin(x)) ---",
    "Atan":      "\t// --- Atan ---",
    "Atan2":     "\t// --- Atan2 (inA=y numerator, inB=x denominator, quadrant-corrected) ---",
    "Cosh":      "\t// --- Cosh ---",
    "Tanh":      "\t// --- Tanh (derived: sinh(x) / cosh(x)) ---",
    "Acosh":     "\t// --- Acosh ---",
    "Atanh":     "\t// --- Atanh ---",
    "Erfc":      "\t// --- Erfc ---",
    "Sigmoid":   "\t// --- Sigmoid ---",
    "Rsqrt":     "\t// --- Rsqrt (seed polynomial + 1x Newton-Raphson) ---",
    "Cbrt":      "\t// --- Cbrt (seed polynomial + 1x Newton-Raphson) ---",
    "Pow":       "\t// --- Pow (derived: exp(inB * log(inA)), valid for inA > 0 only) ---",
}

# ============================================================
# HORNER CHAIN BUILDER
# ============================================================

def build_horner_expr(coeffs, var, fp_suffix):
    """Build inlined Horner evaluation: c[0] + x*(c[1] + x*(c[2] + ...))"""
    def lit(c):
        if fp_suffix == "f" and c.endswith("f"):
            return c
        return c + fp_suffix
    expr = lit(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        expr = f"({lit(c)} + {var} * {expr})"
    return expr

def build_horner_body_unary(op_key, type_key, N):
    """
    Emit the loop body for a unary Horner-based Tier 4 op.
    Uses IDX_VAL for staticFor (integral_constant), replaces to 'i' for MSVC loop.
    """
    # Resolve derived ops to their underlying coefficients
    coeff_key = op_key
    if op_key == "acos_derived":
        coeff_key = "asin"
    elif op_key == "tanh_derived":
        coeff_key = None  # handled specially below

    coeffs = COEFFS.get(coeff_key, {}).get(type_key, []) if coeff_key else []
    suffix = FP_LITERAL[type_key]
    T = TYPE_INFO[type_key]["T"]

    idx = IDX_VAL

    if op_key == "sigmoid":
        horner = build_horner_expr(coeffs, f"(-inA[{idx}])", suffix)
        inner = (f"// Balanced Horner: 1 / (1 + exp(-x))\n"
                 f"\t\t\tout[{idx}] = {1.0}{suffix} / ({1.0}{suffix} + {horner});")

    elif op_key == "sqrt":
        horner = build_horner_expr(coeffs, f"inA[{idx}]", suffix)
        inner = (f"// seed polynomial + 1x Newton-Raphson\n"
                 f"\t\t\tconst {T} r = {horner};\n"
                 f"\t\t\tconst {T} s = r * r;\n"
                 f"\t\t\tout[{idx}] = r + {0.5}{suffix} * (inA[{idx}] - s) / r;")

    elif op_key == "rsqrt":
        horner = build_horner_expr(coeffs, f"inA[{idx}]", suffix)
        inner = (f"// seed polynomial + 1x Newton-Raphson\n"
                 f"\t\t\tconst {T} r = {horner};\n"
                 f"\t\t\tconst {T} s = inA[{idx}] * r * r;\n"
                 f"\t\t\tout[{idx}] = r * ({1.5}{suffix} - {0.5}{suffix} * s);")

    elif op_key == "cbrt":
        horner = build_horner_expr(coeffs, f"inA[{idx}]", suffix)
        inner = (f"// seed polynomial + 1x Newton-Raphson\n"
                 f"\t\t\tconst {T} r = {horner};\n"
                 f"\t\t\tconst {T} s = r * r * r;\n"
                 f"\t\t\tout[{idx}] = r * ({2.0/3.0}{suffix} + ({1.0/3.0}{suffix} * inA[{idx}]) / s);")

    elif op_key == "acos_derived":
        # acos(x) = pi/2 - asin(x), reuse asin coefficients
        pi_2 = PI_2_F32 if type_key == "f32" else PI_2_F64
        horner = build_horner_expr(coeffs, f"inA[{idx}]", suffix)
        inner = (f"// Derived: pi/2 - asin(x), Balanced Horner on asin\n"
                 f"\t\t\tout[{idx}] = {pi_2}{suffix} - {horner};")

    elif op_key == "tanh_derived":
        # tanh(x) = sinh(x) / cosh(x), two Horner chains
        sinh_coeffs = COEFFS["sinh"][type_key]
        cosh_coeffs = COEFFS["cosh"][type_key]
        x2 = f"(inA[{idx}] * inA[{idx}])"
        sinh_h = build_horner_expr(sinh_coeffs, x2, suffix)
        cosh_h = build_horner_expr(cosh_coeffs, x2, suffix)
        inner = (f"// Derived: sinh(x) / cosh(x), two Balanced Horner chains\n"
                 f"\t\t\tconst {T} s = inA[{idx}] * {sinh_h};\n"
                 f"\t\t\tconst {T} c = {cosh_h};\n"
                 f"\t\t\tout[{idx}] = s / c;")

    else:
        horner = build_horner_expr(coeffs, f"inA[{idx}]", suffix)
        inner = (f"// Balanced Horner: c[0] + x*(c[1] + x*(...))\n"
                 f"\t\t\tout[{idx}] = {horner};")

    msvc_inner = inner.replace(IDX_VAL, "i")
    return (f"#if LEIBNIZ_COMPILER_MSVC\n"
            f"\t\tfor (size_t i = 0; i < N; i++) {{\n"
            f"\t\t\t{msvc_inner}\n"
            f"\t\t}}\n"
            f"#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC\n"
            f"\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) {{\n"
            f"\t\t\t{inner}\n"
            f"\t\t}});\n"
            f"#endif")

def build_atan2_body(type_key, N):
    """Binary special emitter for atan2: inA=y, inB=x, quadrant-corrected."""
    coeffs = COEFFS["atan"][type_key]
    suffix = FP_LITERAL[type_key]
    T = TYPE_INFO[type_key]["T"]
    pi   = PI_F32   if type_key == "f32" else PI_F64
    pi_2 = PI_2_F32 if type_key == "f32" else PI_2_F64
    idx = IDX_VAL

    horner = build_horner_expr(coeffs, f"(inA[{idx}] / inB[{idx}])", suffix)
    inner = (f"// atan2(y,x): atan(y/x) with quadrant correction\n"
             f"\t\t\tconst {T} r = {horner};\n"
             f"\t\t\tconst {T} result =\n"
             f"\t\t\t\t(inB[{idx}] > {T}(0))                                  ? r :\n"
             f"\t\t\t\t(inB[{idx}] < {T}(0) && inA[{idx}] >= {T}(0)) ? r + {pi}{suffix} :\n"
             f"\t\t\t\t(inB[{idx}] < {T}(0) && inA[{idx}] <  {T}(0)) ? r - {pi}{suffix} :\n"
             f"\t\t\t\t(inA[{idx}] > {T}(0))                                  ? {pi_2}{suffix} :\n"
             f"\t\t\t\t(inA[{idx}] < {T}(0))                                  ? -{pi_2}{suffix} : {T}(0);\n"
             f"\t\t\tout[{idx}] = result;")

    msvc_inner = inner.replace(IDX_VAL, "i")
    return (f"#if LEIBNIZ_COMPILER_MSVC\n"
            f"\t\tfor (size_t i = 0; i < N; i++) {{\n"
            f"\t\t\t{msvc_inner}\n"
            f"\t\t}}\n"
            f"#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC\n"
            f"\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) {{\n"
            f"\t\t\t{inner}\n"
            f"\t\t}});\n"
            f"#endif")

def build_pow_body(type_key, N):
    """Binary special emitter for pow: exp(inB * log(inA)). Valid for inA > 0."""
    log_coeffs = COEFFS["log"][type_key]
    exp_coeffs = COEFFS["exp"][type_key]
    suffix = FP_LITERAL[type_key]
    T = TYPE_INFO[type_key]["T"]
    idx = IDX_VAL

    log_h = build_horner_expr(log_coeffs, f"inA[{idx}]", suffix)
    exp_h = build_horner_expr(exp_coeffs, f"(inB[{idx}] * log_val)", suffix)
    inner = (f"// pow(x,y) = exp(y * log(x)) — valid for inA (base) > 0 only\n"
             f"\t\t\tconst {T} log_val = {log_h};\n"
             f"\t\t\tout[{idx}] = {exp_h};")

    msvc_inner = inner.replace(IDX_VAL, "i")
    return (f"#if LEIBNIZ_COMPILER_MSVC\n"
            f"\t\tfor (size_t i = 0; i < N; i++) {{\n"
            f"\t\t\t{msvc_inner}\n"
            f"\t\t}}\n"
            f"#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC\n"
            f"\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) {{\n"
            f"\t\t\t{inner}\n"
            f"\t\t}});\n"
            f"#endif")

# ============================================================
# FUNCTION EMITTER
# ============================================================

def emit_params_binary(ti):
    return (f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inA,\n"
            f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inB,\n"
            f"\t\t{ti['ptr']} LEIBNIZ_RESTRICT out")

def emit_params_unary(ti):
    return (f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inA,\n"
            f"\t\t{ti['ptr']} LEIBNIZ_RESTRICT out")

def emit_params_ternary(ti, kind="select"):
    if kind == "fused":
        return (f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inA,\n"
                f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inB,\n"
                f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inC,\n"
                f"\t\t{ti['ptr']} LEIBNIZ_RESTRICT out")
    else:  # select
        return (f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT mask,\n"
                f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inA,\n"
                f"\t\t{ti['ptr_const']} LEIBNIZ_RESTRICT inB,\n"
                f"\t\t{ti['ptr']} LEIBNIZ_RESTRICT out")

def emit_asserts_binary(ti):
    return (f"\t\tLEIBNIZ_ASSERT(inA != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inA, 32);\n"
            f"\t\tLEIBNIZ_ASSERT(inB != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inB, 32);\n"
            f"\t\tLEIBNIZ_ASSERT(out != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(out, 32);")

def emit_asserts_unary(ti):
    return (f"\t\tLEIBNIZ_ASSERT(inA != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inA, 32);\n"
            f"\t\tLEIBNIZ_ASSERT(out != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(out, 32);")

def emit_asserts_ternary(ti, kind="select"):
    if kind == "fused":
        return (f"\t\tLEIBNIZ_ASSERT(inA != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inA, 32);\n"
                f"\t\tLEIBNIZ_ASSERT(inB != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inB, 32);\n"
                f"\t\tLEIBNIZ_ASSERT(inC != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inC, 32);\n"
                f"\t\tLEIBNIZ_ASSERT(out != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(out, 32);")
    else:
        return (f"\t\tLEIBNIZ_ASSERT(mask != nullptr);\tLEIBNIZ_ASSUME_ALIGNED(mask, 32);\n"
                f"\t\tLEIBNIZ_ASSERT(inA != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inA, 32);\n"
                f"\t\tLEIBNIZ_ASSERT(inB != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(inB, 32);\n"
                f"\t\tLEIBNIZ_ASSERT(out != nullptr);\t\tLEIBNIZ_ASSUME_ALIGNED(out, 32);")

def emit_msvc_body_expr(expr, idx_var="i"):
    e = expr.replace("{i}", idx_var)
    return (f"\t\tfor (size_t {idx_var} = 0; {idx_var} < N; {idx_var}++)\n"
            f"\t\t\tout[{idx_var}] = {e};")

def emit_gcc_body_expr(expr):
    # Use IDX_VAL for expressions; bare idx works for array subscript but
    # we standardize on idx.value everywhere for consistency and safety.
    e = expr.replace("{i}", IDX_VAL)
    return (f"\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) {{\n"
            f"\t\t\tout[idx] = {e};\n"
            f"\t\t}});")

def emit_function(op, type_key, N=8):
    ti = TYPE_INFO[type_key]
    arity = op["arity"]
    fn_name = f"unrolledScalar{op['name']}Elements_{type_key}"
    T = ti["T"]
    kind = op.get("ternary_kind", "select")

    tier_tag = get_tier_tag(op["name"])
    desc = get_op_desc(op, type_key)
    doc_comment = f"\t// [{tier_tag}] {op['fn']}_{type_key} — {desc}"

    # params & asserts
    if arity == "binary":
        params  = emit_params_binary(ti)
        asserts = emit_asserts_binary(ti)
    elif arity == "unary":
        params  = emit_params_unary(ti)
        asserts = emit_asserts_unary(ti)
    else:  # ternary
        params  = emit_params_ternary(ti, kind)
        asserts = emit_asserts_ternary(ti, kind)

    # body
    horner_key = op.get("horner")
    fn_key     = op.get("fn")

    if horner_key:
        body = build_horner_body_unary(horner_key, type_key, N)
    elif fn_key == "atan2":
        body = build_atan2_body(type_key, N)
    elif fn_key == "pow":
        body = build_pow_body(type_key, N)
    elif "expr" in op:
        expr = op["expr"].replace("{T}", T)
        msvc_body = emit_msvc_body_expr(expr)
        gcc_body  = emit_gcc_body_expr(expr)
        body = (f"#if LEIBNIZ_COMPILER_MSVC\n"
                f"{msvc_body}\n"
                f"#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC\n"
                f"{gcc_body}\n"
                f"#endif")
    else:
        return ""

    return (f"{doc_comment}\n"
            f"\ttemplate<size_t N> requires (N <= 8)\n"
            f"\tLEIBNIZ_FORCEINLINE void {fn_name}(\n"
            f"{params}\n"
            f"\t) {{\n"
            f"{asserts}\n\n"
            f"{body}\n"
            f"\t}}\n")

# ============================================================
# MAP ABSTRACTIONS (Tier 5)
# ============================================================

MAP_BLOCK = """
\t// ============================================================
\t// TIER 5 — GENERIC ELEMENTWISE MAP ABSTRACTIONS
\t// map, map2, map3: apply any user lambda elementwise over N elements.
\t// Templated on element type T and operation type Op.
\t// Compiler inlines the lambda and unrolls the staticFor loop.
\t// ============================================================

\ttemplate<size_t N, typename T, typename UnaryOp> requires (N <= 8)
\tLEIBNIZ_FORCEINLINE void unrolledScalarMap(
\t\tconst T* LEIBNIZ_RESTRICT in,
\t\tT* LEIBNIZ_RESTRICT out,
\t\tUnaryOp op
\t) {
\t\tLEIBNIZ_ASSERT(in  != nullptr); LEIBNIZ_ASSUME_ALIGNED(in,  32);
\t\tLEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
\t\tfor (size_t i = 0; i < N; i++) out[i] = op(in[i]);
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = op(in[idx]); });
#endif
\t}

\ttemplate<size_t N, typename T, typename BinaryOp> requires (N <= 8)
\tLEIBNIZ_FORCEINLINE void unrolledScalarMap2(
\t\tconst T* LEIBNIZ_RESTRICT inA,
\t\tconst T* LEIBNIZ_RESTRICT inB,
\t\tT* LEIBNIZ_RESTRICT out,
\t\tBinaryOp op
\t) {
\t\tLEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
\t\tLEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
\t\tLEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
\t\tfor (size_t i = 0; i < N; i++) out[i] = op(inA[i], inB[i]);
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = op(inA[idx], inB[idx]); });
#endif
\t}

\ttemplate<size_t N, typename T, typename TernaryOp> requires (N <= 8)
\tLEIBNIZ_FORCEINLINE void unrolledScalarMap3(
\t\tconst T* LEIBNIZ_RESTRICT inA,
\t\tconst T* LEIBNIZ_RESTRICT inB,
\t\tconst T* LEIBNIZ_RESTRICT inC,
\t\tT* LEIBNIZ_RESTRICT out,
\t\tTernaryOp op
\t) {
\t\tLEIBNIZ_ASSERT(inA != nullptr); LEIBNIZ_ASSUME_ALIGNED(inA, 32);
\t\tLEIBNIZ_ASSERT(inB != nullptr); LEIBNIZ_ASSUME_ALIGNED(inB, 32);
\t\tLEIBNIZ_ASSERT(inC != nullptr); LEIBNIZ_ASSUME_ALIGNED(inC, 32);
\t\tLEIBNIZ_ASSERT(out != nullptr); LEIBNIZ_ASSUME_ALIGNED(out, 32);

#if LEIBNIZ_COMPILER_MSVC
\t\tfor (size_t i = 0; i < N; i++) out[i] = op(inA[i], inB[i], inC[i]);
#elif LEIBNIZ_COMPILER_CLANG || LEIBNIZ_COMPILER_GCC
\t\tUtils::Unrolled::staticFor<0, N>([=](auto idx) { out[idx] = op(inA[idx], inB[idx], inC[idx]); });
#endif
\t}
"""

# ============================================================
# HEADER BOILERPLATE
# ============================================================

HEADER_TOP = """\
#pragma once
// ============================================================
// UnrolledScalarKernels.h
// AUTO-GENERATED by gen_unrolled_kernels.py — DO NOT EDIT BY HAND
//
// Unrolled scalar fallback kernels for N <= 8 elements.
//
// DISPATCH MODEL:
//   Primary path:  Vectorize* kernels (AVX2, full SIMD lane width)
//   Fallback path: These unrolled scalar kernels (N <= 8, tail handler)
//
//   The fallback triggers when tensor size is not a multiple of the
//   AVX2 lane count (8 x FP32 / 4 x FP64). N is a compile-time constant
//   so the compiler fully unrolls the staticFor loop and can pipeline
//   all N independent element chains via ILP.
//
// TIER STRUCTURE:
//   Tier 0 — Core Arithmetic:    add, sub, mul, div, neg
//   Tier 0 — Comparison:         eq, neq, lt, le, gt, ge (bitmask output)
//   Tier 1 — Mask / Selection:   select, where
//   Tier 1 — Basic Numeric:      min, max, clamp, abs, sign
//   Tier 2 — Fused / Perf:       fma, square, reciprocal
//   Tier 3 — Rounding:           floor, ceil, round, trunc, frac
//   Tier 3 — Classification:     isNan, isInf, isFinite, replaceNan, replaceInf
//   Tier 4 — Transcendental:     exp, exp2, exp10, expm1, log, log2, log10,
//                                log1p, sin, cos, tan, asin, acos, atan, atan2,
//                                sinh, cosh, tanh, asinh, acosh, atanh,
//                                erf, erfc, sqrt, rsqrt, cbrt, pow, sigmoid
//   Tier 5 — Generic Map:        map, map2, map3
//
// MASK CONVENTION:
//   All comparison and classification ops produce SIMD-style bitmasks:
//     true  -> Leibniz::Mask::trueVal<T>()   (all bits set)
//     false -> Leibniz::Mask::falseVal<T>()  (all bits clear)
//   For integer types this is ~T(0) / T(0).
//   For float types this is std::bit_cast from 0xFFFFFFFF / 0x00000000.
//
// TRANSCENDENTAL ACCURACY:
//   All Tier 4 ops use Balanced-tier Chebyshev minimax coefficients
//   from LookUpTables.h. Accuracy matches the Vectorize* SIMD path.
//   sqrt, rsqrt, cbrt include one Newton-Raphson refinement step.
//
// EPSILON COMPARISON:
//   Float eq/neq use Leibniz::Numerics::Limits::NumericLimits<T>::epsilon()
//   for the tolerance threshold (machine epsilon, 2^-mantissaBits).
//
// DEPENDENCIES:
//   FloatIntrin.h       — platform wrapper for isInf, isFinite, floor etc.
//   LeibnizMask.h       — bitmask value helpers (trueVal / falseVal)
//   LeibnizInt.h        — Numerics::Int32/64, UInt32/64
//   LeibnizFloat.h      — Numerics::Float32/64
//   NumericLimits.h     — NumericLimits<T>::epsilon()
//   <bit>               — std::bit_cast (C++20, accepted Leibniz exception)
// ============================================================

#include <bit>
#include "Leibniz.h"
#include "StaticPipelines.h"
#include "LeibnizCompiler.h"
#include "LebnizDiagnostics.h"
#include "LeibnizInt.h"
#include "LeibnizFloat.h"
#include "FloatIntrin.h"
#include "LeibnizMask.h"
#include "NumericLimits.h"

namespace Leibniz::LinAlg::Internal::Kernels::Scalar {

\tusing cst_u32_ptr = const Numerics::UInt32*;
\tusing cst_u64_ptr = const Numerics::UInt64*;
\tusing u32_ptr = Numerics::UInt32*;
\tusing u64_ptr = Numerics::UInt64*;

\tusing cst_i32_ptr = const Numerics::Int32*;
\tusing cst_i64_ptr = const Numerics::Int64*;
\tusing i32_ptr = Numerics::Int32*;
\tusing i64_ptr = Numerics::Int64*;

\tusing cst_f32_ptr = const Numerics::Float32*;
\tusing cst_f64_ptr = const Numerics::Float64*;
\tusing f32_ptr = Numerics::Float32*;
\tusing f64_ptr = Numerics::Float64*;

"""

HEADER_BOTTOM = "\n} // namespace Leibniz::LinAlg::Internal::Kernels::Scalar\n"

# ============================================================
# MAIN GENERATOR
# ============================================================

# Track which op names have had their tier banner already emitted
# (CmpEqual appears twice — int4 and fp — so we only want one banner)
_emitted_tier_banners = set()
_emitted_op_banners   = set()

def generate():
    global _emitted_tier_banners, _emitted_op_banners
    _emitted_tier_banners = set()
    _emitted_op_banners   = set()

    out = [HEADER_TOP]

    for op in OPS:
        name = op["name"]
        type_keys = TYPE_SETS[op["types"]]

        # Tier banner (once per tier-entry op name)
        if name in TIER_COMMENTS and name not in _emitted_tier_banners:
            out.append(TIER_COMMENTS[name] + "\n")
            _emitted_tier_banners.add(name)

        # Per-op section comment (once per op name)
        if name not in _emitted_tier_banners and name in OP_COMMENTS and name not in _emitted_op_banners:
            out.append(OP_COMMENTS[name] + "\n")
            _emitted_op_banners.add(name)
        elif name in OP_COMMENTS and name not in _emitted_op_banners:
            out.append(OP_COMMENTS[name] + "\n")
            _emitted_op_banners.add(name)

        # For ops that are tier-banner openers, also emit their own op comment on first pass
        if name in TIER_COMMENTS and name not in _emitted_op_banners:
            # Tier opener ops don't need a redundant op comment — tier banner is sufficient
            _emitted_op_banners.add(name)

        for tk in type_keys:
            out.append(emit_function(op, tk))

    out.append(MAP_BLOCK)
    out.append(HEADER_BOTTOM)
    return "".join(out)


if __name__ == "__main__":
    import sys
    content = generate()
    output_path = sys.argv[1] if len(sys.argv) > 1 else "UnrolledScalarKernels.h"
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated {output_path} ({content.count(chr(10))} lines)")
