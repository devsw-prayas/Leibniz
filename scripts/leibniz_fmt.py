"""
Leibniz HPC Math Library — shared formatting utilities for code generators.
"""
import mpmath
import math

def set_prec(mbits: int):
    mpmath.mp.prec = mbits + 64

def sig_digits(mbits: int) -> int:
    return int(math.ceil(mbits * math.log10(2))) + 6

def to_hex_float(v, mbits: int, suffix: str) -> str:
    if v == 0:
        return f"0x0p+0{suffix}"
    set_prec(mbits)
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

def fmt_coeff(v, mbits: int, suffix: str) -> str:
    return to_hex_float(v, mbits, suffix)
