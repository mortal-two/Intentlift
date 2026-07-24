import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def scrc_k0(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5: tl.constexpr,
    p6: tl.constexpr,
    c0: tl.constexpr,
):
    i0 = tl.program_id(0)
    i1 = i0.to(tl.int64)
    i3 = i1 * c0
    i5 = i3 + tl.arange(0, c0)

    i7 = tl.program_id(1)
    i8 = i7.to(tl.int64)

    i10 = p0 + i8 * p6
    i12 = p1 + i8 * p6

    i13 = p2 + 0
    i14 = tl.load(i13)

    i16 = p3 + i5
    i17 = tl.load(i16, mask=i5 < p6, other=0.0)

    i18 = i10 + i5
    i19 = tl.load(i18, mask=i5 < p6, other=0.0)

    v0 = i14 * i19
    v1 = libdevice.tanh(v0)
    v2 = v1 * i17

    if p5:
        i22 = p4 + i5
        i23 = tl.load(i22, mask=i5 < p6, other=0.0)
        v3 = v2 + i23
        t5 = v3
    else:
        t5 = v2

    i24 = i12 + i5
    tl.store(i24, t5, mask=i5 < p6)
