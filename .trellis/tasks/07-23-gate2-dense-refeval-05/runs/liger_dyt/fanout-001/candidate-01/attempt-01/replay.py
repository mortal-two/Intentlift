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
    i4 = tl.arange(0, c0)
    i5 = i3 + i4

    i6 = tl.program_id(1)
    i7 = i6.to(tl.int64)

    i9 = i7 * p6
    i10 = i7 * p6

    i11 = p2
    i12 = tl.load(i11)

    i13 = p3
    i14 = i13 + i5
    i15 = tl.load(i14, mask=i5 < p6, other=0.0)

    i16 = p0 + (i9 + i5)
    i17 = tl.load(i16, mask=i5 < p6, other=0.0)

    i18 = i12 * i17
    i19 = libdevice.tanh(i18)
    i20 = i19 * i15

    if p5:
        i22 = p4
        i23 = i22 + i5
        i24 = tl.load(i23, mask=i5 < p6, other=0.0)
        i25 = i20 + i24
        i26 = i25
    else:
        i26 = i20

    i27 = p1 + (i10 + i5)
    tl.store(i27, i26, mask=i5 < p6)
