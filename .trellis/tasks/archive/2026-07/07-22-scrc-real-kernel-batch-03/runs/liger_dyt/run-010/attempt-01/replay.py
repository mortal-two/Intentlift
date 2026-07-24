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
    i2 = i0 * c0
    i3 = tl.arange(0, c0)
    i4 = i2 + i3

    i5 = tl.program_id(1)

    i8 = p0 + i5 * p6
    i9 = i8 + i2
    i10 = i9 + i3
    i11 = tl.load(i10, mask=i4 < p6, other=0.0)

    i13 = p1 + i5 * p6
    i14 = i13 + i2
    i15 = i14 + i3

    i18 = p3 + i2
    i19 = i18 + i3
    i20 = tl.load(i19, mask=i4 < p6, other=0.0)

    i22 = p4 + i2
    i23 = i22 + i3
    i24 = tl.load(i23, mask=p5 & (i4 < p6), other=0.0)

    i26 = tl.load(p2)

    i28 = i26 * i11
    i29 = libdevice.tanh(i28)
    i30 = i29 * i20
    i31 = i30 + i24
    i32 = tl.where(p5, i31, i30)

    tl.store(i15, i32, mask=i4 < p6)
