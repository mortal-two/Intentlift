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
    i0 = tl.full((1,), 0, tl.int32)
    i1 = i0.to(tl.int64)

    i3 = tl.arange(0, c0)
    i4 = i3.to(tl.int64)

    i5 = i1 * c0 + i4
    i7 = i5 < p6

    i8 = tl.program_id(0)
    i9 = i8.to(tl.int64)
    i10 = i9 * p6

    i11 = p0 + i10
    i12 = p1 + i10

    i14 = tl.load(p2)
    i15 = i14

    i16 = p3 + i5
    i17 = tl.load(i16, mask=i7, other=0.0)
    i18 = i17

    i19 = i11 + i5
    i20 = tl.load(i19, mask=i7, other=0.0)
    i21 = i20

    i22 = i15 * i21
    i23 = libdevice.tanh(i22)
    i24 = i23 * i18

    if p5:
        i26 = p4 + i5
        i27 = tl.load(i26, mask=i7, other=0.0)
        i28 = i27
        i29 = i24 + i28
        i30 = i29
    else:
        i30 = i24

    i31 = i12 + i5
    tl.store(i31, i30, mask=i7)
