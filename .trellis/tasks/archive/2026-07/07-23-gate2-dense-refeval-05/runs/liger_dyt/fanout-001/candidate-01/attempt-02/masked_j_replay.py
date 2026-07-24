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
    i10 = p0 + i9
    i11 = i7 * p6
    i12 = p1 + i11

    i14 = tl.load(p2)

    i16 = p3 + i5
    i17 = tl.load(i16, mask=i5 < p6, other=0.0)

    i18 = i10 + i5
    i19 = tl.load(i18, mask=i5 < p6, other=0.0)

    i20 = i14 * i19
    i21 = libdevice.tanh(i20)
    i22 = i21 + i17

    if p5:
        i25 = p4 + i5
        i26 = tl.load(i25, mask=i5 < p6, other=0.0)
        i27 = i22 + i26
        i28 = i27
    else:
        i28 = i22

    i29 = i12 + i5
    tl.store(i29, i28, mask=i5 < p6)
