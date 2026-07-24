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
    i8 = tl.program_id(0)
    i9 = i8.to(tl.int64)
    i10 = i9 * c0

    i11 = tl.arange(0, c0)
    i12 = i11.to(tl.int64)
    i13 = i10 + i12
    i14 = i13 < p6

    i15 = tl.program_id(1)
    i16 = i15.to(tl.int64)

    i17 = p0 + i16 * p6
    i18 = p1 + i16 * p6

    i19 = tl.load(p2)

    i20 = p3 + i13
    i21 = tl.load(i20, mask=i14, other=0.0)

    i22 = i17 + i13
    i23 = tl.load(i22, mask=i14, other=0.0)

    v0 = i19 * i23
    v1 = libdevice.tanh(v0)
    v2 = v1 * i21

    if p5:
        i25 = p4 + i13
        i26 = tl.load(i25, mask=i14, other=0.0)
        v3 = v2 + i26
        v4 = v3
    else:
        v4 = v2

    i27 = i18 + i13
    tl.store(i27, v4, mask=i14)
