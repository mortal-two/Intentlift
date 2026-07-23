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

    i14 = tl.program_id(1)
    i15 = i14.to(tl.int64)
    i16 = i15 * p6
    i17 = i15 * p6

    i18 = p0 + i16
    i19 = p1 + i17

    i20 = tl.load(p2)

    i21 = p3 + i13
    i22 = tl.load(i21, mask=i13 < p6, other=0.0)

    i23 = i18 + i13
    i24 = tl.load(i23, mask=i13 < p6, other=0.0)

    i25 = i20 * i24
    i26 = libdevice.tanh(i25)
    i27 = i26 * i22

    if p5:
        i28 = p4 + i13
        i29 = tl.load(i28, mask=i13 < p6, other=0.0)
        i30 = i27 + i29
        i31 = i30
    else:
        i31 = i27

    i32 = i19 + i13
    tl.store(i32, i31, mask=i13 < p6)
