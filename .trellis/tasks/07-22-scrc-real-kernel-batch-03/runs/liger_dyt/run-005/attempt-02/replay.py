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
    i12 = tl.program_id(0)
    i13 = tl.cast(i12, tl.int64)

    i14 = i13 * p6
    i15 = i13 * p6

    i16 = p0 + i14
    i17 = p1 + i15

    i22 = tl.load(p2)

    for i7 in range(0, tl.cdiv(p6, c0)):
        i8 = tl.cast(i7, tl.int64)
        i9 = i8 * c0
        i10 = tl.arange(0, c0)
        i11 = i9 + i10

        mask = i11 < p6

        i18 = p3 + i11
        i19 = i16 + i11
        i20 = p4 + i11
        i21 = i17 + i11

        i23 = tl.load(i18, mask=mask, other=0.0)
        i24 = tl.load(i19, mask=mask, other=0.0)
        i25 = tl.load(i20, mask=mask, other=0.0)

        i26 = i22 * i24
        i27 = libdevice.tanh(i26)
        i28 = i27 * i23
        i29 = i28 + i25

        tl.store(i21, i29, mask=mask)
