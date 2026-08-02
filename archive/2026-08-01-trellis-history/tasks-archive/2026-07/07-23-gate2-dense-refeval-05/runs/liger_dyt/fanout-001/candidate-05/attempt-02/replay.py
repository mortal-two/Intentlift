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
    i2 = i1 * c0
    i4 = i2 + tl.arange(0, c0)

    i5 = tl.program_id(1)
    i6 = i5.to(tl.int64)
    i8 = i6 * p6

    i15 = p0 + i8
    i16 = i15 + i4

    i18 = p1 + i8
    i19 = i18 + i4

    mask = i4 < p6

    i9 = tl.load(p2)
    i11 = tl.load(p3 + i4, mask=mask, other=0.0)
    i10 = tl.load(i16, mask=mask, other=0.0)
    i12 = tl.load(p4 + i4, mask=mask & p5, other=0.0)

    v0 = i10 * i9
    v1 = libdevice.tanh(v0)
    v2 = v1 * i11
    v3 = v2 + i12
    result = tl.where(p5, v3, v2)

    tl.store(i19, result, mask=mask)
