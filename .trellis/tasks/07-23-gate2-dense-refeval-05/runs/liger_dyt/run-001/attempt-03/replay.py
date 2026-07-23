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
    i6 = i5 < p6

    i7 = tl.program_id(1)
    i8 = i7.to(tl.int64)
    i10 = i8 * p6

    i12 = p0 + i10
    i14 = p1 + i10

    i16 = tl.load(p2)
    i17 = i16.to(tl.float32)

    i19 = p3 + i5
    i20 = tl.load(i19, mask=i6, other=0.0)
    i21 = i20.to(tl.float32)

    i22 = i12 + i5
    i23 = tl.load(i22, mask=i6, other=0.0)
    i24 = i23.to(tl.float32)

    i25 = i17 * i24
    i26 = libdevice.tanh(i25)
    i27 = i26 * i21

    if p5:
        i30 = p4 + i5
        i31 = tl.load(i30, mask=i6, other=0.0)
        i32 = i31.to(tl.float32)
        i33 = i27 + i32
    else:
        i33 = i27

    i34 = tl.where(p5, i33, i27)
    i35 = i14 + i5
    tl.store(i35, i34, mask=i6)
