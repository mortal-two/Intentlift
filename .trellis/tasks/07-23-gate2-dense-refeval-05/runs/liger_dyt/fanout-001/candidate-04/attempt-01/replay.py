import triton
import triton.language as tl


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
    i2 = tl.program_id(0)
    i3 = i2.to(tl.int64)
    i4 = i3 * c0

    i5 = tl.arange(0, c0)
    i6 = i5.to(tl.int64)
    i7 = i4 + i6

    i8 = tl.program_id(1)
    i9 = i8.to(tl.int64)
    i10 = i9 * p6

    i12 = p0 + i10
    i14 = p1 + i10

    i16 = tl.load(p2)
    i17 = i16.to(tl.float32)

    i19 = tl.load(p3 + i7, mask=i7 < p6, other=0.0)
    i20 = i19.to(tl.float32)

    i21 = tl.load(i12 + i7, mask=i7 < p6, other=0.0)
    i22 = i21.to(tl.float32)

    i23 = i17 * i22
    i24 = tl.tanh(i23)
    i25 = i24 * i20

    if p5:
        i28 = tl.load(p4 + i7, mask=i7 < p6, other=0.0)
        i29 = i28.to(tl.float32)
        i30 = i25 + i29
        i31 = i30
    else:
        i31 = i25

    tl.store(i14 + i7, i31, mask=i7 < p6)
