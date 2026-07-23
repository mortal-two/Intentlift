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
    i0 = tl.program_id(0)
    i1 = i0.to(tl.int64)

    i3 = tl.arange(0, c0)
    i4 = i3.to(tl.int64)

    i5 = i1 * c0 + i4
    i7 = i5 < p6

    i8 = tl.program_id(1)
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
    i23 = 2.0
    i24 = i22 * i23
    i25 = tl.sigmoid(i24)
    i26 = i23 * i25
    i27 = 1.0
    i28 = i26 - i27
    i29 = i28 * i18

    if p5:
        i31 = p4 + i5
        i32 = tl.load(i31, mask=i7, other=0.0)
        i33 = i32
        i34 = i29 + i33
        i35 = i34
    else:
        i35 = i29

    i36 = i12 + i5
    tl.store(i36, i35, mask=i7)
