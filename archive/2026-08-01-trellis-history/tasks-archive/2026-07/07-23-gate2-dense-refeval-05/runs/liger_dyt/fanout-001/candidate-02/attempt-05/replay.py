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
    i3 = i1 * c0
    i4 = i3

    i5 = tl.arange(0, c0)
    i6 = i5.to(tl.int64)
    i7 = i4 + i6

    i8 = tl.program_id(1)
    i9 = i8.to(tl.int64)
    i11 = i9 * p6
    i12 = i11

    lane_mask = i7 < p6

    i13 = p2
    i14 = tl.load(i13)
    i25 = i14.to(tl.float32)

    i15 = p3 + i7
    i16 = tl.load(i15, mask=lane_mask, other=0.0)
    i26 = i16.to(tl.float32)

    i17 = i12
    i18 = p0 + (i17 + i7)
    i19 = tl.load(i18, mask=lane_mask, other=0.0)
    i27 = i19.to(tl.float32)

    i29 = i25 * i27
    i35 = i29 + i29
    i30 = tl.sigmoid(i35)
    i36 = i30 + i30
    i38 = 1.0
    i37 = i36 - i38
    i31 = i37 * i26

    if p5:
        i20 = p4 + i7
        i21 = tl.load(i20, mask=p5 & lane_mask, other=0.0)
        i28 = i21.to(tl.float32)
        i32 = i31 + i28
        i34 = i32
    else:
        i34 = i31

    i22 = i12
    i23 = p1 + (i22 + i7)
    tl.store(i23, i34, mask=lane_mask)
