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
    i8 = tl.program_id(0)
    i9 = tl.cast(i8, tl.int64)
    i10 = i9 * c0
    i11 = tl.arange(0, c0)
    i12 = i10 + tl.cast(i11, tl.int64)
    i13 = i12 < p6

    i14 = tl.program_id(1)
    i15 = tl.cast(i14, tl.int64)
    i16 = p0 + i15 * p6
    i17 = p1 + i15 * p6

    i18 = tl.load(p2)
    i19 = i18

    i20 = p3 + i12
    i21 = tl.load(i20, mask=i13, other=0.0)
    i22 = i21

    i23 = i16 + i12
    i24 = tl.load(i23, mask=i13, other=0.0)
    i25 = i24

    i26 = i19 * i25
    i27 = 2.0
    i28 = i26 * i27
    i29 = tl.sigmoid(i28)
    i30 = i29 * i27
    i31 = 1.0
    i32 = i30 - i31
    i33 = i32 * i22

    if p5:
        i35 = p4 + i12
        i36 = tl.load(i35, mask=i13, other=0.0)
        i37 = i36
        i38 = i33 + i37
        i39 = i38
    else:
        i39 = i33

    i40 = i17 + i12
    tl.store(i40, i39, mask=i13)
