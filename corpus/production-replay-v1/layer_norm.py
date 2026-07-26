import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0, p1,
    p2, p3,
    p4, p5,
    p6, p7,
    p8, p9,
    p10, p11,
    p12, p13,
    c0: tl.constexpr,
):
    v16 = tl.program_id(0).to(tl.int64)
    v17 = tl.arange(0, c0)

    i4 = p4 + v17
    i5 = tl.load(i4, mask=v17 < p12, other=0.0)

    i7 = p6 + v17
    i8 = tl.load(i7, mask=v17 < p12, other=0.0)

    i11 = v16 * p3
    i12 = p2 + i11
    i13 = i12 + v17
    i14 = tl.load(i13, mask=v17 < p12, other=0.0)

    v2 = tl.sum(i14, axis=0)
    v3 = p12.to(tl.float32)
    t5 = v2 / v3

    i17 = v16 * p9
    i18 = p8 + i17
    tl.store(i18, t5)

    v5_1 = i14 - t5

    v0 = tl.arange(0, c0)
    v1 = v0 < p12
    v6_1 = tl.zeros((c0,), tl.float32)
    v7_1 = tl.where(v1, v5_1, v6_1)
    v8_1 = v7_1 * v7_1
    v9_1 = tl.sum(v8_1, axis=0)
    v10_1 = v9_1 / v3
    v11_1 = p13 + v10_1
    t6 = tl.rsqrt(v11_1)

    i22 = v16 * p11
    i23 = p10 + i22
    tl.store(i23, t6)

    v13 = v5_1 * t6
    v14 = v13 * i5
    t7 = v14 + i8

    i27 = v16 * p1
    i28 = p0 + i27
    i29 = i28 + v17
    tl.store(i29, t7, mask=v17 < p12)
