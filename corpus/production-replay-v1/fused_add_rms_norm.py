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
    p12, p13, p14,
    p15: tl.constexpr,
    c0: tl.constexpr,
):
    v13 = tl.program_id(0).to(tl.int64)
    v14 = tl.arange(0, c0)

    a5_base = p0 + v13 * p1
    a2_base = p2 + v13 * p3
    a0_base = p4 + v13 * p5
    a1_base = p6 + v13 * p7
    a4_ptr = p10 + v13 * p11

    mask = v14 < p12

    a0_ptr = a0_base + v14
    x0 = tl.load(a0_ptr, mask=mask, other=0.0)

    a1_ptr = a1_base + v14
    x1 = tl.load(a1_ptr, mask=mask, other=0.0)

    t8 = x0 + x1

    a2_ptr = a2_base + v14
    tl.store(a2_ptr, t8, mask=mask)

    a3_ptr = p8 + v14
    t2 = tl.load(a3_ptr, mask=mask, other=0.0)

    v1_1 = t8 * t8
    v2_1 = tl.sum(v1_1, axis=0)
    v3_1 = p12.to(tl.float32)
    v4_1 = v2_1 / v3_1

    v5_1 = -1
    v6_1 = p15 == v5_1
    v7_1 = tl.where(v6_1, p13, p13)

    v8_1 = v4_1 + v7_1
    t9 = tl.rsqrt(v8_1)
    tl.store(a4_ptr, t9)

    v10 = t8 * t9
    v11 = p14 + t2
    t7 = v10 * v11

    a5_ptr = a5_base + v14
    tl.store(a5_ptr, t7, mask=mask)
