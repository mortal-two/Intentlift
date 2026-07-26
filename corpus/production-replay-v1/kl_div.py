import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8: tl.constexpr,
    p9: tl.constexpr,
    c0: tl.constexpr,
):
    d0 = tl.program_id(0)
    d0_wide = d0.to(tl.int64)

    a0_base = p0 + d0_wide * p1
    a1_base = p2 + d0_wide * p3
    a2_base = p4 + d0_wide * p5

    d1_lane = tl.arange(0, c0)
    b0 = 0.0

    for q1 in range(0, p6, c0):
        d1 = q1 + d1_lane

        a0 = a0_base + d1
        v1 = tl.load(a0, mask=d1 < p6, other=0.0)

        a1 = a1_base + d1
        v0 = tl.load(a1, mask=d1 < p6, other=0.0)

        v4_1 = tl.exp(v0)
        v5_1 = v0 - v1
        v6_1 = v4_1 * v5_1
        v7_1 = tl.maximum(v0, p7)
        v8_1 = tl.log(v7_1)
        v9_1 = v8_1 - v1
        v10_1 = v0 * v9_1
        v11_1 = tl.where(p8, v6_1, v10_1)
        v15_1 = tl.sum(v11_1, axis=0)
        b0 = b0 + v15_1

    tl.store(a2_base, b0)
