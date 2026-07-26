import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
    c0: tl.constexpr,
):
    # One program realizes one row of each rank-two logical tensor.
    v42_i32 = tl.program_id(0)
    v42 = v42_i32.to(tl.int64)
    v43 = tl.arange(0, c0)

    # Preserve the separate program-uniform base and lane address stages.
    a0_base = p2 + v42 * p3
    a0_lane = a0_base + v43
    x = tl.load(a0_lane, mask=v43 < p8, other=0.0)

    a1_lane = p4 + 0
    scale0 = tl.load(a1_lane)
    a2_lane = p4 + 1
    scale1 = tl.load(a2_lane)
    a3_lane = p4 + 2
    scale2 = tl.load(a3_lane)
    bias = tl.load(p5)

    v7 = x * x
    v8 = v7 * x
    v9 = v8 * v8
    v10 = tl.sum(v9, axis=0)

    v11 = p8.to(tl.float32)
    v12 = v10 / v11
    v13 = v12 + p9
    v14 = tl.rsqrt(v13)
    v15 = v8 * v14

    v16 = v7 * v7
    v17 = tl.sum(v16, axis=0)
    v18 = v17 / v11
    v19 = v18 + p9
    v20 = tl.rsqrt(v19)
    v21 = v7 * v20

    v22 = tl.sum(v7, axis=0)
    v23 = v22 / v11
    v24 = v23 + p9
    v25 = tl.rsqrt(v24)

    # The first output term precedes the three distinct direct statistic stores.
    v35 = x * v25
    v36 = scale0 * v15

    a5_base = p6 + v42 * p7
    a5_lane = a5_base + 0
    tl.store(a5_lane, v14)
    a6_lane = a5_base + 1
    tl.store(a6_lane, v20)
    a7_lane = a5_base + 2
    tl.store(a7_lane, v25)

    v37 = scale1 * v21
    v38 = v36 + v37
    v39 = scale2 * v35
    v40 = v38 + v39
    v41 = v40 + bias

    a8_base = p0 + v42 * p1
    a8_lane = a8_base + v43
    tl.store(a8_lane, v41, mask=v43 < p8)
