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
    p8,
    p9: tl.constexpr,
    p10,
    p11: tl.constexpr,
    p12,
    p13: tl.constexpr,
    c0: tl.constexpr,
):
    v36 = tl.program_id(0)
    v36_i64 = v36.to(tl.int64)

    a0_base = p0 + v36_i64 * p1
    a1_base = p2 + v36_i64 * p3
    a2_addr = p8 + v36_i64
    v_t2 = tl.load(a2_addr)
    a3_base = p4 + v36_i64 * p5
    a4_base = p6 + v36_i64 * p7

    for v37_base in range(0, p12, c0):
        v37 = v37_base + tl.arange(0, c0)
        v37_mask = v37 < p12

        a0_addr = a0_base + v37
        v_t0 = tl.load(a0_addr, mask=v37_mask, other=-float("inf"))
        a1_addr = a1_base + v37
        v_t1 = tl.load(a1_addr, mask=v37_mask, other=-float("inf"))

        v4_1 = tl.max(v_t0, axis=0)
        v5_1 = tl.max(v_t1, axis=0)
        v6_1 = tl.maximum(v4_1, v5_1)
        v7_1 = v_t0 - v6_1
        v8_1 = v_t1 - v6_1
        v9_1 = tl.exp(v6_1)
        v10_1 = tl.exp(v7_1)
        v11_1 = v10_1 * v9_1
        v12_1 = tl.exp(v8_1)
        v13_1 = v12_1 * v9_1
        v14_1 = p9 * v13_1
        v15_1 = 1.0
        v16_1 = v15_1 - p9
        v17_1 = v16_1 * v11_1
        v18_1 = v14_1 + v17_1
        v19_1 = tl.log(v18_1)
        v20_1 = v14_1 * v_t1
        v21_1 = v17_1 * v_t0
        v22_1 = v20_1 + v21_1
        v23_1 = v18_1 * v19_1
        v24_1 = v22_1 - v23_1
        v25_1 = v_t0 - v19_1
        v26_1 = v17_1 * v25_1
        v27_1 = p10.to(tl.float32)
        v28 = 1.0
        v29 = v28 / v27_1
        v30_1 = v24_1 * v29
        v31_1 = v26_1 * v29
        v32_1 = v_t2 == p11
        v33_1 = 0.0
        v34 = tl.where(v32_1, v33_1, v31_1)
        v35 = tl.where(p13, v34, v31_1)

        a3_addr = a3_base + v37
        tl.store(a3_addr, v30_1, mask=v37_mask)
        a4_addr = a4_base + v37
        tl.store(a4_addr, v35, mask=v37_mask)
