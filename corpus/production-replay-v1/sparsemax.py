import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, p5, p6, c0: tl.constexpr, c1: tl.constexpr):
    v26 = tl.program_id(0)
    v27 = tl.arange(0, c0)

    a0_base = p2 + v26 * p3
    a0_ptr = a0_base + v27
    t2 = tl.load(a0_ptr, mask=v27 < p6, other=-float("inf"))

    a1_base = p0 + v26 * p1
    a1_ptr = a1_base + v27
    t0 = tl.load(a1_ptr, mask=v27 < p6, other=0.0)

    v0_1 = v27
    v1_1 = v0_1 < p6
    v2_1 = tl.full((c0,), 0.0, tl.float32)
    v3_1 = tl.where(v1_1, t2, v2_1)
    v4_1 = tl.cumsum(v3_1, axis=0)
    v5_1 = tl.full((c0,), 1.0, tl.float32)
    v6_1 = v4_1 - v5_1
    v7_1 = tl.full((c0,), 1, tl.int32)
    v8_1 = v0_1 + v7_1
    v9_1 = v8_1.to(tl.float32)
    v10_1 = v6_1 / v9_1
    v11_1 = t2 > v10_1
    v12_1 = v11_1 & v1_1
    v13_1 = v12_1.to(tl.int32)
    v14_1 = tl.sum(v13_1, axis=0)
    v15_1 = 1
    v16_1 = tl.maximum(v14_1, v15_1)
    v17_1 = v16_1.to(tl.float32)
    v18_1 = tl.where(v12_1, t2, v2_1)
    v19_1 = tl.sum(v18_1, axis=0)
    v20_1 = 1.0
    v21_1 = v19_1 - v20_1
    v22_1 = v21_1 / v17_1
    v23_1 = t0 - v22_1
    v24_1 = tl.full((c0,), 0.0, tl.float32)
    t6 = tl.maximum(v23_1, v24_1)

    a2_base = p4 + v26 * p5
    a2_ptr = a2_base + v27
    tl.store(a2_ptr, t6, mask=v27 < p6)
