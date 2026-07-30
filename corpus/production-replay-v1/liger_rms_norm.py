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
    p9,
    p10,
    p11: tl.constexpr,
    p12: tl.constexpr,
    c0: tl.constexpr,
):
    v30_i32 = tl.program_id(0)
    v30_i64 = v30_i32.to(tl.int64)
    v31 = tl.arange(0, c0)
    in_bounds = v31 < p8

    a0_base = p2 + v30_i64 * p3
    a0_ptrs = a0_base + v31
    t0_tile = tl.load(a0_ptrs, mask=in_bounds, other=0.0)

    a1_ptrs = p4 + v31
    t1_tile = tl.load(a1_ptrs, mask=in_bounds, other=0.0)

    v0 = -1
    v1 = 0
    v2 = 1
    v3 = p11 == v1
    v4 = p11 == v2
    v5 = p11 == v0

    v6 = t0_tile.to(tl.float32)
    v7 = tl.where(v3, v6, t0_tile)
    v8 = t1_tile.to(tl.float32)
    v9 = tl.where(v4, v8, t1_tile)
    v10 = v7.to(tl.float32)
    v11 = tl.where(v4, v10, v7)
    v12 = p9.to(tl.float32)
    v13 = tl.where(v5, v12, p9)
    v14 = p10.to(tl.float32)
    v15 = tl.where(v5, v14, p10)
    v16 = v11 * v11
    v17 = tl.sum(v16, axis=0)
    v18 = p8.to(tl.float32)
    v19 = v17 / v18
    v20 = v19 + v13
    t8_value = tl.rsqrt(v20)

    a2_ptr = p6 + v30_i64 * p7
    tl.store(a2_ptr, t8_value)

    v22_1 = v11 * t8_value
    v23_1 = v22_1.to(tl.float32)
    v24_1 = tl.where(v4, v23_1, v22_1)
    v25_1 = v15 + v9
    v26_1 = v24_1 * v25_1
    v27_1 = tl.where(p12, v26_1, v24_1)
    v28 = v27_1.to(tl.float32)
    t7_tile = tl.where(v4, v28, v27_1)

    a3_base = p0 + v30_i64 * p1
    a3_ptrs = a3_base + v31
    tl.store(a3_ptrs, t7_tile, mask=in_bounds)
