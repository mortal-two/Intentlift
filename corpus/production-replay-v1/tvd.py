import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0, p1, p2, p3, p4, p5, p6, p7, p8,
    p9: tl.constexpr, p10, p11, p12: tl.constexpr,
    p13: tl.constexpr, c0: tl.constexpr,
):
    v18_i32 = tl.program_id(0)
    v18_i64 = v18_i32.to(tl.int64)
    lane = tl.arange(0, c0)

    row_offset0 = v18_i64 * p1
    base0 = p0 + row_offset0
    row_offset1 = v18_i64 * p3
    base1 = p2 + row_offset1
    row_offset9 = v18_i64 * p7
    base9 = p6 + row_offset9
    row_offset8 = v18_i64 * p5
    base8 = p4 + row_offset8

    for loop_offset in range(0, p10, c0):
        v19 = loop_offset + lane
        in_bounds = v19 < p10

        ptr0 = base0 + v19
        x = tl.load(ptr0, mask=in_bounds, other=0.0)
        ptr1 = base1 + v19
        y = tl.load(ptr1, mask=in_bounds, other=0.0)

        v4_1 = x - y
        v5_1 = tl.abs(v4_1)
        v6_1 = 0.5
        v7_1 = v6_1 * v5_1
        v8_1 = x > y
        v9 = v6_1 * p11
        v10 = -0.5
        v11_1 = v10 * p11
        v12_1 = tl.where(v8_1, v9, v11_1)

        effect_predicate = tl.full((), p12, tl.int1)
        if effect_predicate:
            conditional_value = tl.load(p8 + v18_i64)
            v13_1 = conditional_value == p9
        else:
            pass

        v14 = tl.where(p12, v12_1, v12_1)
        v15 = 0
        v16 = p13 == v15
        v17 = tl.where(v16, v7_1, v7_1)

        ptr9 = base9 + v19
        tl.store(ptr9, v14, mask=in_bounds)
        ptr8 = base8 + v19
        tl.store(ptr8, v17, mask=in_bounds)
