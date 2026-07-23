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
    program_tile = tl.program_id(0)
    program_tile_i64 = program_tile.to(tl.int64)
    column_base = program_tile_i64 * c0

    lane = tl.arange(0, c0)
    lane_i64 = lane.to(tl.int64)
    column = column_base + lane_i64
    column_mask = column < p6

    program_row = tl.program_id(1)
    program_row_i64 = program_row.to(tl.int64)

    input_row_base = p0 + program_row_i64 * p6
    output_row_base = p1 + program_row_i64 * p6

    scalar = tl.load(p2)

    scale_ptrs = p3 + column
    scale = tl.load(scale_ptrs, mask=column_mask, other=0.0)

    input_ptrs = input_row_base + column
    value = tl.load(input_ptrs, mask=column_mask, other=0.0)

    n0 = value + scalar
    n1 = tl.maximum(n0, 0.0)
    n2 = n1 * scale

    if p5:
        bias_ptrs = p4 + column
        bias = tl.load(bias_ptrs, mask=column_mask, other=0.0)
        n3 = n2 + bias
        n4 = n3
    else:
        n4 = n2

    output_ptrs = output_row_base + column
    tl.store(output_ptrs, n4, mask=column_mask)
