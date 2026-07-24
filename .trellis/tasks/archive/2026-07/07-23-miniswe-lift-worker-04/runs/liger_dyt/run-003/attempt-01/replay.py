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
    program_x = tl.program_id(0)
    program_x_i64 = program_x.to(tl.int64)
    tile_base = program_x_i64 * c0

    lane = tl.arange(0, c0)
    lane_i64 = lane.to(tl.int64)
    x = tile_base + lane_i64

    program_row = tl.program_id(1)
    program_row_i64 = program_row.to(tl.int64)
    input_row_offset = program_row_i64 * p6
    output_row_offset = program_row_i64 * p6

    input_row = p0 + input_row_offset
    output_row = p1 + output_row_offset

    scale = tl.load(p2)

    multiplier_ptrs = p3 + x
    multiplier = tl.load(
        multiplier_ptrs,
        mask=(0 <= x) & (x < p6),
        other=0.0,
    )

    input_ptrs = input_row + x
    input_value = tl.load(
        input_ptrs,
        mask=(0 <= x) & (x < p6),
        other=0.0,
    )

    scaled_input = scale * input_value
    activated = tl.tanh(scaled_input)
    multiplied = activated * multiplier

    if p5:
        addend_ptrs = p4 + x
        addend = tl.load(
            addend_ptrs,
            mask=(0 <= x) & (x < p6),
            other=0.0,
        )
        added = multiplied + addend
        selected = added
    else:
        selected = multiplied

    output_ptrs = output_row + x
    tl.store(
        output_ptrs,
        selected,
        mask=(0 <= x) & (x < p6),
    )
