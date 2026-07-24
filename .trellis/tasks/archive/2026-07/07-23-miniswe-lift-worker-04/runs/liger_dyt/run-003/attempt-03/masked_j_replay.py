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
    column_program = tl.program_id(0)
    widened_column_program = column_program.to(tl.int64)
    block_column = widened_column_program * c0

    lane = tl.arange(0, c0)
    widened_lane = lane.to(tl.int64)
    column = block_column + widened_lane

    row_program = tl.program_id(1)
    widened_row_program = row_program.to(tl.int64)
    input_row_offset = widened_row_program * p6
    output_row_offset = widened_row_program * p6

    input_row_base = p0 + input_row_offset
    output_row_base = p1 + output_row_offset

    scalar_value = tl.load(p2)

    scale_pointer = p3 + column
    scale_value = tl.load(
        scale_pointer,
        mask=column < p6,
        other=0.0,
    )

    input_pointer = input_row_base + column
    input_value = tl.load(
        input_pointer,
        mask=column < p6,
        other=0.0,
    )

    combined_value = scalar_value + input_value
    activated_value = tl.maximum(combined_value, 0.0)
    scaled_value = activated_value * scale_value

    if p5:
        branch_pointer = p4 + column
        branch_operand = tl.load(
            branch_pointer,
            mask=column < p6,
            other=0.0,
        )
        branch_value = scaled_value + branch_operand
        result_value = branch_value
    else:
        result_value = scaled_value

    output_pointer = output_row_base + column
    tl.store(
        output_pointer,
        result_value,
        mask=column < p6,
    )
