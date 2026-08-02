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
    program_x_wide = program_x.to(tl.int64)
    block_base = program_x_wide * c0

    lane = tl.arange(0, c0)
    offsets = block_base + lane
    mask = offsets < p6

    program_y = tl.program_id(1)
    program_y_wide = program_y.to(tl.int64)

    input_base = p0 + program_y_wide * p6
    output_base = p1 + program_y_wide * p6

    tile_ptrs = p2 + lane
    tile_value = tl.load(tile_ptrs)

    scale_ptrs = p3 + offsets
    scale_value = tl.load(scale_ptrs, mask=mask, other=0.0)

    input_ptrs = input_base + offsets
    input_value = tl.load(input_ptrs, mask=mask, other=0.0)

    r0 = input_value + tile_value
    r1 = tl.sigmoid(r0)
    r2 = r1 * scale_value

    if p5:
        branch_ptrs = p4 + offsets
        branch_value = tl.load(branch_ptrs, mask=mask, other=0.0)
        r3 = r2 + branch_value
    else:
        r3 = r2

    output_ptrs = output_base + offsets
    tl.store(output_ptrs, r3, mask=mask)
