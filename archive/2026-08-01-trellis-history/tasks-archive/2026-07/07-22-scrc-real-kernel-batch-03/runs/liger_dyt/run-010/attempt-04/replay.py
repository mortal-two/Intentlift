import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


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
    pid_x = tl.program_id(0)
    x_base = pid_x * c0
    x_lane = tl.arange(0, c0)
    x_index = x_base + x_lane

    pid_row = tl.program_id(1)

    p0_row_base = p0 + pid_row * p6
    p0_block_base = p0_row_base + x_base
    p0_address = p0_block_base + x_lane
    p0_value = tl.load(p0_address, mask=x_index < p6, other=0.0)

    p1_row_base = p1 + pid_row * p6
    p1_block_base = p1_row_base + x_base
    p1_address = p1_block_base + x_lane

    p3_block_base = p3 + x_base
    p3_address = p3_block_base + x_lane
    p3_value = tl.load(p3_address, mask=x_index < p6, other=0.0)

    p4_block_base = p4 + x_base
    p4_address = p4_block_base + x_lane
    p4_value = tl.load(
        p4_address,
        mask=p5 & (x_index < p6),
        other=0.0,
    )

    p2_value = tl.load(p2)

    v0 = p2_value * p0_value
    v1 = libdevice.tanh(v0)
    v2 = v1 * p3_value
    v3 = v2 + p4_value
    v4 = tl.where(p5, v3, v2)

    tl.store(p1_address, v4, mask=x_index < p6)
