import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0, p1, p2,
    p3, p4, p5, p6, p7, p8,
    p9, p10, p11, p12,
    c0: tl.constexpr, c1: tl.constexpr,
):
    d4_program = tl.program_id(0)
    d1_program = tl.program_id(1)

    d0_index = d1_program // p9
    d1_index = d1_program % p9

    d4_base = d4_program * c0
    d4_lane = d4_base + tl.arange(0, c0)
    reduction_lane = tl.arange(0, c1)

    p0_base_0 = p0 + d0_index * p3
    p0_base_1 = p0_base_0 + d1_index * p4
    p0_lane_0 = p0_base_1 + d4_lane[:, None] * p5
    p0_lane_1 = p0_lane_0 + reduction_lane[None, :]
    load_mask = (d4_lane[:, None] < p10) & (reduction_lane[None, :] < p12)
    lhs = tl.load(p0_lane_1, mask=load_mask, other=0.0)
    lhs_f32 = lhs.to(tl.float32)

    p1_base_0 = p1 + d0_index * p6
    p1_base_1 = p1_base_0 + d1_index * p7
    p1_lane_0 = p1_base_1 + d4_lane[:, None] * p8
    p1_lane_1 = p1_lane_0 + reduction_lane[None, :]
    rhs = tl.load(p1_lane_1, mask=load_mask, other=0.0)
    rhs_f32 = rhs.to(tl.float32)

    products = lhs_f32 * rhs_f32
    reduced = tl.sum(products, axis=1)

    p2_base = p2 + d1_program * p11
    p2_lane = p2_base + d4_lane
    tl.store(p2_lane, reduced)
