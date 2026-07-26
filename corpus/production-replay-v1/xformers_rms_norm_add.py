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
    p6: tl.constexpr,
    p7: tl.constexpr,
    c0: tl.constexpr,
):
    program_row = tl.program_id(0)
    row_base = program_row * p5
    lane = tl.arange(0, c0)

    p0_base = p0 + row_base
    p1_base = p1 + row_base
    p2_base = p2 + row_base

    p0_lane = p0_base + lane
    p1_lane = p1_base + lane
    p2_lane = p2_base + lane
    p3_lane = p3 + lane

    lane_mask = lane < p6

    lhs = tl.load(
        p0_lane,
        mask=lane_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    rhs = tl.load(p1_lane, mask=lane_mask, other=0.0)
    added = lhs + rhs
    tl.store(p0_lane, added, mask=lane_mask)

    squared = added * added
    square_sum = tl.sum(squared, axis=0)
    normalized_sum = square_sum / p6
    shifted_sum = normalized_sum + p4
    inverse_root = tl.rsqrt(shifted_sum)

    updated = tl.load(
        p0_lane,
        mask=lane_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    normalized = updated * inverse_root
    scale = tl.load(p3_lane, mask=lane_mask)
    scaled = normalized * scale
    selected = tl.where(p7, scaled, normalized)
    tl.store(p2_lane, selected, mask=lane_mask)
