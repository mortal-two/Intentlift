import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0,
    p1,
    p2,
    c0: tl.constexpr,
):
    schedule = c0
    v1 = tl.program_id(0) * schedule + tl.arange(0, schedule)
    predicate = (0 <= v1) & (v1 < 128)

    p0_base = p0
    p0_lane = p0_base + v1
    t0_value = tl.load(p0_lane, mask=predicate)

    p1_base = p1
    p1_lane = p1_base + v1
    t1_value = tl.load(p1_lane, mask=predicate)

    p2_base = p2
    p2_lane = p2_base + v1

    v0 = t0_value + t1_value
    tl.store(p2_lane, v0, mask=predicate)
