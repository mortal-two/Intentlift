import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v2 = tl.program_id(0)
    a0_base = p0 + v2 * p2

    v3 = tl.arange(0, c0)[None, :]
    a0_lane = a0_base + v3
    v0_input = tl.load(a0_lane, mask=v3 < p2, other=0.0)

    v0 = v0_input * v0_input
    v1 = tl.sum(v0, axis=1)

    a1_base = p1 + v2
    tl.store(a1_base, v1)
