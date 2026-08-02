import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    d0 = tl.program_id(0)
    d1 = tl.arange(0, c0)

    a0_base = p0 + d0 * p2
    a0_lane = a0_base + d1
    loaded = tl.load(a0_lane, mask=d1 < p2, other=0.0)

    n0 = loaded * loaded
    n1 = tl.sum(n0, axis=0)

    a1_base = p1 + d0
    tl.store(a1_base, n1)
