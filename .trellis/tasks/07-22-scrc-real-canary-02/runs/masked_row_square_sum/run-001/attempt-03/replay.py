import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    d0 = tl.arange(0, 2)
    d1 = tl.arange(0, c0)

    a0_mask = d1 < p2
    a0_row_offset = d0 * p2

    a0_base = p0 + a0_row_offset[:, None]
    a0_lane = a0_base + d1[None, :]
    t0 = tl.load(a0_lane, mask=a0_mask[None, :], other=0.0)

    v0 = t0 * t0
    v1 = tl.sum(v0, axis=1)

    a1_base = p1 + d0
    tl.store(a1_base, v1)
