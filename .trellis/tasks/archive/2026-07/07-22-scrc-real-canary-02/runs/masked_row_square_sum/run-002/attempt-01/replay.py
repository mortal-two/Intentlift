import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v2 = tl.arange(0, 2)
    v3 = tl.arange(0, c0)

    a0_base = p0 + v2 * p2
    a0_lane = a0_base[:, None] + v3[None, :]
    a0_value = tl.load(
        a0_lane,
        mask=v3[None, :] < p2,
        other=0.0,
    )

    v0 = a0_value * a0_value
    v1 = tl.sum(v0, axis=1)

    a1_base = p1 + v2
    tl.store(a1_base, v1)
