import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    d0 = v0 < p2

    a0 = p0 + v0
    loaded = tl.load(a0, mask=d0, other=0.0)

    a1 = p1 + v0
    tl.store(a1, loaded, mask=d0)
