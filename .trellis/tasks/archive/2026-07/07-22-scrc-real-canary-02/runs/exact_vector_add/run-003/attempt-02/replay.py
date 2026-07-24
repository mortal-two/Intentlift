import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v1 = tl.arange(0, c0)

    a0 = p0 + v1
    x0 = tl.load(a0)

    a1 = p1 + v1
    x1 = tl.load(a1)

    v0 = x0 + x1

    a2 = p2 + v1
    tl.store(a2, v0)
