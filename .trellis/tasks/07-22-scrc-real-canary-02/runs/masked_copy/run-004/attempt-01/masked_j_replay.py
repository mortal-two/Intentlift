import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    offsets = tl.arange(0, c0)
    d0 = offsets < p2

    a0 = p0 + offsets
    value = tl.load(a0, mask=d0, other=0.0)

    a1 = p1 + offsets
    tl.store(a1, value, mask=d0)
