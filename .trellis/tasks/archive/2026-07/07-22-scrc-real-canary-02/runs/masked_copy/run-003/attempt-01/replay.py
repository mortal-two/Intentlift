import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    d0 = tl.program_id(0) * c0 + tl.arange(0, c0)
    x0 = d0 < p2

    p0_base = p0
    a0 = p0_base + d0
    value = tl.load(a0, mask=x0, other=0.0)

    p1_base = p1
    a1 = p1_base + d0
    tl.store(a1, value, mask=x0)
