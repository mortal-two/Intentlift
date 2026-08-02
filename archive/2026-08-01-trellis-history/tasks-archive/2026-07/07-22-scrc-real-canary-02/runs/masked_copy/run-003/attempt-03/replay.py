import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    i0 = tl.program_id(axis=0) * c0 + tl.arange(0, c0)
    i1 = p2
    i2 = i0 < i1

    i3 = p0
    i4 = i3 + i0
    i5 = tl.load(i4, mask=i2, other=0.0)

    i6 = p1
    i7 = i6 + i0
    tl.store(i7, i5, mask=i2)
