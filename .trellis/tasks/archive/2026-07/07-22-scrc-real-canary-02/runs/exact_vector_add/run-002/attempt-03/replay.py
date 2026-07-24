import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    i0 = tl.arange(0, c0)

    i1 = p0
    i2 = i1 + i0
    i3 = tl.load(i2)

    i4 = p1
    i5 = i4 + i0
    i6 = tl.load(i5)

    i7 = p2
    i8 = i7 + i0

    i9 = i3 + i6
    tl.store(i8, i9)
