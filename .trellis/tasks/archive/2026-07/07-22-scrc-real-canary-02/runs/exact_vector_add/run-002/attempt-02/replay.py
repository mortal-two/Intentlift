import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    i0: tl.constexpr = c0
    v1 = tl.arange(0, i0)
    d0 = (v1 >= 0) & (v1 < 128)

    i1 = p0
    i2 = i1 + v1
    i3 = tl.load(i2, mask=d0)

    i4 = p1
    i5 = i4 + v1
    i6 = tl.load(i5, mask=d0)

    i7 = p2
    i8 = i7 + v1

    i9 = i3 + i6
    tl.store(i8, i9, mask=d0)
