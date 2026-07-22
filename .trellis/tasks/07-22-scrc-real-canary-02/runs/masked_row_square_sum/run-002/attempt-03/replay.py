import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    i0 = p0
    i1 = p1
    i2 = p2

    i3 = tl.arange(0, 2)
    i4 = c0
    i5 = i2

    i6 = tl.arange(0, i4) < i5
    i7 = i3 * i2

    i8 = i0 + i7
    i9 = tl.expand_dims(i8, 1)
    i10 = i9 + tl.arange(0, i4)
    i11 = tl.load(i10, mask=i6, other=0.0)

    i12 = i11 * i11
    i13 = tl.sum(i12, axis=1)

    i14 = i1 + i3
    tl.store(i14, i13)
