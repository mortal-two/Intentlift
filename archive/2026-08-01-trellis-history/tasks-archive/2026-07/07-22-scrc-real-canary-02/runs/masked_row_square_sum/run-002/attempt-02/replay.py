import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    i0 = p0
    i1 = p1
    i2 = p2
    i3 = c0

    i4 = tl.arange(0, 2)
    i5 = tl.arange(0, i3)
    i6 = i2

    i7 = i5 < i6
    i8 = i4 * i2

    i9 = i0 + i8
    i10 = i9[:, None]
    i11 = i10 + i5[None, :]
    i12 = tl.load(i11, mask=i7[None, :], other=0.0)

    i13 = i12 * i12
    i14 = tl.sum(i13, axis=1)

    i15 = i1 + i4
    tl.store(i15, i14)
