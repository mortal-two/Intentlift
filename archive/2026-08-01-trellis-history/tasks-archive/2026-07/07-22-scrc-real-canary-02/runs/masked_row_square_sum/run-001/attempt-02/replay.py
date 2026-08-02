import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    i0 = p0
    i1 = p1
    i2 = p2

    i3 = tl.arange(0, 2)
    i4 = c0

    i5 = i3[:, None] * i2
    i6 = i0 + i5

    i7_lane = tl.arange(0, i4)[None, :]
    i7 = i6 + i7_lane

    i8 = tl.load(i7, mask=i7_lane < i2, other=0.0)

    v0 = i8 * i8
    v1 = tl.sum(v0, axis=1)

    i9 = i1 + i3
    tl.store(i9, v1)
