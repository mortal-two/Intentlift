import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0,
    p1,
    p2,
    c0: tl.constexpr,
):
    i4 = tl.program_id(0)
    i5 = tl.arange(0, c0)
    i6 = tl.broadcast_to(p2, (c0,))
    i7 = i5 < i6
    i8 = i4 * p2

    i9 = p0 + i8
    i10 = tl.broadcast_to(i9, (c0,))
    i11 = i10 + i5
    i12 = tl.zeros((c0,), dtype=tl.float32)
    i13 = tl.load(i11, mask=i7, other=i12)

    i14 = i13 * i13
    i15 = tl.sum(i14, axis=0)

    i16 = p1 + i4
    tl.store(i16, i15)
