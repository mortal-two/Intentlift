import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, c0: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, c0)

    in_offset = row * p3 + lane
    x = tl.load(p2 + in_offset)

    r2 = tl.max(x)
    r4 = x - r2 + p4
    r5 = tl.sum(r4)

    out_offset = row * p1 + lane
    tl.store(p0 + out_offset, r5)
