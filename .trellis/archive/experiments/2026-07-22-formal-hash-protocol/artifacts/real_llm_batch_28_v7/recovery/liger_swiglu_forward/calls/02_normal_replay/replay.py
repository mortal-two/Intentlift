import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4: tl.constexpr, c0: tl.constexpr):
    row = tl.program_id(0)
    row_offset = row * p3

    base0 = p0 + row_offset
    base1 = p1 + row_offset
    base2 = p2 + row_offset

    cols = tl.arange(0, c0)
    mask = cols < 1024

    addr0 = base0 + cols
    x0 = tl.load(addr0, mask=mask, other=0.0)

    v0 = x0.to(tl.float32)
    v1 = tl.sigmoid(v0)
    v2 = v0 * v1
    v3 = v2.to(tl.float16)

    addr1 = base1 + cols
    x1 = tl.load(addr1, mask=mask, other=0.0)
    v4 = v3 * x1

    addr2 = base2 + cols
    tl.store(addr2, v4, mask=mask)
