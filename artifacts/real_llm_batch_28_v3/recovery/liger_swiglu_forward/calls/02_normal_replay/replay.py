import triton
import triton.language as tl


@triton.jit
def scrc_k0(
    p0,
    p1,
    p2,
    p3,
    c0: tl.constexpr,
    c1: tl.constexpr,
):
    block_size: tl.constexpr = 1024
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < 131072

    x = tl.load(p0 + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(p1 + offsets, mask=mask, other=0.0)

    activated = (x * tl.sigmoid(x)).to(tl.float16)
    result = activated * y

    tl.store(p2 + offsets, result, mask=mask)
