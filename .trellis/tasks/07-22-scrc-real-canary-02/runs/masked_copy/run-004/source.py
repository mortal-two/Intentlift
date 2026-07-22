import triton
import triton.language as tl

@triton.jit
def source_masked_copy(X, Y, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
