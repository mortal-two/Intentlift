import triton
import triton.language as tl

@triton.jit
def source_masked_row_square_sum(X, Y, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + row * N + offsets, mask=mask, other=0.0)
    squared = value * value
    total = tl.sum(squared, axis=0)
    tl.store(Y + row, total)
