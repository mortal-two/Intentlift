import triton
import triton.language as tl

@triton.jit
def source_exact_vector_add(X, Y, Out, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    left = tl.load(X + offsets)
    right = tl.load(Y + offsets)
    tl.store(Out + offsets, left + right)
