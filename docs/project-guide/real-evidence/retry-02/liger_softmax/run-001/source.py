import triton
import triton.language as tl

@triton.jit
def _softmax_single_block_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float('inf'), cache_modifier='.ca')
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask, cache_modifier='.cs')
