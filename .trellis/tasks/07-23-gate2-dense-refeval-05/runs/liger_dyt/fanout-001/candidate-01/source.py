import operator
import triton
import triton.language as tl
from liger_kernel.ops.utils import compare_version
from liger_kernel.utils import is_npu_available
if compare_version('triton', operator.ge, '3.0.0') and (not is_npu_available()):
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh

@triton.jit
def _dyt_fwd_kernel(X, Y, Alpha, Gamma, Beta, HAVE_BETA: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    row_id = tl.cast(tl.program_id(1), tl.int64)
    X += row_id * N
    Y += row_id * N
    alpha = tl.load(Alpha).to(tl.float32)
    gamma = tl.load(Gamma + col, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + col, mask=mask, other=0.0).to(tl.float32)
    tanh_x = tanh(alpha * x)
    y = tanh_x * gamma
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=mask, other=0.0).to(tl.float32)
        y += beta
    tl.store(Y + col, y, mask=mask)
