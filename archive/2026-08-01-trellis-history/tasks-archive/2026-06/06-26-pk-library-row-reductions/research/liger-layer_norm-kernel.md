# Research: liger layer_norm Triton kernel (forward) — implementable spec for MLIR mirror + head-to-head PK

- **Query**: Extract a precise, implementable spec of the REAL liger layer_norm Triton forward kernel (grid, masking, reductions, dtype, eps, affine, saved Mean/RSTD), how to call the forward launcher directly bypassing autograd, and the fp32 reference math.
- **Scope**: internal (installed `liger_kernel==0.8.0`, pure Python + Triton) + on-GPU verification (5090D)
- **Date**: 2026-06-26

## Source files

| File Path | Description |
|---|---|
| `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/layer_norm.py` | `_layer_norm_forward_kernel` (jit), `layer_norm_forward` launcher, backward, `LigerLayerNormFunction` (autograd) |
| `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/utils.py` | `calculate_settings(n)` (BLOCK_SIZE / num_warps policy + MAX_FUSED_SIZE cap) |

**REFERENCE ONLY** — both live under site-packages; read, never modify. Nothing was installed for this research.

---

## 1. The `@triton.jit` forward kernel — FULL body

Verbatim, `layer_norm.py:26-86`:

```python
@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pre-load weights and bias in fp32 to avoid repeated conversions
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    W_f32 = W_row.to(tl.float32)
    B_f32 = B_row.to(tl.float32)

    # Calculate pointers for this row
    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride
    row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
    row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

    # Load input data and convert to fp32 for numerical stability
    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)

    # Compute statistics in fp32 for numerical stability
    mean = tl.sum(X_f32, axis=0) / n_cols
    X_centered = X_f32 - mean
    # Apply mask to variance calculation to exclude contributions from masked elements
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols
    rstd = rsqrt(var + eps)

    # Store statistics (convert back to original dtype only once)
    tl.store(row_Mean_ptr, mean.to(X_row.dtype))
    tl.store(row_RSTD_ptr, rstd.to(X_row.dtype))

    # Fused normalization and affine transformation
    # Y = (X - mean) * rstd * W + B = X_centered * rstd * W + B
    Y_f32 = X_centered * rstd * W_f32 + B_f32

    # Store output (single conversion back to original dtype)
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)
```

Annotated semantics:

- **Grid shape: one block per row.** `row_idx = tl.program_id(0)` (cast int64 to avoid 32-bit pointer-offset overflow on large tensors), launched with `grid = (n_rows,)` (line 206). Each program processes exactly one row; rows are independent.
- **Row load + masking for non-power-of-2 n_cols.** `col_offsets = tl.arange(0, BLOCK_SIZE)`; `mask = col_offsets < n_cols`. The **entire row is loaded in a single `tl.load`** (line 66) — there is **NO column loop** in the forward kernel. This requires the row to fit in one block, i.e. `BLOCK_SIZE = next_pow2(n_cols) >= n_cols` (see §2). Masked-out lanes (indices `>= n_cols`, present only when n_cols is not a power of two) load `other=0.0`. W and B are loaded the same way (lines 54-55, `other=0.0`).
- **Reductions — EXACTLY two passes, plain arithmetic (NOT Welford, NOT one-pass).**
  1. **Mean pass** (line 70): `mean = tl.sum(X_f32, axis=0) / n_cols`. Block-cooperative `tl.sum` over the BLOCK_SIZE-wide vector, fp32 accumulation, divided by `n_cols` (the REAL column count, not BLOCK_SIZE — masked lanes contributed 0.0 so they don't bias the sum).
  2. **Variance pass** (lines 71-74): `X_centered = X_f32 - mean`; then a masked square-sum `var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols`. Again block-cooperative fp32 `tl.sum`, divided by `n_cols`. This is **population (biased) variance** (divide by N, not N-1) — matches `torch.var(..., unbiased=False)`.
- **The variance masking trick (line 73) — the one non-obvious correctness detail.** `X_centered_masked = tl.where(mask, X_centered, 0.0)`. It is required because `X_centered = X_f32 - mean` is **nonzero on masked lanes** (equals `0.0 - mean = -mean`) even though X was loaded with `other=0.0`. Without re-zeroing, `(-mean)^2` from each padding lane would be added into the variance sum and inflate `var`. The mean pass needed no such fix because masked X is genuinely 0.0 and adds nothing to the sum.
- **dtype.** All statistics computed in fp32 for numerical stability (X, W, B each cast to fp32 up front). `eps` added in fp32.
- **eps handling.** `rstd = rsqrt(var + eps)` (line 75) — eps added to variance **before** the reciprocal-sqrt. `rsqrt` is imported from libdevice (`layer_norm.py:15-23`: `from triton.language.extra.libdevice import rsqrt`, falling back to `...extra.cuda.libdevice`, or `triton.language.math` for old triton / NPU).
- **Saved outputs Mean and RSTD.** Stored once per row (lines 78-79), cast back to `X_row.dtype` (the input dtype). For fp32 input these stay fp32. Mean/RSTD are allocated in **X.dtype, not fp32** (lines 190-191) — this differs from the backward path, whose `_DW/_DB` accumulators are fp32. They are consumed by the backward kernel.
- **Final affine write.** `Y_f32 = X_centered * rstd * W_f32 + B_f32` (line 83), i.e. `y = (x - mean) * rstd * weight + bias`. Note **both weight AND bias** participate (unlike rms_norm, which has weight only and no bias, no mean subtraction). Result cast to `X_row.dtype` and stored with `mask` (line 86) so padding lanes are not written.

---

## 2. `calculate_settings(n)` → (BLOCK_SIZE, num_warps)

Verbatim, `utils.py:45-62`:

```python
def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/.../unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps
```

Policy:
- `BLOCK_SIZE = next_power_of_2(n_cols)` (so `BLOCK_SIZE >= n_cols` always; the single un-looped row load fits).
- **MAX_FUSED_SIZE = 65536 cap.** If `next_pow2(n_cols) > 65536` it **raises RuntimeError** — there is **NO large-n_cols / column-loop fallback path** in the forward kernel. Max supported feature dim = 65536.
- num_warps ladder (by BLOCK_SIZE): `>=32768 -> 32` (16 on HIP); `>=8192 -> 16`; `>=2048 -> 8`; else `4`.
- Observed on the 5090D: n_cols=4096 -> BLOCK_SIZE=4096, num_warps=8; n_cols=3000 -> BLOCK_SIZE=4096, num_warps=8.

---

## 3. The forward launcher

Verbatim signature + launch, `layer_norm.py:169-227`:

```python
def layer_norm_forward(X, W, B, eps):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    if X.shape[1] != W.shape[0]:
        raise ValueError(...)  # input feature size must match weight size

    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    grid = (n_rows,)
    _layer_norm_forward_kernel[grid](
        Y, Y.stride(0),
        X, X.stride(0),
        W, W.stride(0),
        B, B.stride(0),
        Mean, Mean.stride(0),
        RSTD, RSTD.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,
    )

    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps
```

- **Signature:** `layer_norm_forward(X, W, B, eps)`. X is reshaped to 2D `(-1, dim)`; the leading dims are flattened into rows.
- **grid** = `(n_rows,)` — one program per row.
- **BLOCK_SIZE / num_warps** = `calculate_settings(n_cols)`.
- **Allocations:** Y `(n_rows, n_cols)` in X.dtype; Mean `(n_rows,)` and RSTD `(n_rows,)` in **X.dtype** (not fp32).
- **Return value (6-tuple):** `(Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps)`. `Y` (index 0) is already reshaped back to the original input shape; `X` (index 1) is the 2D-viewed contiguous input saved for backward.
- `kernel_args` only carries XPU GRF tuning; on CUDA it is empty.

The autograd wrapper `LigerLayerNormFunction.forward` (line 310) merely calls this and saves `(X, W, B, Mean, RSTD)`. **For PK we skip `.apply` entirely** (see §4) since `Function.apply` crashes on this torch with `module 'torch.distributed' has no attribute 'tensor'`.

---

## 4. Calling the forward kernel directly for a fair head-to-head PK (VERIFIED on GPU)

Use the module-level `layer_norm_forward` launcher directly (analogous to liger's `rms_norm_forward`). This runs ONLY the forward kernel — no autograd graph, no `Function.apply`. Minimal snippet (verified to run on the 5090D — runs the real kernel and reads output back):

```python
import torch
import torch.nn.functional as F
from liger_kernel.ops.layer_norm import layer_norm_forward

rows, cols, eps = 4, 4096, 1e-5
x = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
w = torch.randn(cols, dtype=torch.float32, device="cuda")   # weight, shape (cols,)
b = torch.randn(cols, dtype=torch.float32, device="cuda")   # bias,   shape (cols,)

# Forward-only call. Returns a 6-tuple; Y (index 0) is reshaped to x.shape.
Y, X_out, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(x, w, b, eps)
# Y    : (rows, cols) fp32 -- the normalized+affine output (read this back for PK)
# X_out: (rows, cols) fp32 -- 2D view of input (autograd bookkeeping, ignore for fwd PK)
# Mean : (rows,) fp32      -- per-row mean   (EXTRA allocated output)
# RSTD : (rows,) fp32      -- per-row rsqrt(var+eps) (EXTRA allocated output)
# BLOCK_SIZE, num_warps    -- launch config actually used
```

**Extra allocated outputs to be aware of:** the launcher unconditionally allocates and writes **Mean `(n_rows,)`** and **RSTD `(n_rows,)`** (both X.dtype). A fair PK should account for these two extra global writes per row, OR have the MLIR mirror also emit per-row mean+rstd to match the memory-traffic profile.

**Verified results (5090D, fp32, seed 0):**
- return tuple length 6; `Y.shape == (4, 4096)` fp32; `Mean/RSTD.shape == (4,)` fp32.
- cols=4096 -> BLOCK_SIZE=4096, num_warps=8.
- `torch.allclose(Y, F.layer_norm(x.float(),(cols,),w.float(),b.float(),eps))` -> **True**, max abs err **9.54e-7** (single-ulp fp32 rounding).
- `Mean` matches `x.mean(dim=1)`; `RSTD` matches `torch.rsqrt(x.var(dim=1, unbiased=False)+eps)` -> both True.
- Non-power-of-2 cols=3000 -> BLOCK_SIZE=4096; allclose True, max err 9.54e-7 (masking path correct).

---

## 5. Exact fp32 reference math to certify against torch

```python
import torch.nn.functional as F
ref = F.layer_norm(x.float(), (n_cols,), weight.float(), bias.float(), eps)
# certify: torch.allclose(Y, ref, rtol=1e-4, atol=1e-5)   # passes, max err ~1e-6 in fp32
```

Equivalent explicit math (population/biased variance, divide by N):

```python
mean = x.float().mean(dim=-1, keepdim=True)
var  = x.float().var(dim=-1, unbiased=False, keepdim=True)   # /N, NOT /(N-1)
rstd = torch.rsqrt(var + eps)
ref  = (x.float() - mean) * rstd * weight.float() + bias.float()
```

Both forms agree with the kernel: mean over feature dim, biased variance, `eps` added before rsqrt, affine `*W + B`.

---

## MLIR-mirroring notes (5 lines)

1. **Grid:** one `gpu.launch` block per row (grid = n_rows blocks); threads strided over the `n_cols` feature columns, accumulating in **fp32** registers.
2. **Two reduction passes in fp32:** pass 1 `mean = (gpu.all_reduce add of x over the row) / n_cols`; pass 2 `var = (gpu.all_reduce add of (x-mean)^2) / n_cols` (population variance, divide by real n_cols).
3. **Normalize:** `rstd = rsqrt(var + eps)` (eps added before rsqrt; use libdevice/`math.rsqrt` equivalent); optionally write per-row `mean` and `rstd` to match liger's extra global outputs for a fair memory-traffic PK.
4. **Affine write pass:** second strided loop `y[c] = (x[c] - mean) * rstd * weight[c] + bias[c]` (both weight AND bias, unlike rms_norm); cast to output dtype on store.
5. **Divergence from Triton:** liger needs the `tl.where(mask, X-mean, 0)` trick only because its power-of-2 BLOCK_SIZE over-reads padding lanes; an MLIR kernel that bounds its thread loop at `n_cols` never touches lanes >= n_cols, so it needs **no masking trick** for variance — it simply iterates exactly the real columns.

## Caveats

- Only the **forward** kernel was specced in depth (PK target). Backward (`_layer_norm_backward_kernel`, lines 89-166; `layer_norm_backward`, lines 230-304) exists but is out of scope; note it uses an SM-count grid with `rows_per_program` row-loop and fp32 `_DW/_DB` partial-accumulator buffers reduced afterward.
- liger_kernel under site-packages is reference-only; not modified. Nothing installed.
