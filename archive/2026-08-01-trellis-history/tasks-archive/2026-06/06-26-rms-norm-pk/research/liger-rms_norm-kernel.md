# Research: liger rms_norm Triton kernel (forward) — implementable spec for MLIR mirror + PK

- **Query**: Extract a precise, implementable spec of the REAL liger `rms_norm` Triton forward kernel(s) to mirror in MLIR and PK head-to-head.
- **Scope**: internal (read of installed `liger_kernel==0.8.0`, REFERENCE ONLY) + empirical verification on the 5090D.
- **Date**: 2026-06-26

## Environment (verified live)

| Thing | Value |
|---|---|
| liger_kernel | `0.8.0` (pure Python + Triton) |
| triton | `3.6.0` |
| is_hip | `False` (CUDA path) |
| Source file | `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/rms_norm.py` |
| Settings file | `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/utils.py` |
| `rsqrt` source | `triton.language.extra.libdevice.rsqrt` (triton>=3.0.0, non-NPU) — `rms_norm.py:28-36` |
| casting mode ints | `{'llama': 0, 'gemma': 1, 'none': -1}` — `rms_norm.py:402-406` |

> CONSTRAINT: the file under site-packages is READ ONLY. Nothing was installed or modified.

---

## 1. The per-row forward kernel `_rms_norm_forward_kernel` (`rms_norm.py:44-119`)

This is the kernel that fires for **both** PK shapes (see §3). Grid is `(n_rows,)` — **one program/block per row**, BLOCK_SIZE covers the whole row.

```python
@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    n_cols, eps, offset,
    casting_mode: tl.constexpr,        # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)
    """
    row_idx = tl.program_id(0).to(tl.int64)            # one program == one row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols                         # tail mask for non-pow2 n_cols

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)   # masked lanes -> 0
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, then casts the output back to original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols        # block-cooperative reduction
    rstd = rsqrt(mean_square + eps)

    tl.store(rstd_base, rstd)                                   # cache rstd (1 scalar / row)

    X_row = X_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)                           # <-- DOWNCAST before weight mul

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)                        # shared for ALL casting modes
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)
```

Precise mechanics:

- **Grid / block-per-row.** Launched as `[(n_rows,)]` (`rms_norm.py:442`). `tl.program_id(0)` (cast to int64 to avoid stride overflow, line 70) is the row index. Each program owns exactly one row; there is **no inter-program reduction**.
- **Row load.** `x_base = X_ptr + row_idx * X_row_stride`; the row is a single contiguous tile `x_base + col_offsets`, `col_offsets = tl.arange(0, BLOCK_SIZE)` (lines 71, 74-78).
- **BLOCK_SIZE / masking.** `BLOCK_SIZE = next_power_of_2(n_cols) >= n_cols` (see §2), so the whole row fits in one tile. `mask = col_offsets < n_cols` (line 72). For non-power-of-2 `n_cols`, masked lanes load as `other=0` (line 78) so they contribute 0 to the sum-of-squares; the store uses the same mask to avoid OOB writes (line 119). **For the two PK shapes (1024, 4096) BLOCK_SIZE == n_cols, so the mask is all-true and masking is a no-op.**
- **Reduction.** `tl.sum(X_row * X_row, axis=0)` (line 97) is **one block-cooperative reduction over the single whole-row tile** (Triton lowers it to warp-shuffle + shared-mem tree across the block's lanes). Because `BLOCK_SIZE >= n_cols`, one program reduces one full row — this is the load-bearing fact for the MLIR mirror. Not per-thread; not multi-block.
- **Reduction dtype.** For `casting_mode == llama` (=0), `X_row` is cast to fp32 *before* the square/sum (lines 84-85), so `mean_square`, `+eps`, and `rsqrt` all run in **fp32**. For `gemma` (=1) everything (incl. weight) is fp32 (lines 88-91). For `none` (=-1) the reduction runs in the input dtype and `eps`/`offset` are cast down to the input dtype (lines 93-95).
- **eps / rsqrt.** `rstd = rsqrt(mean_square + eps)` (line 98). `eps` is added inside the rsqrt (NOT to the variance-before-mean). `rsqrt` = libdevice rsqrt.
- **Final write.** `X_row = X_row * rstd` (line 105). For llama, `X_row` is **downcast back to the input dtype** (line 109) *before* the weight multiply. Then `Y_row = X_row * (offset + W_row)` (line 112) — **this single line is shared by all casting modes**; the only per-mode difference is the surrounding casts. Gemma downcasts the result at line 117. Store at line 119.
- **`offset` output.** `offset` is a runtime scalar arg (not a tensor); always folded as `(offset + W_row)` at line 112. See §5 for semantics.
- **`RSTD` output.** A second output tensor, shape `(n_rows,)`, holding the per-row `rstd` (the reciprocal RMS, i.e. `1/RMS`, NOT the RMS). Cached for the backward pass (lines 100-103). dtype is fp32 for llama/gemma, else input dtype (`rms_norm.py:425`). The forward kernel *always* allocates and writes it even if you only want `Y`.

---

## 2. `calculate_settings(n_cols)` — the Layer-C schedule policy (`utils.py:45-62`)

```python
def calculate_settings(n):
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

Exact mapping `n_cols -> (BLOCK_SIZE, num_warps)`:

| n_cols range (BLOCK_SIZE = next_pow2(n_cols)) | num_warps (CUDA) |
|---|---|
| BLOCK_SIZE < 2048 | 4 |
| 2048 <= BLOCK_SIZE < 8192 | 8 |
| 8192 <= BLOCK_SIZE < 32768 | 16 |
| BLOCK_SIZE >= 32768 | 32 (16 on HIP) |

- **MAX_FUSED_SIZE cap = 65536.** If `next_power_of_2(n_cols) > 65536` (i.e. `n_cols > 65536`), `calculate_settings` **raises `RuntimeError`** — the row does not fit one block, so liger refuses to launch. This is the hard upper bound on hidden size for this kernel family.
- Verified for PK shapes: `n_cols=1024 -> (1024, 4)`; `n_cols=4096 -> (4096, 8)`.

This is the schedule hint to mirror in Layer-C: pick block/thread count from `next_pow2(n_cols)` with the warp ladder above (warps*32 = thread count → 128 / 256 / 512 / 1024 threads).

---

## 3. `rms_norm_forward(...)` launcher — both branches (`rms_norm.py:409-482`)

Signature (exact):

```python
def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
```

- `casting_mode` accepts either a string (`"llama"`/`"gemma"`/`"none"`) or the int (0/1/-1); strings map via `_str_to_casting_mode` (lines 410-414).
- `X` is reshaped to 2D `(-1, dim)`; `n_rows, n_cols = X.shape` (lines 416-419).
- `BLOCK_SIZE, num_warps = calculate_settings(n_cols)` (line 420).
- `Y` allocated as `(n_rows, n_cols)` input dtype; `RSTD` as `(n_rows,)` fp32 for llama/gemma (lines 422-426).
- `elementwise_affine = (W is not None)` (lines 428-435).

The branch (`rms_norm.py:441`):

```python
if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
    _rms_norm_forward_kernel[(n_rows,)]( ... BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps ... )   # PER-ROW
else:
    BLOCK_ROW = 16
    _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)]( ... BLOCK_ROW=16 ... )    # BLOCKED / MULTI-ROW
```

**Branch A — per-row `_rms_norm_forward_kernel`** (taken when `BLOCK_SIZE > 256` OR `n_rows < 32768` OR `row_mode`): grid `(n_rows,)`, one program per row. Documented in §1.

**Branch B — blocked `_block_rms_norm_forward_kernel`** (`rms_norm.py:215-297`): grid `(cdiv(n_rows, 16),)`, each program handles a `BLOCK_ROW=16`-row tile, reducing along `axis=1`:

```python
@triton.jit
def _block_rms_norm_forward_kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_ROW: tl.constexpr):
    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols
    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :], other=0,
    )                                                          # [BLOCK_ROW, BLOCK_SIZE] 2D tile
    ... (same casting policy) ...
    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols       # per-row reduction along cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)
    X_row = X_row * rstd[:, None]
    if casting_mode == _CASTING_MODE_LLAMA: X_row = X_row.to(X_row_dtype)
    if elementwise_affine: Y_row = X_row * (offset + W_row)[None, :]
    else: Y_row = X_row
    if casting_mode == _CASTING_MODE_GEMMA: Y_row = Y_row.to(X_row_dtype)
    tl.store(Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :], Y_row,
             mask=row_mask[:, None] & col_mask[None, :])
```

**Which branch fires for our PK shapes (verified live):**

| Shape (rows, cols) | BLOCK_SIZE | num_warps | `BLOCK_SIZE>256`? | `n_rows<32768`? | Branch |
|---|---|---|---|---|---|
| (32768, 1024) | 1024 | 4 | True | False | **A — per-row** |
| (16384, 4096) | 4096 | 8 | True | True | **A — per-row** |

Both PK shapes hit the per-row kernel (because `BLOCK_SIZE=1024/4096 > 256`). **Branch B is effectively dead for any realistic transformer hidden size**: it requires `BLOCK_SIZE <= 256` (i.e. `n_cols <= 256`) AND `n_rows >= 32768` AND `row_mode` falsy — a narrow tall-skinny regime. **Mirror branch A only.**

**Return value (6-tuple):** `(Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode)` (`rms_norm.py:482`). Note `Y` is reshaped back to the original input shape; `X` returned is the 2D-viewed contiguous input (saved for backward); `casting_mode` is the resolved int.

---

## 4. Direct kernel call for a fair head-to-head PK (bypassing `LigerRMSNormFunction.apply`)

`LigerRMSNormFunction.apply` references `torch.distributed.tensor.DTensor` (`rms_norm.py:612`), which raises `module 'torch.distributed' has no attribute 'tensor'` on this torch build. **Bypass it entirely** by calling the launcher `rms_norm_forward` directly. Verified working on fp32 CUDA tensors:

```python
import torch, triton
from liger_kernel.ops.rms_norm import rms_norm_forward

rows, cols = 32768, 1024                       # or 16384, 4096
x = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
w = torch.randn(cols,       dtype=torch.float32, device="cuda")
eps = 1e-6

# Launch ONLY the forward kernel. casting_mode="llama" -> 0 internally; row_mode=False; offset=0.0
Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(x, w, eps, 0.0, "llama", False)
# Y: (rows, cols) fp32 result. RSTD: (rows,) fp32 reciprocal-RMS cache (allocated even if unused).

# Standard Triton timing harness for the PK:
ms = triton.testing.do_bench(lambda: rms_norm_forward(x, w, eps, 0.0, "llama", False))
print(f"liger fwd: {ms:.4f} ms")
```

Notes:
- Each call allocates a fresh `Y` `(rows, cols)` and a fresh `RSTD` `(rows,)` fp32 — both count as part of liger's launch cost in the PK. (If you want to exclude alloc, you'd have to refactor; for a fair end-to-end PK, leave it.)
- `casting_mode="llama"` is the apples-to-apples mode for a plain RMSNorm. Pass `offset=0.0`.
- Pass `row_mode=False` to leave the branch decision to the heuristic (which selects per-row for PK shapes anyway).

---

## 5. fp32 reference math (`casting_mode="llama"`) + offset semantics (verified)

For **fp32 input**, llama mode reduces to:

```python
rstd = torch.rsqrt(torch.mean(x.float()**2, dim=-1, keepdim=True) + eps)   # (rows, 1)
y    = (x.float() * rstd) * w.float()                                       # offset=0 for llama
```

Empirically certified against the live kernel (5090D):

| Shape | rel-err `Y` vs torch fp32 ref | rel-err `RSTD` |
|---|---|---|
| (32768, 1024) | 2.07e-7 | 1.09e-7 |
| (16384, 4096) | 1.48e-7 | 1.14e-7 |

Both ~1e-7 = fp32 round-off; the kernel is bit-faithful to the fp32 reference. Use this reference to certify BOTH liger and our MLIR kernel.

**Offset semantics — code-level truth (corrected):** there is **no llama-vs-gemma branch on the offset**. Line 112 `Y_row = X_row * (offset + W_row)` is shared by all casting modes. The only difference is *convention*: llama passes `offset=0.0` (so `(0+w) == w`), gemma passes `offset=1.0` (so `(w+1.0)`). Verified: gemma with `offset=1.0` matches `(x.float()*rstd)*(w.float()+1.0)` at rel-err 8.1e-8. So the reference for any mode is `y = (x_norm) * (offset + w)`; pick `offset` per the model convention.

**fp16/bf16 caveat (does NOT affect this fp32 PK, but document for future):** for llama with a half-precision input, `X_row * rstd` runs in fp32 but is **downcast back to the input dtype at line 109 *before* the weight multiply**. So `y = (x.float()*rstd)*w.float()` is exact only for fp32 input; for fp16/bf16 the correct reference must insert that intermediate downcast: `y = ((x.float()*rstd).to(dtype).float()) * w.float()`. A future half-precision PK using the naive fp32 reference would wrongly flag liger as inexact.

---

## MLIR-mirroring notes (per-row kernel — Branch A, the only one PK uses)

1. **One `gpu.launch` block per row**: grid `(n_rows,)`, block size = `num_warps*32` threads (= `calculate_settings(n_cols)` ladder: 128/256/512/1024); each block owns one full row of `n_cols` (BLOCK_SIZE = next_pow2(n_cols) >= n_cols).
2. **Threads strided over cols**: each thread accumulates a partial sum-of-squares `sum(x_i^2)` over its strided lanes, computing in **fp32** (load input, cast to f32 for llama before squaring); tail lanes (non-pow2 n_cols) contribute 0.
3. **`gpu.all_reduce add`** (block-cooperative, fp32) over the partials to get `sum(x^2)`; then `mean = sum / n_cols`, `rstd = rsqrt(mean + eps)` in fp32 (one thread / broadcast). Optionally write `rstd` to a `(n_rows,)` fp32 RSTD buffer to match liger's output (or skip if PK doesn't need backward — but liger always pays that store).
4. **Second pass write**: each thread reloads/holds its `x_i`, computes `x_i * rstd` (fp32), **downcast to input dtype for llama before the weight mul** (no-op for fp32), then `y_i = (x_i * rstd) * (offset + w_i)`; store with the same tail mask. offset is a scalar (0.0 for llama).
5. **Certify** the MLIR output against `y = (x.float()*rstd)*(offset+w.float())` with `rstd = rsqrt(mean(x.float()**2,-1)+eps)`; target rel-err ~1e-7 (matches liger). PK both at fp32, shapes (32768,1024) and (16384,4096), via `triton.testing.do_bench` vs the equivalent MLIR launch.

## Caveats / Not Found

- Backward kernels (`_rms_norm_backward_kernel` lines 122-212, `_block_rms_norm_backward_kernel` lines 300-399) were read but are out of scope for the forward PK; documented only insofar as they explain why RSTD is cached. The backward `BLOCK_SIZE > 256 or n_rows < 4096*8 or row_mode` branch mirrors forward (line 522) but uses grid `(sm_count,)`.
- `RSTD` holds reciprocal RMS (`1/RMS`), not RMS — confirm if your MLIR caches RMS instead.
- All numbers verified on this machine's CUDA device (5090D) with liger 0.8.0 / triton 3.6.0; warp ladder differs on HIP (32->16) but `is_hip()==False` here.
