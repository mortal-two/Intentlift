# Research: Liger Softmax Triton Kernel (forward) — implementable spec for MLIR mirror + head-to-head PK

- **Query**: Extract a precise, implementable spec of the real liger softmax Triton forward kernel so we can mirror it as an MLIR kernel and PK it head-to-head.
- **Scope**: internal (installed site-packages source) + empirical verification (ran python, no installs)
- **Date**: 2026-06-26
- **liger_kernel version**: 0.8.0 (pure Python + Triton)
- **Environment verified on**: torch 2.10.0+cu130, triton 3.6.0, NVIDIA GeForce RTX 5090 D, capability (12, 0)

## TL;DR / Two premise corrections (read first)

1. **The `_softmax_multi_block_forward_kernel` (online softmax) is effectively DEAD CODE** in 0.8.0.
   `calculate_settings` sets `BLOCK_SIZE = triton.next_power_of_2(n_cols)`, which is always `>= n_cols`.
   The launcher branches on `if n_cols <= BLOCK_SIZE:` → that is **always true**, so the **single-block**
   kernel is always chosen. The only way to reach the `else` (multi-block) branch would be `n_cols > BLOCK_SIZE`,
   which never happens; and `n_cols` large enough that `next_power_of_2(n_cols) > 65536` (i.e. `n_cols > 65536`)
   **raises** in `calculate_settings` before any launch. Verified:
   `n=65536 → next_pow2=65536 (single)`, `n=65537 → next_pow2=131072 → RuntimeError`.
   → **The LIVE PK target is `_softmax_single_block_forward_kernel`.** The multi-block kernel is included
   below for completeness/large-n discussion but is unreachable through `_softmax_forward`.
   (Corroborating smell: line 51 `new_m = tl.max(m, blk_max)` passes a 2nd positional arg where `tl.max`'s
   2nd positional is `axis` — this would not compile as written; consistent with "never instantiated.")

2. **`LigerSoftmaxFunction.apply` did NOT crash** on this machine (torch 2.10.0+cu130) — it returned a correct
   `(8,1024)` output. The task's stated `module 'torch.distributed' has no attribute 'tensor'` crash was **not
   reproduced**. We still PK against the module-level `_softmax_forward` launcher because it is the clean,
   autograd-free, forward-only entry point (equivalent to liger's `rms_norm_forward`) — the target is unchanged,
   only the justification.

---

## 1. The forward kernel(s) — FULL bodies (verbatim)

File: `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/softmax.py`

### 1a. LIVE PK target — `_softmax_single_block_forward_kernel` (lines 11–29)

```python
@triton.jit
def _softmax_single_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float("inf"), cache_modifier=".ca")
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask, cache_modifier=".cs")
```

**Precise behavior (this is what the MLIR mirror must match):**

- **Grid shape**: `(n_rows,)` — exactly **one program (block) per row** (`tl.program_id(0) == row_id`). See launcher line 125.
- **Row load + BLOCK_SIZE / masking**: `offs = tl.arange(0, BLOCK_SIZE)` covers the **entire row in a single
  vector** because `BLOCK_SIZE = next_power_of_2(n_cols) >= n_cols`. `mask = offs < n_cols` handles the
  non-power-of-two tail. Masked (padding) lanes load `other=-float("inf")` (line 24).
- **TWO reductions, both block-cooperative over the whole row (axis=0)**:
  1. **Row max (numerical stability)**: `m = tl.max(x, axis=0)` (line 25) — a single `tl.max` reduction over the row.
  2. **Sum of exp**: `d = tl.sum(e, axis=0)` (line 27) where `e = tl.exp(x - m)` (line 26) — a single `tl.sum`
     reduction over the row.
  Both are block-cooperative tree reductions across the lanes of one program (Triton lowers these to
  warp-shuffle + shared-memory reductions).
- **Reduction dtype**: **inherits the input dtype** — the single-block kernel does **NOT** explicitly cast to
  fp32 (unlike the multi-block kernel which uses `tl.float32` accumulators). For an **fp32** input tensor the
  loaded `x`, `m`, `e`, `d` are all fp32, so reductions are fp32. (Matters for the MLIR accumulator choice: for
  the fp32 PK, fp32 accumulators are the faithful match. If you later PK fp16/bf16 inputs, note liger would
  reduce in the *input* dtype here, which is less stable — but our PK is fp32, so fp32 reductions are correct.)
- **-inf / mask over the max**: padded lanes = `-inf` → they never win `tl.max` (so `m` is the true row max
  over valid cols); then `exp(-inf - m) = exp(-inf) = 0` → padded lanes contribute 0 to `tl.sum` (so `d` is the
  true denominator); the final `tl.store` is **masked**, so the `nan/0` padded outputs are never written back.
  This three-step chain (`-inf load → no perturb max → exp→0 → no perturb sum → masked store`) is the entire
  correctness story for non-power-of-two `n_cols`.
- **Final write**: `y = e / d` (line 28), then masked store (line 29). **No second global read** — `x` and `e`
  stay register-resident; the divide reuses the registers from the load/exp. Single load, single store per row.
- **Cache hints (load-bearing for a faithful high-perf mirror)**: load uses `cache_modifier=".ca"`
  (cache at all levels — normal cached load); store uses `cache_modifier=".cs"` (cache-streaming /
  evict-first, signaling write-once-no-reuse).

### 1b. UNREACHABLE via `_softmax_forward` — `_softmax_multi_block_forward_kernel` (lines 32–60)

Included for completeness and the "large-n path" question. This is the classic **online (one-pass running-max)
softmax** for when a row exceeds BLOCK_SIZE — but as shown in the TL;DR it is never launched in 0.8.0.

```python
@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    m = tl.float32(-float("inf"))
    d = tl.float32(0.0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"), cache_modifier=".ca")
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.max(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
        m = new_m

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"), cache_modifier=".ca")
        yblk = tl.exp(xblk - m) / d
        tl.store(Y_ptr + row_id * Y_row_stride + idx, yblk, mask=mask, cache_modifier=".cs")
```

Notes: fp32 accumulators (`m`, `d`) are explicit (lines 44–45); running-max rescale of `d` (line 52); a
**second pass that RELOADS each block from global** (line 58) to write `exp(xblk - m)/d`. This two-pass-reload
structure is what the prescribed MLIR note was based on — but it does NOT match the live single-block kernel
(see §6). Line 51 `tl.max(m, blk_max)` is a likely-non-compiling call (wrong arg shape), reinforcing that this
path is never exercised.

(Backward kernels `_softmax_single_block_backward_kernel` lines 64–82 and `_softmax_multi_block_backward_kernel`
lines 86–113 exist but are out of scope for a forward PK; the backward formula is `dx = y * (dy - sum(dy*y))`.)

## 2. `calculate_settings(n_cols)` — BLOCK_SIZE / num_warps policy (verbatim)

File: `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/utils.py` lines 45–62
(shared util; imported into softmax.py line 7). Originally from Unsloth.

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

Policy summary:
- `BLOCK_SIZE = next_power_of_2(n_cols)`.
- **MAX_FUSED_SIZE cap = 65536**; `n_cols > 65536` (→ `BLOCK_SIZE > 65536`) **raises RuntimeError** (this is the
  only "large-n path" — there is no fallback; it just refuses).
- `num_warps` ladder (NVIDIA): `< 2048 → 4`, `[2048, 8192) → 8`, `[8192, 32768) → 16`, `>= 32768 → 32`
  (HIP caps the top at 16).

Empirically observed (rows×cols → BLOCK_SIZE, num_warps):
`(128,1024)→1024,4` · `(64,1000)→1024,4` · `(32,4096)→4096,8` · `(16,33000)→65536,32` (all single-block).

## 3. Forward launcher — `_softmax_forward` (verbatim, lines 116–135)

```python
def _softmax_forward(x: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y2d = torch.empty_like(x2d)

    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        multi_block_launch = False
    else:
        _softmax_multi_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        multi_block_launch = True

    return y2d.view(*batch, n_cols), BLOCK_SIZE, num_warps, multi_block_launch
```

- **Signature**: `_softmax_forward(x: torch.Tensor) -> Tuple[Tensor, int, int, bool]`.
- Flattens any leading batch dims to a 2D `(n_rows, n_cols)` view (`x.contiguous().view(-1, n_cols)`).
- **Grid** = `(n_rows,)` (one program per row). **BLOCK_SIZE / num_warps** from `calculate_settings(n_cols)`.
  Branch `n_cols <= BLOCK_SIZE` → single-block (always, see §1 TL;DR).
- **Return value**: `(y, BLOCK_SIZE, num_warps, multi_block_launch)` where `y` is reshaped back to the original
  `(*batch, n_cols)` shape, same dtype/device as `x`. For a 2D fp32 `(rows, cols)` input, `y[0]` is `(rows, cols)` fp32.

## 4. How to call the forward kernel directly for a fair head-to-head PK (VERIFIED)

Use the module-level **`_softmax_forward`** launcher (analogous to liger's `rms_norm_forward`). It bypasses
`torch.autograd.Function.apply` entirely — no autograd, no `ensure_contiguous` ctx wrapper, no distributed import
path. Minimal, verified-runnable snippet:

```python
import torch
from liger_kernel.ops.softmax import _softmax_forward   # module-level launcher; NO autograd

x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)   # (rows, cols) fp32 CUDA
y, BLOCK_SIZE, num_warps, multi_block = _softmax_forward(x)        # y: (rows, cols) fp32, summed to 1 per row
torch.cuda.synchronize()
# y is the liger softmax output; read back / compare directly.
```

For timing (drop the 4-tuple, keep `[0]`):
```python
import time
for _ in range(3): _softmax_forward(x)            # warmup / JIT compile
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): y = _softmax_forward(x)[0]
torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 50 * 1e3
```

**Verification (actually ran, no installs):**
- Correctness vs torch (max abs err over `_softmax_forward` output vs `torch.softmax(x.float(), -1)`):
  `(128,1024)→5.6e-9` · `(64,1000)→3.7e-9` · `(32,4096)→1.9e-9` · `(16,33000)→2.3e-10`; every row sums to exactly
  `1.000000`. Output shape/dtype preserved.
- Timing: `(4096,4096)` fp32 → **~0.486 ms/iter**; `torch.allclose(..., atol=1e-6, rtol=1e-5)` → True.
- `LigerSoftmaxFunction.apply(x)` also ran fine here (no crash), but `_softmax_forward` is the preferred PK entry.

If you ever needed to invoke only the autograd Function's forward without a real launcher (not needed here, since
`_softmax_forward` exists): `LigerSoftmaxFunction.forward(None, x)` — passing a dummy `ctx=None` works because
`forward` only calls `ctx.save_for_backward(...)`/sets attrs after computing `y`; but **prefer `_softmax_forward`**
(it returns `y` directly and avoids touching ctx). The clean launcher makes the dummy-ctx hack unnecessary.

## 5. Exact fp32 reference math to certify against torch

```python
ref = torch.softmax(x.float(), dim=-1)      # numerically-stable softmax over last dim
# equivalent explicit form the kernel computes:
m   = x.float().amax(dim=-1, keepdim=True)
e   = (x.float() - m).exp()
ref = e / e.sum(dim=-1, keepdim=True)
```
Certify: `torch.allclose(y, ref, atol=1e-6, rtol=1e-5)` and `assert y.sum(-1) ≈ 1`. Observed errors ~1e-9–1e-10
(see §4), well inside tolerance — the MLIR mirror should target the same.

## 6. MLIR-mirroring notes (5 lines) — corrected to match the LIVE single-block kernel

1. **One `gpu.launch` block per row** (grid = `n_rows`); each block handles exactly one `(n_cols)` row — matches
   `grid=(n_rows,)`, `tl.program_id(0)=row_id`.
2. **Single load of the whole row into registers/shared** (BLOCK_SIZE = next_pow2(n_cols) >= n_cols), with a
   `col < n_cols` predicate and a `-inf` fill for the padded tail; do **NOT** stream/re-tile — liger loads the
   row once and keeps `x` (and `e`) resident.
3. **TWO `gpu.all_reduce` over the row, in order: `max` then `add`** — first reduce the row max `m` (fp32, the
   numerical-stability shift), then reduce `d = Σ exp(x - m)`; both fp32 accumulators for the fp32 PK.
4. **`exp` via the native `ex2.approx` peephole** already in the toolchain (i.e. `exp(t) = ex2(t * log2e)`), used
   for `e = exp(x - m)` and reused from registers for the final divide.
5. **Second "pass" is a register-resident divide + masked store — NOT a global reload.** The live kernel writes
   `y = e / d` directly from the register-held `e` (single store, cache-streaming `.cs`); a two-pass-reload MLIR
   variant (like the unreachable multi-block kernel) would be a *different, slower* algorithm and break apples-to-
   apples PK. **Decide deliberately:** mirror = single-load register/shared-resident (faithful); avoid reload.

## Files / references

| Path | Description |
|---|---|
| `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/softmax.py` | Kernels (lines 11–29 single-block fwd = LIVE target; 32–60 multi-block fwd = dead) + `_softmax_forward` launcher (116–135) + `LigerSoftmaxFunction` (179–202). REFERENCE ONLY. |
| `/home/kingdom/.local/lib/python3.10/site-packages/liger_kernel/ops/utils.py` | `calculate_settings` (45–62), `ensure_contiguous` (32–42). REFERENCE ONLY. |

## Caveats / Not found

- Multi-block forward kernel is unreachable via `_softmax_forward` in 0.8.0 (see §1 TL;DR) and was not (could not
  be) exercised; its `tl.max(m, blk_max)` (line 51) likely would not even compile. Treat it as documentation only.
- The stated `torch.distributed has no attribute 'tensor'` crash on `Function.apply` was **not reproduced** on
  torch 2.10.0+cu130 here; `_softmax_forward` is used regardless as the clean forward-only PK entry point.
- Single-block reductions inherit input dtype (no explicit fp32 cast). Fine for the fp32 PK; revisit if PKing
  fp16/bf16 inputs (liger would reduce in input dtype there).
