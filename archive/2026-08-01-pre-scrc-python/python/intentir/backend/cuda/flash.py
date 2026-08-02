"""Parameterized fused-FlashAttention MLIR emitter + retune / vs-Triton harness.

This module generalizes the hand-written ``test/flash_fused.mlir`` (Sq=2,Sk=4,D=2,
BK=1) into a parameterized emitter ``emit_fused_flash(sq,sk,d,bk)`` and a launch
path ``run_fused_flash(q,k,v,scale,bk)`` that compiles the emitted module through
the device-side pipeline (``lower_memref_module_to_kernels``) and launches the
single resulting kernel on torch device pointers.

The kernel is the FlashAttention forward as ONE fused kernel:

* ``scf.parallel (%qi)`` over the ``sq`` query rows  -> GPU grid (one block/query).
* an inner ``scf.for %kt = 0 to sk step bk`` over key TILES, carrying the online-
  softmax state (running max ``m``, running sum ``l``, accumulator ``acc_0..acc_{d-1}``)
  through scalar ``iter_args``.  Each tile processes ``bk`` keys (the emitter
  UNROLLS the ``bk`` lanes), rescaling ``alpha = exp(m_old - m_new)`` and updating
  ``acc = alpha*acc + sum_i p_i*V[j_i]``, ``l = alpha*l + sum_i p_i``.
* after the loop, ``O[qi,c] = acc_c / l``.

``math.exp`` is left in the IR; the device pipeline's ``-lower-gpu-math-to-nvvm``
peephole rewrites it to the native ``ex2.approx.ftz.f32`` PTX instruction.

The init constants (``m=-inf``, ``l=0``, ``acc=0``) are defined INSIDE the
``scf.parallel`` body so they do not become kernel arguments, keeping the kernel
ABI to (loop step, loop lb, Q, K, V, O ptrs, scale, + whatever literal constants
the loop tile arithmetic introduces).  The launch path resolves the ABI GENERICALLY
by parsing the lowered host module (it never hardcodes an arg order), so changing
``d`` / ``bk`` — which changes the constant args — does not break it.

Three properties this validates:

* **correctness** — ``run_fused_flash`` matches exact ``softmax((Q@K.T)*scale)@V``.
* **tile-invariance (retune safety)** — varying ``bk`` (the K-tile schedule) gives
  the identical Layer-A output: the schedule is non-binding.
* **vs Triton** — agrees with an independent Triton golden in the benchmark
  suite.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from intentir.backend.cuda.compile import lower_memref_module_to_kernels
from intentir.backend.cuda.launch import load_module
from intentir.backend.cuda.orchestrate import (
    _F32Arg,
    _I64Arg,
    _PtrArg,
    _host_func_text,
    _launch,
    _map_input_aligned_ptrs,
    _parse_constants,
    _parse_f32,
    _parse_launches,
    _parse_signature_blockargs,
    _resolve_const_index,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


_NINF_HEX = "0xFF800000"  # IEEE-754 -inf as MLIR prints it.


# --------------------------------------------------------------------------- #
# DELIVERABLE 1 — the emitter.                                                 #
# --------------------------------------------------------------------------- #
def emit_fused_flash(sq: int, sk: int, d: int, bk: int) -> str:
    """Emit MLIR text of a fused flash kernel for the given (sq, sk, d, bk).

    Generalizes ``test/flash_fused.mlir``.  For ``bk == 1`` the emitted body is
    structurally equivalent to that hand-written reference (same single-key online
    rescale).  For ``bk > 1`` the inner ``scf.for`` steps by ``bk`` and the emitter
    unrolls a real tiled online softmax over the ``bk`` keys of each tile, with a
    cmpi/select tail guard emitted only when ``sk % bk != 0``.
    """
    if sq <= 0 or sk <= 0 or d <= 0 or bk <= 0:
        raise ValueError("sq, sk, d, bk must all be positive")

    has_tail = (sk % bk) != 0

    L: list[str] = []
    ind = "      "  # body indentation inside scf.parallel
    fb = ind + "  "  # for-body indentation

    # collect the index constants we need: 0, 1, sq, sk, and one per key-lane
    # offset 0..bk-1 (and the index for sk-1 used by the tail clamp).
    needed_idx: set[int] = {0, 1, sq, sk, bk}  # bk = the scf.for step
    needed_idx.update(range(d))      # column indices 0..d-1
    needed_idx.update(range(bk))     # lane offsets 0..bk-1
    if has_tail:
        needed_idx.add(sk - 1)

    L.append("module {")
    L.append(
        f"  func.func @flash(%Q: memref<{sq}x{d}xf32>, "
        f"%K: memref<{sk}x{d}xf32>, %V: memref<{sk}x{d}xf32>, "
        f"%O: memref<{sq}x{d}xf32>, %scale: f32) {{"
    )

    # ---- index/value constants at func top (these are index/loop literals; they
    # are fine to hoist — the parallel-loop bound `sq` becomes the grid extent and
    # the loop step/lb become i64 kernel args, exactly as in flash_fused.mlir). ----
    for v in sorted(needed_idx):
        L.append(f"    %c{v} = arith.constant {v} : index")

    L.append(f"    scf.parallel (%qi) = (%c0) to (%c{sq}) step (%c1) {{")

    # ---- init carries INSIDE the body so they are NOT kernel args -------------
    L.append(f"{ind}%ninf = arith.constant {_NINF_HEX} : f32")
    L.append(f"{ind}%zero = arith.constant 0.0 : f32")

    # ---- load q[0..d-1] -------------------------------------------------------
    for c in range(d):
        L.append(
            f"{ind}%q{c} = memref.load %Q[%qi, %c{c}] : memref<{sq}x{d}xf32>"
        )

    # ---- outer scf.for over key tiles, step = bk ------------------------------
    init_args = ["%m = %ninf", "%l = %zero"] + [
        f"%a{c} = %zero" for c in range(d)
    ]
    carry_tys = ", ".join(["f32"] * (2 + d))
    L.append(
        f"{ind}%res:{2 + d} = scf.for %kt = %c0 to %c{sk} step %c{bk}"
    )
    L.append(f"{ind}    iter_args({', '.join(init_args)})")
    L.append(f"{ind}    -> ({carry_tys}) {{")

    # For each lane i in 0..bk-1, compute key index j = kt + i and the score s_i.
    # We need an index SSA `%j_i` for the load.  When the tile is exactly divisible
    # (no tail) every j is in range, so we load directly.  With a tail we clamp the
    # load index to sk-1 and mask s_i to -inf for out-of-range j.
    s_vals: list[str] = []
    for i in range(bk):
        # key index j = kt + c{i}
        L.append(f"{fb}%j{i} = arith.addi %kt, %c{i} : index")
        if has_tail:
            # predicate: j < sk
            L.append(
                f"{fb}%inb{i} = arith.cmpi slt, %j{i}, %c{sk} : index"
            )
            # clamped load index: select(j<sk, j, sk-1)
            L.append(
                f"{fb}%jc{i} = arith.select %inb{i}, %j{i}, %c{sk - 1} : index"
            )
            jidx = f"%jc{i}"
        else:
            jidx = f"%j{i}"

        # dot(q, K[j]) over d
        for c in range(d):
            L.append(
                f"{fb}%k{i}_{c} = memref.load %K[{jidx}, %c{c}] "
                f": memref<{sk}x{d}xf32>"
            )
        # accumulate the dot product
        prev = None
        for c in range(d):
            L.append(f"{fb}%pp{i}_{c} = arith.mulf %q{c}, %k{i}_{c} : f32")
            if prev is None:
                prev = f"%pp{i}_{c}"
            else:
                L.append(
                    f"{fb}%dot{i}_{c} = arith.addf {prev}, %pp{i}_{c} : f32"
                )
                prev = f"%dot{i}_{c}"
        dot = prev  # the full dot product SSA
        L.append(f"{fb}%sraw{i} = arith.mulf {dot}, %scale : f32")
        if has_tail:
            # mask out-of-range lanes to -inf so they contribute p = 0.
            L.append(
                f"{fb}%s{i} = arith.select %inb{i}, %sraw{i}, %ninf : f32"
            )
            s_vals.append(f"%s{i}")
        else:
            # no rename needed; the scaled dot IS the score.
            s_vals.append(f"%sraw{i}")

    # m_tile = max over i of s_i ; m_new = max(m, m_tile)
    m_running = "%m"
    for i in range(bk):
        L.append(
            f"{fb}%mt{i} = arith.maximumf {m_running}, {s_vals[i]} : f32"
        )
        m_running = f"%mt{i}"
    mnew = m_running  # max(m, s_0..s_{bk-1}) is exactly m_new

    # alpha = exp(m_old - m_new)
    L.append(f"{fb}%md = arith.subf %m, {mnew} : f32")
    L.append(f"{fb}%alpha = math.exp %md : f32")

    # p_i = exp(s_i - m_new)
    for i in range(bk):
        L.append(f"{fb}%sd{i} = arith.subf {s_vals[i]}, {mnew} : f32")
        L.append(f"{fb}%p{i} = math.exp %sd{i} : f32")

    # l_new = alpha*l + sum_i p_i
    L.append(f"{fb}%la = arith.mulf %alpha, %l : f32")
    l_acc = "%la"
    for i in range(bk):
        L.append(f"{fb}%lp{i} = arith.addf {l_acc}, %p{i} : f32")
        l_acc = f"%lp{i}"
    lnew = l_acc  # alpha*l + sum p_i

    # acc_c_new = alpha*acc_c + sum_i p_i * V[j_i, c]
    acc_new: list[str] = []
    # load V[j_i, c] (reuse the clamped index jc{i} under tail; masked lanes have
    # p_i = 0 so their V contribution is multiplied by 0 regardless of the value).
    for c in range(d):
        L.append(f"{fb}%aa{c} = arith.mulf %alpha, %a{c} : f32")
        cur = f"%aa{c}"
        for i in range(bk):
            jidx = f"%jc{i}" if has_tail else f"%j{i}"
            L.append(
                f"{fb}%v{i}_{c} = memref.load %V[{jidx}, %c{c}] "
                f": memref<{sk}x{d}xf32>"
            )
            L.append(f"{fb}%pv{i}_{c} = arith.mulf %p{i}, %v{i}_{c} : f32")
            L.append(f"{fb}%ac{c}_{i} = arith.addf {cur}, %pv{i}_{c} : f32")
            cur = f"%ac{c}_{i}"
        acc_new.append(cur)

    yields = [mnew, lnew] + acc_new
    L.append(f"{fb}scf.yield {', '.join(yields)} : {carry_tys}")
    L.append(f"{ind}}}")

    # ---- O[qi, c] = acc_c / l -------------------------------------------------
    for c in range(d):
        L.append(f"{ind}%o{c} = arith.divf %res#{2 + c}, %res#1 : f32")
        L.append(
            f"{ind}memref.store %o{c}, %O[%qi, %c{c}] : memref<{sq}x{d}xf32>"
        )

    L.append(f"{ind}scf.reduce")
    L.append("    }")
    L.append("    return")
    L.append("  }")
    L.append("}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Compile cache (keyed on the kernel shape) — so retune_sweep / repeated runs  #
# compile ONCE and time warm launches.  lower_memref_module_to_kernels is not  #
# itself lru_cached (only compile_all_kernels is), so we cache here.           #
# --------------------------------------------------------------------------- #
_KERNEL_CACHE: dict[tuple[int, int, int, int], "_LoweredFlash"] = {}


class _LoweredFlash:
    """A compiled flash kernel + its parsed, resolved launch plan."""

    __slots__ = ("func", "grid", "block", "plan", "n_inputs")

    def __init__(self, func, grid, block, plan):
        self.func = func
        self.grid = grid
        self.block = block
        # plan: list of ("ptr", input_index) | ("scale",) | ("i64", val) | ("f32", val)
        self.plan = plan


def _build_plan(sq: int, sk: int, d: int, bk: int) -> "_LoweredFlash":
    """Emit -> lower -> parse the host module -> resolve a generic launch plan."""
    mlir = emit_fused_flash(sq, sk, d, bk)
    kernels = lower_memref_module_to_kernels(mlir)
    if len(kernels) != 1:
        raise RuntimeError(
            f"flash_emit: expected exactly one kernel, got {len(kernels)}"
        )
    ck = kernels[0]
    func = load_module(ck.cubin).function(ck.kernel_name)

    binary_mlir = ck.binary_mlir
    _name, func_text = _host_func_text(binary_mlir)
    _sig_name, _ordered, ptr_blockargs = _parse_signature_blockargs(func_text)
    aligned_map = _map_input_aligned_ptrs(ptr_blockargs)  # %aligned_ptr -> input idx
    consts = _parse_constants(func_text)
    launches = _parse_launches(func_text)
    if len(launches) != 1:
        raise RuntimeError(
            f"flash_emit: expected one gpu.launch_func, got {len(launches)}"
        )
    site = launches[0]

    # The original func signature order is Q, K, V, O (4 memrefs) then %scale (f32).
    # _map_input_aligned_ptrs gives each input's aligned-ptr block-arg -> its index
    # 0..3 in that order, so a ptr operand resolves directly to Q/K/V/O by index.
    #
    # We build a HostModule-like shim so _resolve_const_index works on our consts.
    class _Shim:
        constants = consts

    shim = _Shim()

    grid = tuple(_resolve_const_index(shim, s) for s in site.grid_ssa)
    block = tuple(_resolve_const_index(shim, s) for s in site.block_ssa)

    plan: list[tuple] = []
    for op in site.args:
        ty = op.ty
        if ty.startswith("!llvm.ptr"):
            if op.ssa not in aligned_map:
                raise RuntimeError(
                    f"flash_emit: launch ptr {op.ssa} is not an input aligned-ptr "
                    f"(have {sorted(aligned_map)}); the kernel uses a buffer we "
                    "did not allocate"
                )
            plan.append(("ptr", aligned_map[op.ssa]))
        elif ty in ("i64", "index"):
            if op.ssa not in consts:
                raise RuntimeError(
                    f"flash_emit: i64 launch arg {op.ssa} is not a literal constant"
                )
            plan.append(("i64", int(consts[op.ssa][0])))
        elif ty == "f32":
            # A block-arg-derived f32 is the func's %scale; an f32 *constant* is a
            # baked literal init (e.g. a leaked carry init) to pass by value.
            if op.ssa in consts:
                plan.append(("f32", _parse_f32(consts[op.ssa][0])))
            else:
                plan.append(("scale",))
        else:
            raise RuntimeError(
                f"flash_emit: unhandled launch-arg type {ty!r} (operand {op.ssa})"
            )

    return _LoweredFlash(func, grid, block, plan)


def _get_lowered(sq: int, sk: int, d: int, bk: int) -> "_LoweredFlash":
    key = (sq, sk, d, bk)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_plan(sq, sk, d, bk)
    return _KERNEL_CACHE[key]


# --------------------------------------------------------------------------- #
# DELIVERABLE 2 — run the fused flash kernel.                                  #
# --------------------------------------------------------------------------- #
def run_fused_flash(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    scale: float,
    bk: int,
) -> "torch.Tensor":
    """Compile + launch the fused flash kernel for these inputs.

    q[sq,d], k[sk,d], v[sk,d] must be contiguous CUDA f32 tensors.  Returns
    O[sq,d] on CUDA.  The grid/block/arg order is resolved generically from the
    lowered host module (cached per (sq,sk,d,bk)).
    """
    import torch

    sq, d = q.shape
    sk = k.shape[0]
    assert k.shape[1] == d and v.shape == (sk, d), "shape mismatch q/k/v"

    low = _get_lowered(sq, sk, d, bk)

    o = torch.empty(sq, d, device="cuda", dtype=torch.float32).contiguous()
    inputs = [q, k, v, o]  # index order Q, K, V, O

    call_args = []
    for entry in low.plan:
        kind = entry[0]
        if kind == "ptr":
            call_args.append(_PtrArg(inputs[entry[1]].data_ptr()))
        elif kind == "scale":
            call_args.append(_F32Arg(float(scale)))
        elif kind == "i64":
            call_args.append(_I64Arg(entry[1]))
        elif kind == "f32":
            call_args.append(_F32Arg(entry[1]))
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"flash_emit: bad plan entry {entry!r}")

    _launch(low.func, low.grid, low.block, call_args)
    return o


# --------------------------------------------------------------------------- #
# DELIVERABLE 4 — retune sweep over the K-tile schedule.                       #
# --------------------------------------------------------------------------- #
def retune_sweep(
    sq: int,
    sk: int,
    d: int,
    candidates=(1, 2, 4, 8, 16),
    warmup: int = 5,
    iters: int = 20,
    seed: int = 0,
):
    """Sweep the K-tile size ``bk`` and time each candidate; return latencies+best.

    This demonstrates the retune MECHANISM: a schedule sweep where every schedule
    produces the identical Layer-A output (tile-invariance) and the selection is
    correctness-preserving.  Each candidate is COMPILED ONCE (outside the timing
    loop, via ``_get_lowered``'s cache); timing uses ``torch.cuda.Event`` over a
    warmup + measured-iters window on fixed random inputs.

    NOTE: this lowering is one-thread-per-query (one GPU block per query row, the
    bring-up granularity that mirrors Triton's BLOCK_M=1 golden), so the absolute
    latency deltas between ``bk`` values are small — the point is the *mechanism*
    (sweep schedules, pick the best while preserving correctness), not a tuned-perf
    claim.

    Returns ``{"latencies_ms": {bk: ms}, "best_bk": int}``.  Candidates that fail
    to compile/launch are skipped (recorded as ``float("inf")``).
    """
    import torch

    torch.manual_seed(seed)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float32).contiguous()
    scale = 1.0 / (d ** 0.5)

    latencies: dict[int, float] = {}
    for bk in candidates:
        try:
            _get_lowered(sq, sk, d, bk)  # compile once, outside timing
        except Exception:
            latencies[bk] = float("inf")
            continue
        # warmup
        for _ in range(warmup):
            run_fused_flash(q, k, v, scale, bk)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_fused_flash(q, k, v, scale, bk)
        end.record()
        torch.cuda.synchronize()
        latencies[bk] = start.elapsed_time(end) / iters

    best_bk = min(latencies, key=lambda b: latencies[b])
    return {"latencies_ms": latencies, "best_bk": best_bk}


__all__ = ["emit_fused_flash", "run_fused_flash", "retune_sweep"]
