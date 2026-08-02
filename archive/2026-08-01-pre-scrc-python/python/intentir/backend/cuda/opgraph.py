"""General op-graph dispatch: sequence a multi-op IntentIR program across kernels.

The keystone the coverage task deferred. Today each *cluster* pass
(gather/scatter/sort/scan/pool/conv/compact) lowers a func whose body is a
SINGLE cluster op, while the linalg / row-reduction leg fuses map/reduce/cast/
select chains. A program that mixes a cluster op with a fused map/reduce chain
(quantile = sort -> gather -> lerp; topk_softmax = softmax -> topk) crosses that
single-op boundary and cannot lower as one func.

This module dispatches such a program by SEQUENCING it into stages:

* the program is described as an ordered list of ``Stage``s (the outlined form of
  the op-graph — one cluster op, or one fused map/reduce chain, per stage);
* each stage is a single-op / single-chain IntentIR func that ONE existing pass
  already lowers unchanged (``convert-intent-sort-to-gpu``,
  ``convert-intent-gather-to-gpu``, ``convert-intent-row-reduction-to-gpu``, ...);
* stages are wired by *named intermediate device buffers*: a stage reads program
  inputs and prior-stage outputs from ``env`` and writes its outputs back into
  ``env``, so the topological data-flow is threaded through torch CUDA tensors
  (the driver-level realization of orchestrate's multi-kernel buffer threading —
  the same mechanism P2a's 3-launch compaction used, here across separate funcs).

Each stage compiles through :class:`RowReductionKernel`, which drives its chosen
C++ pass and resolves a bare-ptr launch plan; the stage's func signature is the
appended-output ABI (inputs first, then one memref per result), so
``thunk([*inputs, *outputs])`` binds them positionally.

Correctness first: every stage kernel is the already-verified single-op lowering;
this module only orders them and threads the buffers. Fusion across the cluster
boundary (materializing fewer intermediates) is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from intentir.backend.cuda.kernel_driver import RowReductionKernel

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_DTYPES = {"f32": "float32", "i32": "int32", "i64": "int64"}


@dataclass
class Buf:
    """A named intermediate/output device buffer produced by a stage."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # "f32" | "i32" | "i64"


@dataclass
class Stage:
    """One outlined op-graph node: a single-op / single-chain IntentIR func.

    ``inputs`` names (from ``env``) feed the func's leading tensor args in order;
    ``outputs`` are the appended-output memrefs the kernel writes, allocated fresh
    per :meth:`OpGraphProgram.run` and published back into ``env`` under their
    names. ``pass_name`` selects the C++ lowering pass; ``block_threads`` its
    block size.
    """

    name: str
    pass_name: str
    mlir_text: str
    block_threads: int
    inputs: list[str]
    outputs: list[Buf]


@dataclass
class OpGraphProgram:
    """A sequenced multi-kernel program. Compile once, run many."""

    stages: list[Stage]
    _kernels: list[RowReductionKernel] = field(default_factory=list, repr=False)

    def compile(self) -> "OpGraphProgram":
        self._kernels = [
            RowReductionKernel(
                st.name, {}, st.block_threads,
                pass_name=st.pass_name, mlir_text=st.mlir_text,
            )
            for st in self.stages
        ]
        return self

    def run(self, env: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        """Execute every stage in order, threading device buffers through ``env``.

        ``env`` maps program-input names to CUDA tensors. Returns a fresh dict of
        every buffer (inputs + all stage outputs) so callers can read any result.
        """
        import torch

        if not self._kernels:
            self.compile()
        env = dict(env)
        for st, k in zip(self.stages, self._kernels):
            for b in st.outputs:
                env[b.name] = torch.empty(
                    b.shape, dtype=getattr(torch, _DTYPES[b.dtype]), device="cuda"
                ).contiguous()
            tensors = [env[n] for n in st.inputs] + [env[b.name] for b in st.outputs]
            k.thunk(tensors)()
        torch.cuda.synchronize()
        return env


# --------------------------------------------------------------------------- #
# Target builders — hand-authored outlined op-graphs for the deferred targets. #
# --------------------------------------------------------------------------- #
def _threads_pow2(n: int, cap: int = 256) -> int:
    """Largest power-of-two <= cap that divides next_pow2(n) (sort geometry)."""
    p = 1
    while p < n:
        p <<= 1
    t = cap
    while t > p or p % t != 0:
        t >>= 1
    return t


def _div_threads(cols: int, cap: int = 256) -> int:
    """Largest power-of-two <= cap dividing ``cols`` (gather/reduce geometry)."""
    t = min(cap, cols)
    while t > 1 and cols % t != 0:
        t >>= 1
    return t


def quantile_plan(M: int, N: int, q: float) -> tuple[OpGraphProgram, dict, int, int, float]:
    """quantile(x[M,N], q) = sort(asc) -> gather(fractional rank lo,hi) -> lerp.

    Linear interpolation (torch default): pos = q*(N-1), lo=floor(pos),
    hi=min(lo+1, N-1), frac=pos-lo; out = v[lo]*(1-frac) + v[hi]*frac.

    Three outlined stages exercising sort (C6b) + gather (C3) + reduce/map:
      1. sort asc, keep the first W=hi+1 columns          -> sorted[M,W]
      2. gather columns [lo,hi] (along axis 1)             -> g[M,2]
      3. weighted sum g*w over axis 1 (w = [1-frac,frac])  -> out[M,1]

    Returns (program, static_index/weight tensors builder args) — the index and
    weight constants are shape-fixed, so the caller supplies them via ``env``.
    """
    import math

    pos = q * (N - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, N - 1)
    frac = pos - lo
    W = hi + 1

    Ts = _threads_pow2(N)
    Tg = _div_threads(2)
    Tr = _div_threads(2)

    sort_stage = Stage(
        name="q_sort", pass_name="convert-intent-sort-to-gpu",
        mlir_text=f"""
func.func @q_sort(%x: tensor<{M}x{N}xf32>) -> tensor<{M}x{W}xf32> {{
  %v, %i = intent.sort %x {{descending = false, axis = 1 : i64}}
      : (tensor<{M}x{N}xf32>) -> (tensor<{M}x{W}xf32>, tensor<{M}x{W}xi32>)
  return %v : tensor<{M}x{W}xf32>
}}
""",
        block_threads=Ts, inputs=["x"],
        outputs=[Buf("sorted", (M, W), "f32")],
    )
    gather_stage = Stage(
        name="q_gather", pass_name="convert-intent-gather-to-gpu",
        mlir_text=f"""
func.func @q_gather(%data: tensor<{M}x{W}xf32>, %index: tensor<{M}x2xi64>)
    -> tensor<{M}x2xf32> {{
  %g = intent.gather %data, %index {{axis = 1 : i64, mode = "along_axis"}}
      : (tensor<{M}x{W}xf32>, tensor<{M}x2xi64>) -> tensor<{M}x2xf32>
  return %g : tensor<{M}x2xf32>
}}
""",
        block_threads=Tg, inputs=["sorted", "qindex"],
        outputs=[Buf("g", (M, 2), "f32")],
    )
    lerp_stage = Stage(
        name="q_lerp", pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @q_lerp(%g: tensor<{M}x2xf32>, %w: tensor<{M}x2xf32>)
    -> tensor<{M}x1xf32> {{
  %p = intent.map_binary %g, %w {{fn = "mul"}}
      : (tensor<{M}x2xf32>, tensor<{M}x2xf32>) -> tensor<{M}x2xf32>
  %o = intent.reduce %p {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}}
      : (tensor<{M}x2xf32>) -> tensor<{M}x1xf32>
  return %o : tensor<{M}x1xf32>
}}
""",
        block_threads=Tr, inputs=["g", "qweight"],
        outputs=[Buf("out", (M, 1), "f32")],
    )
    prog = OpGraphProgram([sort_stage, gather_stage, lerp_stage])
    consts = {"lo": lo, "hi": hi, "frac": frac, "W": W}
    return prog, consts, lo, hi, frac


def topk_softmax_plan(M: int, N: int, k: int) -> OpGraphProgram:
    """topk_softmax(gating[M,N], k) = softmax(experts) -> topk (MoE gating).

    Matches the real ``flag_gems.topk_softmax`` (vLLM fused-MoE gating): softmax
    over the expert axis, then the top-k routing weights + expert indices. Two
    outlined stages exercising the reduce -> sort dispatch:
      1. row-local safe-softmax over [M,N]                  -> probs[M,N]
      2. sort descending, keep W=k                          -> weights[M,k] + idx[M,k]
    """
    Ts = _threads_pow2(N)
    Tr = _div_threads(N)
    softmax_stage = Stage(
        name="tk_softmax", pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @tk_softmax(%t: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %m = intent.reduce %t {{fn = "max", axes = [1], assoc = "associative"}}
      : (tensor<{M}x{N}xf32>) -> tensor<{M}x1xf32>
  %sm = intent.map_binary %t, %m {{fn = "sub"}}
      : (tensor<{M}x{N}xf32>, tensor<{M}x1xf32>) -> tensor<{M}x{N}xf32>
  %e = intent.map_unary %sm {{fn = "exp"}}
      : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %d = intent.reduce %e {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}}
      : (tensor<{M}x{N}xf32>) -> tensor<{M}x1xf32>
  %o = intent.map_binary %e, %d {{fn = "div"}}
      : (tensor<{M}x{N}xf32>, tensor<{M}x1xf32>) -> tensor<{M}x{N}xf32>
  return %o : tensor<{M}x{N}xf32>
}}
""",
        block_threads=Tr, inputs=["gating"],
        outputs=[Buf("probs", (M, N), "f32")],
    )
    topk_stage = Stage(
        name="tk_sort", pass_name="convert-intent-sort-to-gpu",
        mlir_text=f"""
func.func @tk_sort(%x: tensor<{M}x{N}xf32>)
    -> (tensor<{M}x{k}xf32>, tensor<{M}x{k}xi32>) {{
  %v, %i = intent.sort %x {{descending = true, axis = 1 : i64}}
      : (tensor<{M}x{N}xf32>) -> (tensor<{M}x{k}xf32>, tensor<{M}x{k}xi32>)
  return %v, %i : tensor<{M}x{k}xf32>, tensor<{M}x{k}xi32>
}}
""",
        block_threads=Ts, inputs=["probs"],
        outputs=[Buf("weights", (M, k), "f32"), Buf("indices", (M, k), "i32")],
    )
    return OpGraphProgram([softmax_stage, topk_stage])


def cross_entropy_plan(M: int, V: int, block_threads: int) -> OpGraphProgram:
    """cross_entropy(X[M,V], Y[M]) = log_softmax + nll, fused to ONE kernel.

    The classic gather(target logit) + reduce composition. Here the whole DAG
    fuses into a single row-reduction ``gpu.launch``: the gather of the target
    logit is a masked row-sum ``reduce(select(iota == Y, X, 0))`` (the same gather
    mechanism cross_entropy's fused kernel uses), the log-sum-exp is max/sum-exp,
    and the per-row loss = (lse - ori)/M with the in-place softmax-minus-onehot
    gradient. A one-stage op-graph — proof the dispatch subsumes the fully-fused
    case as the degenerate single-stage program.

    Inputs (env): X[M,V] f32, Y[M] i32, zero/one/ninv 0-d f32 scalars.
    Outputs: loss[M,1], grad[M,V] (softmax - onehot)/M.
    """
    R = f"tensor<{M}x{V}xf32>"
    P = f"tensor<{M}x1xf32>"
    S = "tensor<f32>"
    YI = f"tensor<{M}xi32>"
    II = f"tensor<{V}xi32>"
    I = f"tensor<{M}x{V}xi1>"
    stage = Stage(
        name="cross_entropy", pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @cross_entropy(%X: {R}, %Y: {YI}, %zero: {S}, %one: {S}, %ninv: {S})
    -> ({P}, {R}) {{
  %iota = intent.iota : {II}
  %cmp = intent.map_binary %iota, %Y {{fn = "eq"}} : ({II}, {YI}) -> {I}
  %m = intent.reduce %X {{fn = "max", axes = [1], assoc = "associative"}} : ({R}) -> {P}
  %xm = intent.map_binary %X, %m {{fn = "sub"}} : ({R}, {P}) -> {R}
  %ex = intent.map_unary %xm {{fn = "exp"}} : ({R}) -> {R}
  %se = intent.reduce %ex {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %logse = intent.map_unary %se {{fn = "log"}} : ({P}) -> {P}
  %lse = intent.map_binary %m, %logse {{fn = "add"}} : ({P}, {P}) -> {P}
  %selo = intent.select %cmp, %X, %zero : ({I}, {R}, {S}) -> {R}
  %ori = intent.reduce %selo {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %lnum = intent.map_binary %lse, %ori {{fn = "sub"}} : ({P}, {P}) -> {P}
  %loss = intent.map_binary %lnum, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %xlse = intent.map_binary %X, %lse {{fn = "sub"}} : ({R}, {P}) -> {R}
  %g = intent.map_unary %xlse {{fn = "exp"}} : ({R}) -> {R}
  %oh = intent.select %cmp, %one, %zero : ({I}, {S}, {S}) -> {R}
  %gnum = intent.map_binary %g, %oh {{fn = "sub"}} : ({R}, {R}) -> {R}
  %grad = intent.map_binary %gnum, %ninv {{fn = "mul"}} : ({R}, {S}) -> {R}
  return %loss, %grad : {P}, {R}
}}
""",
        block_threads=block_threads,
        inputs=["X", "Y", "zero", "one", "ninv"],
        outputs=[Buf("loss", (M, 1), "f32"), Buf("grad", (M, V), "f32")],
    )
    return OpGraphProgram([stage])


def randperm_plan(n: int, seed: int) -> OpGraphProgram:
    """randperm(n) = argsort of n philox random keys (RNG C5 -> sort C6b).

    A random permutation of ``0..n-1`` is the argsort of ``n`` i.i.d. uniform
    keys: distinct keys w.h.p., and even under a tie the bitonic argsort emits
    each source index exactly once, so the index result is ALWAYS a valid
    permutation. Two outlined stages, no program inputs (the RNG func takes zero
    tensor args — the philox counter is seed+linear-index only):

      1. rand keys[1,n]  (``convert-intent-rng-to-gpu``, seeded philox)
      2. argsort ascending -> perm[1,n] i32  (``convert-intent-sort-to-gpu``)

    Verify: perm is a valid permutation (sorted == arange), STATISTICALLY uniform
    per position over many seeds, and DETERMINISTIC (same seed reproducible).
    Matched-seed parity vs torch is NOT expected (philox != torch's RNG).
    """
    Tr = _div_threads(n)
    Ts = _threads_pow2(n)
    rng_stage = Stage(
        name="rp_rng", pass_name="convert-intent-rng-to-gpu",
        mlir_text=f"""
func.func @rp_rng() -> tensor<1x{n}xf32> {{
  %r = intent.rand {{seed = {seed} : i64}} : tensor<1x{n}xf32>
  return %r : tensor<1x{n}xf32>
}}
""",
        block_threads=Tr, inputs=[],
        outputs=[Buf("keys", (1, n), "f32")],
    )
    sort_stage = Stage(
        name="rp_sort", pass_name="convert-intent-sort-to-gpu",
        mlir_text=f"""
func.func @rp_sort(%x: tensor<1x{n}xf32>)
    -> (tensor<1x{n}xf32>, tensor<1x{n}xi32>) {{
  %v, %i = intent.sort %x {{descending = false, axis = 1 : i64}}
      : (tensor<1x{n}xf32>) -> (tensor<1x{n}xf32>, tensor<1x{n}xi32>)
  return %v, %i : tensor<1x{n}xf32>, tensor<1x{n}xi32>
}}
""",
        block_threads=Ts, inputs=["keys"],
        outputs=[Buf("sorted", (1, n), "f32"), Buf("perm", (1, n), "i32")],
    )
    return OpGraphProgram([rng_stage, sort_stage])


def multinomial_plan(N: int, k: int, seed: int) -> OpGraphProgram:
    """multinomial(p[N], k) = inverse-CDF categorical sampling (scan C6 + RNG C5).

    Draw ``k`` category indices from a discrete distribution ``p`` (normalized) by
    the textbook inverse-CDF method: build the cumulative distribution once, draw
    ``k`` uniforms, and for each map ``u`` to the first category whose CDF crosses
    it. ``searchsorted(cdf, u)`` is realized as a COUNT reduction —
    ``idx = #{ j : cdf[j] < u }`` — the exclusive rank of ``u`` in the CDF. Four
    outlined stages, each an already-verified single-op / single-chain lowering:

      1. cumsum(p)          -> cdf[1,N]     (scan, ``row-reduction`` pass)
      2. tile cdf to [k,N]  -> cdfk[k,N]    (``layout`` pass; one CDF row per draw)
      3. rand u[k,1]        (seeded philox, ``rng`` pass)
      4. tile u to [k,N]    -> uk[k,N]      (``layout``; the draw broadcast over N)
      5. searchsorted (row-reduction chain): cmp = (cdfk < uk) elementwise ->
         select(cmp,1,0) -> reduce sum over N -> counts[k,1] = the drawn indices.

    Verify: the empirical category frequencies over a large ``k`` match ``p``
    (large-sample / chi-square within tolerance) + DETERMINISM. Statistical, not
    matched-seed (philox != torch's RNG).
    """
    Tc = _div_threads(N)
    Tt = _div_threads(N)
    Tr = 1  # rng over [k,1]: one column, one thread
    Ts = _div_threads(N)
    cumsum_stage = Stage(
        name="mn_cumsum", pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @mn_cumsum(%p: tensor<1x{N}xf32>) -> tensor<1x{N}xf32> {{
  %c = intent.scan %p {{fn = "sum", axis = 1 : i64}}
      : (tensor<1x{N}xf32>) -> tensor<1x{N}xf32>
  return %c : tensor<1x{N}xf32>
}}
""",
        block_threads=Tc, inputs=["p"],
        outputs=[Buf("cdf", (1, N), "f32")],
    )
    tile_stage = Stage(
        name="mn_tile", pass_name="convert-intent-layout-to-gpu",
        mlir_text=f"""
func.func @mn_tile(%c: tensor<1x{N}xf32>) -> tensor<{k}x{N}xf32> {{
  %t = intent.tile %c {{reps = [{k}, 1]}}
      : (tensor<1x{N}xf32>) -> tensor<{k}x{N}xf32>
  return %t : tensor<{k}x{N}xf32>
}}
""",
        block_threads=Tt, inputs=["cdf"],
        outputs=[Buf("cdfk", (k, N), "f32")],
    )
    rng_stage = Stage(
        name="mn_rng", pass_name="convert-intent-rng-to-gpu",
        mlir_text=f"""
func.func @mn_rng() -> tensor<{k}x1xf32> {{
  %u = intent.rand {{seed = {seed} : i64}} : tensor<{k}x1xf32>
  return %u : tensor<{k}x1xf32>
}}
""",
        block_threads=Tr, inputs=[],
        outputs=[Buf("u", (k, 1), "f32")],
    )
    utile_stage = Stage(
        name="mn_utile", pass_name="convert-intent-layout-to-gpu",
        mlir_text=f"""
func.func @mn_utile(%u: tensor<{k}x1xf32>) -> tensor<{k}x{N}xf32> {{
  %t = intent.tile %u {{reps = [1, {N}]}}
      : (tensor<{k}x1xf32>) -> tensor<{k}x{N}xf32>
  return %t : tensor<{k}x{N}xf32>
}}
""",
        block_threads=Tt, inputs=["u"],
        outputs=[Buf("uk", (k, N), "f32")],
    )
    search_stage = Stage(
        name="mn_search", pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @mn_search(%cdfk: tensor<{k}x{N}xf32>, %uk: tensor<{k}x{N}xf32>,
    %zero: tensor<f32>, %one: tensor<f32>) -> tensor<{k}x1xf32> {{
  %lt = intent.map_binary %cdfk, %uk {{fn = "lt"}}
      : (tensor<{k}x{N}xf32>, tensor<{k}x{N}xf32>) -> tensor<{k}x{N}xi1>
  %f = intent.select %lt, %one, %zero
      : (tensor<{k}x{N}xi1>, tensor<f32>, tensor<f32>) -> tensor<{k}x{N}xf32>
  %idx = intent.reduce %f {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}}
      : (tensor<{k}x{N}xf32>) -> tensor<{k}x1xf32>
  return %idx : tensor<{k}x1xf32>
}}
""",
        block_threads=Ts, inputs=["cdfk", "uk", "zero", "one"],
        outputs=[Buf("counts", (k, 1), "f32")],
    )
    return OpGraphProgram(
        [cumsum_stage, tile_stage, rng_stage, utile_stage, search_stage])


def per_token_group_quant_fp8_plan(G: int, group_size: int) -> OpGraphProgram:
    """per_token_group_quant_fp8(x) = grouped abs-max scale + clamp/divide (fp8).

    Matches ``flag_gems.per_token_group_quant_fp8`` (row-major, the vLLM
    block-fp8 quantizer). ``x`` is viewed as ``[G, group_size]`` (one group per
    row, groups contiguous). Per group:

      amax  = max(max(|x|), eps)          (grouped abs-max reduce, C-reduce)
      scale = amax / fp8_max              (fp8_max = 448 for e4m3fn)
      q     = clamp(x / scale, fp8_min, fp8_max)   then rounded to the e4m3 grid.

    The genuinely-IntentIR part is ONE row-reduction kernel: the grouped abs-max
    reduce + the eps-clamp + the per-group scale + the broadcast clamp/divide
    (the same reduce -> broadcast -> map shape softmax/cross_entropy use). The
    final e4m3-grid rounding is a *dtype cast* — emitting an ``f8E4M3FN`` store
    through NVVM/ptxas is a separate large mechanism (deferred, named), so the
    caller realizes the cast with the arch's ``float8_e4m3fn`` and verifies the
    DEQUANTIZED result (``q_fp8 * scale``) vs the real flag_gems dequant, plus
    the scales exactly.

    Inputs (env): x[G,group_size] f32; eps, invmax(=1/fp8_max), fmin, fmax
    rank-0 f32 scalars. Outputs: scale[G,1], q[G,group_size] (pre-cast clamp).
    """
    R = f"tensor<{G}x{group_size}xf32>"
    P = f"tensor<{G}x1xf32>"
    S = "tensor<f32>"
    T = _div_threads(group_size)
    stage = Stage(
        name="ptgq", pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @ptgq(%x: {R}, %eps: {S}, %invmax: {S}, %fmin: {S}, %fmax: {S})
    -> ({P}, {R}) {{
  %ax = intent.map_unary %x {{fn = "abs"}} : ({R}) -> {R}
  %amax = intent.reduce %ax {{fn = "max", axes = [1], assoc = "associative"}} : ({R}) -> {P}
  %ampos = intent.map_binary %amax, %eps {{fn = "max"}} : ({P}, {S}) -> {P}
  %scale = intent.map_binary %ampos, %invmax {{fn = "mul"}} : ({P}, {S}) -> {P}
  %div = intent.map_binary %x, %scale {{fn = "div"}} : ({R}, {P}) -> {R}
  %clo = intent.map_binary %div, %fmin {{fn = "max"}} : ({R}, {S}) -> {R}
  %q = intent.map_binary %clo, %fmax {{fn = "min"}} : ({R}, {S}) -> {R}
  return %scale, %q : {P}, {R}
}}
""",
        block_threads=T,
        inputs=["x", "eps", "invmax", "fmin", "fmax"],
        outputs=[Buf("scale", (G, 1), "f32"), Buf("q", (G, group_size), "f32")],
    )
    return OpGraphProgram([stage])


def reshape_and_cache_run(key, value, key_cache, value_cache, slot_mapping,
                          block_size: int, x: int):
    """reshape_and_cache (non-flash) = the x-vectorized cache-layout C3 scatter.

    The extra reshape bookkeeping the flash/MLA row-scatter (P6) deferred. Matches
    ``flag_gems.reshape_and_cache`` byte-for-byte. Each token ``t`` (slot
    ``s=slot_mapping[t]``, block ``s//block_size``, offset ``s%block_size``)
    writes its ``[num_heads, head_size]`` key/value into the vectorized cache:

      key_cache[num_blocks, num_heads, head_size/x, block_size, x]
      value_cache[num_blocks, num_heads, head_size, block_size]

    The per-element destination is a pure index remap (see the flag_gems kernel's
    ``tgt_key_idx``/``tgt_value_idx``): a deterministic function of (t, i, slot),
    NOT a plain row scatter. We materialize that destination flat index host-side
    and drive ONE ``intent.scatter`` (C3, the gather pass) into the flattened
    cache — the identical scatter mechanism the flash variant used, with the
    x-vectorized coordinate as the data-dependent address.

    Args are the flag_gems tensors (``key``/``value`` [Nt,H,hd], the pre-allocated
    caches, ``slot_mapping`` [Nt]). Writes the caches in place; returns nothing.
    """
    import torch

    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    Nt = int(slot_mapping.shape[0])
    H = int(key.shape[1])
    hd = int(key.shape[2])
    D = H * hd
    nblk = int(key_cache.shape[0])
    total = nblk * H * hd * block_size

    dev = key.device
    slot = slot_mapping.to(torch.int64)
    valid = slot >= 0
    t = torch.arange(Nt, device=dev, dtype=torch.int64).view(Nt, 1)
    i = torch.arange(D, device=dev, dtype=torch.int64).view(1, D)  # [1, H*hd]
    block_idx = (slot // block_size).view(Nt, 1)
    block_off = (slot % block_size).view(Nt, 1)
    head_idx = i // hd
    head_off = i % hd
    x_idx = head_off // x
    x_off = head_off % x
    tgt_key = (block_idx * H * (hd // x) * block_size * x
               + head_idx * (hd // x) * block_size * x
               + x_idx * block_size * x
               + block_off * x
               + x_off)
    tgt_val = (block_idx * H * hd * block_size
               + head_idx * hd * block_size
               + head_off * block_size
               + block_off)
    # invalid (slot<0) tokens: park them past the end so the scatter drops them
    # (flag_gems early-returns; we route to a throwaway tail row instead).
    park = torch.full_like(tgt_key, total)
    tgt_key = torch.where(valid.view(Nt, 1), tgt_key, park).reshape(Nt * D, 1).contiguous()
    tgt_val = torch.where(valid.view(Nt, 1), tgt_val, park).reshape(Nt * D, 1).contiguous()

    kc_flat = key_cache.reshape(total, 1)
    vc_flat = value_cache.reshape(total, 1)
    # pad one throwaway tail row for parked (dropped) writes.
    kc_pad = torch.cat([kc_flat, torch.zeros(1, 1, device=dev, dtype=kc_flat.dtype)], 0).contiguous()
    vc_pad = torch.cat([vc_flat, torch.zeros(1, 1, device=dev, dtype=vc_flat.dtype)], 0).contiguous()

    scat = RowReductionKernel(
        "rac", {}, 1, pass_name="convert-intent-gather-to-gpu",
        mlir_text=f"""
func.func @rac(%data: tensor<{total + 1}x1xf32>, %index: tensor<{Nt * D}x1xi64>,
    %src: tensor<{Nt * D}x1xf32>) -> tensor<{total + 1}x1xf32> {{
  %o = intent.scatter %data, %index, %src {{axis = 0 : i64, mode = "along_axis", reduce = "none"}}
      : (tensor<{total + 1}x1xf32>, tensor<{Nt * D}x1xi64>, tensor<{Nt * D}x1xf32>) -> tensor<{total + 1}x1xf32>
  return %o : tensor<{total + 1}x1xf32>
}}
""")
    ksrc = key.reshape(Nt * D, 1).contiguous()
    vsrc = value.reshape(Nt * D, 1).contiguous()
    scat.thunk([kc_pad, tgt_key, ksrc])()
    scat.thunk([vc_pad, tgt_val, vsrc])()
    torch.cuda.synchronize()
    key_cache.reshape(total, 1).copy_(kc_pad[:total])
    value_cache.reshape(total, 1).copy_(vc_pad[:total])


def moe_align_block_size_run(topk_ids, block_size: int, num_experts: int):
    """moe_align_block_size = sort tokens by expert + block-padded scatter.

    Matches ``flag_gems.moe_align_block_size`` EXACTLY (the vLLM fused-MoE routing
    bookkeeping). Returns ``(sorted_ids, expert_ids, num_tokens_post_pad)``.

    The core is the by-expert sort (C6b) + the block-padded cumulative scan (C6) +
    the placement scatter (C3):

      1. sort the composite key ``expert*numel + token`` ascending -> tokens
         GROUPED by expert, STABLE within an expert by original flat index (the
         unique token term breaks bitonic ties, so the argsort == flag_gems'
         stable order). The argsort indices are the by-expert token order.  [C6b]
      2. per-expert counts (histogram) -> block round-up ``round_up(cnt, bs)`` ->
         exclusive scan = ``cumsum_padded[e]`` = each expert's block-aligned
         start; the inclusive total is ``num_tokens_post_pad``.               [C6]
      3. each sorted position ``p`` (expert ``e``, within-expert rank ``r``) lands
         at ``rank_post_pad = cumsum_padded[e] + r``; scatter the token id there;
         unwritten slots stay ``numel`` (the padding marker).                 [C3]
      4. ``expert_ids[block] = e`` for blocks in expert ``e``'s padded region.

    The sort + scan + scatter are IntentIR kernels; the integer round-up / block
    bookkeeping is thin host glue (the opgraph-driver pattern, cf. ``unique_run``).
    """
    import torch

    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = int(flat.numel())
    max_pad = numel + num_experts * (block_size - 1)
    max_blocks = (max_pad + block_size - 1) // block_size

    # ---- 1. by-expert sort (C6b): composite key = expert*numel + token idx ----
    tok = torch.arange(numel, device=dev, dtype=torch.int64)
    key_f = (flat * numel + tok).to(torch.float32).reshape(1, numel).contiguous()
    Ts = _threads_pow2(numel)
    sort_k = RowReductionKernel(
        "mab_sort", {}, Ts, pass_name="convert-intent-sort-to-gpu",
        mlir_text=f"""
func.func @mab_sort(%x: tensor<1x{numel}xf32>)
    -> (tensor<1x{numel}xf32>, tensor<1x{numel}xi32>) {{
  %v, %i = intent.sort %x {{descending = false, axis = 1 : i64}}
      : (tensor<1x{numel}xf32>) -> (tensor<1x{numel}xf32>, tensor<1x{numel}xi32>)
  return %v, %i : tensor<1x{numel}xf32>, tensor<1x{numel}xi32>
}}
""")
    sorted_key = torch.empty(1, numel, device=dev, dtype=torch.float32).contiguous()
    order = torch.empty(1, numel, device=dev, dtype=torch.int32).contiguous()
    sort_k.thunk([key_f, sorted_key, order])()
    torch.cuda.synchronize()
    order = order.reshape(numel).to(torch.int64)          # by-expert token order
    sorted_expert = flat[order]                           # expert per sorted pos

    # ---- 2. counts -> block round-up -> exclusive scan (C6) ----
    counts = torch.bincount(flat, minlength=num_experts)[:num_experts]
    padded = ((counts + block_size - 1) // block_size) * block_size
    pad_row = padded.to(torch.float32).reshape(1, num_experts).contiguous()
    Tc = _div_threads(num_experts)
    scan_k = RowReductionKernel(
        "mab_scan", {}, Tc, pass_name="convert-intent-row-reduction-to-gpu",
        mlir_text=f"""
func.func @mab_scan(%p: tensor<1x{num_experts}xf32>) -> tensor<1x{num_experts}xf32> {{
  %c = intent.scan %p {{fn = "sum", axis = 1 : i64}}
      : (tensor<1x{num_experts}xf32>) -> tensor<1x{num_experts}xf32>
  return %c : tensor<1x{num_experts}xf32>
}}
""")
    incl = torch.empty(1, num_experts, device=dev, dtype=torch.float32).contiguous()
    scan_k.thunk([pad_row, incl])()
    torch.cuda.synchronize()
    incl = incl.reshape(num_experts).round().to(torch.int64)  # inclusive padded scan
    cumsum_padded = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), incl])
    num_post_pad = int(incl[-1].item()) if num_experts > 0 else 0

    # within-expert rank r[p] = p - group_start[e_p] (unpadded exclusive count scan)
    grp_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                           counts.cumsum(0)])[:num_experts]
    pos = torch.arange(numel, device=dev, dtype=torch.int64)
    rank = pos - grp_start[sorted_expert]
    dest = cumsum_padded[sorted_expert] + rank            # rank_post_pad

    # ---- 3. placement scatter (C3): sorted_ids[dest[p]] = order[p] ----
    sorted_ids = torch.full((max_pad,), numel, device=dev, dtype=torch.int64)
    padrow = torch.zeros(1, device=dev, dtype=torch.int64)   # throwaway tail
    buf = torch.cat([sorted_ids.to(torch.float32), padrow.to(torch.float32)]).reshape(max_pad + 1, 1).contiguous()
    idx = dest.reshape(numel, 1).contiguous()
    src = order.to(torch.float32).reshape(numel, 1).contiguous()
    scat = RowReductionKernel(
        "mab_scat", {}, 1, pass_name="convert-intent-gather-to-gpu",
        mlir_text=f"""
func.func @mab_scat(%data: tensor<{max_pad + 1}x1xf32>, %index: tensor<{numel}x1xi64>,
    %src: tensor<{numel}x1xf32>) -> tensor<{max_pad + 1}x1xf32> {{
  %o = intent.scatter %data, %index, %src {{axis = 0 : i64, mode = "along_axis", reduce = "none"}}
      : (tensor<{max_pad + 1}x1xf32>, tensor<{numel}x1xi64>, tensor<{numel}x1xf32>) -> tensor<{max_pad + 1}x1xf32>
  return %o : tensor<{max_pad + 1}x1xf32>
}}
""")
    scat.thunk([buf, idx, src])()
    torch.cuda.synchronize()
    sorted_ids = buf[:max_pad, 0].round().to(torch.int32)

    # ---- 4. expert_ids per block ----
    expert_ids = torch.zeros(max_blocks, device=dev, dtype=torch.int32)
    for e in range(num_experts):
        b0 = int(cumsum_padded[e].item()) // block_size
        b1 = int(cumsum_padded[e + 1].item()) // block_size
        if b1 > b0:
            expert_ids[b0:b1] = e
    npp = torch.tensor([num_post_pad], device=dev, dtype=torch.int32)
    return sorted_ids, expert_ids, npp


def unique_run(x, block_threads: int | None = None):
    """unique(x) = sort(asc) -> adjacent-dedup -> C8 compaction (sort+dedup graph).

    The op-graph the coverage task named: a data-dependent output extent (# of
    distinct values). ``x`` is a 1-D CUDA f32 tensor; returns the sorted distinct
    values (== ``torch.unique(x, sorted=True)``). Four hand-authored stages wired
    by device buffers (the last is the C8 multi-launch compaction, so this is
    driven directly rather than through :class:`OpGraphProgram`):

      1. sort ascending                          -> sorted[1,N]   (sort C6b)
      2. gather prev[j] = sorted[max(j-1,0)]     -> prev[1,N]     (gather C3)
      3. keep[j] = max(sorted[j]!=prev[j], j==0) -> keep[1,N] f32 (elementwise:
         eq + select + max with a host [1,0,..] boundary mask — the neighbor
         compare dedup map; j==0 is always kept, being the row-major boundary)
      4. masked_select(sorted, keep)             -> distinct[nnz] (C8 compaction)
    """
    import torch

    from intentir.backend.cuda.kernel_driver import CompactKernel, RowReductionKernel

    N = int(x.numel())
    row = x.reshape(1, N).contiguous().float()
    Ts = _threads_pow2(N)
    Tg = _div_threads(N)
    Tk = _div_threads(N)
    Tc = _div_threads(N)

    sort_k = RowReductionKernel(
        "uq_sort", {}, Ts, pass_name="convert-intent-sort-to-gpu",
        mlir_text=f"""
func.func @uq_sort(%x: tensor<1x{N}xf32>)
    -> (tensor<1x{N}xf32>, tensor<1x{N}xi32>) {{
  %v, %i = intent.sort %x {{descending = false, axis = 1 : i64}}
      : (tensor<1x{N}xf32>) -> (tensor<1x{N}xf32>, tensor<1x{N}xi32>)
  return %v, %i : tensor<1x{N}xf32>, tensor<1x{N}xi32>
}}
""")
    gather_k = RowReductionKernel(
        "uq_gather", {}, Tg, pass_name="convert-intent-gather-to-gpu",
        mlir_text=f"""
func.func @uq_gather(%data: tensor<1x{N}xf32>, %index: tensor<1x{N}xi64>)
    -> tensor<1x{N}xf32> {{
  %g = intent.gather %data, %index {{axis = 1 : i64, mode = "along_axis"}}
      : (tensor<1x{N}xf32>, tensor<1x{N}xi64>) -> tensor<1x{N}xf32>
  return %g : tensor<1x{N}xf32>
}}
""")
    keep_k = RowReductionKernel(
        "uq_keep", {}, Tk, pass_name="convert-intent-elementwise-to-gpu",
        mlir_text=f"""
func.func @uq_keep(%sorted: tensor<1x{N}xf32>, %prev: tensor<1x{N}xf32>,
    %first: tensor<1x{N}xf32>, %zero: tensor<f32>, %one: tensor<f32>)
    -> tensor<1x{N}xf32> {{
  %eqp = intent.map_binary %sorted, %prev {{fn = "eq"}}
      : (tensor<1x{N}xf32>, tensor<1x{N}xf32>) -> tensor<1x{N}xi1>
  %neqf = intent.select %eqp, %zero, %one
      : (tensor<1x{N}xi1>, tensor<f32>, tensor<f32>) -> tensor<1x{N}xf32>
  %keep = intent.map_binary %neqf, %first {{fn = "max"}}
      : (tensor<1x{N}xf32>, tensor<1x{N}xf32>) -> tensor<1x{N}xf32>
  return %keep : tensor<1x{N}xf32>
}}
""")
    sel_k = CompactKernel(
        "uq_sel", Tc,
        mlir_text=f"""
func.func @uq_sel(%data: tensor<1x{N}xf32>, %mask: tensor<1x{N}xf32>)
    -> tensor<?xf32> {{
  %r = intent.masked_select %data, %mask
      : (tensor<1x{N}xf32>, tensor<1x{N}xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}}
""")

    sorted_v = torch.empty(1, N, device="cuda", dtype=torch.float32).contiguous()
    perm = torch.empty(1, N, device="cuda", dtype=torch.int32).contiguous()
    sort_k.thunk([row, sorted_v, perm])()

    previdx = torch.arange(-1, N - 1, device="cuda", dtype=torch.int64)
    previdx.clamp_(min=0)
    previdx = previdx.reshape(1, N).contiguous()
    prev = torch.empty(1, N, device="cuda", dtype=torch.float32).contiguous()
    gather_k.thunk([sorted_v, previdx, prev])()

    first = torch.zeros(1, N, device="cuda", dtype=torch.float32).contiguous()
    first[0, 0] = 1.0  # the row-major boundary element is always distinct
    zero = torch.zeros((), device="cuda", dtype=torch.float32)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    keep = torch.empty(1, N, device="cuda", dtype=torch.float32).contiguous()
    keep_k.thunk([sorted_v, prev, first, zero, one, keep])()

    out = torch.zeros(N, device="cuda", dtype=torch.float32).contiguous()
    count = torch.zeros(1, device="cuda", dtype=torch.int64).contiguous()
    sel_k.thunk([sorted_v, keep, out, count])()
    torch.cuda.synchronize()
    nnz = int(count.item())
    return out[:nnz]


__all__ = [
    "Buf", "Stage", "OpGraphProgram",
    "quantile_plan", "topk_softmax_plan", "cross_entropy_plan",
    "randperm_plan", "multinomial_plan", "unique_run",
    "per_token_group_quant_fp8_plan", "reshape_and_cache_run",
    "moe_align_block_size_run",
]
