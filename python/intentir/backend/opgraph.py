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

from intentir.benchmarks.row_reduction import RowReductionKernel

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


__all__ = [
    "Buf", "Stage", "OpGraphProgram",
    "quantile_plan", "topk_softmax_plan", "cross_entropy_plan",
]
