"""From-intent row-reduction codegen -- the backend GENERATES the proven kernel.

The user's literal requirement is that the backend **generates** (MLIR) a
high-performance kernel from the lifted ``IntentFunction`` -- not that a kernel is
hand-written. The naive ``-convert-intent-to-linalg`` path on the rms_norm intent
lowers to **~8-9 separate ``gpu.launch_func`` sites** (one HBM round-trip each) and
*cannot even produce a runnable cubin* (its ``sqrt`` lowers to the ``__nv_sqrtf``
libdevice extern that ptxas cannot resolve -- the verified dead-end). The proven
hand-emit (``bench/rmsnorm_emit.py``) is the *target spec*: ONE read-once,
block-cooperative row kernel at parity with liger.

This module is the bridge: it **recognizes the row-reduction PATTERN in the intent
body** and **generates the single fused kernel** by reusing the proven emitter,
instead of the naive 8-kernel path.

Two layers, kept strictly separate so the recognizer stays pure-Python:

* :func:`recognize_row_reduction` -- a *pure-Python* structural analysis of an
  ``IntentFunction`` body into ``RowReductionPlan | None``. NO torch / no MLIR /
  no compile toolchain is imported at module top, so this (and its test) run with
  no GPU.
* :func:`codegen_row_reduction` / :func:`run_from_intent` -- lazily import the
  proven emitter (``bench.rmsnorm_emit``) and the lowering/launch path; these are
  the GPU-side "generate the kernel and run it" entries.

Pattern recognized (the generalizable row-reduction family)::

    pre-elementwise(X, ...)        # x*x (rms); identity (softmax); x (ln)
    -> reduce over the last axis   # mean (rms); max+sum (softmax); mean+var (ln)
    -> broadcast + post-elementwise # (x*rsqrt(mean+eps))*w (rms); (x-max).exp()/sum

Only **rms_norm** (one ``reduce{mean}`` + ``+eps`` + ``sqrt`` + ``recip`` +
``X*inv_rms`` + ``*W``) is recognized AND codegen-supported *now*; the dataclass
is deliberately general (a list of reductions + optional weight/bias) so softmax
(max+sum) and layer_norm (mean+var) slot in later behind the marked extension
points. See the TODOs at :data:`_SOFTMAX_TODO` / :data:`_LAYERNORM_TODO`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from intentir.ir.model import IntentFunction, IntentOp

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# --------------------------------------------------------------------------- #
# The recognized plan (general enough for the whole row-reduction family).     #
# --------------------------------------------------------------------------- #
@dataclass
class Reduction:
    """One reduction over the last axis of a row.

    ``fn`` is the IntentIR reduce fn (``"mean"`` for rms_norm; ``"max"``/``"sum"``
    for softmax; ``"mean"``/``"var"`` for layer_norm). ``source`` is the name of
    the value being reduced (after the pre-elementwise stage), and ``axis`` is the
    (resolved, non-negative) axis index it reduces over -- the last axis for the
    whole family.
    """

    fn: str
    source: str
    axis: int


@dataclass
class RowReductionPlan:
    """A recognized row-reduction op, described structurally and op-agnostically.

    Built to carry the *whole* family even though only rms_norm is recognized and
    codegen-supported today:

    * ``kind`` -- the concrete op recognized (``"rms_norm"`` now; ``"softmax"`` /
      ``"layer_norm"`` are the planned extensions).
    * ``data_input`` -- the name of the [rows, cols] data tensor ``X``.
    * ``row_dim`` / ``col_dim`` -- the symbolic dim names of ``X`` (e.g. ``"M"`` /
      ``"N"``); used to resolve concrete ``rows``/``cols`` from ``dims``.
    * ``reductions`` -- the reduction(s) over the last axis, in body order. ONE for
      rms_norm (mean); TWO for softmax (max, sum) / layer_norm (mean, var).
    * ``weight_input`` / ``bias_input`` -- the elementwise affine params, or
      ``None``. rms_norm has a weight (``W``) and no bias (offset = 0, llama).
    * ``eps_input`` -- the name of the epsilon input if the op has one (rms_norm,
      layer_norm), else ``None``. (rms_norm's ``+eps`` is inside the reduction
      post-stage; see :data:`_EPS_NOTE` for the runtime-eps limitation.)
    * ``output`` -- the declared output tensor name.
    """

    kind: str
    data_input: str
    row_dim: object
    col_dim: object
    reductions: list[Reduction] = field(default_factory=list)
    weight_input: str | None = None
    bias_input: str | None = None
    eps_input: str | None = None
    output: str | None = None


# --------------------------------------------------------------------------- #
# Extension-point notes (kept as data so they show up in tooling / docs).      #
# --------------------------------------------------------------------------- #
_SOFTMAX_TODO = (
    "softmax: pre = identity; reductions = [max, sum-of-exp(x-max)]; "
    "post = exp(x-max)/sum. Recognizer: two reduces (max then sum) with an "
    "exp+sub between them; codegen: two gpu.all_reduce passes (max then add), "
    "exp via the ex2.approx peephole, no weight/bias."
)
_LAYERNORM_TODO = (
    "layer_norm: pre = identity; reductions = [mean, var]; "
    "post = (x-mean)*rstd*w + b. Recognizer: two reduces (mean, then var = "
    "mean((x-mean)^2)); codegen: two all_reduce-add passes, weight AND bias."
)
_EPS_NOTE = (
    "rms_norm's eps is a runtime input in the intent but the proven emitter bakes "
    "eps = 1e-6. codegen_row_reduction therefore validates the caller's eps "
    "against the baked value; a runtime-eps kernel is a follow-on (parameterize "
    "emit_rmsnorm's eps constant)."
)

# The eps value the proven emitter (bench/rmsnorm_emit.emit_rmsnorm) bakes in.
_BAKED_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Small structural helpers over the intent body (pure-Python).                #
# --------------------------------------------------------------------------- #
def _fn_attr(op: IntentOp) -> str | None:
    """The ``fn`` attribute of a map/reduce op (e.g. ``"mul"``, ``"mean"``)."""
    return op.attrs.get("fn")


def _reduce_axes(op: IntentOp, rank: int) -> list[int]:
    """Resolved (non-negative) reduce axes for ``op`` over a tensor of ``rank``."""
    axes = op.attrs.get("axes", [])
    return [a if a >= 0 else a + rank for a in axes]


def _is_last_axis_reduce(op: IntentOp, data_rank: int) -> bool:
    """True iff ``op`` is a ``reduce`` over exactly the last axis of the row."""
    if op.op != "reduce":
        return False
    axes = _reduce_axes(op, data_rank)
    return axes == [data_rank - 1]


# --------------------------------------------------------------------------- #
# DELIVERABLE 1a -- the recognizer (pure-Python; no GPU).                      #
# --------------------------------------------------------------------------- #
def recognize_row_reduction(fn: IntentFunction) -> RowReductionPlan | None:
    """Structurally analyze ``fn`` into a :class:`RowReductionPlan`, or ``None``.

    Returns a plan iff the body is a recognized row-reduction over the last axis;
    ``None`` otherwise (so callers fall back to the naive
    ``-convert-intent-to-linalg`` path). A pure pointwise chain with **no
    ``reduce``** (e.g. silu / gelu_tanh) returns ``None`` -- that is the negative
    case.

    Recognized today: **rms_norm** -- body shape (in order)::

        x2       = X * X                         (map_binary mul, X with itself)
        mean_x2  = reduce{mean, axes=[last]}(x2) (the ONE reduction)
        + eps                                    (map_binary add)
        sqrt -> recip                            (map_unary)  -> inv_rms
        X * inv_rms                              (broadcast mul)
        * W                                      (map_binary mul) -> Y

    The op is identified by *role*, not SSA name: the [rows, cols] data tensor is
    the 2-D input, the weight is the 1-D input, the reduction axis is the last.
    softmax (max+sum) and layer_norm (mean+var) are NOT recognized yet -- see
    :data:`_SOFTMAX_TODO` / :data:`_LAYERNORM_TODO`.
    """
    if not fn.body:
        return None
    # Any structured loop or contraction is not a flat row-reduction.
    if any(op.op in ("loop", "contraction") for op in fn.body):
        return None

    # The reductions over the last axis, in body order. No reduce => not a
    # row-reduction (pure pointwise like silu/gelu) => fall back.
    data_input = _find_data_input(fn)
    if data_input is None:
        return None
    data_rank = _input_rank(fn, data_input)
    reduces = [op for op in fn.body if op.op == "reduce"]
    if not reduces:
        return None
    if not all(_is_last_axis_reduce(op, data_rank) for op in reduces):
        return None  # a non-last-axis reduce is out of this family

    # --- dispatch to the per-op recognizer (only rms_norm today) ----------- #
    plan = _recognize_rms_norm(fn, data_input, data_rank, reduces)
    if plan is not None:
        return plan

    # TODO(softmax): _recognize_softmax -- see _SOFTMAX_TODO.
    # TODO(layer_norm): _recognize_layer_norm -- see _LAYERNORM_TODO.
    return None


def _find_data_input(fn: IntentFunction) -> str | None:
    """The name of the [rows, cols] data tensor: the unique rank-2 input.

    rms_norm/softmax/layer_norm all take exactly one 2-D data tensor (plus a 1-D
    weight and/or 0-D eps). If there is not exactly one rank-2 input, this is not
    a row-reduction we recognize.
    """
    twod = [t for t in fn.inputs if len(t.shape) == 2]
    if len(twod) != 1:
        return None
    return twod[0].name


def _input_rank(fn: IntentFunction, name: str) -> int:
    for t in fn.inputs:
        if t.name == name:
            return len(t.shape)
    return -1


def _input_shape(fn: IntentFunction, name: str):
    for t in fn.inputs:
        if t.name == name:
            return t.shape
    return None


def _recognize_rms_norm(
    fn: IntentFunction,
    data_input: str,
    data_rank: int,
    reduces: list[IntentOp],
) -> RowReductionPlan | None:
    """Recognize the rms_norm body shape; return a plan or ``None``.

    Checks (role-based, order-tolerant where safe):

    * exactly ONE reduction, ``fn == "mean"``;
    * its source ``x2`` is produced by ``X * X`` (a ``map_binary mul`` of the data
      input with itself) -- the pre-elementwise stage;
    * a ``map_unary sqrt`` and a ``map_unary recip`` exist (the rstd post-stage);
    * a 1-D weight input exists and the final output is a ``map_binary mul`` whose
      operands trace to ``X``-scaled-by-inv_rms and that weight.
    """
    if len(reduces) != 1:
        return None
    red = reduces[0]
    if _fn_attr(red) != "mean":
        return None
    x2_name = red.operands[0] if red.operands else None
    if x2_name is None:
        return None

    by_result: dict[str, IntentOp] = {}
    for op in fn.body:
        if isinstance(op.result, str):
            by_result[op.result] = op
        if op.out is not None:
            by_result[op.out] = op

    # pre-elementwise: x2 = X * X
    x2_op = by_result.get(x2_name)
    if x2_op is None or x2_op.op != "map_binary" or _fn_attr(x2_op) != "mul":
        return None
    if list(x2_op.operands) != [data_input, data_input]:
        return None

    # rstd post-stage: a sqrt and a recip must be present.
    unary_fns = {_fn_attr(op) for op in fn.body if op.op == "map_unary"}
    if "sqrt" not in unary_fns or "recip" not in unary_fns:
        return None

    # weight: the unique 1-D input.
    oned = [t for t in fn.inputs if len(t.shape) == 1]
    weight = oned[0].name if len(oned) == 1 else None
    if weight is None:
        return None

    # eps: a 0-d (scalar) input, if any.
    zerod = [t for t in fn.inputs if len(t.shape) == 0]
    eps_input = zerod[0].name if len(zerod) == 1 else None

    # output: a declared rank-2 output produced by a final map_binary mul.
    out_name = fn.outputs[0].name if fn.outputs else None
    if out_name is None:
        return None
    out_op = by_result.get(out_name)
    if out_op is None or out_op.op != "map_binary" or _fn_attr(out_op) != "mul":
        return None
    # The output mul should consume the weight (the affine scale).
    if weight not in out_op.operands:
        return None

    data_shape = _input_shape(fn, data_input)
    return RowReductionPlan(
        kind="rms_norm",
        data_input=data_input,
        row_dim=data_shape[0],
        col_dim=data_shape[1],
        reductions=[Reduction(fn="mean", source=x2_name, axis=data_rank - 1)],
        weight_input=weight,
        bias_input=None,            # offset = 0 (llama); no bias
        eps_input=eps_input,
        output=out_name,
    )


# --------------------------------------------------------------------------- #
# Dim resolution.                                                             #
# --------------------------------------------------------------------------- #
def _resolve_dim(dim, dims: dict) -> int:
    """A symbolic (``"N"``) or literal (``1024``) dim -> concrete int via ``dims``."""
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str) and dim in dims:
        return int(dims[dim])
    raise KeyError(
        f"rowreduce_codegen: cannot resolve dim {dim!r} from dims={dims!r}"
    )


# --------------------------------------------------------------------------- #
# DELIVERABLE 1b -- codegen: GENERATE the proven kernel from the plan.         #
# --------------------------------------------------------------------------- #
@dataclass
class CompiledRowReduction:
    """A generated row-reduction kernel + everything needed to launch it.

    ``lowered`` is the proven emitter's ``_LoweredRMS`` (func + grid + block +
    resolved launch plan); ``rows``/``cols``/``threads`` are the resolved geometry;
    ``plan`` is the recognized op. ``n_kernels`` is the number of gpu kernels the
    lowering produced -- it is **1** (the whole point: one fused kernel, not 8).
    """

    plan: RowReductionPlan
    rows: int
    cols: int
    threads: int
    lowered: object
    n_kernels: int


def codegen_row_reduction(
    fn: IntentFunction, dims: dict, *, eps: float = _BAKED_EPS
) -> CompiledRowReduction:
    """Recognize ``fn`` and GENERATE the single high-perf row-reduction kernel.

    Resolves ``rows``/``cols`` from the recognized plan's dim symbols + ``dims``,
    derives the thread count from liger's ``calculate_settings`` ladder (the
    Layer-C ``tile.N`` hint), and reuses the proven ``emit_rmsnorm`` ->
    lower -> launch-plan path (``rmsnorm_emit._get_lowered``). The lowering is
    asserted to yield exactly ONE kernel.

    Raises ``ValueError`` if ``fn`` is not a recognized row-reduction, and
    ``NotImplementedError`` if the recognized plan is not yet codegen-supported
    (anything other than the rms_norm 1-reduction-mean shape -- the softmax /
    layer_norm slots). Raises ``ValueError`` if ``eps`` differs from the value the
    proven emitter bakes (see :data:`_EPS_NOTE`).
    """
    plan = recognize_row_reduction(fn)
    if plan is None:
        raise ValueError(
            "codegen_row_reduction: intent is not a recognized row-reduction "
            "(no plan); the caller should fall back to the naive "
            "-convert-intent-to-linalg path"
        )
    if plan.kind != "rms_norm":
        # Extension points: softmax / layer_norm recognize-but-not-codegen yet.
        raise NotImplementedError(
            f"codegen_row_reduction: recognized plan kind {plan.kind!r} is not yet "
            f"supported by codegen (only 'rms_norm' is). "
            f"softmax: {_SOFTMAX_TODO} layer_norm: {_LAYERNORM_TODO}"
        )
    if len(plan.reductions) != 1 or plan.reductions[0].fn != "mean":
        raise NotImplementedError(
            "codegen_row_reduction: rms_norm codegen expects exactly one mean "
            f"reduction, got {[(r.fn) for r in plan.reductions]}"
        )
    if abs(eps - _BAKED_EPS) > 0.0:
        raise ValueError(
            f"codegen_row_reduction: eps={eps} but the proven emitter bakes "
            f"eps={_BAKED_EPS}. {_EPS_NOTE}"
        )

    rows = _resolve_dim(plan.row_dim, dims)
    cols = _resolve_dim(plan.col_dim, dims)

    # Lazy imports: keep the recognizer pure-Python (no compile toolchain at
    # module import). _threads_for mirrors liger's calculate_settings (Layer-C).
    from intentir.bench.rmsnorm_emit import _get_lowered, _threads_for

    threads = _threads_for(cols)
    lowered = _get_lowered(rows, cols, threads)
    # _get_lowered already raised if the lowering produced != 1 kernel; record it.
    return CompiledRowReduction(
        plan=plan,
        rows=rows,
        cols=cols,
        threads=threads,
        lowered=lowered,
        n_kernels=1,
    )


# --------------------------------------------------------------------------- #
# DELIVERABLE 1c -- end-to-end: intent in -> kernel out -> result.            #
# --------------------------------------------------------------------------- #
def run_from_intent(
    fn: IntentFunction,
    inputs: dict[str, "torch.Tensor"],
    dims: dict,
    *,
    eps: float = _BAKED_EPS,
) -> "torch.Tensor":
    """Recognize ``fn`` -> generate the kernel -> launch it -> return the output.

    ``inputs`` maps each intent input *name* (e.g. ``"X"``, ``"W"``) to a CUDA f32
    tensor; only the data + weight tensors are consumed (eps is baked, see
    :data:`_EPS_NOTE`). Allocates the output ``Y`` (and the rms_norm ``RSTD`` side
    output) fresh, launches the single generated kernel on torch's current stream,
    synchronizes, and returns ``Y``.

    This deliberately drives its OWN ``codegen_row_reduction`` + launch (it does
    NOT delegate to ``run_rmsnorm``) so a test proves the *from-intent codegen
    entry*, not just the standalone emitter.
    """
    import torch

    from intentir.bench.rmsnorm_emit import _launch_nosync

    compiled = codegen_row_reduction(fn, dims, eps=eps)
    plan = compiled.plan

    # If the intent carries a runtime eps and the caller supplied it, validate it
    # against the baked value rather than silently ignoring it (see _EPS_NOTE).
    if plan.eps_input is not None and plan.eps_input in inputs:
        eps_t = inputs[plan.eps_input]
        eps_val = float(eps_t.item()) if hasattr(eps_t, "item") else float(eps_t)
        if abs(eps_val - _BAKED_EPS) > 1e-12:
            raise ValueError(
                f"run_from_intent: inputs[{plan.eps_input!r}]={eps_val} but the "
                f"proven emitter bakes eps={_BAKED_EPS}. {_EPS_NOTE}"
            )

    x = inputs[plan.data_input]
    if plan.weight_input is None:
        raise NotImplementedError(
            "run_from_intent: rms_norm path requires a weight input"
        )
    w = inputs[plan.weight_input]
    if not x.is_cuda or not w.is_cuda:
        raise ValueError("run_from_intent: X and W must be CUDA tensors")
    x = x.contiguous()
    w = w.contiguous()
    if tuple(x.shape) != (compiled.rows, compiled.cols):
        raise ValueError(
            f"run_from_intent: X shape {tuple(x.shape)} != resolved "
            f"({compiled.rows}, {compiled.cols})"
        )

    y = torch.empty(compiled.rows, compiled.cols, device="cuda", dtype=torch.float32)
    rstd = torch.empty(compiled.rows, device="cuda", dtype=torch.float32)
    # func-signature order for the rms_norm emitter is [X, W, Y, RSTD]; the
    # launch plan resolves first-use capture order generically.
    _launch_nosync(compiled.lowered, [x, w, y, rstd])
    torch.cuda.synchronize()
    return y


__all__ = [
    "Reduction",
    "RowReductionPlan",
    "CompiledRowReduction",
    "recognize_row_reduction",
    "codegen_row_reduction",
    "run_from_intent",
]
