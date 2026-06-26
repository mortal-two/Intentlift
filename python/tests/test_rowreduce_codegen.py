"""From-intent row-reduction codegen -- recognizer + generated-kernel correctness.

This is the *generation* test for task ``06-26-rowreduce-codegen-from-intent``: the
backend, given only the rms_norm INTENT (an ``IntentFunction``, not a hand-written
MLIR string), must recognize the row-reduction pattern and GENERATE the single
high-perf kernel -- not the ~8-9 naive ``linalg.generic`` kernels the
``-convert-intent-to-linalg`` path emits (that path additionally cannot even ptxas
past the ``sqrt`` -> ``__nv_sqrtf`` libdevice dead-end).

Two tiers:

* **Recognizer tests (always-on, pure-Python, no GPU):** load the real
  ``corpus/rms_norm/intent.json`` -> ``IntentFunction``; assert
  ``recognize_row_reduction`` returns a well-formed plan, and returns ``None`` for
  a pure pointwise chain (silu / gelu_tanh -- no ``reduce``). Module top imports
  only the model + the codegen module, so these run with no CUDA.
* **From-intent codegen tests (GPU-gated):** assert the generated path produces
  exactly ONE gpu kernel and that its result certifies vs the fp32 rms_norm
  reference AND (if liger importable) vs ``rms_norm_forward`` at rel < 1e-3
  (expect ~1e-7), at (4096,1024) and the non-power-of-2 (4096,1000).
"""

import json
from pathlib import Path

import pytest

from intentir.backend.rowreduce_codegen import (
    RowReductionPlan,
    codegen_row_reduction,
    recognize_row_reduction,
    run_from_intent,
)
from intentir.ir.model import IntentFunction

_CORPUS = Path(__file__).resolve().parents[2] / "corpus"
_EPS = 1e-6


def _load(name: str) -> IntentFunction:
    return IntentFunction.from_json(json.loads((_CORPUS / name / "intent.json").read_text()))


# --------------------------------------------------------------------------- #
# Tier 1 -- recognizer (always-on, pure-Python).                              #
# --------------------------------------------------------------------------- #
def test_recognize_rms_norm_returns_plan():
    """The real rms_norm intent is recognized as a row-reduction plan."""
    fn = _load("rms_norm")
    plan = recognize_row_reduction(fn)
    assert isinstance(plan, RowReductionPlan)
    assert plan.kind == "rms_norm"
    assert plan.data_input == "X"
    assert plan.weight_input == "W"
    assert plan.bias_input is None              # offset = 0 (llama), no bias
    assert plan.output == "Y"
    # exactly one reduction over the last axis: mean.
    assert len(plan.reductions) == 1
    red = plan.reductions[0]
    assert red.fn == "mean"
    assert red.axis == 1                          # last axis of [M, N]
    # dim symbols carried for later resolution from `dims`.
    assert (plan.row_dim, plan.col_dim) == ("M", "N")


@pytest.mark.parametrize("name", ["silu", "gelu_tanh"])
def test_recognize_pointwise_returns_none(name):
    """A pure pointwise chain (no reduce) is NOT a row-reduction -> None.

    This is the negative case that lets callers fall back to the naive path.
    """
    fn = _load(name)
    assert recognize_row_reduction(fn) is None


def test_codegen_rejects_non_row_reduction():
    """codegen on a non-recognized intent raises (no silent wrong kernel)."""
    fn = _load("silu")
    with pytest.raises(ValueError):
        codegen_row_reduction(fn, {"M": 4096, "N": 1024})


# --------------------------------------------------------------------------- #
# Tier 2 -- from-intent codegen on the GPU (gated at function level so the      #
# recognizer tests above stay always-on).                                     #
# --------------------------------------------------------------------------- #
def _gpu_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA GPU")
    return torch


@pytest.mark.parametrize("rows,cols", [(4096, 1024), (4096, 1000)])
def test_from_intent_one_kernel_and_correct(rows, cols):
    """Intent in -> ONE generated kernel out -> correct vs fp32 ref (and liger).

    (4096,1000) is non-power-of-2, exercising the BLOCK_SIZE=next_pow2 tail mask.
    Asserts the generated path produces exactly ONE gpu kernel (vs the ~8-9 naive
    linalg.generic kernels) and certifies the result.
    """
    torch = _gpu_or_skip()
    fn = _load("rms_norm")
    dims = {"M": rows, "N": cols}

    # GENERATION: the backend emits/lowers ONE kernel from the intent.
    compiled = codegen_row_reduction(fn, dims)
    assert (compiled.rows, compiled.cols) == (rows, cols)

    # Measure the headline "ONE kernel, not ~8 naive ones" claim at the MLIR
    # level: lower the generated kernel text and count the gpu kernels. (The
    # naive -convert-intent-to-linalg path lowers this same op to 9
    # gpu.launch_func sites and additionally fails ptxas on the sqrt libdevice
    # dead-end -- so one runnable kernel is the win, not just a speedup.)
    from intentir.backend.compile import lower_memref_module_to_kernels
    from intentir.bench.rmsnorm_emit import emit_rmsnorm

    n_kernels = len(
        lower_memref_module_to_kernels(emit_rmsnorm(rows, cols, compiled.threads))
    )
    assert n_kernels == 1, f"expected ONE generated kernel, got {n_kernels}"
    assert compiled.n_kernels == 1

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    w = torch.randn(cols, device="cuda", dtype=torch.float32)
    eps = torch.tensor(_EPS, device="cuda", dtype=torch.float32)

    y = run_from_intent(fn, {"X": x, "W": w, "eps": eps}, dims)
    assert tuple(y.shape) == (rows, cols)

    # fp32 reference (llama): rstd = rsqrt(mean(x^2)+eps); y = (x*rstd)*w.
    rstd = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + _EPS)
    y_ref = (x.float() * rstd) * w.float()

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

    rel_ref = rel(y, y_ref)
    assert rel_ref < 1e-3, f"({rows},{cols}) from-intent Y vs fp32 ref rel={rel_ref}"

    # If the real liger kernel is importable, certify against it too (~1e-7).
    try:
        from liger_kernel.ops.rms_norm import rms_norm_forward
    except Exception:  # pragma: no cover - liger optional
        rms_norm_forward = None
    if rms_norm_forward is not None:
        y_lig, *_ = rms_norm_forward(x, w, _EPS, 0.0, "llama", False)
        torch.cuda.synchronize()
        rel_lig = rel(y, y_lig)
        assert rel_lig < 1e-3, f"({rows},{cols}) from-intent Y vs liger rel={rel_lig}"
