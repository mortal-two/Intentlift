"""Tie execution back to our OWN executable spec — two distinct checks.

Every other execution test compares against an EXTERNAL reference (torch / Triton
/ an inline numpy recompute). Those prove the *algorithm* is right. The two tests
here instead compare against ``verify/interpret.py`` run on the JSON IR
(``test/flashattention_fwd.intent.json``) — our executable spec.

1. ``test_lifting_certificate_intent_dialect`` — THE certified-lifting check.
   ``run_program("attention.mlir")`` runs ``convert-intent-to-linalg`` on a real
   ``intent``-dialect op-graph (contraction/reduce/map_binary/map_unary), so the
   executed result IS *derived from the IR through the compiler's lowering*.
   Agreement with ``interpret(JSON)`` certifies that lowering preserved the IR's
   semantics. (attention.mlir takes host-pre-scaled Q, so softmax(q·kᵀ)·v equals
   flash with that scale.)

2. ``test_fused_emitter_matches_interpreter`` — a cross-IMPLEMENTATION correctness
   check, NOT a lowering certificate. ``run_fused_flash`` → ``emit_fused_flash``
   emits standard ``scf.parallel``+``memref`` MLIR directly and never lowers the
   intent dialect — it is an independent re-implementation. Its agreement with the
   interpreter across BK groupings confirms the JSON spec is realizable and
   numerically faithful (and that the fused online-softmax recurrence is tile-
   invariant), but it does not exercise the dialect lowering. The C++ ``intent.loop``
   lowering that *would* put the fused path through the compiler is M8 (unbuilt).

Skips cleanly without a GPU / CUDA / build.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT

_TEST = pathlib.Path(__file__).resolve().parents[2] / "test"
_JSON = _TEST / "flashattention_fwd.intent.json"

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def _flash_fn():
    from intentir.ir.model import IntentFunction
    return IntentFunction.from_json(json.loads(_JSON.read_text()))


@pytest.mark.parametrize("scale", [1.0, 0.25, 1.0 / np.sqrt(16)])
def test_lifting_certificate_intent_dialect(scale) -> None:
    """interpret(JSON IR) == execute(intent-dialect attention.mlir, lowered via
    convert-intent-to-linalg). The real certified-lifting check: the executed
    side goes through the compiler's dialect lowering."""
    from intentir.backend.orchestrate import run_program
    from intentir.verify.interpret import interpret

    fn = _flash_fn()
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((1, 1, 4, 16)).astype(np.float32)   # match attention.mlir
    K = rng.standard_normal((1, 1, 8, 16)).astype(np.float32)
    V = rng.standard_normal((1, 1, 8, 16)).astype(np.float32)

    o_interp = interpret(
        fn, {"Q": Q, "K": K, "V": V, "scale": np.float32(scale)}, {}
    )["O"][0, 0]                                                  # [4,16] numpy spec

    # attention.mlir consumes host-pre-scaled Q -> softmax(q_ps·kᵀ)·v == flash(scale)
    q_ps = torch.tensor(Q[0, 0] * scale, device="cuda").contiguous()
    k = torch.tensor(K[0, 0], device="cuda").contiguous()
    v = torch.tensor(V[0, 0], device="cuda").contiguous()
    o_exec = next(iter(
        run_program(str(_TEST / "attention.mlir"), {"q": q_ps, "k": k, "v": v}).values()
    )).cpu().numpy()

    rel = np.abs(o_exec - o_interp).max() / (np.abs(o_interp).max() + 1e-6)
    assert rel < 1e-3, f"lifting certificate rel={rel:.2e}"


# Shapes span D=2..8 and BK both dividing and NOT dividing Sk (the last two are
# genuine tail tiles: 10%4=2, 12%5=2).
_FUSED_CASES = [
    (2, 4, 2, 0.5, 1, 0),
    (4, 8, 4, 1.0, 2, 1),
    (8, 16, 8, 0.125, 4, 2),
    (6, 10, 4, 0.7, 4, 3),    # Sk % BK = 2 -> tail iteration
    (4, 12, 4, 0.9, 5, 4),    # Sk % BK = 2 -> tail iteration
]


@pytest.mark.parametrize("sq,sk,d,scale,bk,seed", _FUSED_CASES)
def test_fused_emitter_matches_interpreter(sq, sk, d, scale, bk, seed) -> None:
    """interpret(JSON IR) == execute(independent fused emitter). A cross-
    implementation correctness + tile-invariance check; NOT a lowering certificate
    (the emitter bypasses the intent dialect — see module docstring)."""
    from intentir.emitters.flash_emit import run_fused_flash
    from intentir.verify.interpret import interpret

    fn = _flash_fn()
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((1, 1, sq, d)).astype(np.float32)
    K = rng.standard_normal((1, 1, sk, d)).astype(np.float32)
    V = rng.standard_normal((1, 1, sk, d)).astype(np.float32)

    o_interp = interpret(
        fn, {"Q": Q, "K": K, "V": V, "scale": np.float32(scale)}, {"BK": bk}
    )["O"][0, 0]

    q = torch.tensor(Q[0, 0], device="cuda").contiguous()
    k = torch.tensor(K[0, 0], device="cuda").contiguous()
    v = torch.tensor(V[0, 0], device="cuda").contiguous()
    o_exec = run_fused_flash(q, k, v, scale, bk).cpu().numpy()

    rel = np.abs(o_exec - o_interp).max() / (np.abs(o_interp).max() + 1e-6)
    assert rel < 1e-3, f"fused emitter vs interpreter rel={rel:.2e}"
