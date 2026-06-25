"""Front-end: JSON -> intent-dialect MLIR -> (lower / execute).

These tests exercise the missing 前端 leg — :mod:`intentir.frontend.emit_mlir`
turning ``flashattention_fwd.intent.json`` into real `intent`-dialect text:

* ``test_emit_loop_module_lowers`` — the structured ``intent.loop`` module parses,
  verifies, and lowers to ``scf.for`` via ``-convert-intent-to-linalg`` (no GPU).
* ``test_emit_body_lowers`` — the executable body function lowers cleanly (no GPU).
* ``test_emitted_body_executes_vs_interpret`` — THE end-to-end front+back proof:
  the EMITTED body runs on the 5090D (host-driven trip loop, carries in CUDA
  buffers) and matches ``interpret(JSON)`` — i.e. JSON-sourced intent IR executes
  the FlashAttention recurrence on hardware, certified against our own spec.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT
from intentir.ir.model import IntentFunction

_TEST = pathlib.Path(__file__).resolve().parents[2] / "test"
_JSON = _TEST / "flashattention_fwd.intent.json"
_DIMS = {"B": 1, "H": 1, "Sq": 4, "Sk": 8, "D": 4, "BK": 2}


def _fn() -> IntentFunction:
    return IntentFunction.from_json(json.loads(_JSON.read_text()))


def _intent_opt(mlir: str, *passes: str) -> str:
    p = subprocess.run([INTENT_OPT, *passes], input=mlir, capture_output=True,
                       text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


_needs_opt = pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                                reason="intent-opt not built")


@_needs_opt
def test_emit_loop_module_lowers() -> None:
    from intentir.frontend import emit_loop_module

    mlir = emit_loop_module(_fn(), _DIMS)
    assert "intent.loop" in mlir and "intent.yield" in mlir
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "scf.for" in lowered
    assert "tensor.extract_slice" in lowered
    assert "intent." not in lowered  # everything lowered to standard MLIR


@_needs_opt
def test_emit_body_lowers() -> None:
    from intentir.frontend import emit_body_function

    mlir = emit_body_function(_fn(), _DIMS)
    # body args: outer (Q, scale) + scan tiles (Kj, Vj) + carries (m, l, Oacc)
    for name in ("%Q", "%scale", "%Kj", "%Vj", "%m", "%l", "%Oacc"):
        assert f"{name}:" in mlir, name
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "linalg.generic" in lowered
    assert "intent." not in lowered


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(), reason="intent-opt not built")
@pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(), reason="CUDA toolkit not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_emitted_body_executes_vs_interpret(tmp_path) -> None:
    """The whole front+back path: emit intent-dialect MLIR from the JSON, run the
    body once per K-tile on the 5090D with carries in persistent CUDA buffers,
    and check the result matches the numpy interpreter run on the same JSON IR."""
    from intentir.backend.orchestrate import run_program
    from intentir.frontend import emit_body_function
    from intentir.verify.interpret import interpret

    fn = _fn()
    Sq, Sk, D, BK = _DIMS["Sq"], _DIMS["Sk"], _DIMS["D"], _DIMS["BK"]
    body_path = tmp_path / "flash_body_fwd.mlir"
    body_path.write_text(emit_body_function(fn, _DIMS))
    nt = Sk // BK

    for seed, scale in [(0, 1.0), (1, 0.25), (2, 0.5)]:
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((1, 1, Sq, D)).astype(np.float32)
        K = rng.standard_normal((1, 1, Sk, D)).astype(np.float32)
        V = rng.standard_normal((1, 1, Sk, D)).astype(np.float32)
        o_interp = interpret(
            fn, {"Q": Q, "K": K, "V": V, "scale": np.float32(scale)}, {"BK": BK}
        )["O"][0, 0]

        def t(x):
            return torch.tensor(x, device="cuda").contiguous()

        Qd, Kd, Vd, sc = t(Q[0, 0]), t(K[0, 0]), t(V[0, 0]), t(np.float32(scale))
        m = torch.full((Sq, 1), float("-inf"), device="cuda").contiguous()
        l = torch.zeros((Sq, 1), device="cuda").contiguous()
        Oacc = torch.zeros((Sq, D), device="cuda").contiguous()
        for ti in range(nt):
            out = run_program(str(body_path), {
                "Q": Qd, "scale": sc,
                "Kj": Kd[ti * BK:(ti + 1) * BK].contiguous(),
                "Vj": Vd[ti * BK:(ti + 1) * BK].contiguous(),
                "m": m, "l": l, "Oacc": Oacc})
            m = out["out0"].reshape(Sq, 1).contiguous()
            l = out["out1"].reshape(Sq, 1).contiguous()
            Oacc = out["out2"].reshape(Sq, D).contiguous()
        o_exec = (Oacc / l).cpu().numpy()
        rel = np.abs(o_exec - o_interp).max() / (np.abs(o_interp).max() + 1e-6)
        assert rel < 1e-3, f"seed={seed} scale={scale}: rel={rel}"
