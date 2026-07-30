"""Multi-frontend: CUDA C++ kernels lift to the same IntentIR (committed JSON).

Deterministic, GPU-free, no LLM: re-verify the committed corpus/cuda/<name>/
intent.json (extracted earlier from CUDA source by the live provider) against a
numpy reference of the op. Proves the extractor handled a different input
language. A gated test (IIR_LIVE_LLM=1) re-lifts CUDA live.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "corpus" / "kernels" / "cuda"


def _np_ref(name, arrs):
    x = arrs[0]
    if name == "cuda_softmax":
        z = x - x.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    if name == "cuda_silu":
        return x * (1.0 / (1.0 + np.exp(-x)))
    if name == "cuda_rmsnorm":
        x, w, eps = arrs[0], arrs[1], float(arrs[2])
        inv = 1.0 / np.sqrt((x * x).mean(1, keepdims=True) + eps)
        return x * inv * w
    raise KeyError(name)


@pytest.mark.skipif(not CUDA_DIR.exists(), reason="no committed CUDA artifacts")
@pytest.mark.parametrize("name,shapes,dims", [
    ("cuda_softmax", [(4, 6)], {"M": 4, "N": 6}),
    ("cuda_silu", [(4, 6)], {"M": 4, "N": 6}),
    ("cuda_rmsnorm", [(4, 6), (6,), ()], {"M": 4, "N": 6}),
])
def test_cuda_extraction_matches_reference(name, shapes, dims):
    from intentir.ir.model import IntentFunction
    from intentir.verify.back.legacy_v0.interpret import interpret

    jf = CUDA_DIR / name / "intent.json"
    if not jf.exists():
        pytest.skip(f"no artifact for {name}")
    rng = np.random.default_rng(1)
    arrs = []
    for s in shapes:
        if s == ():
            arrs.append(np.float32(1e-5))
        else:
            arrs.append(rng.standard_normal(s).astype(np.float32))
    fn = IntentFunction.from_json(json.loads(jf.read_text()))
    interp_in = {t.name: a for t, a in zip(fn.inputs, arrs)}
    y = interpret(fn, interp_in)[fn.outputs[0].name]
    ref = _np_ref(name, arrs)
    rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 1e-3, f"{name}: lifted CUDA vs reference rel {rel}"


@pytest.mark.skipif(os.environ.get("IIR_LIVE_LLM") != "1",
                    reason="set IIR_LIVE_LLM=1 to lift CUDA live")
def test_live_cuda_frontend():
    from intentir.extraction.lifters.cuda_frontend import run_cuda_frontend

    rows = run_cuda_frontend(save=False)
    assert rows, "no CUDA kernels"
    assert all(r["extracted"] and r["verified"] for r in rows), rows
