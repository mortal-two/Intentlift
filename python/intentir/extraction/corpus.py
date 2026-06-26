"""Kernel corpus: real high-performance-library Triton kernels lifted end-to-end.

Each entry names a REAL kernel source (FlagGems / Triton tutorials), a runnable
entry point (the library's own wrapper), and an input contract. The pipeline for
one kernel is:

    source --(gpt-5.5)--> IntentIR JSON --validate--> interpret(JSON)
        == run(the real kernel)          [dynamic certificate]
    JSON --emit--> intent dialect MLIR --convert-intent-to-linalg--> linalg
        [lower; no residual intent.*]

The committed artifact is corpus/<name>/{intent.json, provenance.json}. The JSON
is produced by the live provider, NOT hand-authored -- provenance records that.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..frontend.emit_mlir import emit_loop_module, emit_module
from ..ir.model import IntentFunction
from ..verify.interpret import interpret
from .extract import extract_intent
from .provider import REPO_ROOT

FG_OPS = Path("/home/kingdom/.local/lib/python3.10/site-packages/flag_gems/ops")
CORPUS_DIR = REPO_ROOT / "corpus" / "kernels"
REPORTS_DIR = REPO_ROOT / "corpus" / "reports"
INTENT_OPT = REPO_ROOT / "build" / "tools" / "intent-opt" / "intent-opt"


@dataclass
class KernelSpec:
    name: str
    library: str
    source_file: str
    shape_hint: str
    # positional inputs (torch), declared in the same order the hint pins
    inputs: Callable[[], list]
    # run the actual library kernel on those inputs -> torch.Tensor
    run: Callable[[list], object]
    note: str = ""


def _src(spec: KernelSpec) -> str:
    return Path(spec.source_file).read_text()


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def _dims_from(fn: IntentFunction, arrays: list[np.ndarray]) -> dict[str, int]:
    """Bind the LLM's symbolic dim names to the concrete input extents."""
    dims: dict[str, int] = {}
    for t, arr in zip(fn.inputs, arrays):
        for d, n in zip(t.shape, np.shape(arr)):
            if isinstance(d, str):
                dims[d] = int(n)
    return dims


@dataclass
class KernelResult:
    name: str
    library: str
    extracted: bool
    verified: bool
    lowered: bool
    rel_err: float | None
    attempts: int
    op_kinds: list[str]
    residual_intent: int | None
    error: str = ""

    def status_row(self) -> str:
        def mk(b):  # checkmark
            return "PASS" if b else "----"
        rel = f"{self.rel_err:.2e}" if self.rel_err is not None else "  -  "
        return (f"{self.name:16s} {self.library:8s} extract={mk(self.extracted)} "
                f"verify={mk(self.verified)} lower={mk(self.lowered)} "
                f"rel={rel} ops={len(self.op_kinds)}"
                + (f"  !{self.error}" if self.error else ""))


def lower_to_linalg(fn: IntentFunction, dims: dict[str, int]) -> tuple[int, list[str]]:
    """Emit dialect MLIR and run convert-intent-to-linalg. Returns
    (residual intent.* op count, lowered op kinds)."""
    has_loop = any(o.op == "loop" for o in fn.body)
    mlir = (emit_loop_module(fn, dims) if has_loop
            else emit_module(fn, dims))
    proc = subprocess.run(
        [str(INTENT_OPT), "-convert-intent-to-linalg"],
        input=mlir, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"intent-opt failed: {proc.stderr[:400]}")
    out = proc.stdout
    residual = sum(1 for line in out.splitlines() if "intent." in line)
    kinds = sorted({tok for line in out.splitlines()
                    for tok in _op_tokens(line)})
    return residual, kinds


def _op_tokens(line: str) -> list[str]:
    import re
    return re.findall(r"\b(?:linalg|tensor|scf|arith|math)\.[a-z_]+", line)


def run_kernel(spec: KernelSpec, *, save: bool = True,
               max_repairs: int = 3) -> KernelResult:
    """Full extract -> verify-vs-real-kernel -> lower for one kernel."""
    import torch

    res = KernelResult(spec.name, spec.library, False, False, False,
                       None, 0, [], None)
    try:
        ex = extract_intent(_src(spec), spec.name, shape_hint=spec.shape_hint,
                            max_repairs=max_repairs)
        res.extracted = True
        res.attempts = ex.provenance.attempts
        res.op_kinds = [o.op + (":" + o.attrs.get("fn", "") if o.attrs.get("fn") else "")
                        for o in ex.fn.body]

        ti = spec.inputs()
        arrays = [t.detach().cpu().numpy() for t in ti]
        if len(arrays) != len(ex.fn.inputs):
            raise RuntimeError(
                f"input arity mismatch: provided {len(arrays)}, "
                f"LLM declared {len(ex.fn.inputs)} ({[t.name for t in ex.fn.inputs]})")
        interp_in = {t.name: a for t, a in zip(ex.fn.inputs, arrays)}
        y_interp = interpret(ex.fn, interp_in)[ex.fn.outputs[0].name]
        y_real = spec.run(ti).detach().cpu().numpy()
        res.rel_err = _relerr(y_interp, y_real)
        res.verified = res.rel_err < 1e-3

        dims = _dims_from(ex.fn, arrays)
        res.residual_intent, kinds = lower_to_linalg(ex.fn, dims)
        res.lowered = res.residual_intent == 0
        res.op_kinds_lowered = kinds  # type: ignore[attr-defined]

        if save:
            d = CORPUS_DIR / spec.name
            d.mkdir(parents=True, exist_ok=True)
            (d / "intent.json").write_text(json.dumps(ex.json_obj, indent=2))
            prov = {
                "name": spec.name,
                "library": spec.library,
                "source_file": spec.source_file,
                "model": ex.provenance.model,
                "prompt_sha256": ex.provenance.prompt_sha256,
                "attempts": ex.provenance.attempts,
                "repairs": ex.provenance.repairs,
                "verified_rel_err": res.rel_err,
                "verified_against": f"{spec.library}.{spec.name}",
                "lowered_residual_intent": res.residual_intent,
                "generated_by": "intentir.extraction (live provider, NOT hand-authored)",
            }
            (d / "provenance.json").write_text(json.dumps(prov, indent=2))
    except Exception as e:  # noqa: BLE001 - record, never crash the sweep
        res.error = f"{type(e).__name__}: {str(e)[:120]}"
    return res


def verify_saved(spec: KernelSpec) -> KernelResult:
    """Re-verify a COMMITTED extraction artifact: load corpus/<name>/intent.json
    (provider-generated earlier), interpret vs the real kernel, and lower. No LLM
    call -- this is the deterministic regression path over the committed JSON."""
    res = KernelResult(spec.name, spec.library, False, False, False,
                       None, 0, [], None)
    jf = CORPUS_DIR / spec.name / "intent.json"
    try:
        fn = IntentFunction.from_json(json.loads(jf.read_text()))
        res.extracted = True
        res.op_kinds = [o.op + (":" + o.attrs.get("fn", "") if o.attrs.get("fn") else "")
                        for o in fn.body]
        ti = spec.inputs()
        arrays = [t.detach().cpu().numpy() for t in ti]
        interp_in = {t.name: a for t, a in zip(fn.inputs, arrays)}
        y_interp = interpret(fn, interp_in)[fn.outputs[0].name]
        y_real = spec.run(ti).detach().cpu().numpy()
        res.rel_err = _relerr(y_interp, y_real)
        res.verified = res.rel_err < 1e-3
        dims = _dims_from(fn, arrays)
        res.residual_intent, _ = lower_to_linalg(fn, dims)
        res.lowered = res.residual_intent == 0
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {str(e)[:120]}"
    return res
