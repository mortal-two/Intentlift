"""Legacy C1 universe harness: resumable extract -> lower -> interpreter check.

Historical behavior and reports remain reproducible, but this module is not a
Gate1/Gate2 entry point.  The original staged failure description follows.

Batch, resumable extract -> lower -> verify with STAGED
failure attribution + a coverage manifest. Gates whether C2 can blindly scale.

    PYTHONPATH=python IIR_LIVE_LLM=1 python -m intentir.extraction.data.run_universe
    PYTHONPATH=python python -m intentir.extraction.data.run_universe saved   # replay

Unlike ``run_corpus`` (which bails a kernel on the first exception and so masks the
composability signal), each op is scored in THREE INDEPENDENT stages:

  extract  -> LLM produces valid IntentIR after bounded-repair          (extraction_fail)
  lower    -> that IntentIR reaches ZERO residual intent.* through the
              general ``-convert-intent-to-linalg`` pass  == COMPOSABILITY (lowering_fail)
  verify   -> interpret(JSON) == the real flag_gems.<op>, rel<1e-3       (verify_fail)

failure_class precedence: extraction_fail > lowering_fail > verify_fail > ok.
THE gate number = lowering_fail / (extract-ok)  == the composability-failure rate.

Resumable: the manifest is rewritten after EVERY op, and any op already recorded
is skipped (pass op names on the CLI, or delete the manifest, to force a re-run).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

from intentir.ir.model import IntentFunction
from intentir.verify.back.legacy_v0.interpret import interpret
from intentir.extraction.lifters.extract import extract_intent, _concretize_dims
from intentir.extraction.providers.provider import REPO_ROOT
from .corpus import (GPU_PASS_OPS, KernelSpec, _dims_from, _relerr, _src,
                     lower_to_linalg)
from .universe_specs import bulk_specs, pilot_specs

PILOT_DIR = REPO_ROOT / "corpus" / "pilot"
MANIFEST = PILOT_DIR / "manifest.json"
REPORT = PILOT_DIR / "REPORT.md"


@dataclass
class OpResult:
    op: str
    family: str
    extracted: bool = False
    lowered: bool = False
    verified: bool = False
    rel: float | None = None
    residual_intent: int | None = None
    residual_ops: list[str] = field(default_factory=list)
    gpu_pass_lowered: bool = False
    attempts: int = 0
    # ok | extraction_fail | lowering_fail | gpu_pass_lower | verify_fail
    failure_class: str = ""
    ops: list[str] = field(default_factory=list)
    error: str = ""

    def classify(self) -> None:
        if not self.extracted:
            self.failure_class = "extraction_fail"
        elif not self.lowered and self.gpu_pass_lowered:
            # Composes via a fused GPU pass (iota/scan/layout/... never lower on
            # the generic linalg leg by design) — NOT a composability failure.
            self.failure_class = "gpu_pass_lower"
        elif not self.lowered:
            self.failure_class = "lowering_fail"
        elif not self.verified:
            self.failure_class = "verify_fail"
        else:
            self.failure_class = "ok"

    def row(self) -> str:
        def mk(b):
            return "PASS" if b else "----"
        rel = f"{self.rel:.2e}" if self.rel is not None else "  -  "
        return (f"{self.op:16s} {self.family:9s} extract={mk(self.extracted)} "
                f"lower={mk(self.lowered)} verify={mk(self.verified)} "
                f"rel={rel} [{self.failure_class}]"
                + (f"  !{self.error}" if self.error else ""))


def _arrays(spec: KernelSpec):
    """Best-effort concrete inputs for the verify stage (may be []/wrong for
    hard ops -> that only costs the verify column, never extract/lower)."""
    ti = spec.inputs()
    return ti, [t.detach().cpu().numpy() for t in ti]


def run_op(spec: KernelSpec, *, saved: bool, max_repairs: int = 3) -> OpResult:
    res = OpResult(spec.name, spec.family)
    d = PILOT_DIR / spec.name

    # --- stage 1: extract (or load the committed artifact in saved mode) -------
    fn: IntentFunction | None = None
    try:
        if saved:
            fn = IntentFunction.from_json(
                json.loads((d / "intent.json").read_text()))
        else:
            ex = extract_intent(_src(spec), spec.name, shape_hint=spec.shape_hint,
                                max_repairs=max_repairs)
            fn = ex.fn
            res.attempts = ex.provenance.attempts
            d.mkdir(parents=True, exist_ok=True)
            (d / "intent.json").write_text(json.dumps(ex.json_obj, indent=2))
            (d / "provenance.json").write_text(json.dumps({
                "name": spec.name, "family": spec.family,
                "library": spec.library, "source_file": spec.source_file,
                "model": ex.provenance.model, "attempts": ex.provenance.attempts,
                "repairs": ex.provenance.repairs,
                "generated_by": "intentir.extraction (live provider, NOT hand-authored)",
            }, indent=2))
        res.extracted = True
        res.ops = [o.op + (":" + o.attrs.get("fn", "") if o.attrs.get("fn") else "")
                   for o in fn.body]
    except Exception as e:  # noqa: BLE001
        res.error = f"extract: {type(e).__name__}: {str(e)[:140]}"
        res.classify()
        return res

    # concrete inputs (best-effort) shared by lower(dims) + verify --------------
    try:
        ti, arrays = _arrays(spec)
    except Exception as e:  # noqa: BLE001
        ti, arrays = [], []
        res.error = f"inputs: {type(e).__name__}: {str(e)[:80]}"

    # --- stage 2: lower == the composability gate ------------------------------
    try:
        dims = _concretize_dims(fn, default=8)
        dims.update(_dims_from(fn, arrays))  # concrete extents win over defaults
        res.residual_intent, _, res.residual_ops = lower_to_linalg(fn, dims)
        res.lowered = res.residual_intent == 0
        # A non-zero residual made up ENTIRELY of GPU-pass-lowered ops (iota/scan/
        # layout/gather/...) means the op composes on its fused pass, not the
        # generic linalg leg — not counted as a composability failure.
        res.gpu_pass_lowered = (not res.lowered and bool(res.residual_ops)
                                and all(o in GPU_PASS_OPS
                                        for o in res.residual_ops))
        if not res.lowered:
            tag = "gpu-pass composes" if res.gpu_pass_lowered else "residual"
            res.error = (res.error + "; " if res.error else "") + \
                f"lower: {res.residual_intent} {tag} intent.* op(s) " \
                f"{res.residual_ops}"
    except Exception as e:  # noqa: BLE001
        res.error = (res.error + "; " if res.error else "") + \
            f"lower: {type(e).__name__}: {str(e)[:120]}"

    # --- stage 3: verify vs the real flag_gems.<op> ----------------------------
    try:
        if len(arrays) != len(fn.inputs):
            raise RuntimeError(
                f"arity: provided {len(arrays)} inputs, LLM declared {len(fn.inputs)}")
        interp_in = {t.name: a for t, a in zip(fn.inputs, arrays)}
        y_interp = interpret(fn, interp_in)[fn.outputs[0].name]
        y_real = spec.run(ti).detach().cpu().numpy()
        res.rel = _relerr(y_interp, y_real)
        res.verified = res.rel < 1e-3
    except Exception as e:  # noqa: BLE001
        res.error = (res.error + "; " if res.error else "") + \
            f"verify: {type(e).__name__}: {str(e)[:120]}"

    res.classify()
    return res


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"summary": {}, "ops": []}


def _write(results: list[OpResult]) -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    ext_ok = [r for r in results if r.extracted]
    n_lower_fail = sum(1 for r in results if r.failure_class == "lowering_fail")
    denom = max(1, len(ext_ok))
    comp_rate = 100.0 * n_lower_fail / denom
    n_gpu_pass = sum(1 for r in results if r.failure_class == "gpu_pass_lower")
    summary = {
        "total": len(results),
        "extracted": len(ext_ok),
        "lowered": sum(1 for r in results if r.lowered),
        "gpu_pass_lower": n_gpu_pass,
        "verified": sum(1 for r in results if r.verified),
        "ok": sum(1 for r in results if r.failure_class == "ok"),
        "extraction_fail": sum(1 for r in results if r.failure_class == "extraction_fail"),
        "lowering_fail": n_lower_fail,
        "verify_fail": sum(1 for r in results if r.failure_class == "verify_fail"),
        "extraction_success_pct": round(100.0 * len(ext_ok) / max(1, len(results)), 1),
        # composability = the fraction of extract-ok ops that FAIL to compose on
        # the generic linalg leg. GPU-pass-lowered ops (iota/scan/layout/...) DO
        # compose (on their fused pass) so they are NOT counted here.
        "composability_fail_pct": round(comp_rate, 1),
        "composes_pct": round(100.0 * (len(ext_ok) - n_lower_fail)
                              / max(1, len(ext_ok)), 1),
        "verify_pass_pct": round(100.0 * sum(1 for r in results if r.verified)
                                 / max(1, len(results)), 1),
    }
    MANIFEST.write_text(json.dumps(
        {"summary": summary, "ops": [asdict(r) for r in results]}, indent=2))

    lines = [
        "# C1 universe pilot — composability-measurement gate", "",
        f"- extraction success: **{summary['extraction_success_pct']}%** "
        f"({summary['extracted']}/{summary['total']})",
        f"- composability-failure rate (extract-ok but linalg-lower-fail): "
        f"**{summary['composability_fail_pct']}%** "
        f"({summary['lowering_fail']}/{summary['extracted']})",
        f"- composes (linalg leg + GPU-pass): **{summary['composes_pct']}%** "
        f"(of which {summary['gpu_pass_lower']} ride a fused GPU pass: "
        f"iota/scan/layout/gather/...)",
        f"- verify pass: **{summary['verify_pass_pct']}%** "
        f"({summary['verified']}/{summary['total']})",
        "",
        "| op | family | extract | lower | verify | rel | class |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        def mk(b):
            return "✅" if b else "❌"
        rel = f"{r.rel:.2e}" if r.rel is not None else "—"
        lines.append(f"| {r.op} | {r.family} | {mk(r.extracted)} | {mk(r.lowered)} | "
                     f"{mk(r.verified)} | {rel} | {r.failure_class} |")
    gate = ("**C2 CAN scale**" if summary["composability_fail_pct"] < 20
            else "**C2 needs pass-boundary triage first**")
    lines += ["", f"Gate: composability-fail {summary['composability_fail_pct']}% "
              f"(< 20% ? {gate})", ""]
    REPORT.write_text("\n".join(lines))


def main() -> int:
    global PILOT_DIR, MANIFEST, REPORT
    args = sys.argv[1:]
    saved = "saved" in args
    bulk = "bulk" in args
    only = {a for a in args if a not in ("saved", "--fresh", "bulk")}
    fresh = "--fresh" in args

    if bulk:  # SCALE run: the extraction-only bulk, its own manifest dir
        PILOT_DIR = REPO_ROOT / "corpus" / "bulk"
        MANIFEST = PILOT_DIR / "manifest.json"
        REPORT = PILOT_DIR / "REPORT.md"

    if not saved and not os.environ.get("IIR_LIVE_LLM"):
        print("IIR_LIVE_LLM not set: falling back to saved-mode replay.", flush=True)
        saved = True

    spec_fn = bulk_specs if bulk else pilot_specs
    all_specs = spec_fn()
    prior = {} if fresh else {o["op"]: o for o in _load_manifest().get("ops", [])}

    def _load_prior(name: str) -> OpResult:
        return OpResult(**{k: v for k, v in prior[name].items()
                           if k in OpResult.__dataclass_fields__})

    mode = "saved-replay" if saved else "configured live-provider extraction"
    n_req = len(only) if only else len(all_specs)
    print(f"=== C1 pilot: {n_req} requested / {len(all_specs)} ops ({mode}) ===\n",
          flush=True)

    # `only` MERGES: re-run exactly the named ops fresh, carry every other op's
    # prior manifest entry forward (so a targeted re-verify never clobbers the
    # full manifest). Without `only`, resume normally (skip recorded, run new).
    results: list[OpResult] = []
    for s in all_specs:
        if only and s.name not in only:
            if s.name in prior:
                results.append(_load_prior(s.name))  # carry forward untouched
            continue
        if s.name in prior and not only:
            r = _load_prior(s.name)
            print(f"    (skip, recorded) {r.row()}", flush=True)
        else:
            print(f">>> {s.name} ({s.family}) ...", flush=True)
            r = run_op(s, saved=saved)
            print("   ", r.row(), flush=True)
        results.append(r)
        _write(results)  # resumable: persist after every op
    _write(results)

    ext = sum(1 for r in results if r.extracted)
    lf = sum(1 for r in results if r.failure_class == "lowering_fail")
    comp = 100.0 * lf / max(1, ext)
    print(f"\n=== extraction {ext}/{len(results)} | "
          f"composability-fail {comp:.1f}% ({lf}/{ext}) | "
          f"verify {sum(1 for r in results if r.verified)}/{len(results)} ===")
    print(f"wrote {MANIFEST} + {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
