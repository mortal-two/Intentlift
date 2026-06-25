"""Run the whole kernel corpus: extract (live provider) -> verify-vs-real-kernel
-> lower, save artifacts, and emit a status matrix + manifest.

    PYTHONPATH=python python -m intentir.extraction.run_corpus
"""

from __future__ import annotations

import json
import sys

from .corpus import CORPUS_DIR, run_kernel, verify_saved
from .corpus_specs import kernel_specs


def main() -> int:
    args = sys.argv[1:]
    saved = "saved" in args  # deterministic re-verify of committed artifacts
    only = {a for a in args if a != "saved"}
    specs = [s for s in kernel_specs() if not only or s.name in only]
    results = []
    mode = "deterministic re-verify of committed JSON" if saved \
        else "live gpt-5.5 extraction"
    print(f"=== IntentIR corpus: {len(specs)} kernels ({mode}) ===\n")
    for s in specs:
        print(f">>> {s.name} ({s.library}) ...", flush=True)
        r = verify_saved(s) if saved else run_kernel(s)
        print("   ", r.status_row(), flush=True)
        results.append(r)

    npass = sum(1 for r in results if r.extracted and r.verified and r.lowered)
    print(f"\n=== {npass}/{len(results)} kernels: extracted + verified + lowered ===")

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "summary": {
            "total": len(results),
            "fully_passing": npass,
            "verified": sum(1 for r in results if r.verified),
            "lowered": sum(1 for r in results if r.lowered),
        },
        "kernels": [
            {
                "name": r.name, "library": r.library,
                "extracted": r.extracted, "verified": r.verified,
                "lowered": r.lowered, "rel_err": r.rel_err,
                "attempts": r.attempts, "ops": r.op_kinds,
                "residual_intent": r.residual_intent, "error": r.error,
            }
            for r in results
        ],
    }
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [
        "# IntentIR kernel corpus",
        "",
        "Real high-performance-library Triton kernels lifted to IntentIR by the "
        "**live provider agent** (gpt-5.5, OpenAI responses API), then verified "
        "against the actual library kernel and lowered to linalg.",
        "",
        "- **extract**: `intentir.extraction` calls the provider; JSON is NOT "
        "hand-authored (see each `provenance.json`).",
        "- **verify**: `interpret(JSON)` vs the real `flag_gems.<op>` output, "
        "max relative error < 1e-3 (the dynamic lifting certificate).",
        "- **lower**: emitted `intent` dialect MLIR through "
        "`-convert-intent-to-linalg`, zero residual `intent.*` ops.",
        "",
        "| kernel | library | extract | verify | lower | rel err | #ops |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        def mk(b):
            return "✅" if b else "❌"
        rel = f"{r.rel_err:.2e}" if r.rel_err is not None else "—"
        lines.append(f"| {r.name} | {r.library} | {mk(r.extracted)} | "
                     f"{mk(r.verified)} | {mk(r.lowered)} | {rel} | {len(r.op_kinds)} |")
    lines += ["", f"**{npass}/{len(results)} kernels** pass all three stages.", ""]
    (CORPUS_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {CORPUS_DIR}/REPORT.md + manifest.json")
    return 0 if npass >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
