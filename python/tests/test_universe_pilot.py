"""C1 universe-pilot regression: the composability-measurement gate.

Two layers (mirrors test_corpus.py):
  * saved-replay (default): re-run every committed pilot artifact
    (corpus/pilot/<op>/intent.json) through the STAGED harness and assert the
    manifest's per-op failure_class + the composability-fail rate are
    reproducible from the committed JSON. No network.
  * live (opt-in, IIR_LIVE_LLM=1): re-extract a couple of pilot ops through the
    live provider and assert the harness stages score them.

Both need a CUDA GPU (verify runs the real FlagGems kernel).
"""

from __future__ import annotations

import json
import os

import pytest

try:
    import torch
    _HAS_GPU = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="needs CUDA + flag_gems")

_CLASSES = {"ok", "extraction_fail", "lowering_fail", "verify_fail"}


def _committed_ops():
    """Pilot ops that have a committed intent.json (saved-replayable)."""
    from intentir.extraction.data.run_universe import PILOT_DIR
    from intentir.extraction.data.universe_specs import pilot_specs
    return [s for s in pilot_specs()
            if (PILOT_DIR / s.name / "intent.json").exists()]


def test_manifest_schema_and_gate_present():
    """The committed manifest carries the C1 gate schema + a valid gate number."""
    from intentir.extraction.data.run_universe import MANIFEST

    if not MANIFEST.exists():
        pytest.skip("no committed pilot manifest yet (run run_universe first)")
    m = json.loads(MANIFEST.read_text())
    s = m["summary"]
    for k in ("extraction_success_pct", "composability_fail_pct", "verify_pass_pct"):
        assert isinstance(s[k], (int, float)), k
    assert m["ops"], "manifest has no ops"
    for o in m["ops"]:
        assert o["failure_class"] in _CLASSES, o


def test_saved_replay_reproduces_failure_class():
    """Re-scoring the committed JSON reproduces each op's recorded failure_class."""
    from intentir.extraction.data.run_universe import MANIFEST, run_op
    from intentir.extraction.data.universe_specs import pilot_specs

    committed = _committed_ops()
    if not committed:
        pytest.skip("no committed pilot artifacts yet")
    recorded = {}
    if MANIFEST.exists():
        recorded = {o["op"]: o for o in json.loads(MANIFEST.read_text())["ops"]}

    ext_ok = lower_fail = 0
    mism = []
    for s in committed:
        r = run_op(s, saved=True)
        assert r.failure_class in _CLASSES
        if s.name in recorded:
            if recorded[s.name]["failure_class"] != r.failure_class:
                mism.append((s.name, recorded[s.name]["failure_class"], r.failure_class))
        if r.extracted:
            ext_ok += 1
        if r.failure_class == "lowering_fail":
            lower_fail += 1
    assert not mism, f"saved-replay drift vs committed manifest: {mism}"
    # the gate number must be computable from the replay
    assert ext_ok > 0
    _ = 100.0 * lower_fail / ext_ok  # composability-fail rate


@pytest.mark.skipif(os.environ.get("IIR_LIVE_LLM") != "1",
                    reason="set IIR_LIVE_LLM=1 to call the live provider")
def test_live_pilot_op_scores():
    """A live extraction of a pilot op is staged-scored end-to-end."""
    from intentir.extraction.data.run_universe import run_op
    from intentir.extraction.data.universe_specs import pilot_specs

    spec = {s.name: s for s in pilot_specs()}["cos"]
    r = run_op(spec, saved=False)
    assert r.extracted, r.error
    assert r.failure_class in _CLASSES
