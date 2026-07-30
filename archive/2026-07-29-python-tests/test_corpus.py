"""Corpus regression: every committed extraction artifact still verifies + lowers.

Two layers:
  * deterministic (default): re-verify the COMMITTED corpus/<name>/intent.json
    (provider-generated earlier) against the real kernel + lower it. No network.
  * live (opt-in, IIR_LIVE_LLM=1): re-extract one kernel through the live
    provider and assert it still validates + matches -- proves the JSON really is
    provider-agent-generated, not hand-authored.

Both need a CUDA GPU (the dynamic check runs the real FlagGems kernel).
"""

from __future__ import annotations

import os

import pytest

try:
    import torch
    _HAS_GPU = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="needs CUDA + flag_gems")


def _specs():
    from intentir.extraction.data.corpus_specs import kernel_specs
    return kernel_specs()


def test_corpus_artifacts_verify_and_lower():
    """All committed extractions interpret==real-kernel and lower clean."""
    from intentir.extraction.data.corpus import verify_saved

    results = [verify_saved(s) for s in _specs()]
    full = [r for r in results if r.extracted and r.verified and r.lowered]
    failed = [(r.name, r.error or f"rel={r.rel_err}") for r in results
              if not (r.extracted and r.verified and r.lowered)]
    assert len(full) >= 10, f"only {len(full)} kernels fully pass; failures: {failed}"
    # In fact every committed artifact should pass.
    assert not failed, f"regressions: {failed}"
    for r in full:
        assert r.rel_err < 1e-3
        assert r.residual_intent == 0


@pytest.mark.skipif(os.environ.get("IIR_LIVE_LLM") != "1",
                    reason="set IIR_LIVE_LLM=1 to call the live provider")
def test_live_extraction_is_provider_generated():
    """The JSON is produced by the live provider agent (not hand-authored)."""
    from intentir.extraction.data.corpus import run_kernel

    spec = {s.name: s for s in _specs()}["softmax"]
    r = run_kernel(spec, save=False)
    assert r.extracted and r.verified and r.lowered, r.error
    assert r.rel_err < 1e-3
