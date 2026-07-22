from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from direct_fixtures import make_copy_artifact


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tools/scrc_canary.py"
_SPEC = importlib.util.spec_from_file_location("intentir_scrc_canary_tool", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
canary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = canary
_SPEC.loader.exec_module(canary)


def test_canary_catalog_is_small_ordered_and_uses_real_harness_coordinates() -> None:
    assert tuple(canary.CANARIES) == (
        "exact_vector_add",
        "masked_copy",
        "masked_row_square_sum",
    )
    request, mapping = canary.prepare_request(canary.CANARIES["masked_copy"])
    assert request.binding.values == {"c0": 1024}
    assert mapping["BLOCK_SIZE"] == "c0"
    code, _origin = request.source.gate_source.read_code()
    assert "def scrc_k0(p0, p1, p2, c0" in code
    assert "source_masked_copy" not in code
    assert "BLOCK_SIZE" not in code


def test_gate2_suite_is_constructed_from_accepted_j_and_same_c0() -> None:
    artifact = make_copy_artifact()
    artifact.c.values["c0"] = 1024
    request, _mapping = canary.prepare_request(canary.CANARIES["masked_copy"])
    suite = canary._build_gate2_suite(
        canary.CANARIES["masked_copy"], artifact, request.binding
    )

    assert suite.c_values == request.binding.c_values == (("c0", 1024),)
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.j is artifact.j
    assert case.refeval.j is artifact.j
    assert case.output_keys == ("output0",)
    assert case.triton.schedule == {"c0": 1024}


def test_run_directories_use_plain_monotonic_numbers(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(canary, "TASK_ROOT", tmp_path)
    spec = canary.CANARIES["exact_vector_add"]
    first = canary._next_run_dir(spec)
    second = canary._next_run_dir(spec)
    assert first.name == "run-001"
    assert second.name == "run-002"
