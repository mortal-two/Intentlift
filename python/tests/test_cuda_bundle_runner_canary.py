from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "run_cuda_benchmark_bundle.py"
    )
    spec = importlib.util.spec_from_file_location("intentir_cuda_bundle_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_output_canaries_cover_float_integer_and_bool() -> None:
    runner = _load_runner()
    outputs = [
        torch.empty(4, dtype=torch.float32),
        torch.empty(4, dtype=torch.int32),
        torch.empty(4, dtype=torch.bool),
    ]

    runner._poison_cuda_outputs(torch, outputs, variant=0)
    first = [item.clone() for item in outputs]
    runner._poison_cuda_outputs(torch, outputs, variant=1)

    assert all(torch.all(left != right) for left, right in zip(first, outputs))

    aliased = torch.full((4,), 7.0)
    runner._poison_cuda_outputs(
        torch, [aliased], variant=0, protected=[aliased]
    )
    assert torch.all(aliased == 7.0)


def test_full_selection_contains_the_measured_schedule_spine() -> None:
    runner = _load_runner()
    structural = {"candidate": "structural", "latency_us": 101.0}
    schedule = {"candidate": "schedule", "latency_us": 100.0}

    assert runner._select_full_record(structural, schedule) is schedule
    assert runner._select_full_record(structural, None) is structural
    assert runner._select_full_record(None, None) is None
