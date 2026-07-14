from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    KernelParameter,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.replay import ReplayGeneration


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

from scrc_live_e2e import (  # noqa: E402
    _compile_case,
    _gate2_case,
    _gate1_evidence_json,
    _lift_attempt_evidence,
    _write_report,
    build_case,
)


def _args(kernel: str, **overrides) -> Namespace:
    values = {
        "kernel": kernel,
        "m": 7,
        "n": 1003,
        "block": None,
        "num_warps": 4,
        "num_stages": 2,
        "eps": 1.0e-5,
    }
    values.update(overrides)
    return Namespace(**values)


def test_row_sum_case_carries_full_trusted_compiler_view() -> None:
    case = build_case(_args("row_sum"))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["name"] for item in view["arguments"]] == [
        "x_ptr",
        "out_ptr",
        "M",
        "N",
    ]
    assert view["arguments"][0]["shape"] == [7, 1003]
    assert view["arguments"][1]["shape"] == [7]
    assert view["grid"] == [7]
    assert view["grid_contract"] == ["M"]
    assert view["parameter_order"] == [
        "x_ptr",
        "out_ptr",
        "M",
        "N",
        "BLOCK_N",
    ]
    assert view["constexpr_parameters"] == ["BLOCK_N"]
    assert view["schedule_values"]["BLOCK_N"] == 1024
    assert compile_case.grid == (7,)
    assert compile_case.arguments["M"].dtype == "int32"


def test_row_sum_case_rejects_multi_program_column_reduction() -> None:
    with pytest.raises(ValueError, match="one BLOCK_N tile"):
        build_case(_args("row_sum", block=512))


def test_softmax_case_preserves_two_dimensional_output_and_tail_tile() -> None:
    case = build_case(_args("softmax", m=4, n=37, block=64))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["shape"] for item in view["arguments"][:2]] == [
        [4, 37],
        [4, 37],
    ]
    assert view["grid"] == [4]
    assert view["grid_contract"] == ["M"]
    assert view["schedule_values"]["BLOCK_N"] == 64
    assert compile_case.arguments["N"].value == 37
    assert case.tolerance.rtol == 2.0e-4
    assert case.tolerance.atol == 2.0e-5


def test_softmax_case_rejects_multi_program_column_reduction() -> None:
    with pytest.raises(ValueError, match="one BLOCK_N tile"):
        build_case(_args("softmax", block=512))


def test_rmsnorm_case_preserves_runtime_eps_and_tail_tile() -> None:
    case = build_case(_args("rmsnorm", m=4, n=37, block=64, eps=1.0e-5))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["name"] for item in view["arguments"]] == [
        "x_ptr",
        "w_ptr",
        "out_ptr",
        "M",
        "N",
        "eps",
    ]
    assert [item["shape"] for item in view["arguments"][:3]] == [
        [4, 37],
        [37],
        [4, 37],
    ]
    assert view["parameter_order"] == [
        "x_ptr",
        "w_ptr",
        "out_ptr",
        "M",
        "N",
        "eps",
        "BLOCK_N",
    ]
    assert view["constexpr_parameters"] == ["BLOCK_N"]
    assert view["schedule_values"]["BLOCK_N"] == 64
    assert compile_case.arguments["eps"].dtype == "float32"
    assert compile_case.arguments["eps"].value == 1.0e-5
    assert case.tolerance.rtol == 3.0e-4
    assert case.tolerance.atol == 3.0e-5


def test_rmsnorm_case_rejects_multi_program_column_reduction() -> None:
    with pytest.raises(ValueError, match="one BLOCK_N tile"):
        build_case(_args("rmsnorm", block=512))


def test_rmsnorm_gate2_case_binds_eps_as_f32_semantic_scalar() -> None:
    case = build_case(_args("rmsnorm", m=4, n=37, block=64, eps=1.0e-5))
    intent = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/rms_norm/intent.json").read_text())
    )
    parameters = [
        KernelParameter(
            "x_ptr", ParameterKind.POINTER, ParameterMode.INPUT,
            "f32", 2, tensor_ref="X",
        ),
        KernelParameter(
            "w_ptr", ParameterKind.POINTER, ParameterMode.INPUT,
            "f32", 1, tensor_ref="W",
        ),
        KernelParameter(
            "out_ptr", ParameterKind.POINTER, ParameterMode.OUTPUT,
            "f32", 2, tensor_ref="Y",
        ),
        KernelParameter("M", ParameterKind.SCALAR, dtype="i32", rank=0),
        KernelParameter("N", ParameterKind.SCALAR, dtype="i32", rank=0),
        KernelParameter(
            "eps", ParameterKind.SCALAR, dtype="f32", rank=0,
            tensor_ref="eps",
        ),
        KernelParameter("BLOCK_N", ParameterKind.CONSTEXPR, dtype="i32", rank=0),
    ]
    artifact = SimpleNamespace(
        interface=SimpleNamespace(parameters=parameters),
        schedule=SimpleNamespace(values=case.schedule_values),
        j=SimpleNamespace(intent=intent),
    )

    gate2 = _gate2_case(artifact, case, 20260715)

    assert isinstance(gate2.values["eps"], np.float32)
    assert gate2.values["eps"] == np.float32(1.0e-5)
    eps_arg = next(item for item in gate2.triton.arguments if item.name == "eps")
    assert eps_arg.value_key == "eps"
    assert gate2.lowering.input_bindings["eps"] == "eps"
    assert gate2.lowering.dims == {"M": 4, "N": 37}


def test_add_compatibility_case_keeps_existing_defaults() -> None:
    case = build_case(_args("add"))
    view = case.compiler_view().to_json()

    assert view["grid"] == [4]
    assert view["grid_contract"] == ["ceil_div(n_elements, BLOCK)"]
    assert view["schedule_values"]["BLOCK"] == 256


def _generation(code: str, *, masked: bool) -> ReplayGeneration:
    return ReplayGeneration(
        code=code,
        prompt_sha256="prompt",
        requested_model="model",
        generation_id=f"generation:{masked}",
        code_sha256="digest",
        masked=masked,
    )


def test_failure_evidence_keeps_candidates_but_redacts_source() -> None:
    normal = _generation("NORMAL_REPLAY\n", masked=False)
    masked = _generation("OPAQUE_MASKED_REPLAY\n", masked=True)
    gate1 = SimpleNamespace(
        to_json=lambda: {
            "artifact": {"source": "SECRET_SOURCE", "kernel_id": "k"},
            "candidates": [{"generation": {"code_sha256": "digest"}}],
            "masked_generation": {"code_sha256": "digest"},
        },
        candidates=(SimpleNamespace(generation=normal),),
        masked_generation=masked,
    )

    evidence = _gate1_evidence_json(gate1)
    assert "source" not in evidence["artifact"]
    assert evidence["artifact"]["source_bytes"] == len("SECRET_SOURCE")
    assert evidence["candidates"][0]["generation"]["code"] == normal.code
    assert evidence["masked_generation"]["code"] == masked.code


def test_lift_failure_evidence_hashes_raw_model_response() -> None:
    evidence = _lift_attempt_evidence(
        SimpleNamespace(
            attempt=0,
            model="model",
            response_id="response",
            diagnostic="invalid artifact",
            response_text="SECRET_MODEL_RESPONSE",
        )
    )

    assert "response_text" not in evidence
    assert evidence["response_bytes"] == len("SECRET_MODEL_RESPONSE")
    assert len(evidence["response_sha256"]) == 64


def test_report_write_is_atomic_private_and_creates_explicit_parent(tmp_path) -> None:
    target = tmp_path / "nested" / "report.json"
    _write_report(str(target), {"status": "ok"})

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(target.parent.iterdir()) == [target]
