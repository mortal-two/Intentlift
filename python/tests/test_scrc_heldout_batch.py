from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from intentir.verify.gate1.rules import default_triton_rule_registry


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

from scrc_heldout_batch import (  # noqa: E402
    CANDIDATES,
    READY_CASES,
    _batch_identity,
    _load_cached_lift,
    _response_ids,
    _resume_row,
    _result_status,
    build_heldout_case,
)
from scrc_live_e2e import _compile_case  # noqa: E402


def test_first_batch_manifest_has_reviewed_20_to_40_real_candidates() -> None:
    assert 20 <= len(CANDIDATES) <= 40
    assert len({item.name for item in CANDIDATES}) == len(CANDIDATES)
    assert READY_CASES == (
        "fg_dot",
        "fg_addr",
        "fg_ones",
        "fg_sum",
        "fg_mean",
    )
    assert all(item.module.startswith("flag_gems.") for item in CANDIDATES)
    assert all(item.function.isidentifier() for item in CANDIDATES)


def test_real_flag_gems_dot_case_preserves_rank_zero_output_contract() -> None:
    case = build_heldout_case("fg_dot")
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert case.source_path.name == "dot.py"
    assert case.function_name == "dot_kernel"
    assert "def dot_kernel" in case.source_text
    assert view["arguments"][2] == {
        "name": "out_ptr",
        "kind": "tensor",
        "dtype": "f32",
        "shape": [],
    }
    assert view["grid"] == [1]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024
    assert compile_case.arguments["out_ptr"].shape == ()


def test_real_flag_gems_addr_case_records_runtime_scalars_and_strides() -> None:
    case = build_heldout_case("fg_addr")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "addr.py"
    assert view["grid"] == [1, 1]
    assert view["grid_contract"] == [
        "ceil_div(M, BLOCK_SIZE_M)",
        "ceil_div(N, BLOCK_SIZE_N)",
    ]
    by_name = {item["name"]: item for item in view["arguments"]}
    assert by_name["beta"]["value"] == 0.75
    assert by_name["alpha"]["value"] == -1.25
    assert by_name["stride_input_m"]["value"] == 13
    assert by_name["stride_output_n"]["value"] == 1


def test_real_flag_gems_ones_case_has_multi_program_tail_contract() -> None:
    case = build_heldout_case("fg_ones")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "ones.py"
    assert case.function_name == "ones_kernel"
    assert "def ones_kernel" in case.source_text
    assert view["arguments"] == [
        {
            "name": "output_ptr",
            "kind": "tensor",
            "dtype": "f32",
            "shape": [2051],
        },
        {
            "name": "n_elements",
            "kind": "scalar",
            "dtype": "i32",
            "value": 2051,
        },
    ]
    assert view["grid"] == [3]
    assert view["grid_contract"] == [
        "ceil_div(n_elements, BLOCK_SIZE)"
    ]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024


def test_real_flag_gems_sum_case_uses_final_rank_zero_stage() -> None:
    case = build_heldout_case("fg_sum")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "sum.py"
    assert case.function_name == "sum_kernel_2"
    assert "def sum_kernel_2" in case.source_text
    assert view["arguments"][0]["shape"] == [1003]
    assert view["arguments"][1] == {
        "name": "out",
        "kind": "tensor",
        "dtype": "f32",
        "shape": [],
    }
    assert view["arguments"][2]["value"] == 1003
    assert view["grid"] == [1, 1, 1]
    assert view["schedule_values"]["BLOCK_MID"] == 1024


def test_real_flag_gems_mean_case_keeps_total_distinct_from_mid_size() -> None:
    case = build_heldout_case("fg_mean")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "mean.py"
    assert case.function_name == "mean_kernel_2"
    assert "mean_val = sum_val / M" in case.source_text
    assert by_name["mid"]["shape"] == [37]
    assert by_name["out"]["shape"] == []
    assert by_name["M"]["value"] == 1003
    assert by_name["MID_SIZE"]["value"] == 37
    assert view["schedule_values"]["BLOCK_MID"] == 64


def test_batch_status_uses_existing_gate_verdicts() -> None:
    assert _result_status(
        {"preflight": {"ok": False}}
    ) == "source_compile_failed"
    assert _result_status({"lift": {"ok": False}}) == "lift_failed"
    assert _result_status(
        {"lift": {"ok": True}, "gate1": {"status": "no_closure"}}
    ) == "no_closure"
    assert _result_status(
        {
            "lift": {"ok": True},
            "gate1": {"status": "replay_closed"},
            "gate2": {"status": "lowering_unsupported"},
        }
    ) == "lowering_unsupported"


def test_batch_response_ids_tolerate_failed_generation_and_absent_masked() -> None:
    normal, masked = _response_ids(
        {
            "gate1": {
                "candidates": [
                    {"generation": None},
                    {"generation": {"response_id": "resp-normal"}},
                ],
                "masked_generation": None,
            }
        }
    )
    assert normal == ["resp-normal"]
    assert masked is None


def _batch_args(**overrides) -> Namespace:
    values = {
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "seed": 20260715,
        "lift_repairs": 2,
        "rewrite_repairs": 2,
        "diagnose_rejected": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resume_requires_the_exact_current_batch_identity() -> None:
    identity = _batch_identity(build_heldout_case("fg_dot"), _batch_args())
    row = {"case": "fg_dot", "status": "leakage_rejected"}
    saved = {
        "batch": {"terminal": True, "identity": identity, "row": row}
    }

    assert _resume_row(saved, identity) == row
    stale = dict(identity)
    stale["seed"] = identity["seed"] + 1
    assert _resume_row(saved, stale) is None
    assert (
        _resume_row({"batch": {"terminal": True, "row": row}}, identity)
        is None
    )


def test_lift_cache_rejects_a_different_rule_registry(tmp_path: Path) -> None:
    case = build_heldout_case("fg_dot")
    path = tmp_path / "fg_dot.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.6-sol",
                "effort": "xhigh",
                "source": {
                    "sha256": hashlib.sha256(
                        case.source_text.encode("utf-8")
                    ).hexdigest()
                },
                "compiler_view": case.compiler_view().to_json(),
                "lift": {
                    "ok": True,
                    "rule_registry_digest": "stale-registry",
                    "artifact": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rule registry differs"):
        _load_cached_lift(
            path,
            case,
            Namespace(model="gpt-5.6-sol", effort="xhigh"),
        )


def test_lift_cache_rejects_a_different_initial_prompt(tmp_path: Path) -> None:
    case = build_heldout_case("fg_dot")
    path = tmp_path / "fg_dot.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.6-sol",
                "effort": "xhigh",
                "source": {
                    "sha256": hashlib.sha256(
                        case.source_text.encode("utf-8")
                    ).hexdigest()
                },
                "compiler_view": case.compiler_view().to_json(),
                "lift": {
                    "ok": True,
                    "rule_registry_digest": (
                        default_triton_rule_registry().digest
                    ),
                    "prompt_sha256": "stale-prompt",
                    "artifact": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt differs"):
        _load_cached_lift(
            path,
            case,
            Namespace(model="gpt-5.6-sol", effort="xhigh"),
        )
