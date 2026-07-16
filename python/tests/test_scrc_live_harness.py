from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.model import (
    Gate1Status,
    KernelParameter,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.replay import ReplayGeneration
from intentir.verify.gate2 import Gate2Status


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

import scrc_live_e2e as live  # noqa: E402
from scrc_live_e2e import (  # noqa: E402
    _compile_case,
    _gate2_case,
    _gate1_evidence_json,
    _independent_gate2_candidate,
    _lift_attempt_evidence,
    _parser,
    _pi_ablation_control_evidence,
    _pi_ablation_evidence_json,
    _relift_evidence_json,
    _write_report,
    build_case,
)


def _args(kernel: str, **overrides) -> Namespace:
    values = {
        "kernel": kernel,
        "m": 7,
        "n": 1003,
        "k": 19,
        "block": None,
        "block_m": 16,
        "block_n": 16,
        "block_k": 16,
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


def test_matmul_case_preserves_runtime_dims_three_c_slots_and_2d_grid() -> None:
    case = build_case(_args("matmul", m=17, n=13, k=19))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["name"] for item in view["arguments"]] == [
        "a_ptr",
        "b_ptr",
        "out_ptr",
        "M",
        "N",
        "K",
    ]
    assert [item["dtype"] for item in view["arguments"][:3]] == [
        "f16",
        "f16",
        "f32",
    ]
    assert [item["shape"] for item in view["arguments"][:3]] == [
        [17, 19],
        [19, 13],
        [17, 13],
    ]
    assert view["grid"] == [2, 1]
    assert view["grid_contract"] == [
        "ceil_div(M, BLOCK_M)",
        "ceil_div(N, BLOCK_N)",
    ]
    assert view["parameter_order"] == [
        "a_ptr",
        "b_ptr",
        "out_ptr",
        "M",
        "N",
        "K",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
    ]
    assert view["constexpr_parameters"] == [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
    ]
    assert view["schedule_values"] == {
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_K": 16,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert compile_case.arguments["M"].value == 17
    assert compile_case.arguments["N"].value == 13
    assert compile_case.arguments["K"].value == 19
    assert case.tolerance.rtol == 2.0e-3
    assert case.tolerance.atol == 2.0e-3


def test_matmul_case_rejects_nonpositive_logical_or_block_extent() -> None:
    with pytest.raises(ValueError, match="K must be positive"):
        build_case(_args("matmul", k=0))
    with pytest.raises(ValueError, match="block sizes must be positive"):
        build_case(_args("matmul", block_k=0))


def test_matmul_entry_uses_reviewed_probe_defaults() -> None:
    args = _parser("matmul").parse_args([])
    assert (args.m, args.n, args.k) == (17, 13, 19)
    assert (args.block_m, args.block_n, args.block_k) == (16, 16, 16)
    assert args.model == "gpt-5.6-sol"
    assert args.effort == "xhigh"
    assert args.relift_rounds == 3


def test_live_parser_rejects_nonpositive_relift_rounds() -> None:
    with pytest.raises(SystemExit):
        _parser("add").parse_args(["--relift-rounds", "0"])


def test_live_case_wires_real_relifter_policy_into_gate1(monkeypatch) -> None:
    case = build_case(_args("add"))
    provider = ProviderConfig(
        "test",
        "https://example.invalid",
        "secret",
        "base-model",
    )
    artifact = SimpleNamespace(
        interface=SimpleNamespace(function_name=case.function_name),
        to_json=lambda: {
            "kernel_id": case.function_name,
            "source": case.source_text,
        },
    )
    extraction = SimpleNamespace(
        artifact=artifact,
        provenance=SimpleNamespace(
            model="gpt-5.6-sol",
            prompt_sha256="lift-prompt",
            attempts=1,
            repairs=(),
            response_id="lift-response",
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "get_provider", lambda: provider)

    def fake_extract(*args, **kwargs):
        captured["lift_provider"] = kwargs["provider"]
        captured["lift_effort"] = kwargs["effort"]
        return extraction

    monkeypatch.setattr(live, "extract_scrc_artifact", fake_extract)
    monkeypatch.setattr(
        live,
        "ResponsesReplayWriter",
        lambda **kwargs: SimpleNamespace(),
    )

    class FakeRelifter:
        def __init__(self, **kwargs):
            captured["relifter_kwargs"] = kwargs
            self.records = ()

    class FakeRunner:
        def __init__(self, *, config):
            captured["gate1_config"] = config

        def run(self, *args, relifter=None, **kwargs):
            captured["gate1_relifter"] = relifter
            return SimpleNamespace(
                status=Gate1Status.NO_CLOSURE,
                replay_closed=False,
                candidates=(),
                masked_generation=None,
                to_json=lambda: {
                    "status": Gate1Status.NO_CLOSURE.value,
                    "artifact": artifact.to_json(),
                    "candidates": [],
                    "masked_generation": None,
                    "diagnostics": [],
                },
            )

    monkeypatch.setattr(live, "ResponsesSCRCRelifter", FakeRelifter)
    monkeypatch.setattr(live, "Gate1Runner", FakeRunner)
    args = Namespace(
        model="gpt-5.6-sol",
        effort="xhigh",
        seed=20260715,
        lift_repairs=2,
        relift_rounds=4,
        gate2_matrix=False,
        gate2_seeds=None,
        diagnose_rejected=False,
    )

    code, report = live.run_case(case, args)

    configured_provider = captured["lift_provider"]
    assert isinstance(configured_provider, ProviderConfig)
    assert configured_provider.model == "gpt-5.6-sol"
    assert configured_provider.reasoning_effort == "xhigh"
    assert captured["lift_effort"] == "xhigh"
    relifter_kwargs = captured["relifter_kwargs"]
    assert isinstance(relifter_kwargs, dict)
    assert relifter_kwargs["provider"] is configured_provider
    assert relifter_kwargs["effort"] == "xhigh"
    assert captured["gate1_relifter"] is not None
    assert captured["gate1_config"].max_relifts == 4
    assert captured["gate1_config"].run_semantic_interventions
    assert captured["gate1_config"].max_semantic_interventions == 3
    assert report["relift_policy"]["protocol"] == (
        "relift_only_pi_audit_on_strict_opaque_refs_causal_v7"
    )
    assert report["relift_policy"]["maximum_semantic_relifts"] == 3
    assert report["relift_policy"]["normal_recovery"] == (
        "relift_only_restart_full_pipeline"
    )
    assert report["relift_policy"]["replay_code_repair"] == "disabled"
    assert report["gate1_anti_leakage_policy"] == {
        "paired_semantic_interventions": True,
        "maximum_interventions": 3,
        "paired_replay_control": "normal_model_generated_candidate",
        "maximum_provider_attempts_per_generation": 3,
        "masked_j_role": "diagnostic_or_fallback",
        "masked_j_fallback_policy": "exact_semantic_hard_degradation_v1",
        "masked_j_score_role": "diagnostic_only",
        "ttir_score_semantics": "active_evidence_weighted_v2",
        "replay_reference_naming": "opaque_replay_refs_v1",
        "pi_ablations": {
            "enabled": True,
            "default_policy": "enabled_for_formal_live_evidence",
            "maximum": 3,
            "maximum_replay_repairs": 0,
            "requires_full_pi_replay_closed": True,
            "kinds": [
                "j_only",
                "plan_step_dropout",
                "plan_mechanism_substitution",
            ],
            "verdict_role": "diagnostic",
            "opaque_reference_discipline": "resolve_handles_never_emit_v1",
            "replay_code_repair": "disabled",
        },
    }
    assert report["pi_ablation_control"] == {
        "valid": False,
        "reason": "source_fingerprint_unavailable",
        "gate1_status": "no_closure",
    }
    assert report["pi_ablations"] == []
    assert report["relifts"] == []
    assert code == 3


def test_add_compatibility_case_keeps_existing_defaults() -> None:
    case = build_case(_args("add"))
    view = case.compiler_view().to_json()

    assert view["grid"] == [4]
    assert view["grid_contract"] == ["ceil_div(n_elements, BLOCK)"]
    assert view["schedule_values"]["BLOCK"] == 256
    assert case.source_path == ROOT / "test" / "scrc_probe_sources.py"
    assert case.function_name == "add_kernel"


def test_live_parser_records_independent_gate2_for_rejected_replays_by_default() -> None:
    assert _parser("add").parse_args([]).diagnose_rejected
    assert not _parser("add").parse_args(
        ["--no-diagnose-rejected"]
    ).diagnose_rejected


def test_live_parser_enables_final_pi_audit_by_default() -> None:
    assert _parser("add").parse_args([]).pi_ablations
    assert not _parser("add").parse_args(
        ["--no-pi-ablations"]
    ).pi_ablations


def test_pi_ablation_report_keeps_source_blind_generated_code() -> None:
    generation = ReplayGeneration(
        code="import triton\n",
        prompt_sha256="prompt",
        requested_model="model",
        generation_id="generation",
        code_sha256="code",
    )
    record = SimpleNamespace(
        generation=generation,
        to_json=lambda: {"generation": {"code_sha256": "code"}},
    )

    payload = _pi_ablation_evidence_json((record,))[0]
    assert payload["generation"]["code"] == "import triton\n"
    assert payload["generation"]["code_bytes"] == 14


def test_pi_ablation_requires_a_closed_full_pi_control() -> None:
    rejected = SimpleNamespace(
        source_fingerprint=object(),
        replay_closed=False,
        status=Gate1Status.NO_CLOSURE,
    )
    accepted = SimpleNamespace(
        source_fingerprint=object(),
        replay_closed=True,
        status=Gate1Status.REPLAY_CLOSED,
    )

    assert _pi_ablation_control_evidence(rejected) == {
        "valid": False,
        "reason": "full_pi_control_not_replay_closed",
        "gate1_status": "no_closure",
    }
    assert _pi_ablation_control_evidence(accepted)["valid"]


def test_independent_gate2_candidate_prefers_hard_close_then_final_round_best() -> None:
    earlier_best = SimpleNamespace(
        generation=object(),
        comparison=None,
        compiled=True,
        relift_round=0,
        rank_tuple=(1,),
    )
    final_weaker = SimpleNamespace(
        generation=object(),
        comparison=None,
        compiled=True,
        relift_round=1,
        rank_tuple=(2,),
    )
    final_best = SimpleNamespace(
        generation=object(),
        comparison=None,
        compiled=True,
        relift_round=1,
        rank_tuple=(3,),
    )
    gate1 = SimpleNamespace(
        candidates=(earlier_best, final_weaker, final_best)
    )
    assert _independent_gate2_candidate(gate1) is final_best

    hard_closed = SimpleNamespace(
        generation=object(),
        comparison=SimpleNamespace(hard_pass=True),
        compiled=True,
        relift_round=0,
        rank_tuple=(0,),
    )
    gate1 = SimpleNamespace(candidates=(final_best, hard_closed))
    assert _independent_gate2_candidate(gate1) is hard_closed


def test_rejected_gate1_records_independent_gate2_without_overwriting_status(
    monkeypatch,
) -> None:
    case = build_case(_args("add"))
    provider = ProviderConfig(
        "test", "https://example.invalid", "secret", "base-model"
    )
    artifact = SimpleNamespace(
        interface=SimpleNamespace(function_name=case.function_name),
        to_json=lambda: {
            "kernel_id": case.function_name,
            "source": case.source_text,
        },
    )
    extraction = SimpleNamespace(
        artifact=artifact,
        provenance=SimpleNamespace(
            model="gpt-5.6-sol",
            prompt_sha256="lift-prompt",
            attempts=1,
            repairs=(),
            response_id="lift-response",
        ),
    )
    generation = _generation(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\ndef add_kernel(a_ptr, b_ptr, out_ptr, n_elements, "
        "BLOCK: tl.constexpr):\n    return\n",
        masked=False,
    )
    candidate = SimpleNamespace(
        generation=generation,
        comparison=SimpleNamespace(hard_pass=True),
        compiled=True,
        relift_round=0,
        rank_tuple=(1,),
    )
    gate1 = SimpleNamespace(
        status=Gate1Status.LEAKAGE_REJECTED,
        replay_closed=False,
        artifact=artifact,
        candidates=(candidate,),
        masked_generation=None,
        to_json=lambda: {
            "status": Gate1Status.LEAKAGE_REJECTED.value,
            "artifact": artifact.to_json(),
            "candidates": [{"generation": {"code_sha256": "digest"}}],
            "masked_generation": None,
            "diagnostics": [],
        },
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "get_provider", lambda: provider)
    monkeypatch.setattr(live, "extract_scrc_artifact", lambda *a, **k: extraction)
    monkeypatch.setattr(
        live, "ResponsesReplayWriter", lambda **kwargs: SimpleNamespace()
    )

    class FakeRelifter:
        def __init__(self, **kwargs):
            self.records = ()

    class FakeGate1Runner:
        def __init__(self, *, config):
            pass

        def run(self, *args, **kwargs):
            return gate1

    class FakeGate2Runner:
        def run(self, source, replay, gate2_case):
            captured["gate2_case"] = gate2_case
            captured["replay"] = replay
            return SimpleNamespace(status=Gate2Status.PASS)

    monkeypatch.setattr(live, "ResponsesSCRCRelifter", FakeRelifter)
    monkeypatch.setattr(live, "Gate1Runner", FakeGate1Runner)
    monkeypatch.setattr(live, "Gate2Runner", FakeGate2Runner)
    monkeypatch.setattr(live, "_gate2_case", lambda *args: "gate2-case")
    monkeypatch.setattr(
        live,
        "_gate2_evidence_json",
        lambda result: {"status": result.status.value},
    )
    args = Namespace(
        model="gpt-5.6-sol",
        effort="xhigh",
        seed=20260715,
        lift_repairs=2,
        relift_rounds=2,
        gate2_matrix=False,
        gate2_seeds=None,
        diagnose_rejected=True,
    )

    code, report = live.run_case(case, args)

    assert code == 3
    assert report["gate1"]["status"] == Gate1Status.LEAKAGE_REJECTED.value
    assert report["gate2_diagnostic"] == {
        "kind": "offline_gate2_for_gate1_rejected_sample",
        "independent": True,
        "gate1_status": Gate1Status.LEAKAGE_REJECTED.value,
        "status": Gate2Status.PASS.value,
    }
    assert captured["gate2_case"] == "gate2-case"


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


def test_relift_report_links_the_normal_no_closure_request() -> None:
    relifter = SimpleNamespace(
        records=(
            SimpleNamespace(
                relift_round=1,
                extraction=None,
                request={
                    "trigger": "normal_no_closure",
                    "evaluation_round": 0,
                    "candidate_count": 3,
                    "compiled_candidate_count": 3,
                    "best_generation_round": 0,
                    "hard_difference_categories": ["dataflow"],
                },
                diagnostic="relift failed",
                rejected_attempts=(),
            ),
        )
    )

    evidence = _relift_evidence_json(relifter)

    assert evidence[0]["request"]["trigger"] == "normal_no_closure"
    assert evidence[0]["request"]["hard_difference_categories"] == [
        "dataflow"
    ]
    assert evidence[0]["diagnostic"] == "relift failed"


def test_report_write_is_atomic_private_and_creates_explicit_parent(tmp_path) -> None:
    target = tmp_path / "nested" / "report.json"
    _write_report(str(target), {"status": "ok"})

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(target.parent.iterdir()) == [target]
