from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from direct_fixtures import make_copy_artifact
from intentir.extraction.providers.provider import ProviderConfig
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.gate1.compiler import CompileResult
from intentir.verify.gate1.model import Gate1Artifact, Gate1Status
from intentir.verify.gate1.orchestrate import (
    Gate1ArmEvidence,
    Gate1ExecutionResult,
    canonical_artifact_digest,
)
from intentir.verify.gate1.replay import ReplayAttempt, ReplayGeneration
from intentir.verify.gate2.model import (
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    Gate2SuiteResult,
)
from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import (
    CallBudget,
    CandidateAttemptReport,
    FeedbackKind,
    LiftFeedback,
    MainStatus,
    SCRCRunReport,
)
from intentir.verify.scrc_workload_run import (
    GATE1_DIAGNOSTIC_RECHECK_PROTOCOL,
    WorkloadExecutionError,
    build_workload_gate2_suite,
    classify_scrc_report,
    initialize_workload_gate1_recheck_evidence,
    initialize_workload_evidence,
    next_candidate_fanout_dir,
    next_gate1_recheck_dir,
    next_workload_run_dir,
    recheck_prepared_workload_gate1,
    save_workload_gate1_recheck,
    save_workload_failure,
    scrc_attempt_stage_counts,
    scrc_report_diagnostic,
    validate_workload_gate1_recheck_parent,
)
from intentir.verify.typecheck import TypeCheckStatus
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferRole,
    NoneBinding,
    PreparedTritonWorkload,
    ScalarBinding,
    TRITON_WORKLOAD_ADAPTERS,
    TritonWorkloadAdapter,
    prepare_triton_workload,
)


def test_evidence_directory_allocation_is_atomic_for_one_case(tmp_path) -> None:
    case_id = "high_performance:allocation_race"

    with ThreadPoolExecutor(max_workers=8) as pool:
        run_dirs = tuple(
            pool.map(
                lambda _index: next_workload_run_dir(tmp_path, case_id),
                range(8),
            )
        )
        fanout_dirs = tuple(
            pool.map(
                lambda _index: next_candidate_fanout_dir(tmp_path, case_id),
                range(8),
            )
        )
        recheck_dirs = tuple(
            pool.map(
                lambda _index: next_gate1_recheck_dir(tmp_path, case_id),
                range(8),
            )
        )

    assert sorted(path.name for path in run_dirs) == [
        f"run-{index:03d}" for index in range(1, 9)
    ]
    assert sorted(path.name for path in fanout_dirs) == [
        f"fanout-{index:03d}" for index in range(1, 9)
    ]
    assert sorted(path.name for path in recheck_dirs) == [
        f"recheck-{index:03d}" for index in range(1, 9)
    ]
    assert len(set(run_dirs)) == len(run_dirs)
    assert len(set(fanout_dirs)) == len(fanout_dirs)
    assert len(set(recheck_dirs)) == len(recheck_dirs)


def test_request_evidence_records_independent_lift_and_replay_backends(
    tmp_path,
) -> None:
    prepared = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    run_dir = next_workload_run_dir(tmp_path, prepared.entry.case_id)
    provider = ProviderConfig(
        "provider", "https://provider.invalid/v1", "key", "model"
    )

    initialize_workload_evidence(
        run_dir,
        prepared,
        provider=provider,
        effort="high",
        run_masked_j=True,
        lift_backend="responses",
        replay_backend="miniswe",
    )

    request = json.loads((run_dir / "request.json").read_text())
    assert request["lift_backend"] == "responses"
    assert request["replay_backend"] == "miniswe"


def _write_recheck_parent_lineage(
    tmp_path, prepared: PreparedTritonWorkload, artifact: Gate1Artifact
):
    run_dir = tmp_path / "old-run"
    parent = run_dir / "attempt-03"
    parent.mkdir(parents=True)
    (run_dir / "source.py").write_text(
        prepared.source_slice.code, encoding="utf-8"
    )
    (run_dir / "source_projection.json").write_text(
        json.dumps(
            {
                **prepared.source_slice.to_json(),
                "source_location": prepared.source_location,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "case_id": prepared.entry.case_id,
                "frontend": prepared.entry.frontend,
                "c0": prepared.request.binding.values,
                "grid": list(prepared.adapter.grid),
                "source_to_opaque": dict(prepared.source_to_opaque),
            }
        ),
        encoding="utf-8",
    )
    (parent / "gate1.json").write_text(
        json.dumps({"artifact_digest": canonical_artifact_digest(artifact)}),
        encoding="utf-8",
    )
    return run_dir, parent


def test_gate1_recheck_parent_lineage_binds_case_source_and_artifact(
    tmp_path,
) -> None:
    prepared = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    artifact = make_copy_artifact()
    artifact.c.values = dict(prepared.request.binding.values)
    artifact = canonicalize_gate1_artifact(artifact)
    run_dir, parent = _write_recheck_parent_lineage(
        tmp_path, prepared, artifact
    )

    assert validate_workload_gate1_recheck_parent(
        parent, prepared, artifact
    ) == run_dir.resolve()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cross_case", "request differs from selected workload"),
        ("wrong_artifact", "artifact digest differs"),
        ("wrong_source", "source differs from selected workload"),
        ("wrong_projection", "source projection differs"),
        ("missing_request_field", "request lacks lineage field"),
    ],
)
def test_gate1_recheck_parent_lineage_rejects_mismatch(
    tmp_path, mutation: str, message: str
) -> None:
    prepared = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    artifact = make_copy_artifact()
    artifact.c.values = dict(prepared.request.binding.values)
    artifact = canonicalize_gate1_artifact(artifact)
    run_dir, parent = _write_recheck_parent_lineage(
        tmp_path, prepared, artifact
    )
    selected_artifact = artifact
    if mutation == "cross_case":
        request = json.loads((run_dir / "request.json").read_text())
        request["case_id"] = "high_performance:another_case"
        (run_dir / "request.json").write_text(json.dumps(request))
    elif mutation == "wrong_artifact":
        selected_artifact = copy.deepcopy(artifact)
        selected_artifact.j.kernel_id = "scrc_wrong_parent_artifact"
    elif mutation == "wrong_source":
        (run_dir / "source.py").write_text("# different source\n")
    elif mutation == "wrong_projection":
        projection = json.loads(
            (run_dir / "source_projection.json").read_text()
        )
        projection["function"] = "different_entry"
        (run_dir / "source_projection.json").write_text(
            json.dumps(projection)
        )
    elif mutation == "missing_request_field":
        request = json.loads((run_dir / "request.json").read_text())
        request.pop("source_to_opaque")
        (run_dir / "request.json").write_text(json.dumps(request))
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        validate_workload_gate1_recheck_parent(
            parent, prepared, selected_artifact
        )


def test_fixed_artifact_recheck_canonicalizes_typechecks_and_runs_normal_only(
    monkeypatch,
) -> None:
    import intentir.verify.scrc_workload_run as workload_run

    artifact = make_copy_artifact()
    prepared = SimpleNamespace(request=object())
    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "test-key", "test-model"
    )
    generation = ReplayGeneration(
        code="replay",
        function_name="scrc_k0",
        model="test-model",
        response_id="response-1",
        packet_digest="packet",
        prompt_digest="prompt",
        request_sha256="request",
        response_sha256="response",
    )
    calls: dict[str, object] = {}

    class FakeWriter:
        def __init__(self, *, provider, effort):
            calls["provider"] = provider
            calls["effort"] = effort
            self.records = (generation,)

    class FakeRunner:
        def __init__(self, *, compiler, writer, config):
            calls["compiler"] = compiler
            calls["writer"] = writer
            calls["config"] = config

        def run(self, candidate, *, source, source_case):
            calls["artifact"] = candidate
            calls["source"] = source
            calls["source_case"] = source_case
            return SimpleNamespace(
                artifact_digest=canonical_artifact_digest(candidate),
                status=SimpleNamespace(value="replay_closed"),
                passed=True,
                normal=SimpleNamespace(generation=generation, diagnostic=None),
                masked_j=None,
                masked_j_calls=0,
                to_json=lambda: {"status": "replay_closed"},
            )

    monkeypatch.setattr(workload_run, "ResponsesReplayWriter", FakeWriter)
    monkeypatch.setattr(workload_run, "TritonCompiler", lambda **_kwargs: object())
    monkeypatch.setattr(workload_run, "Gate1Runner", FakeRunner)
    prepared.request = SimpleNamespace(
        source=SimpleNamespace(gate_source="trusted-source"),
        binding=SimpleNamespace(gate1_case="c0-case"),
    )

    result = recheck_prepared_workload_gate1(
        prepared,
        artifact,
        provider=provider,
        effort="xhigh",
    )

    assert result.protocol == GATE1_DIAGNOSTIC_RECHECK_PROTOCOL
    assert result.diagnostic_only
    assert result.typecheck.passed
    assert result.gate1 is not None and result.gate1.passed
    assert result.artifact_digest == canonical_artifact_digest(result.artifact)
    assert result.artifact.to_json() == canonicalize_gate1_artifact(artifact).to_json()
    assert calls["config"].run_masked_j is False
    assert calls["source"] == "trusted-source"
    assert calls["source_case"] == "c0-case"
    assert calls["effort"] == "xhigh"


def test_fixed_artifact_recheck_stops_at_same_typecheck_before_provider(
    monkeypatch,
) -> None:
    import intentir.verify.scrc_workload_run as workload_run

    artifact = make_copy_artifact()
    artifact.j.intent.body[0].attrs["fn"] = "tanh"
    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "test-key", "test-model"
    )

    def forbidden_writer(**_kwargs):
        pytest.fail("TypeCheck failure must not create a Replay writer")

    monkeypatch.setattr(workload_run, "ResponsesReplayWriter", forbidden_writer)

    result = recheck_prepared_workload_gate1(
        SimpleNamespace(request=object()),
        artifact,
        provider=provider,
        effort="xhigh",
    )

    assert result.diagnostic_only
    assert not result.typecheck.passed
    assert result.gate1 is None
    assert result.replay_writer is None


def test_replay_backend_is_independent_and_new_per_formal_round(
    monkeypatch,
) -> None:
    import intentir.verify.scrc_workload_run as workload_run

    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "key", "model"
    )
    created: list[object] = []

    class FakeMiniReplayWriter:
        def __init__(self, *, provider, effort):
            self.provider = provider
            self.effort = effort
            self.records = ()
            created.append(self)

    monkeypatch.setattr(
        workload_run,
        "MiniSWEReplayWriter",
        FakeMiniReplayWriter,
    )
    prepared = SimpleNamespace(
        compiler_view=None,
        entry=SimpleNamespace(symbol="source_symbol"),
    )
    runner, _lift, writers = workload_run._build_workload_runtime(
        prepared,
        provider=provider,
        effort="xhigh",
        lift_backend="responses",
        replay_backend="miniswe",
    )

    first = runner.dependencies.gate1_factory(1)
    second = runner.dependencies.gate1_factory(2)

    assert first is not second
    assert writers == created
    assert len({id(writer) for writer in writers}) == 2
    assert all(writer.provider is provider for writer in writers)
    with pytest.raises(ValueError, match="unknown Replay backend"):
        workload_run._new_replay_writer(
            "lift-backend-name-is-not-replay",
            provider=provider,
            effort=None,
        )


def test_fixed_artifact_recheck_evidence_is_isolated_and_non_certifying(
    tmp_path,
) -> None:
    prepared = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "test-key", "test-model"
    )
    artifact = canonicalize_gate1_artifact(make_copy_artifact())
    typecheck = __import__(
        "intentir.verify.typecheck", fromlist=["type_check"]
    ).type_check(artifact.j, artifact.c, artifact.pi)
    trajectory = {"info": {"api_key": "<redacted>", "fresh_session": True}}
    trajectory_bytes = json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    generation = ReplayGeneration(
        code="normalized replay\n",
        function_name="scrc_k0",
        model="test-model",
        response_id="response-1",
        packet_digest="packet",
        prompt_digest="prompt",
        request_sha256="request",
        response_sha256="response",
        raw_response_sha256=hashlib.sha256(trajectory_bytes).hexdigest(),
        raw_response=trajectory_bytes,
        response_text="exact submitted replay\n",
        trajectory=trajectory,
        backend_protocol="minisweagent.intentir-replay.v2",
        episode_contract_sha256="e" * 64,
    )
    gate1 = SimpleNamespace(
        protocol="llm_direct_pi_gate1_level2_v1",
        status=SimpleNamespace(value="replay_closed"),
        passed=True,
        normal=SimpleNamespace(generation=generation, diagnostic=None),
        masked_j=None,
        masked_j_calls=0,
        to_json=lambda: {"status": "replay_closed", "masked_j": None},
    )
    result = SimpleNamespace(
        protocol=GATE1_DIAGNOSTIC_RECHECK_PROTOCOL,
        diagnostic_only=True,
        artifact=artifact,
        artifact_digest=canonical_artifact_digest(artifact),
        typecheck=typecheck,
        gate1=gate1,
        replay_writer=SimpleNamespace(records=(generation,)),
        passed=True,
    )
    parent = tmp_path / "old-run" / "attempt-03"
    parent.mkdir(parents=True)
    recheck_dir = next_gate1_recheck_dir(tmp_path, prepared.entry.case_id)

    initialize_workload_gate1_recheck_evidence(
        recheck_dir,
        prepared,
        provider=provider,
        effort="xhigh",
        parent_evidence=parent,
        replay_backend="miniswe",
    )
    save_workload_gate1_recheck(recheck_dir, prepared, result)

    metadata = __import__("json").loads(
        (recheck_dir / "recheck.json").read_text()
    )
    assert metadata["protocol"] == GATE1_DIAGNOSTIC_RECHECK_PROTOCOL
    assert metadata["diagnostic_only"] is True
    assert metadata["enters_gate2"] is False
    assert metadata["changes_prior_verdict"] is False
    assert metadata["parent_evidence"] == str(parent.resolve())
    assert metadata["artifact_digest"] == result.artifact_digest
    request = json.loads((recheck_dir / "request.json").read_text())
    assert request["replay_backend"] == "miniswe"
    assert (recheck_dir / "artifact.json").is_file()
    assert (recheck_dir / "typecheck.json").is_file()
    assert (recheck_dir / "gate1.json").is_file()
    assert (recheck_dir / "replay.py").is_file()
    assert (recheck_dir / "replay_receipt.json").is_file()
    assert (recheck_dir / "replay.py").read_text() == (
        "exact submitted replay\n"
    )
    assert (recheck_dir / "replay_normalized.py").read_text() == (
        "normalized replay\n"
    )
    receipt = json.loads((recheck_dir / "replay_receipt.json").read_text())
    assert receipt["backend_protocol"] == "minisweagent.intentir-replay.v2"
    assert receipt["episode_contract_sha256"] == "e" * 64
    assert json.loads((recheck_dir / "replay_trajectory.json").read_text()) == (
        trajectory
    )
    assert not (parent / "gate1.json").exists()


def test_infrastructure_replay_evidence_saves_exact_sources_receipts_and_hashes(
    tmp_path,
) -> None:
    def trajectory_record(label: str) -> tuple[dict, bytes, str]:
        trajectory = {
            "info": {
                "api_key": "<redacted>",
                "fresh_session": True,
                "label": label,
            }
        }
        encoded = json.dumps(
            trajectory,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return trajectory, encoded, hashlib.sha256(encoded).hexdigest()

    normal_trajectory, normal_raw, normal_digest = trajectory_record("normal")
    masked_trajectory, masked_raw, masked_digest = trajectory_record("masked")
    generation = ReplayGeneration(
        code="normalized normal\n",
        function_name="scrc_k0",
        model="model",
        response_id="normal-id",
        packet_digest="packet",
        prompt_digest="prompt",
        response_text="exact normal submission\n",
        raw_response_sha256=normal_digest,
        raw_response=normal_raw,
        trajectory=normal_trajectory,
        backend_protocol="minisweagent.intentir-replay.v2",
        episode_contract_sha256="a" * 64,
    )
    rejected = ReplayAttempt(
        model="model",
        diagnostic="closed rejection",
        response_text="exact masked rejection\n",
        response_id="masked-id",
        raw_response_sha256=masked_digest,
        raw_response=masked_raw,
        trajectory=masked_trajectory,
        backend_protocol="minisweagent.intentir-replay.v2",
        episode_contract_sha256="b" * 64,
    )
    lift = SimpleNamespace(records=())
    writer = SimpleNamespace(records=(generation, rejected))
    lifecycle_dir = tmp_path / "lifecycle"
    lifecycle_dir.mkdir()
    attempt = SimpleNamespace(
        formal_round=1,
        artifact=None,
        typecheck=None,
        gate1=SimpleNamespace(
            normal=SimpleNamespace(generation=generation),
            masked_j=SimpleNamespace(generation=None),
            to_json=lambda: {"status": "masked_rejected"},
        ),
    )
    import intentir.verify.scrc_workload_run as workload_run

    workload_run._save_attempt_evidence(
        lifecycle_dir,
        (attempt,),
        lift,
        (writer,),
    )
    attempt_dir = lifecycle_dir / "attempt-01"
    assert (attempt_dir / "replay.py").read_text() == (
        "exact normal submission\n"
    )
    assert (attempt_dir / "masked_j_replay_rejected.txt").read_text() == (
        "exact masked rejection\n"
    )
    assert (attempt_dir / "replay_receipt.json").is_file()
    assert (attempt_dir / "masked_j_replay_receipt.json").is_file()
    assert (attempt_dir / "replay_trajectory.json").is_file()
    assert (attempt_dir / "masked_j_replay_trajectory.json").is_file()

    runtime = WorkloadExecutionError(
        RuntimeError("infrastructure stopped"),
        lift,
        (writer,),
    )
    run_dir = tmp_path / "infrastructure"
    run_dir.mkdir()

    save_workload_failure(run_dir, runtime, runtime=runtime)

    replay_dir = run_dir / "gate1-call-01"
    assert (replay_dir / "replay.py").read_text() == (
        "exact normal submission\n"
    )
    assert (replay_dir / "replay_normalized.py").read_text() == (
        "normalized normal\n"
    )
    assert (replay_dir / "masked_j_replay_rejected.txt").read_text() == (
        "exact masked rejection\n"
    )
    for prefix, digest in (
        ("replay", "a" * 64),
        ("masked_j_replay", "b" * 64),
    ):
        receipt = json.loads(
            (replay_dir / f"{prefix}_receipt.json").read_text()
        )
        assert receipt["backend_protocol"] == (
            "minisweagent.intentir-replay.v2"
        )
        assert receipt["episode_contract_sha256"] == digest
        assert (replay_dir / f"{prefix}_trajectory.json").is_file()


def test_unaccepted_lift_continuation_is_persisted_as_draft_only(
    tmp_path,
) -> None:
    import intentir.verify.scrc_workload_run as workload_run
    from intentir.extraction.lifters.scrc import SCRCReliftRecord

    draft = {"schema": "draft", "j": {}, "c": {}, "pi": {}}
    lift = SimpleNamespace(
        records=(SCRCReliftRecord(1, continuation_candidate=draft),)
    )
    attempt = SimpleNamespace(
        formal_round=None,
        artifact=None,
        typecheck=None,
        gate1=None,
    )
    lifecycle_dir = tmp_path / "draft-lifecycle"
    lifecycle_dir.mkdir()

    workload_run._save_attempt_evidence(
        lifecycle_dir,
        (attempt,),
        lift,
        (),
    )

    path = lifecycle_dir / "attempt-01/unaccepted_lift_continuation.json"
    assert json.loads(path.read_text()) == draft
    assert not (lifecycle_dir / "attempt-01/artifact.json").exists()


def test_replay_evidence_rejects_trajectory_hash_or_secret_before_source_write(
    tmp_path,
) -> None:
    import intentir.verify.scrc_workload_run as workload_run

    trajectory = {"info": {"api_key": "<redacted>"}}
    record = ReplayAttempt(
        model="model",
        diagnostic="rejected",
        response_text="must not be written",
        raw_response_sha256="0" * 64,
        trajectory=trajectory,
    )
    with pytest.raises(ValueError, match="canonical hash"):
        workload_run._save_replay_record_evidence(
            tmp_path,
            prefix="replay",
            exact_source_name="replay_rejected.txt",
            record=record,
        )
    assert not (tmp_path / "replay_rejected.txt").exists()

    encoded = json.dumps(
        {"info": {"api_key": "not-redacted"}},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    unredacted = replace(
        record,
        trajectory={"info": {"api_key": "not-redacted"}},
        raw_response_sha256=hashlib.sha256(encoded).hexdigest(),
        raw_response=encoded,
    )
    with pytest.raises(ValueError, match="unredacted secret"):
        workload_run._save_replay_record_evidence(
            tmp_path,
            prefix="masked_j_replay",
            exact_source_name="masked_j_replay_rejected.txt",
            record=unredacted,
        )


def _budget() -> CallBudget:
    return CallBudget(1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)


def test_gate2_suite_binds_adapter_roles_to_accepted_j() -> None:
    base = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    adapter = TritonWorkloadAdapter(
        base.adapter.case_id,
        buffers=(
            BufferBinding("X", BufferRole.INPUT, "f32", (1000,)),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(),
        c_values=(("BLOCK", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=17,
    )
    prepared = replace(
        base,
        adapter=adapter,
        request=replace(
            base.request,
            binding=replace(
                base.request.binding,
                c_values=(("c0", 1024),),
            ),
        ),
        source_to_opaque={"X": "p0", "Y": "p1", "N": "p2", "BLOCK": "c0"},
    )
    artifact = make_copy_artifact()

    suite = build_workload_gate2_suite(prepared, artifact)

    case = suite.cases[0]
    assert suite.c_values == (("c0", 1024),)
    assert case.output_keys == ("output",)
    assert case.refeval.input_binding_map == {"t0": "input_p0"}
    assert case.refeval.output_binding_map == {"t1": "output"}
    assert case.triton.schedule == {"c0": 1024}
    assert case.triton.arguments[0].name == "p0"
    assert case.triton.arguments[1].name == "p1"
    assert case.triton.arguments[2].name == "p2"


def test_gate2_suite_shares_explicit_zero_fixture_across_all_three_legs() -> None:
    base = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    adapter = TritonWorkloadAdapter(
        base.adapter.case_id,
        buffers=(
            BufferBinding(
                "X",
                BufferRole.INPUT,
                "f32",
                (1000,),
                initializer=BufferInitializer.ZEROS,
            ),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(),
        c_values=(("BLOCK", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=23,
    )
    prepared = replace(
        base,
        adapter=adapter,
        request=replace(
            base.request,
            binding=replace(base.request.binding, c_values=(("c0", 1024),)),
        ),
        source_to_opaque={"X": "p0", "Y": "p1", "N": "p2", "BLOCK": "c0"},
    )

    case = build_workload_gate2_suite(
        prepared, make_copy_artifact()
    ).cases[0]

    assert np.array_equal(case.values["input_p0"], np.zeros(1000, dtype=np.float32))
    # Source and Replay consume the same Triton spec/value key; RefEval binds
    # that identical key rather than regenerating an independent fixture.
    assert case.triton.arguments[0].value_key == "input_p0"
    assert case.refeval.input_binding_map["t0"] == "input_p0"


def test_gate2_suite_passes_none_as_opaque_without_logical_binding() -> None:
    base = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    adapter = TritonWorkloadAdapter(
        base.adapter.case_id,
        buffers=(
            BufferBinding("X", BufferRole.INPUT, "f32", (1000,)),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(),
        c_values=(("BLOCK", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=29,
        none_arguments=(NoneBinding("OPTIONAL"),),
    )
    prepared = replace(
        base,
        adapter=adapter,
        request=replace(
            base.request,
            binding=replace(base.request.binding, c_values=(("c0", 1024),)),
        ),
        source_to_opaque={
            "X": "p0",
            "Y": "p1",
            "N": "p2",
            "OPTIONAL": "p3",
            "BLOCK": "c0",
        },
    )
    payload = make_copy_artifact().to_json()
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "opaque",
            "mode": "value",
            "dtype": "none",
            "rank": None,
            "alias_of": None,
            "logical_ref": None,
        }
    )

    case = build_workload_gate2_suite(
        prepared, Gate1Artifact.from_json(payload)
    ).cases[0]

    assert case.triton.arguments[-1].name == "p3"
    assert case.triton.arguments[-1].role.value == "none"
    assert "p3" not in case.values


def test_report_classification_preserves_reference_unsupported() -> None:
    execution = ExecutionResult("source", True, {"out": __import__("numpy").zeros(1)})
    gate2 = Gate2SuiteResult(
        "suite",
        (
            Gate2Result(
                "case",
                Gate2Status.REFERENCE_UNSUPPORTED,
                execution,
                ExecutionResult("replay", True, execution.outputs),
                ExecutionResult("j", False, unsupported=True),
            ),
        ),
        (),
    )
    report = SCRCRunReport(
        MainStatus.GATE2_FAILED,
        (),
        1,
        gate2,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.REFERENCE_UNSUPPORTED


def test_gate2_binds_semantic_constexpr_as_rank_zero_j_input() -> None:
    base = prepare_triton_workload(next(iter(TRITON_WORKLOAD_ADAPTERS.values())))
    adapter = TritonWorkloadAdapter(
        base.adapter.case_id,
        buffers=(
            BufferBinding("X", BufferRole.INPUT, "f32", (1000,)),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(("FLAG", True),),
        c_values=(("BLOCK", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=19,
    )
    prepared = replace(
        base,
        adapter=adapter,
        request=replace(
            base.request,
            binding=replace(base.request.binding, c_values=(("c0", 1024),)),
        ),
        source_to_opaque={
            "X": "p0",
            "Y": "p1",
            "N": "p2",
            "FLAG": "p3",
            "BLOCK": "c0",
        },
    )
    payload = make_copy_artifact().to_json()
    function = payload["j"]["intent"]["intent_function"]
    function["inputs"].append({"name": "t1", "dtype": "i1", "shape": []})
    function["outputs"][0]["name"] = "t2"
    function["body"][0]["result"] = "t2"
    payload["j"]["signature"]["parameters"][1]["logical_ref"] = "t2"
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "constexpr",
            "mode": "value",
            "dtype": "i1",
            "rank": 0,
            "logical_ref": "t1",
        }
    )
    artifact = Gate1Artifact.from_json(payload)

    suite = build_workload_gate2_suite(prepared, artifact)

    case = suite.cases[0]
    assert case.refeval.input_binding_map == {
        "t0": "input_p0",
        "t1": "logical_p3",
    }
    assert case.values["logical_p3"].shape == ()
    assert bool(case.values["logical_p3"])


def test_certified_report_classifies_directly() -> None:
    report = SCRCRunReport(
        MainStatus.SCRC_CERTIFIED,
        (),
        1,
        None,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.SCRC_CERTIFIED


def test_terminal_pre_replay_rejection_keeps_stage_and_diagnostic() -> None:
    feedback = LiftFeedback(
        FeedbackKind.LIFT,
        {"diagnostic": "artifact schema rejected"},
    )
    attempt = CandidateAttemptReport(
        lift_call=4,
        formal_round=None,
        typecheck_status=TypeCheckStatus.NOT_RUN,
        feedback=feedback,
    )
    report = SCRCRunReport(
        MainStatus.NO_CLOSURE,
        (attempt,),
        None,
        None,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.LIFT_RESPONSE_REJECTED
    assert scrc_attempt_stage_counts(report) == {"lift_response_rejected": 1}
    assert scrc_report_diagnostic(report) == "artifact schema rejected"


def _gate1_diagnostic_report(diagnostic: str) -> SCRCRunReport:
    source_compile = CompileResult(True, ttir="module {}")
    gate1 = Gate1ExecutionResult(
        Gate1Status.NO_CLOSURE,
        "artifact-digest",
        source_compile,
        None,
        Gate1ArmEvidence(
            "normal", None, None, None, None, diagnostic
        ),
        None,
        0,
        0,
    )
    attempt = CandidateAttemptReport(
        lift_call=1,
        formal_round=1,
        typecheck_status=TypeCheckStatus.PASS,
        gate1=gate1,
    )
    return SCRCRunReport(
        MainStatus.NO_CLOSURE,
        (attempt,),
        None,
        None,
        None,
        _budget(),
    )


def test_source_observer_gap_has_dedicated_batch_and_stage_classification() -> None:
    report = _gate1_diagnostic_report(
        "observer unsupported: source TTIR: TTIRParseError: "
        "multi-block CFG is not supported"
    )

    assert (
        classify_scrc_report(report)
        is BatchClassification.OBSERVER_UNSUPPORTED
    )
    assert scrc_attempt_stage_counts(report) == {"observer_unsupported": 1}


def test_harness_value_error_remains_harness_invalid() -> None:
    report = _gate1_diagnostic_report(
        "harness binding failed: ValueError: schedule differs from C"
    )

    assert classify_scrc_report(report) is BatchClassification.HARNESS_INVALID
    assert scrc_attempt_stage_counts(report) == {"harness_invalid": 1}


def test_replay_observation_failure_is_not_source_observer_unsupported() -> None:
    source_compile = CompileResult(True, ttir="module {}")
    generation = ReplayGeneration(
        code="replay",
        function_name="scrc_k0",
        model="test-model",
        response_id=None,
        packet_digest="packet",
        prompt_digest="prompt",
    )
    gate1 = Gate1ExecutionResult(
        Gate1Status.NO_CLOSURE,
        "artifact-digest",
        source_compile,
        None,
        Gate1ArmEvidence(
            "normal",
            generation,
            source_compile,
            None,
            None,
            "replay observation failed: TTIRParseError: unsupported region",
        ),
        None,
        1,
        0,
    )
    attempt = CandidateAttemptReport(
        lift_call=1,
        formal_round=1,
        typecheck_status=TypeCheckStatus.PASS,
        gate1=gate1,
    )
    report = SCRCRunReport(
        MainStatus.NO_CLOSURE,
        (attempt,),
        None,
        None,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.NO_CLOSURE
    assert scrc_attempt_stage_counts(report) == {"gate1_hard_mismatch": 1}
