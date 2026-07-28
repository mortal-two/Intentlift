from __future__ import annotations

import copy
import hashlib
import json

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1 import replay as replay_module
from intentir.verify.gate1.compiler import CompileResult
from intentir.verify.gate1.model import Frontend, Gate1Artifact
from intentir.verify.gate1.prompt import assert_source_isolated_payload
from intentir.verify.gate1.replay import (
    MINISWE_REPLAY_BACKEND,
    MiniSWEReplayWriter,
    ReplayIngressResult,
    _replay_input_files,
    build_source_blind_replay_compile_case,
    replay_source_ingress,
    source_blind_replay_compile_check,
)
from minisweagent.environments.intentir_replay import (
    INTENTIR_PACKET_QUERY,
    INTENTIR_REPLAY_CHECK,
    INTENTIR_REPLAY_COMMIT,
    INTENTIR_REPLAY_COMPILE_CHECK,
    INTENTIR_REPLAY_INSPECT,
    INTENTIR_REPLAY_ROLLBACK,
    INTENTIR_REPLAY_SUBMIT,
    INTENTIR_REPLAY_SUBMIT_COMMAND,
    REPLAY_CHECK_PROTOCOL,
    REPLAY_INPUT_FILES,
    REPLAY_SANDBOX_HOSTNAME,
    REPLAY_TOOL_APPLETS,
    REPLAY_TRANSACTION_PROTOCOL,
    IntentIRReplayEnvironment,
    IntentIRReplayWorkspace,
)
from minisweagent.exceptions import Submitted
from minisweagent.run.intentir_replay import (
    intentir_replay_episode_contract_sha256,
    run_intentir_replay,
    validated_intentir_replay_config,
)


VALID_REPLAY = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = tl.load(p0 + v0, mask=v0 < p2, other=0.0)
    tl.store(p1 + v0, v1, mask=v0 < p2)
"""

INVALID_CONSTEXPR_REPLAY = VALID_REPLAY.replace(
    "c0: tl.constexpr", "c0"
)


class _ScriptedModel:
    def __init__(
        self, commands: list[str | list[str]], *, secret: str = ""
    ) -> None:
        self.commands = list(commands)
        self.secret = secret
        self.query_inputs: list[list[dict]] = []

    def query(self, messages, **kwargs):
        del kwargs
        self.query_inputs.append(list(messages))
        command = self.commands.pop(0)
        commands = [command] if isinstance(command, str) else command
        return {
            "role": "assistant",
            "content": "",
            "extra": {
                "actions": [{"command": item} for item in commands],
                "cost": 0.0,
            },
        }

    def format_message(self, **kwargs):
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        del message, template_vars
        return [
            {
                "role": "user",
                "content": output["output"],
                "extra": {"returncode": output["returncode"]},
            }
            for output in outputs
        ]

    def get_template_vars(self, **kwargs):
        return dict(kwargs)

    def serialize(self):
        return {
            "info": {
                "config": {
                    "model": "scripted",
                    "api_key": self.secret,
                }
            }
        }


def _write_source(source: str) -> str:
    escaped = source.replace("'", "'\\''")
    return f"printf '%s' '{escaped}' > /output/replay.source"


def _workspace() -> IntentIRReplayWorkspace:
    files, _, _ = _replay_input_files(
        make_reduce_artifact(), frontend=Frontend.TRITON, masked=False
    )
    return IntentIRReplayWorkspace.create(files)


def test_replay_input_is_exact_source_isolated_allowlist() -> None:
    files, packet, _ = _replay_input_files(
        make_reduce_artifact(), frontend=Frontend.TRITON, masked=False
    )

    assert set(files) == REPLAY_INPUT_FILES
    assert_source_isolated_payload(packet)
    assert json.loads(files["packet.json"]) == packet
    assert set(packet) == {"j", "c", "pi", "u_f"}
    assert set(packet["c"]) == {"slots"}
    encoded = json.dumps(packet, sort_keys=True)
    for forbidden in (
        "source_code",
        "anonymous_source",
        "harness",
        "compiler_diff",
        "gate1_diff",
        "prior_replay",
        "conversation",
        "CompileCase",
        "trajectory",
    ):
        assert forbidden not in encoded

    with pytest.raises(ValueError, match="fixed allowlisted files"):
        IntentIRReplayWorkspace.create({**files, "source.py": "forbidden"})


def test_packet_query_exposes_bounded_source_blind_views() -> None:
    files, packet, _ = _replay_input_files(
        make_reduce_artifact(), frontend=Frontend.TRITON, masked=False
    )
    assert INTENTIR_PACKET_QUERY in files["README.md"]
    assert INTENTIR_REPLAY_COMPILE_CHECK in files["TOOL.md"]
    workspace = IntentIRReplayWorkspace.create(files)
    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        overview = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} overview"}
        )
        abi = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} abi 0 48"}
        )
        graph = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} graph 0 12"}
        )
        realization = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} realization"}
        )
        edges = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} edges 0 24"}
        )
        routes = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} routes 0 24"}
        )
        order = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} order 0 24"}
        )
        obligations = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} obligations 0 12"}
        )
        obligation = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} obligation i1"}
        )
        semantic_obligation = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} obligation n0"}
        )
        j_view = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} section j"}
        )
        handle = str(next(iter(packet["j"]["node_ids"])))
        references = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} find {handle}"}
        )
        rejected_compound = environment.execute(
            {"command": f"{INTENTIR_PACKET_QUERY} overview; cat /input/packet.json"}
        )
    finally:
        workspace.cleanup()

    assert overview["returncode"] == 0
    assert len(overview["output"].encode()) <= 10_000
    assert '"operations"' in overview["output"]
    assert abi["returncode"] == 0
    assert '"logical_ref"' in abi["output"]
    assert '"intent_io"' in abi["output"]
    assert graph["returncode"] == 0
    assert '"attrs"' in graph["output"]
    assert realization["returncode"] == 0
    assert '"topology"' in realization["output"]
    assert edges["returncode"] == 0
    assert '"routes"' in edges["output"]
    assert routes["returncode"] == 0
    resolved_routes = json.loads(routes["output"])
    assert resolved_routes["protocol"] == "intentir.replay-resolved-routes.v1"
    assert "infers no edge" in resolved_routes["meaning"]
    assert resolved_routes["items"][0]["source"]["semantic_ref"] == "a0"
    assert resolved_routes["items"][0]["source"]["semantic"][
        "semantic_kind"
    ] == "boundary_access"
    assert resolved_routes["items"][0]["target"]["semantic_ref"] == "n0"
    assert resolved_routes["items"][0]["target_local_slots"] == [0, 1]
    assert resolved_routes["items"][0]["required_flow"] == (
        "i0 MUST flow to i1 at target-local slots [0, 1]"
    )
    assert order["returncode"] == 0
    resolved_order = json.loads(order["output"])
    assert resolved_order["protocol"] == "intentir.replay-resolved-order.v1"
    assert "MUST execute before" in resolved_order["meaning"]
    assert resolved_order["items"][0]["before"]["semantic_ref"] == "a0"
    assert resolved_order["items"][0]["before"]["semantic"][
        "semantic_kind"
    ] == "boundary_access"
    assert resolved_order["items"][0]["required_direction"] == (
        "i0 MUST execute before i1"
    )
    assert obligations["returncode"] == 0
    obligation_page = json.loads(obligations["output"])
    assert obligation_page["protocol"] == (
        "intentir.replay-instance-obligations.v2"
    )
    assert "infers no edge" in obligation_page["meaning"]
    assert obligation_page["columns"][0:4] == [
        "id",
        "semantic_ref",
        "mechanism",
        "region[id,parent,placement]",
    ]
    i0 = obligation_page["items"][0]
    assert i0[0:3] == ["i0", "a0", "realize"]
    assert i0[6][0] == ["i1", [0, 1]]
    assert {row[0] for row in obligation_page["items"]} <= {
        item["id"] for item in packet["pi"]["instances"]
    }
    for forbidden in (
        "source_code",
        "harness",
        "compiler_diff",
        "gate1_diff",
        "prior_replay",
        "trajectory",
    ):
        assert forbidden not in obligations["output"]
    assert obligation["returncode"] == 0
    i1 = json.loads(obligation["output"])["items"][0]
    assert i1["id"] == "i1"
    assert i1["incoming_routes"][0]["source"]["id"] == "i0"
    assert semantic_obligation["returncode"] == 0
    assert json.loads(semantic_obligation["output"])["items"][0][
        "semantic_ref"
    ] == "n0"
    assert j_view["returncode"] == 0
    assert '"intent_function"' in j_view["output"]
    assert references["returncode"] == 0
    assert handle in references["output"]
    assert rejected_compound["returncode"] == 1


def test_own_replay_source_inspect_is_syntax_only_and_diagnostic() -> None:
    workspace = _workspace()
    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        written = environment.execute({"command": _write_source(VALID_REPLAY)})
        overview = environment.execute(
            {
                "command": (
                    f"{INTENTIR_REPLAY_INSPECT} /output/replay.source overview"
                )
            }
        )
        statements = environment.execute(
            {
                "command": (
                    f"{INTENTIR_REPLAY_INSPECT} /output/replay.source "
                    "statements 0 16"
                )
            }
        )
        found = environment.execute(
            {
                "command": (
                    f"{INTENTIR_REPLAY_INSPECT} /output/replay.source find v1"
                )
            }
        )
        rejected_compound = environment.execute(
            {
                "command": (
                    f"{INTENTIR_REPLAY_INSPECT} /output/replay.source overview; "
                    "cat /input/packet.json"
                )
            }
        )
        audit = environment.serialize()["info"]
    finally:
        workspace.cleanup()

    assert written["returncode"] == 0
    assert overview["returncode"] == 0
    summary = json.loads(overview["output"])
    assert summary["protocol"] == "intentir.replay-own-source-inventory.v1"
    assert "current replay.source" in summary["authority"]
    assert "performs no comparison, repair, or generation" in summary["authority"]
    assert summary["functions"][0]["name"] == "scrc_k0"
    assert summary["call_counts"] == {
        "tl.arange": 1,
        "tl.load": 1,
        "tl.store": 1,
    }
    assert summary["control_counts"] == {}
    assert "a0" not in overview["output"]
    assert statements["returncode"] == 0
    statement_items = json.loads(statements["output"])["items"]
    assert any(item["defines"] == ["v1"] for item in statement_items)
    assert any("tl.store" in item["calls"] for item in statement_items)
    assert found["returncode"] == 0
    assert json.loads(found["output"])["match_total"] >= 2
    assert rejected_compound["returncode"] == 1
    inspect_audit = audit["replay_source_inspect"]
    assert inspect_audit["authority"] == (
        "diagnostic_syntax_only_own_replay_source"
    )
    assert inspect_audit["calls"] == 4
    assert inspect_audit["limit"] == 0


def test_synthetic_compile_profile_never_reads_c0_or_harness_values() -> None:
    artifact = make_reduce_artifact()
    artifact.c.values["c0"] = 997

    case = build_source_blind_replay_compile_case(artifact)

    assert case.grid == (1, 1, 1)
    assert case.schedule["c0"] == 16
    assert 997 not in case.schedule.values()
    assert all(
        argument.shape == (1,)
        for argument in case.arguments.values()
        if argument.kind == "tensor"
    )


def test_synthetic_compile_profile_accepts_legacy_tensor_signature_kind() -> None:
    payload = make_reduce_artifact().to_json()
    payload["j"]["signature"]["parameters"][0]["kind"] = "tensor"
    artifact = Gate1Artifact.from_json(payload)

    case = build_source_blind_replay_compile_case(artifact)

    assert case.arguments["p0"].kind == "tensor"
    assert case.arguments["p0"].shape == (1,)


def test_synthetic_compile_check_returns_actionable_own_code_diagnostic() -> None:
    captured = []

    class RejectingCompiler:
        def compile(self, source, case):
            captured.append((source, case))
            return CompileResult(
                False,
                diagnostic=(
                    "triton.compiler.errors.CompilationError: at 42:4:\n"
                    "    tl.store(p1, v0)\n"
                    "Value argument cannot be block type if pointer argument "
                    "is not a block"
                ),
            )

    result = source_blind_replay_compile_check(
        VALID_REPLAY,
        make_reduce_artifact(),
        "scrc_k0",
        compiler=RejectingCompiler(),  # type: ignore[arg-type]
    )

    assert result.status == "fail"
    assert len(captured) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic["rule_code"] == "synthetic_frontend_compile_failed"
    assert diagnostic["line"] == 42
    assert "Value argument cannot be block type" in diagnostic["path"]
    assert diagnostic["anonymous_handles"] == []


@pytest.mark.parametrize(
    "override",
    [
        {"agent": {"output_path": "/tmp/trajectory.json"}},
        {"agent": {"system_template": "replace policy"}},
        {"environment": {"executable": "/bin/bash"}},
        {"environment": {"env": {"SECRET": "leak"}}},
        {"environment": {"input_dir": "/host"}},
        {"model": {"api_key": "leak"}},
    ],
)
def test_worker_overrides_reject_templates_paths_tools_and_secrets(
    override,
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validated_intentir_replay_config(override)


def test_episode_contract_digest_binds_exact_files_prompts_and_budgets() -> None:
    files, _, _ = _replay_input_files(
        make_reduce_artifact(), frontend=Frontend.TRITON, masked=False
    )
    default_config = validated_intentir_replay_config()
    bounded_config = validated_intentir_replay_config(
        {"agent": {"step_limit": 15}}
    )
    digest = intentir_replay_episode_contract_sha256(files, default_config)
    changed_budget = intentir_replay_episode_contract_sha256(
        files, bounded_config
    )
    changed_files = dict(files)
    changed_files["TOOL.md"] += "\n"
    changed_tool = intentir_replay_episode_contract_sha256(
        changed_files, default_config
    )
    changed_prompt_config = copy.deepcopy(default_config)
    changed_prompt_config["agent"]["instance_template"] += "\nchanged"
    changed_prompt = intentir_replay_episode_contract_sha256(
        files, changed_prompt_config
    )

    assert len(digest) == 64
    assert len({digest, changed_budget, changed_tool, changed_prompt}) == 4
    assert default_config["agent"].get("output_path") is None
    assert default_config["agent"]["step_limit"] == 0
    assert default_config["environment"]["replay_compile_check_limit"] == 0
    assert default_config["environment"]["replay_source_inspect_limit"] == 0
    assert default_config["model"]["retry_attempts"] == 3
    assert default_config["model"]["max_tool_calls_per_response"] == 4
    assert default_config["model"]["enable_replay_patch_tool"] is True
    assert default_config["model"]["enable_submit_replay_tool"] is True
    assert default_config["model"]["compact_tool_history"] is True
    assert default_config["model"]["compact_workspace_mirrors"] is False
    assert default_config["model"]["compact_response_history_keep_recent"] == 3
    assert default_config["model"]["use_previous_response_id"] is True
    assert default_config["model"]["routine_reasoning_effort"] == "low"
    assert default_config["model"]["routine_timeout_seconds"] == 180
    assert default_config["model"]["retry_timeouts"] is True
    assert default_config["model"]["semantic_initial_turns"] == 1
    assert default_config["model"]["model_kwargs"]["parallel_tool_calls"] is True
    assert MINISWE_REPLAY_BACKEND == "minisweagent.intentir-replay.v16"
    instance = default_config["agent"]["instance_template"]
    assert "exact_packet_embedded=true" in instance
    assert "do not query" in instance
    assert "value-equivalent flattened formula" in instance
    assert "Construct and direct-submit complete Triton source early" in instance


def test_static_ingress_checks_exact_packet_abi_without_compiling() -> None:
    artifact = make_reduce_artifact()

    accepted = replay_source_ingress(
        VALID_REPLAY, artifact=artifact, frontend=Frontend.TRITON
    )
    rejected = replay_source_ingress(
        INVALID_CONSTEXPR_REPLAY,
        artifact=artifact,
        frontend=Frontend.TRITON,
    )
    wrong_language = replay_source_ingress(
        VALID_REPLAY, artifact=artifact, frontend=Frontend.CUDA
    )

    assert accepted.status == "pass"
    assert accepted.code == VALID_REPLAY
    diagnostic = rejected.to_check_response()["diagnostics"][0]
    assert diagnostic == {
        "rule_code": "constexpr_annotation_required",
        "path": "<parameter:c0>",
        "anonymous_handles": ["c0"],
        "line": 5,
    }
    assert wrong_language.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "replay_static_policy_rejected"
    assert set(rejected.to_check_response()) == {
        "protocol",
        "status",
        "diagnostics",
    }


def test_static_ingress_rejects_nonexistent_triton_program_axis() -> None:
    invalid = VALID_REPLAY.replace(
        "    v0 = tl.arange(0, c0)",
        "    pid = tl.program_id(3)\n    v0 = tl.arange(0, c0) + pid * c0",
    )

    rejected = replay_source_ingress(
        invalid,
        artifact=make_reduce_artifact(),
        frontend=Frontend.TRITON,
    )

    diagnostic = rejected.to_check_response()["diagnostics"][0]
    assert diagnostic["rule_code"] == "program_id_axis_range"
    assert diagnostic["path"] == "<program_id-axis>"
    assert diagnostic["line"] == 6


def test_static_ingress_reports_all_independent_constexpr_annotation_errors() -> None:
    artifact = make_reduce_artifact()
    two_errors = INVALID_CONSTEXPR_REPLAY.replace(
        "def scrc_k0(p0,",
        "def scrc_k0(p0: tl.constexpr,",
    )

    rejected = replay_source_ingress(
        two_errors,
        artifact=artifact,
        frontend=Frontend.TRITON,
    )

    assert [
        item["rule_code"] for item in rejected.to_check_response()["diagnostics"]
    ] == [
        "constexpr_annotation_forbidden",
        "constexpr_annotation_required",
    ]
    assert [
        item["path"] for item in rejected.to_check_response()["diagnostics"]
    ] == ["<parameter:p0>", "<parameter:c0>"]


def test_static_ingress_preserves_but_forbids_reading_true_absent_carrier() -> None:
    payload = make_reduce_artifact().to_json()
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "opaque",
            "mode": "value",
            "dtype": "none",
        }
    )
    artifact = Gate1Artifact.from_json(payload)
    prefix = VALID_REPLAY.replace(
        "def scrc_k0(p0, p1, p2, c0: tl.constexpr):",
        "def scrc_k0(p0, p1, p2, p3, c0: tl.constexpr):",
    )

    accepted = replay_source_ingress(
        prefix,
        artifact=artifact,
        frontend=Frontend.TRITON,
    )
    rejected = replay_source_ingress(
        prefix.replace(
            "    v0 = tl.arange(0, c0)",
            "    absent = p3\n    v0 = tl.arange(0, c0)",
        ),
        artifact=artifact,
        frontend=Frontend.TRITON,
    )

    assert accepted.status == "pass"
    assert rejected.to_check_response()["diagnostics"] == [
        {
            "rule_code": "absent_carrier_use_forbidden",
            "path": "<parameter:p3>",
            "anonymous_handles": ["p3"],
            "line": 6,
        }
    ]


def test_synthetic_compile_rejects_compiled_transfer_occurrence_collapse() -> None:
    unmasked_ttir = """
module {
  tt.func public @scrc_k0(%p0: !tt.ptr<f32>, %p1: !tt.ptr<f32>, %p2: i32) attributes {noinline = false} {
    %value = tt.load %p0 : !tt.ptr<f32>
    tt.store %p1, %value : !tt.ptr<f32>
    tt.return
  }
}
"""

    class PassingCompiler:
        def compile(self, _source, _case):
            return CompileResult(True, ttir=unmasked_ttir)

    payload = make_reduce_artifact().to_json()
    for instance in payload["pi"]["instances"]:
        if instance["semantic_ref"] in {"a0", "a1"}:
            instance["mechanism"] = "transfer_direct"
    artifact = Gate1Artifact.from_json(payload)
    result = source_blind_replay_compile_check(
        VALID_REPLAY,
        artifact,
        "scrc_k0",
        compiler=PassingCompiler(),  # type: ignore[arg-type]
    )

    assert result.status == "fail"
    diagnostic = result.to_check_response()["diagnostics"][0]
    assert diagnostic["rule_code"] == "compiled_transfer_direct_count"
    assert "expected=0, actual=1" in diagnostic["path"]


def _physical_reduce_replay_artifact(
    *, with_carrier: bool = False
) -> Gate1Artifact:
    artifact = make_reduce_artifact()
    artifact.c.slots[0].subject_ref = "n1"
    artifact.c.slots[0].axis_ref = 1
    artifact.c.slots[0].required = True
    artifact.pi.regions.extend(
        [
            {"id": "r1", "parent": "r0", "placement": "program"},
            {"id": "r2", "parent": "r1", "placement": "loop"},
        ]
    )
    artifact.pi.instances.append(
        {
            "id": "i4",
            "semantic_ref": "c0",
            "region": "r1",
            "mechanism": "realize",
        }
    )
    artifact.pi.physical_iterations = [
        {
            "region": "r2",
            "body_instance": "i2",
            "tile_instance": "i4",
            "induction_basis": "extent_step",
            "direction": "forward",
        }
    ]
    if with_carrier:
        artifact.pi.instances.extend(
            [
                {
                    "id": "i5",
                    "semantic_ref": "t1",
                    "region": "r1",
                    "mechanism": "realize",
                },
                {
                    "id": "i6",
                    "semantic_ref": "t1",
                    "region": "r2",
                    "mechanism": "realize",
                },
                {
                    "id": "i7",
                    "semantic_ref": "t1",
                    "region": "r2",
                    "mechanism": "realize",
                },
                {
                    "id": "i8",
                    "semantic_ref": "t1",
                    "region": "r1",
                    "mechanism": "realize",
                },
            ]
        )
        artifact.pi.routes.extend(
            [
                {"source": "i5", "target": "i6", "slots": [0]},
                {"source": "i6", "target": "i7", "slots": [0]},
            ]
        )
        artifact.pi.physical_loop_carriers = [
            {
                "region": "r2",
                "init": "i5",
                "state": "i6",
                "next": "i7",
                "result": "i8",
            }
        ]
    return Gate1Artifact.from_json(artifact.to_json())


def test_synthetic_compile_rejects_missing_authored_physical_iteration() -> None:
    artifact = _physical_reduce_replay_artifact()
    loopless_ttir = """
module {
  tt.func public @scrc_k0(%p0: !tt.ptr<f32>, %p1: !tt.ptr<f32>, %p2: i32) attributes {noinline = false} {
    tt.return
  }
}
"""

    class PassingCompiler:
        def compile(self, _source, _case):
            return CompileResult(True, ttir=loopless_ttir)

    result = source_blind_replay_compile_check(
        VALID_REPLAY,
        artifact,
        "scrc_k0",
        compiler=PassingCompiler(),  # type: ignore[arg-type]
    )

    assert result.status == "fail"
    diagnostic = result.to_check_response()["diagnostics"][0]
    assert diagnostic["rule_code"] == "compiled_iteration_count"
    assert "expected=1, actual=0" in diagnostic["path"]


def test_synthetic_compile_rejects_omitted_authored_physical_carrier() -> None:
    artifact = _physical_reduce_replay_artifact(with_carrier=True)
    carrierless_loop_ttir = """
module {
  tt.func public @scrc_k0(%p0: !tt.ptr<f32>, %p1: !tt.ptr<f32>, %p2: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %p2 step %c1 : i32 {
      scf.yield
    }
    tt.return
  }
}
"""

    class PassingCompiler:
        def compile(self, _source, _case):
            return CompileResult(True, ttir=carrierless_loop_ttir)

    result = source_blind_replay_compile_check(
        VALID_REPLAY,
        artifact,
        "scrc_k0",
        compiler=PassingCompiler(),  # type: ignore[arg-type]
    )

    assert result.status == "fail"
    diagnostic = result.to_check_response()["diagnostics"][0]
    assert diagnostic["rule_code"] == "compiled_iteration_carrier_arity"
    assert "expected=[2], actual=[1]" in diagnostic["path"]


def test_replay_sandbox_has_empty_root_fixed_entrypoints_and_no_network(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTENTIR_REPLAY_SECRET", "must-not-enter")
    workspace = _workspace()
    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "set -e; "
                    "test ! -e /proc; test ! -e /usr; test ! -e /etc; "
                    "test ! -e /lib; "
                    "test ! -e /lib64; test ! -e /usr/local; "
                    "test ! -w /input/packet.json; "
                    "test ! -e /home/kingdom; "
                    "test -z \"${INTENTIR_REPLAY_SECRET+x}\"; "
                    "for name in python python3 gcc g++ clang ld as nvcc "
                    "ptxas triton torch intentir; do ! command -v \"$name\"; done; "
                    "find /input -type f | sed 's#^/input/#I:#' | sort; "
                    "find /bin -type f | sed 's#^/bin/#T:#' | sort; "
                    "printf 'H:%s\\n' \"$(hostname)\""
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 0, result
    lines = result["output"].splitlines()
    assert {line.removeprefix("I:") for line in lines if line.startswith("I:")} == (
        REPLAY_INPUT_FILES
    )
    assert {line.removeprefix("T:") for line in lines if line.startswith("T:")} == set(
        REPLAY_TOOL_APPLETS
    )
    assert f"H:{REPLAY_SANDBOX_HOSTNAME}" in lines
    serialized = environment.serialize()["info"]["config"]["environment"]
    assert serialized["network"] == "new-unshared-namespace-host-probed"
    assert serialized["proc"] == "not-mounted"


def test_static_error_can_be_repaired_in_one_fresh_multi_turn_episode() -> None:
    artifact = make_reduce_artifact()
    workspace = _workspace()
    model = _ScriptedModel(
        [
            _write_source(INVALID_CONSTEXPR_REPLAY),
            f"{INTENTIR_REPLAY_CHECK} /output/replay.source",
            _write_source(VALID_REPLAY),
            f"{INTENTIR_REPLAY_CHECK} /output/replay.source",
            f"{INTENTIR_REPLAY_SUBMIT_COMMAND} /output/replay.source",
        ]
    )
    try:
        result = run_intentir_replay(
            model=model,
            workspace=workspace,
            replay_checker=lambda source: replay_source_ingress(
                source, artifact=artifact, frontend=Frontend.TRITON
                ).to_check_response(),
            )
        assert not tuple(workspace.scratch_dir.rglob("*"))
    finally:
        workspace.cleanup()

    assert result.source_text == VALID_REPLAY
    first_packet = model.query_inputs[0][1]["content"]
    assert "intentir.replay-startup-context.v2" in first_packet
    assert '"exact_packet_embedded":true' in first_packet
    assert '"packet":' in first_packet
    assert '"views":' not in first_packet
    assert "anonymous_source" not in first_packet
    evidence = result.trajectory["info"]["replay_check"]
    assert [item["status"] for item in evidence["results"]] == [
        "fail",
        "pass",
    ]
    assert evidence["evaluations"] == 2


def test_replay_exact_no_progress_loop_terminates_as_orchestration_stall() -> None:
    workspace = _workspace()
    model = _ScriptedModel(["true", "true", "true"])
    try:
        with pytest.raises(Exception) as captured:
            run_intentir_replay(model=model, workspace=workspace)
    finally:
        workspace.cleanup()

    result = captured.value.result
    assert result.exit_status == "OrchestrationStall"
    assert result.model_calls == 3
    stagnation = result.trajectory["info"]["agent_runtime"]["stagnation"]
    assert stagnation["terminated"] is True
    assert stagnation["consecutive"] == 3


def test_replay_multi_action_write_batch_is_rejected_before_execution() -> None:
    workspace = _workspace()
    model = _ScriptedModel(
        [
            [
                "printf forbidden > /output/extra",
                "printf forbidden > /output/replay.source",
            ],
            _write_source(VALID_REPLAY),
            f"{INTENTIR_REPLAY_SUBMIT_COMMAND} /output/replay.source",
        ]
    )
    try:
        result = run_intentir_replay(model=model, workspace=workspace)
    finally:
        workspace.cleanup()

    assert result.source_text == VALID_REPLAY
    policy = result.trajectory["info"]["multi_action_policy"]
    assert policy["rejections"] == 1
    assert "no action in this response was executed" in json.dumps(
        result.trajectory["messages"]
    )


def test_replay_multi_action_runtime_rejects_more_than_four_packet_queries() -> None:
    workspace = _workspace()
    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        diagnostic = environment.validate_actions(
            [
                {"command": f"{INTENTIR_PACKET_QUERY} overview"}
                for _index in range(5)
            ]
        )
        policy = environment.serialize()["info"]["multi_action_policy"]
    finally:
        workspace.cleanup()

    assert diagnostic is not None and "at most four" in diagnostic
    assert policy["rejections"] == 1


def test_construction_compile_rejects_submission_until_exact_bytes_pass() -> None:
    workspace = _workspace()
    compile_allowed = False

    def passed_ingress(_source: str) -> dict:
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    def compile_probe(_source: str) -> dict:
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": "pass" if compile_allowed else "fail",
            "diagnostics": [
                {
                    "rule_code": "synthetic_frontend_compile_failed",
                    "path": "<synthetic-frontend-compile> own-code shape error",
                    "anonymous_handles": [],
                    "line": 7,
                }
            ] if not compile_allowed else [],
        }

    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        replay_checker=passed_ingress,
        replay_compile_checker=compile_probe,
    )
    try:
        rejected = environment.execute(
            {"tool_name": "submit_replay", "source_text": VALID_REPLAY}
        )
        assert rejected["returncode"] == 2
        assert "construction compile rejected" in rejected["output"]
        compile_allowed = True
        with pytest.raises(Submitted):
            environment.execute(
                {"tool_name": "submit_replay", "source_text": VALID_REPLAY}
            )
        info = environment.serialize()["info"]
    finally:
        workspace.cleanup()

    compile_evidence = info["replay_compile_check"]
    assert "no Gate1" in compile_evidence["authority"]
    assert compile_evidence["evaluations"] == 2
    assert [item["status"] for item in compile_evidence["results"]] == [
        "fail",
        "pass",
    ]


def test_check_budget_submission_reserve_and_exact_bytes() -> None:
    workspace = _workspace()
    checked: list[str] = []

    def checker(source: str) -> dict:
        checked.append(source)
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    model = _ScriptedModel(
        [
            _write_source(VALID_REPLAY),
            f"{INTENTIR_REPLAY_CHECK} /output/replay.source",
            "printf '\\n' >> /output/replay.source",
            f"printf '{INTENTIR_REPLAY_SUBMIT}\\n'",
        ]
    )
    try:
        result = run_intentir_replay(
            model=model,
            workspace=workspace,
            replay_checker=checker,
            config_overrides={
                "environment": {
                    "replay_check_limit": 1,
                    "replay_submission_check_limit": 1,
                }
            },
        )
    finally:
        workspace.cleanup()

    assert checked == [VALID_REPLAY, VALID_REPLAY + "\n"]
    evidence = result.trajectory["info"]["replay_check"]
    assert evidence["shared_budget"] == {"used": 1, "limit": 1}
    assert evidence["submission_reserve"] == {"used": 1, "limit": 1}
    assert [item["trigger"] for item in evidence["results"]] == [
        "explicit",
        "submission",
    ]


def test_replay_direct_submit_and_exact_patch_share_submission_preflight() -> None:
    workspace = _workspace()

    def checker(source: str) -> dict:
        accepted = source == VALID_REPLAY
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": "pass" if accepted else "fail",
            "diagnostics": (
                []
                if accepted
                else [
                    {
                        "rule_code": "constexpr_annotation_required",
                        "path": "<parameter:c0>",
                        "anonymous_handles": ["c0"],
                        "line": 5,
                    }
                ]
            ),
        }

    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        replay_checker=checker,
    )
    try:
        rejected = environment.execute(
            {
                "tool_name": "submit_replay",
                "source_text": INVALID_CONSTEXPR_REPLAY,
            }
        )
        assert rejected["returncode"] == 2
        assert workspace.replay_path.read_text() == INVALID_CONSTEXPR_REPLAY
        with pytest.raises(Submitted):
            environment.execute(
                {
                    "tool_name": "apply_replay_patch",
                    "path": "/output/replay.source",
                    "old_text": "c0):",
                    "new_text": "c0: tl.constexpr):",
                    "submit": True,
                }
            )
        evidence = environment.serialize()["info"]["replay_transaction"]
        final_source = workspace.replay_path.read_text()
    finally:
        workspace.cleanup()

    assert final_source == VALID_REPLAY
    assert evidence["direct_submit_count"] == 1
    assert evidence["apply_patch_count"] == 1
    assert evidence["direct_submits"][0]["source_bytes"] == len(
        INVALID_CONSTEXPR_REPLAY.encode()
    )
    assert evidence["apply_patches"][0]["submit_requested"] is True


def test_replay_transaction_preserves_only_explicit_exact_bytes() -> None:
    workspace = _workspace()
    committed = VALID_REPLAY.encode("utf-8")
    changed = (VALID_REPLAY + "\n").encode("utf-8")
    workspace.replay_path.write_bytes(committed)

    def checker(_source: str) -> dict:
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        replay_checker=checker,
    )
    canary = workspace.scratch_dir / "rollback-canary"
    canary.write_bytes(b"must-stay-unchanged")
    (workspace.scratch_dir / ".replay-rollback.tmp").symlink_to(canary)
    try:
        checked = environment.execute(
            {"command": f"{INTENTIR_REPLAY_CHECK} /output/replay.source"}
        )
        committed_result = environment.execute(
            {"command": f"{INTENTIR_REPLAY_COMMIT} /output/replay.source"}
        )
        workspace.replay_path.write_bytes(changed)
        mismatched_commit = environment.execute(
            {"command": f"{INTENTIR_REPLAY_COMMIT} /output/replay.source"}
        )
        rolled_back = environment.execute(
            {"command": f"{INTENTIR_REPLAY_ROLLBACK} /output/replay.source"}
        )
        evidence = environment.serialize()["info"]["replay_transaction"]
        restored = workspace.replay_path.read_bytes()
        canary_bytes = canary.read_bytes()
    finally:
        workspace.cleanup()

    assert checked["returncode"] == committed_result["returncode"] == 0
    assert mismatched_commit["returncode"] == 1
    assert "exact current bytes" in json.loads(mismatched_commit["output"])[
        "diagnostic"
    ]
    assert rolled_back["returncode"] == 0
    assert restored == committed
    assert canary_bytes == b"must-stay-unchanged"
    assert evidence == {
        "protocol": REPLAY_TRANSACTION_PROTOCOL,
        "authority": (
            "exact Replay-LLM-authored bytes and exact-text replacement only; "
            "no source is generated or inferred"
        ),
        "commit_count": 1,
        "commits": [
            {
                "sha256": hashlib.sha256(committed).hexdigest(),
                "bytes": len(committed),
            }
        ],
        "rollback_count": 1,
        "fallback_count": 0,
        "fallbacks": [],
        "apply_patch_count": 0,
        "apply_patches": [],
        "direct_submit_count": 0,
        "direct_submits": [],
    }


def test_step_limit_returns_explicitly_committed_replay_checkpoint() -> None:
    workspace = _workspace()
    model = _ScriptedModel(
        [
            _write_source(VALID_REPLAY),
            f"{INTENTIR_REPLAY_CHECK} /output/replay.source",
            f"{INTENTIR_REPLAY_COMMIT} /output/replay.source",
        ]
    )

    def checker(_source: str) -> dict:
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    try:
        result = run_intentir_replay(
            model=model,
            workspace=workspace,
            replay_checker=checker,
            config_overrides={"agent": {"step_limit": 3}},
        )
    finally:
        workspace.cleanup()

    assert result.source_text == VALID_REPLAY
    assert result.exit_status == "SubmittedCheckpoint"
    assert result.model_calls == 3
    transaction = result.trajectory["info"]["replay_transaction"]
    assert transaction["commit_count"] == 1
    assert transaction["fallback_count"] == 1
    assert transaction["fallbacks"][0]["terminal_status"] == "LimitsExceeded"


def test_submission_rejects_any_additional_output_file() -> None:
    workspace = _workspace()
    environment = IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    _write_source(VALID_REPLAY)
                    + "; printf x > /output/extra; "
                    + f"printf '{INTENTIR_REPLAY_SUBMIT}\\n'"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 2
    assert "Submission rejected by output policy" in result["output"]


def test_writer_uses_fresh_sessions_same_ingress_and_sanitized_receipt(
    monkeypatch,
) -> None:
    from minisweagent.environments.intentir_replay import IntentIRReplayWorkspace as Workspace
    from minisweagent.run import intentir_replay as run_module

    secret = "provider-secret-canary"
    provider = ProviderConfig(
        name="test",
        base_url="https://provider.invalid/v1",
        api_key=secret,
        model="test-model",
        reasoning_effort="high",
    )
    models: list[_ScriptedModel] = []

    def model_factory(**kwargs):
        assert kwargs["api_key"] == secret
        assert len(kwargs["prompt_cache_key"]) <= 64
        model = _ScriptedModel(
            [
                (
                    "test ! -e /work/session-marker; "
                    "touch /work/session-marker; "
                    + _write_source(VALID_REPLAY)
                ),
                f"{INTENTIR_REPLAY_CHECK} /output/replay.source",
                f"{INTENTIR_REPLAY_COMPILE_CHECK} /output/replay.source",
                f"printf '{INTENTIR_REPLAY_SUBMIT}\\n'",
            ],
            secret=secret,
        )
        models.append(model)
        return model

    monkeypatch.setattr(run_module, "create_intentir_replay_model", model_factory)
    captured_files: list[dict[str, str | bytes]] = []
    workspace_roots: list[str] = []
    original_create = Workspace.create.__func__

    def capture_create(cls, files):
        captured_files.append(dict(files))
        workspace = original_create(cls, files)
        workspace_roots.append(str(workspace.root))
        return workspace

    monkeypatch.setattr(Workspace, "create", classmethod(capture_create))
    ingress_calls: list[str] = []
    original_ingress = replay_module.replay_source_ingress

    def counted_ingress(source: str, **kwargs):
        ingress_calls.append(source)
        return original_ingress(source, **kwargs)

    monkeypatch.setattr(replay_module, "replay_source_ingress", counted_ingress)
    compile_checks = []

    def passing_compile_check(code, artifact, function_name):
        compile_checks.append((code, artifact, function_name))
        return ReplayIngressResult(
            "pass",
            code=code,
            entry_function_name=function_name,
        )

    writer = MiniSWEReplayWriter(
        provider=provider,
        compile_checker=passing_compile_check,
    )

    normal = writer.generate(make_reduce_artifact(), masked=False)
    masked = writer.generate(make_reduce_artifact(), masked=True)
    next_formal_round = writer.generate(make_reduce_artifact(), masked=False)

    assert len(models) == 3
    assert len(set(workspace_roots)) == 3
    assert len(writer.records) == 3
    assert len(ingress_calls) == 9  # ingress + compile preflight + formal ingress
    assert len(compile_checks) == 3
    assert normal.context_receipt["backend_protocol"] == MINISWE_REPLAY_BACKEND
    assert normal.context_receipt["fresh_session"] is True
    assert normal.context_receipt["formal_ingress_recheck"] is True
    assert normal.context_receipt["source_blind_synthetic_compile_probe"] is True
    assert normal.context_receipt["model_calls"] == 4
    assert normal.context_receipt["physical_provider_calls"] == 4
    assert normal.context_receipt["transport_retries"] == 0
    assert normal.attempts == 4
    assert normal.physical_provider_calls == 4
    assert normal.transport_retries == 0
    assert normal.context_receipt["replay_check"]["evaluations"] == 1
    assert normal.context_receipt["replay_compile_check"]["evaluations"] == 1
    assert normal.context_receipt["packet_digest"] == normal.packet_digest
    assert normal.context_receipt["episode_contract_sha256"] == (
        normal.episode_contract_sha256
    )
    assert len(normal.episode_contract_sha256) == 64
    assert normal.backend_protocol == MINISWE_REPLAY_BACKEND
    assert secret not in json.dumps(normal.trajectory)
    assert normal.trajectory["info"]["config"]["api_key"] == "<redacted>"
    canonical_trajectory = json.dumps(
        normal.trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert normal.raw_response == canonical_trajectory
    assert normal.raw_response_sha256 == hashlib.sha256(
        canonical_trajectory
    ).hexdigest()
    assert set(captured_files[0]) == set(captured_files[1]) == set(
        captured_files[2]
    ) == {
        "README.md",
        "packet.json",
        "REPLAY_CONTRACT.md",
        "TOOL.md",
    }
    normal_packet = json.loads(captured_files[0]["packet.json"])
    masked_packet = json.loads(captured_files[1]["packet.json"])
    next_packet = json.loads(captured_files[2]["packet.json"])
    assert normal_packet["j"] != masked_packet["j"]
    assert next_packet == normal_packet
    for key in ("c", "pi", "u_f"):
        assert normal_packet[key] == masked_packet[key]
    for files in captured_files:
        assert secret not in json.dumps(files, default=str)
        assert "trajectory" not in files["packet.json"]
    assert next_formal_round.context_receipt["fresh_session"] is True
    assert normal.episode_contract_sha256 != masked.episode_contract_sha256
    assert (
        next_formal_round.episode_contract_sha256
        == normal.episode_contract_sha256
    )

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {
                key
                for item in value.values()
                for key in keys(item)
            }
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    forbidden_keys = {"source", "source_code", "anonymous_source", "harness"}
    assert forbidden_keys.isdisjoint(keys(normal.context_receipt))
    assert forbidden_keys.isdisjoint(keys(normal.trajectory))
