from __future__ import annotations

import copy
import hashlib
import json

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1 import replay as replay_module
from intentir.verify.gate1.model import Frontend
from intentir.verify.gate1.prompt import assert_source_isolated_payload
from intentir.verify.gate1.replay import (
    MINISWE_REPLAY_BACKEND,
    MiniSWEReplayWriter,
    _replay_input_files,
    replay_source_ingress,
)
from minisweagent.environments.intentir_replay import (
    INTENTIR_REPLAY_CHECK,
    INTENTIR_REPLAY_SUBMIT,
    REPLAY_CHECK_PROTOCOL,
    REPLAY_INPUT_FILES,
    REPLAY_SANDBOX_HOSTNAME,
    REPLAY_TOOL_APPLETS,
    IntentIRReplayEnvironment,
    IntentIRReplayWorkspace,
)
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
    def __init__(self, commands: list[str], *, secret: str = "") -> None:
        self.commands = list(commands)
        self.secret = secret
        self.query_inputs: list[list[dict]] = []

    def query(self, messages, **kwargs):
        del kwargs
        self.query_inputs.append(list(messages))
        return {
            "role": "assistant",
            "content": "",
            "extra": {
                "actions": [{"command": self.commands.pop(0)}],
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
    assert default_config["model"]["retry_attempts"] == 3


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
    unsupported = replay_source_ingress(
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
    assert unsupported.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "frontend_not_supported"
    assert set(rejected.to_check_response()) == {
        "protocol",
        "status",
        "diagnostics",
    }


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
            f"printf '{INTENTIR_REPLAY_SUBMIT}\\n'",
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
    evidence = result.trajectory["info"]["replay_check"]
    assert [item["status"] for item in evidence["results"]] == [
        "fail",
        "pass",
    ]
    assert evidence["evaluations"] == 2


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
        model = _ScriptedModel(
            [
                (
                    "test ! -e /work/session-marker; "
                    "touch /work/session-marker; "
                    + _write_source(VALID_REPLAY)
                ),
                f"{INTENTIR_REPLAY_CHECK} /output/replay.source",
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
    writer = MiniSWEReplayWriter(provider=provider)

    normal = writer.generate(make_reduce_artifact(), masked=False)
    masked = writer.generate(make_reduce_artifact(), masked=True)
    next_formal_round = writer.generate(make_reduce_artifact(), masked=False)

    assert len(models) == 3
    assert len(set(workspace_roots)) == 3
    assert len(writer.records) == 3
    assert len(ingress_calls) == 6  # explicit check + formal recheck per episode
    assert normal.context_receipt["backend_protocol"] == MINISWE_REPLAY_BACKEND
    assert normal.context_receipt["fresh_session"] is True
    assert normal.context_receipt["formal_ingress_recheck"] is True
    assert normal.context_receipt["model_calls"] == 3
    assert normal.context_receipt["physical_provider_calls"] == 3
    assert normal.context_receipt["transport_retries"] == 0
    assert normal.attempts == 3
    assert normal.physical_provider_calls == 3
    assert normal.transport_retries == 0
    assert normal.context_receipt["replay_check"]["evaluations"] == 1
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
