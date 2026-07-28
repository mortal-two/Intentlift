from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from direct_fixtures import make_copy_artifact
from intentir.extraction.lifters.codex_cli import (
    CODEX_CLI_LIFT_BACKEND,
    CodexCLILiftAdapter,
)
from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLIError,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.codex_replay import (
    CODEX_CLI_REPLAY_BACKEND,
    CodexCLIReplayWriter,
    _replay_manifest_view,
    _replay_packet_query,
    _replay_realization_view,
    _replay_workspace_view,
)
from intentir.verify.gate1.replay import ReplayIngressResult
from intentir.verify.scrc_runner import (
    FeedbackKind,
    HarnessBindingInstance,
    LiftFeedback,
    SCRCRunRequest,
    SourceProgram,
)


def _workspace_dirs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    values = tuple(tmp_path / name for name in ("input", "output", "work", "tmp"))
    for path in values:
        path.mkdir()
    return values  # type: ignore[return-value]


class _FinishedProcess:
    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        raise AssertionError("completed fake process must not be killed")


class _RunningProcess:
    def __init__(self) -> None:
        self.killed = False

    def poll(self) -> int | None:
        return -9 if self.killed else None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.killed
        return -9

    def kill(self) -> None:
        self.killed = True


def _events(thread_id: str) -> bytes:
    values = (
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "m0", "type": "agent_message", "text": "working"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "t0",
                "type": "command_execution",
                "command": "true",
                "aggregated_output": "",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 8,
                "output_tokens": 2,
            },
        },
    )
    return b"".join(json.dumps(value).encode() + b"\n" for value in values)


def test_codex_cli_command_is_outer_isolated_and_secret_free(
    tmp_path: Path,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    try:
        command = session._command("do work", effort="high", resumed=False)
        encoded = json.dumps(command)
        assert "provider-secret" not in encoded
        assert "--unshare-net" not in command
        assert command[command.index("--ro-bind") + 1] == "/usr"
        input_index = command.index(str(input_dir))
        assert command[input_index - 1] == "--ro-bind"
        assert command[input_index + 1] == "/workspace/input"
        assert command[command.index(str(output_dir)) - 1] == "--bind"
        assert command[command.index(str(work_dir)) - 1] == "--bind"
        assert "--remount-ro" in command
        remount_index = command.index("--remount-ro")
        assert remount_index < input_index
        for mountpoint in ("/workspace/.git", "/workspace/.agents", "/workspace/.codex"):
            assert mountpoint in command
            assert command.index(mountpoint) < remount_index
        assert 'shell_environment_policy.exclude=["^OPENAI_API_KEY$","^CODEX_HOME$"]' in command
        assert "sandbox_workspace_write.network_access=false" in command
        assert command[command.index("-s") + 1] == "workspace-write"
        assert "/home/kingdom" not in encoded
    finally:
        home = session.session_home
        session.close()
    assert not home.exists()


def test_codex_outer_sandbox_has_exact_writable_subdirectories(
    tmp_path: Path,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    script = """from pathlib import Path
for name in ("output", "work", "tmp"):
    Path(name, "probe").write_text("ok")
try:
    Path("forbidden").write_text("bad")
except OSError:
    pass
else:
    raise SystemExit("workspace root unexpectedly writable")
try:
    Path("input", "forbidden").write_text("bad")
except OSError:
    pass
else:
    raise SystemExit("input unexpectedly writable")
"""
    try:
        completed = subprocess.run(
            [*session._sandbox_prefix(), "/usr/bin/python3", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert (output_dir / "probe").read_text() == "ok"
        assert (work_dir / "probe").read_text() == "ok"
        assert (tmp_dir / "probe").read_text() == "ok"
        assert not (tmp_path / "forbidden").exists()
        assert not (input_dir / "forbidden").exists()
    finally:
        session.close()


def test_codex_tool_client_help_is_local_and_does_not_create_a_request(
    tmp_path: Path,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    try:
        completed = subprocess.run(
            [
                *session._sandbox_prefix(),
                "/usr/bin/python3",
                "/workspace/.agents/intentir_agent_tool.py",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        session.close()

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "usage: intentir_agent_tool.py TOOL [ARG ...]"
    assert not (work_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME).exists()


def test_codex_tool_bridge_returns_closed_host_diagnostic_without_writing(
    tmp_path: Path,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    observed: list[str] = []

    def candidate_check() -> dict:
        observed.append((output_dir / "candidate.json").read_text())
        return {
            "protocol": "intentir.scrc.candidate-check.v4",
            "status": "pass",
            "diagnostics": [],
        }

    (output_dir / "candidate.json").write_text("model-authored")
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="xhigh",
    )
    bridge = CodexCLIToolBridge(
        work_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
        {"candidate_check": candidate_check},
    )
    try:
        bridge.start()
        completed = subprocess.run(
            [
                *session._sandbox_prefix(),
                "/usr/bin/python3",
                "/workspace/.agents/intentir_agent_tool.py",
                "candidate_check",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        evidence = bridge.evidence
    finally:
        bridge.close()
        session.close()

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "pass"
    assert observed == ["model-authored"]
    assert (output_dir / "candidate.json").read_text() == "model-authored"
    assert evidence["calls"] == 1
    assert evidence["transport"] == "isolated_workspace_file_mailbox"
    assert evidence["receipts"][0]["tool"] == "candidate_check"
    assert evidence["receipts"][0]["argument_count"] == 0
    assert not (work_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME).exists()


def test_codex_tool_bridge_passes_bounded_arguments_only_to_argument_handler(
    tmp_path: Path,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    observed: list[tuple[str, ...]] = []

    def packet_query(arguments: tuple[str, ...]) -> dict:
        observed.append(arguments)
        return {
            "protocol": "intentir.test-query.v1",
            "status": "pass",
            "arguments": list(arguments),
        }

    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    bridge = CodexCLIToolBridge(
        work_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
        {},
        argument_handlers={"packet_query": packet_query},
    )
    try:
        bridge.start()
        completed = subprocess.run(
            [
                *session._sandbox_prefix(),
                "/usr/bin/python3",
                "/workspace/.agents/intentir_agent_tool.py",
                "packet_query",
                "graph",
                "0",
                "12",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        evidence = bridge.evidence
    finally:
        bridge.close()
        session.close()

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["arguments"] == ["graph", "0", "12"]
    assert observed == [("graph", "0", "12")]
    assert evidence["receipts"][0]["argument_count"] == 3
    assert len(evidence["receipts"][0]["arguments_sha256"]) == 64


def test_codex_replay_realization_view_is_lossless_and_source_isolated() -> None:
    artifact = make_copy_artifact()
    packet = {
        "c": artifact.c.to_json(),
        "j": artifact.j.to_json(),
        "pi": artifact.pi.to_json(),
    }

    view = _replay_realization_view(packet)

    assert view["c"] == packet["c"]
    assert view["j_boundary"] == {
        "accesses": packet["j"]["boundary"]["accesses"],
        "effects": packet["j"]["boundary"]["effects"],
    }
    assert view["pi"] == packet["pi"]
    assert "physical_iterations" in view["pi"]
    assert "physical_loop_carriers" in view["pi"]
    assert set(view) == {"protocol", "status", "c", "j_boundary", "pi", "authority"}
    assert not ({"source", "harness", "gate1", "gate2"} & set(view))


def test_codex_replay_workspace_view_joins_authored_facts_without_source() -> None:
    artifact = make_copy_artifact()
    packet = {
        "c": artifact.c.to_json(),
        "j": artifact.j.to_json(),
        "pi": artifact.pi.to_json(),
        "u_f": {"frontend": "triton", "replay_function": "scrc_k0"},
    }

    view = _replay_workspace_view(packet)

    assert view["abi"]["signature"] == packet["j"]["signature"]
    assert view["abi"]["mask_c"] == packet["c"]
    assert view["j"]["boundary"] == packet["j"]["boundary"]
    assert len(view["j"]["graph"]) == 1
    assert view["protocol"] == "intentir.scrc.replay-workspace-view.v2"
    assert view["j"]["reduction_obligations"] == []
    assert view["pi"]["terminal_controls"] == packet["pi"]["terminal_controls"]
    obligations = view["pi"]["obligations"]
    assert obligations["total"] == len(packet["pi"]["instances"])
    assert len(obligations["items"]) == obligations["total"]
    assert view["status"] == "pass"
    assert not ({"source", "harness", "gate1", "gate2"} & set(view))


def test_codex_replay_bounded_queries_partition_exact_packet_without_source() -> None:
    artifact = make_copy_artifact()
    packet = {
        "c": artifact.c.to_json(),
        "j": artifact.j.to_json(),
        "pi": artifact.pi.to_json(),
        "u_f": {"frontend": "triton", "replay_function": "scrc_k0"},
    }

    manifest = _replay_manifest_view(packet)
    abi = _replay_packet_query(packet, ("abi", "0", "48"))
    graph = _replay_packet_query(packet, ("graph", "0", "12"))
    boundary = _replay_packet_query(packet, ("section", "j.boundary"))
    realization = _replay_packet_query(packet, ("realization",))
    obligations = _replay_packet_query(packet, ("obligations", "0", "12"))

    assert manifest["query_plan"] == [
        "python .agents/intentir_agent_tool.py replay_packet_query abi 0 48",
        "python .agents/intentir_agent_tool.py replay_packet_query graph 0 12",
        "python .agents/intentir_agent_tool.py replay_packet_query realization",
        "python .agents/intentir_agent_tool.py replay_packet_query obligations 0 12",
    ]
    assert manifest["operation_count"] == 1
    assert len(json.dumps(manifest)) < len(json.dumps(_replay_workspace_view(packet)))
    assert abi["view"]["intent_io"]["name"] == "scrc_j0"
    expected_op = artifact.j.to_json()["intent"]["intent_function"]["body"][0][
        "op"
    ]
    assert graph["view"]["items"][0]["op"] == expected_op
    assert boundary["view"] == packet["j"]["boundary"]
    assert realization["view"]["topology"]["terminal_controls"] == packet["pi"][
        "terminal_controls"
    ]
    assert obligations["view"]["total"] == len(packet["pi"]["instances"])
    for view in (manifest, abi, graph, boundary, realization, obligations):
        assert not ({"source", "harness", "gate1", "gate2"} & set(view))


def test_codex_cli_turn_parses_jsonl_and_resumes_same_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    commands: list[list[str]] = []

    def fake_popen(command, *, stdin, stdout, stderr, env):
        del stdin, stderr
        commands.append(list(command))
        assert env["OPENAI_API_KEY"] == "provider-secret"
        assert "provider-secret" not in json.dumps(command)
        stdout.write(_events("thread-1"))
        stdout.flush()
        return _FinishedProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    try:
        first = session.run("first", timeout_seconds=5)
        second = session.run("repair", timeout_seconds=5)
    finally:
        session.close()

    assert first.thread_id == second.thread_id == "thread-1"
    assert first.resumed is False
    assert second.resumed is True
    assert first.usage["cached_input_tokens"] == 8
    assert first.model_segments == first.tool_actions == 1
    assert "resume" not in commands[0]
    assert "resume" in commands[1]
    assert "thread-1" in commands[1]
    assert all(command[command.index("-s") + 1] == "workspace-write" for command in commands)


def test_codex_cli_cancels_inflight_turn_for_first_gate1_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    process = _RunningProcess()

    def fake_popen(_command, *, stdin, stdout, stderr, env):
        del stdin, stdout, stderr, env
        return process

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    try:
        with pytest.raises(CodexCLICancelled) as captured:
            session.run(
                "work",
                timeout_seconds=30,
                stop_requested=lambda: 2,
            )
    finally:
        session.close()

    assert process.killed
    assert captured.value.winner_branch_id == 2
    assert captured.value.physical_calls == 1
    assert len(captured.value.stdout_sha256) == 64


def test_codex_cli_recovers_thread_identity_from_timed_out_initial_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)
    process = _RunningProcess()

    def fake_popen(_command, *, stdin, stdout, stderr, env):
        del stdin, stderr, env
        stdout.write(
            b'{"type":"thread.started","thread_id":"thread-timeout"}\n'
            b'{"type":"turn.started"}\n'
            b'{"type":"item.started"'
        )
        stdout.flush()
        return process

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    try:
        with pytest.raises(CodexCLIError, match="wall-time limit"):
            session.run("first", timeout_seconds=1e-9)
        resumed = session._command("continue", effort="high", resumed=True)
    finally:
        session.close()

    assert process.killed
    assert session.thread_id == "thread-timeout"
    assert "resume" in resumed
    assert "thread-timeout" in resumed


def test_codex_cli_fails_closed_if_secret_reaches_event_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir, output_dir, work_dir, tmp_dir = _workspace_dirs(tmp_path)

    def fake_popen(_command, *, stdin, stdout, stderr, env):
        del stdin, stderr, env
        stdout.write(b'{"type":"error","message":"provider-secret"}\n')
        stdout.flush()
        return _FinishedProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    session = CodexCLISession(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "provider-secret", "test-model"
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        scratch_dir=work_dir,
        tmp_dir=tmp_dir,
        model="test-model",
        effort="high",
    )
    try:
        with pytest.raises(CodexCLIError, match="exposed the provider secret"):
            session.run("first", timeout_seconds=5)
    finally:
        session.close()


def test_codex_replay_repairs_in_one_thread_but_each_generation_is_fresh() -> None:
    valid_replay = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = tl.load(p0 + v0, mask=v0 < p2, other=0.0)
    tl.store(p1 + v0, v1, mask=v0 < p2)
"""
    sessions: list[object] = []
    compile_calls = 0

    def compile_checker(code, _artifact, function_name):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls % 2 == 1:
            return ReplayIngressResult(
                "fail",
                diagnostics=(
                    {
                        "rule_code": "synthetic_frontend_compile_failed",
                        "path": "<synthetic-frontend-compile>",
                        "anonymous_handles": [],
                        "line": 7,
                    },
                ),
            )
        return ReplayIngressResult(
            "pass",
            code=code,
            entry_function_name=function_name,
        )

    class FakeSession:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]
            self.scratch_dir = kwargs["scratch_dir"]
            self.thread_id = None
            self.calls = 0
            self.closed = False
            self.efforts = []
            self.identity = len(sessions) + 1
            sessions.append(self)

        def run(self, prompt, **kwargs):
            self.efforts.append(kwargs["effort"])
            self.calls += 1
            self.thread_id = f"fresh-replay-{self.identity}"
            if self.calls == 1:
                assert "replay_manifest_view" in prompt
                payload = valid_replay.replace("scrc_k0", "wrong_entry")
            else:
                check = json.loads(
                    (self.scratch_dir / "host_replay_check.json").read_text()
                )
                assert check["status"] == "fail"
                assert "same fresh source-blind Replay episode" in prompt
                payload = valid_replay
            (self.output_dir / "replay.source").write_text(payload)
            return CodexCLITurn(
                self.thread_id,
                10.0,
                {"input_tokens": 12, "cached_input_tokens": 9, "output_tokens": 3},
                events=(),
                model_segments=1,
                tool_actions=2,
                resumed=self.calls > 1,
            )

        def close(self):
            self.closed = True

    writer = CodexCLIReplayWriter(
        provider=ProviderConfig(
            "test",
            "http://provider.invalid/v1",
            "provider-secret",
            "test-model",
            reasoning_effort="xhigh",
        ),
        session_factory=FakeSession,
        compile_checker=compile_checker,
    )
    first = writer.generate(make_copy_artifact())
    second = writer.generate(make_copy_artifact())

    assert first.backend_protocol == second.backend_protocol == CODEX_CLI_REPLAY_BACKEND
    assert first.response_id == "fresh-replay-1"
    assert second.response_id == "fresh-replay-2"
    assert first.attempts == second.attempts == 3
    assert first.context_receipt["fresh_session"] is True
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert all(
        session.efforts == ["xhigh", "xhigh", "xhigh"]
        for session in sessions
    )
    assert compile_calls == 4
    assert [
        item["status"]
        for item in first.trajectory["replay_compile_check"]
    ] == ["fail", "pass"]


def _adapter_request() -> SCRCRunRequest:
    source = """def source_copy(a, b, n, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    value = tl.load(a + offsets, mask=mask)
    tl.store(b + offsets, value, mask=mask)
"""
    return SCRCRunRequest(
        source=SourceProgram(source, "gate-source"),
        binding=HarnessBindingInstance.create(
            {"c0": 1024}, gate1_case="same-c0"
        ),
        frontend="triton",
    )


def test_codex_lift_keeps_one_thread_and_repairs_from_host_check() -> None:
    artifact = make_copy_artifact()
    invalid = artifact.to_json()
    invalid["j"]["signature"]["parameters"][0]["dtype"] = "f16"
    sessions: list[object] = []

    class FakeSession:
        def __init__(self, **kwargs):
            self.input_dir = kwargs["input_dir"]
            self.output_dir = kwargs["output_dir"]
            self.thread_id = None
            self.calls = 0
            self.closed = False
            self.efforts = []
            sessions.append(self)

        def run(self, prompt, **kwargs):
            self.efforts.append(kwargs["effort"])
            self.calls += 1
            self.thread_id = "persistent-thread"
            if self.calls == 1:
                assert "lift_harness_abi" in prompt
                assert "lift_candidate_control overview" in prompt
                payload = invalid
            elif self.calls == 2:
                check = json.loads(
                    (self.input_dir / "candidate_check.json").read_text()
                )
                assert check["status"] == "fail"
                assert "CandidateCheck rejected" in prompt
                assert "lift_candidate_control overview" in prompt
                payload = artifact.to_json()
            else:
                external = json.loads(
                    (self.input_dir / "feedback.json").read_text()
                )
                assert external["failure_kind"] == "gate1"
                assert "same workspace and thread" in prompt
                assert "lift_candidate_control overview" in prompt
                payload = artifact.to_json()
            (self.output_dir / "candidate.json").write_text(json.dumps(payload))
            return CodexCLITurn(
                "persistent-thread",
                20.0,
                {"input_tokens": 10, "cached_input_tokens": 8, "output_tokens": 2},
                events=(),
                model_segments=1,
                tool_actions=2,
                resumed=self.calls > 1,
            )

        def close(self):
            self.closed = True

    adapter = CodexCLILiftAdapter(
        provider=ProviderConfig(
            "test",
            "http://provider.invalid/v1",
            "provider-secret",
            "test-model",
            reasoning_effort="xhigh",
        ),
        session_factory=FakeSession,
    )
    request = _adapter_request()
    first = adapter.lift(request, None)
    feedback = LiftFeedback(
        FeedbackKind.GATE1,
        {"observer_diff": {"category": "dataflow"}},
        1,
        artifact,
    )
    second = adapter.lift(request, feedback)

    assert first.artifact.to_json() == second.artifact.to_json()
    assert len(sessions) == 1
    session = sessions[0]
    assert session.calls == 3
    assert session.efforts == ["xhigh", "xhigh", "xhigh"]
    assert first.metadata["backend"] == CODEX_CLI_LIFT_BACKEND
    assert first.metadata["runtime"] == CODEX_CLI_AGENT_RUNTIME
    assert first.metadata["fresh_session"] is True
    assert second.metadata["fresh_session"] is False
    assert first.metadata["cli_turns"] == 2
    assert second.metadata["cli_turns"] == 1
    assert second.metadata["thread_id"] == "persistent-thread"
    assert second.metadata["session_cli_turns"] == 3
    assert len(adapter.records) == 2
    assert adapter.records[0].extraction is not None
    assert "provider-secret" not in json.dumps(
        adapter.records[0].extraction.provenance.trajectory
    )
    workspace_root = adapter._session_workspace.root
    adapter.close()
    assert session.closed
    assert not workspace_root.exists()


def test_codex_lift_accepts_exact_candidate_written_before_turn_timeout() -> None:
    artifact = make_copy_artifact()
    candidate_text = json.dumps(artifact.to_json())

    class FakeSession:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]
            self.thread_id = None
            self.closed = False

        def run(self, _prompt, **_kwargs):
            self.thread_id = "timed-out-lift-thread"
            (self.output_dir / "candidate.json").write_text(
                candidate_text
            )
            raise CodexCLIError("Codex turn exceeded its wall-time limit")

        def close(self):
            self.closed = True

    adapter = CodexCLILiftAdapter(
        provider=ProviderConfig(
            "test",
            "http://provider.invalid/v1",
            "provider-secret",
            "test-model",
            reasoning_effort="high",
        ),
        session_factory=FakeSession,
        model_timeout_seconds=1,
        session_wall_time_limit_seconds=10,
    )
    result = adapter.lift(_adapter_request(), None)

    assert result.artifact.j.kernel_id == artifact.j.kernel_id
    assert (
        adapter.records[-1].extraction.provenance.response_text
        == candidate_text
    )
    assert result.physical_calls == 1
    assert result.metadata["cli_turns"] == 1
    assert result.metadata["timeout_events"] == [
        {
            "turn_timeout_seconds": 1,
            "thread_id": "timed-out-lift-thread",
            "recovery": "current_exact_candidate",
        }
    ]
    adapter.close()


def test_codex_replay_accepts_exact_source_written_before_turn_timeout() -> None:
    valid_replay = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = tl.load(p0 + v0, mask=v0 < p2, other=0.0)
    tl.store(p1 + v0, v1, mask=v0 < p2)
"""

    class FakeSession:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]
            self.thread_id = None

        def run(self, _prompt, **_kwargs):
            self.thread_id = "timed-out-replay-thread"
            (self.output_dir / "replay.source").write_text(valid_replay)
            raise CodexCLIError("Codex turn exceeded its wall-time limit")

        def close(self):
            return None

    writer = CodexCLIReplayWriter(
        provider=ProviderConfig(
            "test",
            "http://provider.invalid/v1",
            "provider-secret",
            "test-model",
            reasoning_effort="high",
        ),
        session_factory=FakeSession,
        compile_checker=lambda code, _artifact, function_name: ReplayIngressResult(
            "pass", code=code, entry_function_name=function_name
        ),
        model_timeout_seconds=1,
        session_wall_time_limit_seconds=10,
    )
    generation = writer.generate(make_copy_artifact())

    assert generation.code == valid_replay
    assert generation.attempts == 1
    assert generation.physical_provider_calls == 1
    assert generation.trajectory["timeout_events"] == [
        {
            "turn_timeout_seconds": 1,
            "thread_id": "timed-out-replay-thread",
            "recovery": "current_exact_replay",
        }
    ]
