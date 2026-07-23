from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from direct_fixtures import make_copy_artifact
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.extraction.lifters.miniswe import (
    MINISWE_LIFT_KNOWLEDGE,
    MiniSWELiftAdapter,
    _input_files,
)
from intentir.extraction.lifters.scrc import (
    SCRCLiftAttempt,
    SCRCReliftRecord,
    prepare_scrc_lift,
)
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.scrc_runner import (
    CandidateLiftError,
    FeedbackKind,
    HarnessBindingInstance,
    LiftFeedback,
    SCRCRunRequest,
    SourceProgram,
)
from minisweagent.environments.intentir_lift import (
    INTENTIR_LIFT_SUBMIT,
    IntentIRLiftEnvironment,
    IntentIRLiftWorkspace,
    MAX_ACTION_OUTPUT_BYTES,
    MAX_CANDIDATE_BYTES,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall_response import (
    parse_toolcall_actions_response,
)
from minisweagent.exceptions import FormatError
from minisweagent.run.intentir_lift import (
    IntentIRLiftRunError,
    IntentIRLiftRunResult,
    create_intentir_lift_model,
    load_intentir_lift_config,
    run_intentir_lift,
)


class _ScriptedModel:
    def __init__(self, commands: list[str]):
        self.commands = list(commands)
        self.query_inputs: list[list[dict]] = []

    def query(self, messages, **kwargs):
        del kwargs
        self.query_inputs.append(list(messages))
        command = self.commands.pop(0)
        return {
            "role": "assistant",
            "content": "",
            "extra": {"actions": [{"command": command}], "cost": 0.0},
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
        return {"info": {"config": {"model": "scripted"}}}


def _workspace() -> IntentIRLiftWorkspace:
    return IntentIRLiftWorkspace.create(
        {
            "README.md": "allowed input",
            "schema/artifact.schema.json": "{}",
        }
    )


def _real_lift_input_files() -> dict[str, str]:
    prepared = prepare_scrc_lift(
        """@triton.jit
def source_copy(a, b, n, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    value = tl.load(a + offsets, mask=mask)
    tl.store(b + offsets, value, mask=mask)
"""
    )
    return _input_files(prepared)


def test_real_input_bundle_has_compact_source_neutral_knowledge() -> None:
    files = _real_lift_input_files()

    assert MINISWE_LIFT_KNOWLEDGE == "intentir.lift-knowledge.v6"
    assert "knowledge/minimal_example.json" in files
    assert "knowledge/intent_ops.md" in files
    assert "tools/schema_query.py" in files
    assert "map_binary" in files["knowledge/intent_ops.md"]
    assert "Do not page through the raw schema" in files["README.md"]
    assert "never names a source/local or J-body v*" in (
        files["knowledge/lift_contract.md"]
    )
    assert "Never replace the program path with constant zero" in (
        files["knowledge/preflight.md"]
    )


def test_compact_schema_queries_work_in_real_bubblewrap() -> None:
    workspace = IntentIRLiftWorkspace.create(_real_lift_input_files())
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json top; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json catalog; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json op map_binary; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section parameter; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section access; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section output_mapping; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section pi_instance"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 0, result
    output = result["output"]
    assert "root: required=['schema', 'j', 'c', 'pi']" in output
    assert "ordinary operations: map_binary, map_unary" in output
    assert 'fn in' in output
    assert '"logical_ref"' in output
    assert "rank-0 J input" in output
    assert "never names a v* SSA value" in output
    assert '"access_id"' in output
    assert "deterministically elaborated" in output
    assert "fill_value is legal only on a load" in output
    assert "leave output_mappings empty" in output
    assert '"semantic_ref"' in output
    assert len(output.splitlines()) < 350


def test_minimal_example_passes_static_schema_and_canonical_parser() -> None:
    files = _real_lift_input_files()
    payload = json.loads(files["knowledge/minimal_example.json"])
    artifact = Gate1Artifact.from_json(payload)
    assert artifact.to_json()["j"]["scopes"][0]["scope_id"] == "q0"

    workspace = IntentIRLiftWorkspace.create(files)
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "cp /input/knowledge/minimal_example.json "
                    "/output/candidate.json && "
                    "python /input/tools/validate_candidate.py "
                    "/input/schema/artifact.schema.json "
                    "/output/candidate.json"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 0, result
    assert "static schema: PASS" in result["output"]


def test_intentir_observation_exposes_remaining_turn_budget() -> None:
    assert load_intentir_lift_config()["agent"]["step_limit"] == 10
    model = create_intentir_lift_model(
        model_name="test",
        api_base="http://provider.invalid/v1",
        api_key="not-used",
        reasoning_effort="xhigh",
        timeout_seconds=1,
    )
    message = {
        "extra": {
            "actions": [
                {"command": "true", "tool_call_id": "call-budget"}
            ]
        }
    }
    observation = model.format_observation_messages(
        message,
        [{"returncode": 0, "output": "ok", "exception_info": ""}],
        {"n_model_calls": 7, "step_limit": 10},
    )[0]
    rendered = json.loads(observation["output"])

    assert rendered["turn_budget"] == {"used": 7, "limit": 10, "remaining": 3}
    assert rendered["directive"].startswith("Stop browsing")


def test_real_default_agent_and_bubblewrap_deliver_one_candidate() -> None:
    workspace = _workspace()
    candidate = '{"j": {}, "c": {}, "pi": {}}'
    model = _ScriptedModel(
        [
            f"printf '%s\\n' '{candidate}' > /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
        ]
    )
    try:
        result = run_intentir_lift(model=model, workspace=workspace)
    finally:
        workspace.cleanup()

    assert json.loads(result.candidate_text or "") == json.loads(candidate)
    assert result.exit_status == "Submitted"
    assert result.model_calls == 2
    assert len(model.query_inputs) == 2
    assert len(model.query_inputs[0]) == 2


def test_bubblewrap_hides_host_env_repo_and_network_and_protects_input(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTENTIR_TEST_CPA_KEY", "must-not-enter-sandbox")
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    host_net_namespace = os.readlink("/proc/self/ns/net")
    try:
        checks = environment.execute(
            {
                "command": (
                    "test -r /input/README.md && "
                    "test ! -w /input/README.md && "
                    "test ! -e /home/kingdom && "
                    "! touch /rootfile && "
                    "! touch /dev/unmonitored && "
                    "if [ -d /dev/shm ]; then ! touch /dev/shm/unmonitored; fi && "
                    "touch /tmp/monitored && "
                    "test -z \"${INTENTIR_TEST_CPA_KEY+x}\" && "
                    "python -c 'import importlib.util; "
                    "assert importlib.util.find_spec(\"intentir\") is None' && "
                    "readlink /proc/self/ns/net"
                )
            }
        )
        monitored_tmp_exists = (workspace.tmp_dir / "monitored").exists()
    finally:
        workspace.cleanup()

    assert checks["returncode"] == 0, checks
    assert checks["output"].splitlines()[-1] != host_net_namespace
    assert monitored_tmp_exists


def test_submission_rejects_extra_output_file_without_ending_session() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "printf '{}\\n' > /output/candidate.json; "
                    "printf 'draft\\n' > /output/extra.txt; "
                    f"printf '{INTENTIR_LIFT_SUBMIT}\\n'"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 2
    assert "exactly candidate.json" in result["output"]


def test_candidate_size_is_rejected_before_unbounded_read() -> None:
    workspace = _workspace()
    try:
        with workspace.candidate_path.open("wb") as stream:
            stream.truncate(MAX_CANDIDATE_BYTES + 1)
        with pytest.raises(ValueError, match="candidate.json size"):
            workspace.read_candidate()
    finally:
        workspace.cleanup()


def test_shell_stdout_is_hard_bounded() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "head -c 2097152 /dev/zero | tr '\\0' x"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] != 0
    assert len(result["output"].encode()) <= MAX_ACTION_OUTPUT_BYTES
    assert "output reached" in result["exception_info"]


def test_writable_workspace_total_size_is_bounded_and_reset() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "for i in $(seq 1 9); do "
                    "truncate -s 8388608 /work/file-$i; done"
                )
            }
        )
        remaining = tuple(workspace.scratch_dir.iterdir())
    finally:
        workspace.cleanup()

    assert result["returncode"] != 0
    assert "workspace exceeded limit" in result["exception_info"]
    assert remaining == ()


def test_workspace_cleanup_handles_depth_beyond_python_recursion_limit() -> None:
    workspace = _workspace()
    root = workspace.root
    current = workspace.scratch_dir
    for _index in range(1100):
        current = current / "d"
        current.mkdir()

    workspace.cleanup()

    assert not root.exists()


def test_agent_created_excessive_directory_depth_is_reset() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "for i in $(seq 1 80); do mkdir d; cd d; done"
                )
            }
        )
        remaining = tuple(workspace.scratch_dir.iterdir())
    finally:
        workspace.cleanup()

    assert result["returncode"] != 0
    assert "workspace exceeded limit" in result["exception_info"]
    assert remaining == ()


def test_five_candidate_workspaces_are_concurrently_isolated() -> None:
    barrier = threading.Barrier(5, timeout=5)

    def run(branch: int) -> tuple[Path, str]:
        workspace = IntentIRLiftWorkspace.create(
            {"branch.txt": f"branch-{branch}"}
        )
        environment = IntentIRLiftEnvironment(
            input_dir=workspace.input_dir,
            output_dir=workspace.output_dir,
            scratch_dir=workspace.scratch_dir,
        )
        root = workspace.root
        try:
            barrier.wait()
            result = environment.execute({"command": "cat /input/branch.txt"})
            assert result["returncode"] == 0
            return root, result["output"].strip()
        finally:
            workspace.cleanup()

    with ThreadPoolExecutor(max_workers=5) as executor:
        outcomes = list(executor.map(run, range(1, 6)))

    assert len({root for root, _value in outcomes}) == 5
    assert {value for _root, value in outcomes} == {
        f"branch-{index}" for index in range(1, 6)
    }
    assert all(not root.exists() for root, _value in outcomes)


def test_step_limit_is_one_candidate_failure_not_an_internal_relift() -> None:
    workspace = _workspace()
    model = _ScriptedModel(["true"])
    try:
        with pytest.raises(IntentIRLiftRunError) as captured:
            run_intentir_lift(
                model=model,
                workspace=workspace,
                config_overrides={"agent": {"step_limit": 1}},
            )
    finally:
        workspace.cleanup()

    assert captured.value.cause is None
    assert captured.value.result.exit_status == "LimitsExceeded"
    assert captured.value.result.model_calls == 1


def test_litellm_model_has_one_explicit_attempt(monkeypatch) -> None:
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        raise RuntimeError("transport failed")

    monkeypatch.setattr("litellm.responses", fail_once)
    model = LitellmResponseModel(
        model_name="openai/test",
        retry_attempts=1,
        cost_tracking="ignore_errors",
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        model.query([])
    assert calls == 1


def test_non_string_shell_action_is_a_model_format_error() -> None:
    with pytest.raises(FormatError) as captured:
        parse_toolcall_actions_response(
            [
                {
                    "type": "function_call",
                    "name": "bash",
                    "call_id": "call-invalid",
                    "arguments": json.dumps({"command": 123}),
                }
            ],
            format_error_template="{{ error }}",
        )
    assert "non-string" in captured.value.messages[0]["content"][0]["text"]


def test_multiple_shell_actions_are_one_turn_format_error() -> None:
    calls = [
        {
            "type": "function_call",
            "name": "bash",
            "call_id": f"call-{index}",
            "arguments": json.dumps({"command": "true"}),
        }
        for index in range(2)
    ]
    with pytest.raises(FormatError) as captured:
        parse_toolcall_actions_response(
            calls,
            format_error_template="{{ error }}",
        )
    assert "exactly one" in captured.value.messages[0]["content"][0]["text"]


def test_model_serialization_redacts_provider_key() -> None:
    model = create_intentir_lift_model(
        model_name="test",
        api_base="http://provider.invalid/v1",
        api_key="not-for-evidence",
        reasoning_effort="xhigh",
        timeout_seconds=1,
    )
    serialized = json.dumps(model.serialize())
    assert "not-for-evidence" not in serialized
    assert "<redacted>" in serialized
    assert model.config.retry_attempts == 1
    assert model.config.use_global_model_stats is False
    assert model.config.model_kwargs["num_retries"] == 0


def test_existing_litellm_responses_model_sends_the_expected_cpa_packet() -> None:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            payload = {
                "id": "response-local",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "test",
                "output": [
                    {
                        "id": "fc-local",
                        "type": "function_call",
                        "call_id": "call-local",
                        "name": "bash",
                        "arguments": json.dumps({"command": "true"}),
                        "status": "completed",
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3,
                },
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = create_intentir_lift_model(
            model_name="test",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-key",
            reasoning_effort="xhigh",
            timeout_seconds=5,
        )
        global_calls_before = GLOBAL_MODEL_STATS.n_calls
        message = model.query(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert message["extra"]["actions"] == [
        {"command": "true", "tool_call_id": "call-local"}
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/v1/responses"
    assert request["authorization"] == "Bearer local-test-key"
    assert request["body"]["model"] == "test"
    assert request["body"]["reasoning"] == {"effort": "xhigh"}
    assert request["body"]["parallel_tool_calls"] is False
    assert request["body"]["tools"][0]["name"] == "bash"
    assert GLOBAL_MODEL_STATS.n_calls == global_calls_before


def test_cpa_http_failure_is_not_retried_by_litellm_or_openai() -> None:
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            request_count += 1
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            payload = json.dumps(
                {"error": {"message": "single failure", "type": "server_error"}}
            ).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = create_intentir_lift_model(
            model_name="test",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-key",
            reasoning_effort="xhigh",
            timeout_seconds=5,
        )
        with pytest.raises(Exception, match="single failure"):
            model.query([{"role": "user", "content": "task"}])
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert request_count == 1


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


def test_adapter_uses_fresh_workspaces_and_canonical_decoder(monkeypatch) -> None:
    artifact = make_copy_artifact()
    captured: list[tuple[Path, dict]] = []

    def fake_model(**kwargs):
        assert kwargs["reasoning_effort"] == "xhigh"
        return object()

    def fake_run(*, model, workspace, config_overrides):
        del model, config_overrides
        feedback = json.loads(
            (workspace.input_dir / "feedback.json").read_text()
        )
        captured.append((workspace.root, feedback))
        response = {
            "object": "response",
            "id": f"response-{len(captured)}",
            "model": "test-model-version",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        return IntentIRLiftRunResult(
            json.dumps(artifact.to_json()),
            {
                "messages": [response],
                "info": {"config": {"authorization": "secret"}},
            },
            "Submitted",
            3,
            25.0,
        )

    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.create_intentir_lift_model", fake_model
    )
    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.run_intentir_lift", fake_run
    )
    adapter = MiniSWELiftAdapter(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "secret", "test-model",
            reasoning_effort="xhigh",
        )
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
    assert first.physical_calls == second.physical_calls == 3
    assert first.metadata["fresh_session"] is True
    assert captured[0][0] != captured[1][0]
    assert captured[0][1] is None
    assert captured[1][1]["failure_kind"] == "gate1"
    assert captured[1][1]["previous_artifact"] == artifact.to_json()
    assert all(not path.exists() for path, _feedback in captured)
    assert len(adapter.records) == 2
    assert adapter.records[0].extraction is not None
    assert adapter.records[0].extraction.provenance.trajectory is not None
    assert "secret" not in json.dumps(
        adapter.records[0].extraction.provenance.trajectory
    )


def test_parser_rejection_enters_only_next_fresh_worker_with_candidate(
    monkeypatch,
) -> None:
    artifact = make_copy_artifact()
    invalid = json.loads(
        _real_lift_input_files()["knowledge/minimal_example.json"]
    )
    invalid["j"]["signature"]["parameters"][2]["logical_ref"] = "v0"
    captured: list[tuple[Path, object]] = []

    def fake_model(**kwargs):
        del kwargs
        return object()

    def fake_run(*, model, workspace, config_overrides):
        del model, config_overrides
        feedback = json.loads(
            (workspace.input_dir / "feedback.json").read_text()
        )
        captured.append((workspace.root, feedback))
        candidate = invalid if len(captured) == 1 else artifact.to_json()
        return IntentIRLiftRunResult(
            json.dumps(candidate),
            {"messages": []},
            "Submitted",
            2,
            10.0,
        )

    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.create_intentir_lift_model", fake_model
    )
    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.run_intentir_lift", fake_run
    )
    adapter = MiniSWELiftAdapter(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "secret", "test-model",
            reasoning_effort="xhigh",
        )
    )
    request = _adapter_request()

    with pytest.raises(CandidateLiftError) as rejected:
        adapter.lift(request, None)
    feedback = LiftFeedback(
        FeedbackKind.LIFT,
        rejected.value.diagnostics,
        None,
        artifact,
    )
    repaired = adapter.lift(request, feedback)

    assert repaired.artifact.to_json() == canonicalize_gate1_artifact(
        artifact
    ).to_json()
    assert captured[0][1] is None
    assert captured[1][1]["failure_kind"] == "lift"
    assert captured[1][1]["previous_artifact"] == invalid
    assert captured[0][0] != captured[1][0]
    assert all(not root.exists() for root, _feedback in captured)


@pytest.mark.parametrize(
    "response_text",
    ["[]", "{not-json", '{"value": NaN}'],
)
def test_non_object_rejection_is_not_reused_as_candidate(
    response_text: str,
) -> None:
    diagnostic = "SchemaError: rejected"
    adapter = MiniSWELiftAdapter()
    adapter._records.append(
        SCRCReliftRecord(
            1,
            diagnostic=diagnostic,
            rejected_attempts=(
                SCRCLiftAttempt("test", diagnostic, response_text),
            ),
        )
    )
    previous = make_copy_artifact()
    context = adapter._relift_context(
        LiftFeedback(
            FeedbackKind.LIFT,
            {
                "failure_class": "lift_response",
                "diagnostic": diagnostic,
            },
            None,
            previous,
        )
    )

    assert context is not None
    assert context.previous_artifact == previous.to_json()


def test_rejected_candidate_state_is_adapter_local_and_adjacent() -> None:
    diagnostic = "SchemaError: rejected"
    rejected = SCRCLiftAttempt("test", diagnostic, '{"candidate": 1}')
    first = MiniSWELiftAdapter()
    first._records.append(
        SCRCReliftRecord(
            1,
            diagnostic=diagnostic,
            rejected_attempts=(rejected,),
        )
    )
    first._records.append(
        SCRCReliftRecord(2, diagnostic="worker produced no candidate")
    )
    feedback = LiftFeedback(
        FeedbackKind.LIFT,
        {
            "failure_class": "lift_response",
            "diagnostic": diagnostic,
        },
    )

    adjacent = first._relift_context(feedback)
    separate = MiniSWELiftAdapter()._relift_context(feedback)

    assert adjacent is not None and adjacent.previous_artifact is None
    assert separate is not None and separate.previous_artifact is None
