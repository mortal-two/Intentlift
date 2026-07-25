"""mini-swe-agent Lift backend; SCRC verification remains entirely external."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.provider import (
    ProviderConfig,
    get_provider,
)

from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCLiftAttempt,
    SCRCLiftRequest,
    SCRCProvenance,
    SCRCReliftContext,
    SCRCReliftRecord,
    decode_scrc_lift_artifact,
    check_scrc_lift_candidate,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)
from .scrc_schema import scrc_lift_core_json_schema
from .prompt import SCHEMA as INTENTIR_OPERATION_REFERENCE


MINISWE_LIFT_BACKEND = "minisweagent.intentir-lift.v1"
MINISWE_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v24"
MAX_RELIFT_CHECK_FEEDBACK_ITEMS = 20
MAX_RELIFT_CHECK_FEEDBACK_CHARS = 2048


def _usage_and_response_identity(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, int], str | None, str | None]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen_usage = False
    response_id: str | None = None
    response_model: str | None = None
    messages = trajectory.get("messages", ())
    if not isinstance(messages, list):
        return {}, None, None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("object") != "response":
            continue
        if isinstance(message.get("id"), str):
            response_id = message["id"]
        if isinstance(message.get("model"), str):
            response_model = message["model"]
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        seen_usage = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return (totals if seen_usage else {}), response_id, response_model


def _trajectory_bytes(trajectory: Mapping[str, Any]) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sanitize_trajectory(value: Any, secret: str, *, key: str = "") -> Any:
    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    } or normalized_key.endswith("_api_key"):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_trajectory(
                item,
                secret,
                key=str(item_key),
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_trajectory(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_trajectory(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _worker_evidence(
    result: Any,
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    info = trajectory.get("info", {})
    candidate_check = (
        info.get("candidate_check", {}) if isinstance(info, Mapping) else {}
    )
    return {
        "backend": MINISWE_LIFT_BACKEND,
        "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
        "exit_status": result.exit_status,
        "model_turns": result.model_calls,
        "physical_provider_calls": result.physical_provider_calls,
        "transport_retries": result.transport_retries,
        "elapsed_ms": result.elapsed_ms,
        "fresh_session": True,
        "candidate_check": (
            dict(candidate_check)
            if isinstance(candidate_check, Mapping)
            else {}
        ),
        "trajectory": trajectory,
    }


def _last_candidate_check_feedback(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return only the last evaluated source-free rule/path summary."""

    candidate_check = evidence.get("candidate_check")
    if not isinstance(candidate_check, Mapping):
        return None
    results = candidate_check.get("results")
    if not isinstance(results, list):
        return None
    for result in reversed(results):
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "fail"
            or result.get("evaluated") is not True
        ):
            continue
        protocol = result.get("protocol")
        rule_paths = result.get("rule_paths")
        if protocol != CANDIDATE_CHECK_PROTOCOL or not isinstance(
            rule_paths, list
        ):
            continue
        summary: list[dict[str, str]] = []
        for item in rule_paths[:MAX_RELIFT_CHECK_FEEDBACK_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            rule_code = item.get("rule_code")
            path = item.get("path")
            if (
                not isinstance(rule_code, str)
                or not isinstance(path, str)
                or rule_code == "check_budget_exhausted"
            ):
                continue
            summary.append(
                {
                    "rule_code": rule_code[:MAX_RELIFT_CHECK_FEEDBACK_CHARS],
                    "path": path[:MAX_RELIFT_CHECK_FEEDBACK_CHARS],
                }
            )
        if summary:
            return {"protocol": protocol, "diagnostics": summary}
    return None


def _input_files(request: SCRCLiftRequest) -> dict[str, str]:
    packet = json.loads(request.user_input)
    anonymous_source = packet.get("anonymous_source")
    if not isinstance(anonymous_source, str) or not anonymous_source:
        raise ValueError("prepared Lift packet lacks anonymous Source")
    harness_view = packet.get("harness_compiler_view", {})
    feedback = packet.get("relift")
    from minisweagent import package_dir

    validator = (
        Path(package_dir) / "config" / "intentir_validate_candidate.py"
    ).read_text(encoding="utf-8")
    preflight = (
        Path(package_dir) / "config" / "intentir_lift_preflight.md"
    ).read_text(encoding="utf-8")
    schema_query = (
        Path(package_dir) / "config" / "intentir_schema_query.py"
    ).read_text(encoding="utf-8")
    context_query = (
        Path(package_dir) / "config" / "intentir_context_query.py"
    ).read_text(encoding="utf-8")
    minimal_example = (
        Path(package_dir) / "config" / "intentir_lift_example.json"
    ).read_text(encoding="utf-8")
    readme = """# IntentIR Lift input

This directory is the complete allowlisted input for one fresh Lift round.

- `source.py`: alpha-renamed Source available only to Lift.
- `harness_view.json`: trusted anonymous compiler/Harness view.
- `feedback.json`: null on the first round, otherwise only the explicitly
  allowed previous parser/TypeCheck/Gate1 feedback, previous candidate, or the
  last worker's bounded source-free CandidateCheck rule/path summary.
- `schema/artifact.schema.json`: closed static JSON envelope for `(J,C,pi)`.
- `knowledge/lift_contract.md`: semantic ownership and construction rules.
- `knowledge/intent_ops.md`: global source-neutral IntentIR operation reference.
- `knowledge/preflight.md`: short source-neutral handle/ownership checklist.
- `knowledge/minimal_example.json`: source-neutral wire/ownership example for one
  integral rank-0 scalar explicitly equated to a logical dimension by a J-owned
  value_dimension_relations entry, explicit sum-then-div stages, one exact cast
  refinement, and a local reduction axis; replace every case-specific semantic,
  handle, shape, value, topology, and multiplicity.
- `tools/validate_candidate.py`: syntax/static-schema validator only.
- `intentir_candidate_check /output/candidate.json`: bounded, read-only
  Decode/TypeCheck diagnostics for the complete candidate (an eleven-evaluation
  ordinary allowance plus one separately reserved submission preflight);
  it never repairs or certifies the artifact.
- `tools/schema_query.py`: compact schema lookup (`top`, `catalog`, `op`,
  `section`) so raw 1400-line schema never needs to be paged end-to-end.
- `tools/context_query.py`: bounded lookup for bootstrap, exact external
  feedback, paged Source, Harness JSON paths, and focused contract/preflight
  matches. `bootstrap` copies an explicitly supplied previous artifact without
  editing it; no command constructs or repairs artifact fields.

Start with this single bounded command so long inputs cannot hide feedback:

`python /input/tools/context_query.py bootstrap`

Then inspect Source and Harness in separate actions with
`python /input/tools/context_query.py source` and
`python /input/tools/context_query.py harness`. On repair rounds, inspect
`python /input/tools/context_query.py feedback` before editing the candidate
copied by bootstrap. Use its `feedback-category`, paged `source`, Harness-path,
and `knowledge` subcommands for narrower views.

Do not page through the raw schema. Use the example plus compact queries such as:

`python /input/tools/schema_query.py /input/schema/artifact.schema.json top`

`python /input/tools/schema_query.py /input/schema/artifact.schema.json catalog`

Query exact `op NAME` or leaf `section NAME` only when needed. Useful leaf
sections include `access`, `effect`, `result_relation`, `authorization`,
`scope`, `c_slot`, `pi_instance`, `pi_route`, `pi_iteration`, and
`pi_refinement`. Create the
first complete candidate early, then let validator errors drive further lookup.
On a fresh retry where `feedback.json.previous_artifact` is an object, copy
that complete candidate first and make only the focused external-diagnostic
repair; do not rebuild it from Source. The repaired full object is still a new
Lift submission and will be checked externally again.
If current Lift/TypeCheck diagnostics contain `unresolved_gate1`, the enclosed
Gate1 counterexample remains an open obligation across the intermediate
construction failure; address both without inventing fields.
Write it to `/output/candidate.json`; then run:

`python /input/tools/validate_candidate.py /input/schema/artifact.schema.json /output/candidate.json`

The validator is not TypeCheck and does not certify the artifact.
After it passes, run the source-free diagnostic check:

`intentir_candidate_check /output/candidate.json`

This host virtual command must be the entire shell action exactly as shown. Do
not prepend `set`, an assignment, a pipe, or another shell command; such a
prefixed action runs inside the sandbox, where it is not an executable.
If it fails, edit the candidate yourself from the closed diagnostics and rerun
within the bounded allowance. The formal Runner independently repeats Decode
and TypeCheck after submission; CandidateCheck is not an acceptance path.
If the exact submitted bytes have no prior CandidateCheck PASS, submission
automatically invokes the same read-only preflight and returns FAIL diagnostics
to this session instead of delivering the candidate. It still never edits the
object, and a PASS still has no authority outside this worker.
"""
    return {
        "README.md": readme,
        "source.py": anonymous_source,
        "harness_view.json": json.dumps(
            harness_view, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "feedback.json": json.dumps(
            feedback, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "schema/artifact.schema.json": json.dumps(
            scrc_lift_core_json_schema(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        "knowledge/lift_contract.md": request.instructions,
        "knowledge/intent_ops.md": INTENTIR_OPERATION_REFERENCE,
        "knowledge/preflight.md": preflight,
        "knowledge/minimal_example.json": minimal_example,
        "tools/validate_candidate.py": validator,
        "tools/schema_query.py": schema_query,
        "tools/context_query.py": context_query,
    }


@dataclass
class MiniSWELiftAdapter:
    """One fresh bounded mini-swe worker per existing LiftAdapter call."""

    provider: ProviderConfig | None = None
    compiler_view: Any = None
    effort: str | None = None
    source_symbol: str | None = None
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def _latest_rejected_candidate(
        self,
        diagnostics: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Return only the adjacent JSON object named by parser feedback."""

        if (
            not self._records
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("failure_class") != "lift_response"
        ):
            return None
        record = self._records[-1]
        if (
            record.relift_round != len(self._records)
            or record.extraction is not None
            or len(record.rejected_attempts) != 1
        ):
            return None
        attempt = record.rejected_attempts[0]
        if diagnostics.get("diagnostic") != attempt.diagnostic:
            return None

        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-finite JSON number: {token}")

        try:
            candidate = json.loads(
                attempt.response_text,
                parse_constant=reject_nonfinite,
            )
            if not isinstance(candidate, dict):
                return None
            json.dumps(candidate, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return None
        return candidate

    def _relift_context(self, feedback: Any) -> SCRCReliftContext | None:
        if feedback is None:
            return None
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        diagnostics = getattr(feedback, "diagnostics", None)
        previous_artifact = getattr(feedback, "previous_artifact", None)
        previous_candidate = (
            self._latest_rejected_candidate(diagnostics)
            if kind == "lift"
            else None
        )
        if previous_candidate is None and hasattr(previous_artifact, "to_json"):
            previous_candidate = previous_artifact.to_json()
        return SCRCReliftContext(
            max(1, len(self._records)),
            kind,
            diagnostics if isinstance(diagnostics, Mapping) else {},
            previous_candidate,
        )

    def lift(self, request: Any, feedback: Any) -> Any:
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("MiniSWE Lift requires SourceProgram.text")
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = self.model_timeout_seconds
        if timeout is None:
            timeout = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=self._relift_context(feedback),
            source_symbol=self.source_symbol,
        )
        round_index = len(self._records) + 1
        candidate_check_context = scrc_candidate_check_context(prepared)
        try:
            from minisweagent.environments.intentir_lift import (
                IntentIRLiftWorkspace,
            )
            from minisweagent.run.intentir_lift import (
                IntentIRLiftRunError,
                create_intentir_lift_model,
                run_intentir_lift,
            )
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "MiniSWE Lift backend requires `pip install -e '.[lift-agent,verify]'`",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc

        workspace = IntentIRLiftWorkspace.create(_input_files(prepared))
        result = None
        try:
            model = create_intentir_lift_model(
                model_name=provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                reasoning_effort=effective_effort,
                timeout_seconds=timeout,
            )
            result = run_intentir_lift(
                model=model,
                workspace=workspace,
                config_overrides=self.worker_config,
                candidate_checker=lambda candidate_text: (
                    check_scrc_lift_candidate(
                        candidate_text,
                        candidate_check_context,
                    )
                ),
            )
        except IntentIRLiftRunError as exc:
            result = exc.result
            safe_trajectory = _sanitize_trajectory(
                result.trajectory, provider.api_key
            )
            evidence = _worker_evidence(result, safe_trajectory)
            diagnostic = str(exc)
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    backend_evidence=evidence,
                )
            )
            if (
                exc.cause is not None
                and result.exit_status != "InvalidSubmission"
            ) or result.model_calls < 1:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    diagnostic,
                    evidence={
                        key: value
                        for key, value in evidence.items()
                        if key != "trajectory"
                    },
                ) from exc
            from intentir.verify.scrc_runner import CandidateLiftError

            rejected_diagnostics: dict[str, Any] = {
                "diagnostic": diagnostic,
                "failure_class": "lift_worker_no_candidate",
            }
            candidate_check_feedback = _last_candidate_check_feedback(evidence)
            if candidate_check_feedback is not None:
                rejected_diagnostics["candidate_check_feedback"] = (
                    candidate_check_feedback
                )
            raise CandidateLiftError(
                rejected_diagnostics,
                physical_calls=result.physical_provider_calls,
            ) from exc
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence={
                        "backend": MINISWE_LIFT_BACKEND,
                        "fresh_session": True,
                    },
                )
            )
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            if isinstance(exc, SCRCInfrastructureError):
                raise
            raise SCRCInfrastructureError(
                f"MiniSWE Lift infrastructure failed: {type(exc).__name__}: {exc}",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc
        finally:
            workspace.cleanup()

        assert result is not None and result.candidate_text is not None
        safe_trajectory = _sanitize_trajectory(
            result.trajectory, provider.api_key
        )
        usage, response_id, response_model = _usage_and_response_identity(
            safe_trajectory
        )
        raw_trajectory = _trajectory_bytes(safe_trajectory)
        worker_evidence = _worker_evidence(result, safe_trajectory)
        try:
            artifact = decode_scrc_lift_artifact(result.candidate_text, prepared)
        except Exception as exc:
            attempt = SCRCLiftAttempt(
                model=provider.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=result.candidate_text,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                response_sha256=hashlib.sha256(
                    result.candidate_text.encode()
                ).hexdigest(),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                elapsed_ms=result.elapsed_ms,
                backend=MINISWE_LIFT_BACKEND,
                termination_status=result.exit_status,
                trajectory=safe_trajectory,
            )
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=attempt.diagnostic,
                    rejected_attempts=(attempt,),
                    backend_evidence=worker_evidence,
                )
            )
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                {
                    "diagnostic": attempt.diagnostic,
                    "failure_class": "lift_response",
                },
                physical_calls=result.physical_provider_calls,
            ) from exc

        provenance = SCRCProvenance(
            model=provider.model,
            prompt_sha256=prepared.prompt_sha256,
            response_id=response_id,
            response_model=response_model,
            usage=usage,
            response_sha256=hashlib.sha256(
                result.candidate_text.encode()
            ).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=result.candidate_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            elapsed_ms=result.elapsed_ms,
            backend=MINISWE_LIFT_BACKEND,
            termination_status=result.exit_status,
            trajectory=safe_trajectory,
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
        )
        self._records.append(
            SCRCReliftRecord(
                round_index,
                extraction=extraction,
                backend_evidence=worker_evidence,
            )
        )
        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            artifact,
            physical_calls=result.physical_provider_calls,
            metadata={
                "backend": MINISWE_LIFT_BACKEND,
                "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
                "model": provider.model,
                "response_model": response_model,
                "response_id": response_id,
                "usage": usage,
                "elapsed_ms": result.elapsed_ms,
                "agent_steps": result.model_calls,
                "physical_provider_calls": result.physical_provider_calls,
                "transport_retries": result.transport_retries,
                "fresh_session": True,
                "artifact_schema": artifact.wire_schema,
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "candidate_check": worker_evidence["candidate_check"],
            },
        )


__all__ = [
    "MINISWE_LIFT_BACKEND",
    "MINISWE_LIFT_KNOWLEDGE",
    "MiniSWELiftAdapter",
]
