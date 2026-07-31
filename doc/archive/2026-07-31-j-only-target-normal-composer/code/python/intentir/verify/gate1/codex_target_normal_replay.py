"""J-only Codex Semantic Agent followed by the deterministic Triton Composer.

This is a separate Replay backend during Task 19's vertical slice.  The Codex
workspace contains only the deterministic S_J view, public frontend capability,
and the Target-Normal contract.  π and MaskC are mounted only after the Agent episode ends, in the
host-side Composer call.  Thus a Composer diagnostic can never become an Agent
repair hint or a semantic shortcut.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
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
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.model import Frontend, Gate1Artifact

from .replay import (
    REPLAY_CHECK_PROTOCOL,
    ReplayAttempt,
    ReplayGeneration,
    ReplayGenerationError,
    _context_receipt,
    _replay_input_files,
    _trajectory_bytes,
    build_source_blind_replay_compile_case,
    replay_source_ingress,
    source_blind_replay_compile_check,
)
from .replay_semantic_core import materialize_replay_semantics
from .target_normal_ir import (
    TARGET_NORMAL_IR_PROTOCOL,
    check_target_normal_json,
    render_target_normal_contract,
)
from .triton_target_composer import (
    compose_triton_target_normal,
)


CODEX_TARGET_NORMAL_REPLAY_BACKEND = "openai.codex-target-normal.intentir-replay.v1"


_INITIAL_PROMPT = """You are the sole semantic author for one fresh source-blind IntentIR Replay episode.
Work only under the isolated /workspace. input/ is immutable; output/, work/, and
tmp/ are writable. The only semantic authority is the deterministic read-only S_J
view of exact J. No Source, Harness/compiler view, concrete schedule value,
realization witness, implementation values, Gate result, prior Replay, repository,
or network is available.

Read input/semantic-program.intent, then read the short input/TARGET_NORMAL_IR.md
and input/frontend.json. Write exactly one complete
`output/target-normal.json`. Do not write target source. The file is a compact
Target-Normal lowering instance: partition every S_J node once in lexical order,
choose only the closed generic primitive family allowed by the contract, preserve
the exact J access order, and preserve output value bindings. You may group a
contiguous semantic region into a generic fusion/reduction/contraction/control
primitive, but you may not invent operands, types, predicates, constants, physical
coordinates, program ids, pointers, layouts, APIs, or target recipes.

Use the exact commands in input/TOOL.md after writing and repair only the J/IR-local
diagnostics. Finish by calling `submit_target_normal`. The checker is diagnostic
only: it never supplies semantic facts or repairs a field. Keep the episode alive
until submission is accepted or the host deadline is reached.
"""

_REPAIR_PROMPT = """Continue the same Target-Normal episode. Read the current output/target-normal.json
and the last J-only checker receipt in work/target-normal-check.json. Repair only
the reported J/IR-local structural issue, then call target_normal_check and
submit_target_normal. Do not ask for or infer any realization, schedule, Source,
or Gate fact; those are outside this episode and are never exposed here.
"""

_TOOL_CONTRACT = """# Target-Normal Agent tools

The only episode tools are:

```text
python .agents/intentir_agent_tool.py target_normal_check
python .agents/intentir_agent_tool.py submit_target_normal
```

Both commands inspect the exact current `output/target-normal.json`.  They only
return J/IR-local diagnostics; neither command sees pi, C, Source, Harness, Gate,
or target source, and neither command generates or edits any field.
"""


def _atomic_write(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    result: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[key] = result.get(key, 0) + value
    return result


@dataclass
class _TargetNormalWorkspace:
    root: Path
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, files: Mapping[str, str]) -> "_TargetNormalWorkspace":
        root = Path(tempfile.mkdtemp(prefix="intentir-target-normal-"))
        workspace = cls(
            root=root,
            input_dir=root / "input",
            output_dir=root / "output",
            scratch_dir=root / "work",
            tmp_dir=root / "tmp",
        )
        for directory in (
            workspace.input_dir,
            workspace.output_dir,
            workspace.scratch_dir,
            workspace.tmp_dir,
        ):
            directory.mkdir()
        try:
            for name, content in files.items():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError(f"unsafe Target-Normal input path: {name!r}")
                destination = workspace.input_dir / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
                destination.chmod(0o400)
        except Exception:
            workspace.cleanup()
            raise
        return workspace

    @property
    def target_normal_path(self) -> Path:
        return self.output_dir / "target-normal.json"

    def read_target_normal(self) -> str:
        entries = tuple(self.output_dir.iterdir())
        if entries != (self.target_normal_path,):
            names = sorted(item.name for item in entries)
            raise ValueError(
                "Target-Normal output must contain exactly target-normal.json; "
                f"found {names}"
            )
        if self.target_normal_path.is_symlink() or not self.target_normal_path.is_file():
            raise ValueError("target-normal.json must be a regular file")
        raw = self.target_normal_path.read_bytes()
        if not 1 <= len(raw) <= 512 * 1024:
            raise ValueError("target-normal.json is outside the closed size bound")
        return raw.decode("utf-8")

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass
class CodexTargetNormalReplayWriter:
    """One persistent J-only Agent episode and one host-side composition."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    model_timeout_seconds: float | None = None
    session_wall_time_limit_seconds: float | None = None
    stop_requested: Callable[[], int | None] | None = field(default=None, repr=False)
    session_factory: Callable[..., Any] = field(default=CodexCLISession, repr=False)
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        provider = self.provider or get_provider()
        effort = self.effort or provider.reasoning_effort
        turn_timeout = self.model_timeout_seconds or (
            XHIGH_RESPONSES_TIMEOUT_SECONDS
            if effort.casefold() == "xhigh"
            else DEFAULT_RESPONSES_TIMEOUT_SECONDS
        )
        episode_limit = self.session_wall_time_limit_seconds or (
            2.0 * XHIGH_RESPONSES_TIMEOUT_SECONDS
            if effort.casefold() == "xhigh"
            else 3.0 * DEFAULT_RESPONSES_TIMEOUT_SECONDS
        )
        if turn_timeout <= 0 or episode_limit <= 0:
            raise ValueError("Target-Normal time limits must be positive")

        files, packet, _instructions = _replay_input_files(
            artifact, frontend=self.frontend, masked=masked
        )
        semantic = materialize_replay_semantics(packet["j"])
        function_name = str(packet["u_f"]["replay_function"])
        frontend_json = {
            "protocol": "intentir.target-normal.frontend.v1",
            "frontend": self.frontend.value,
            "entry": function_name,
            "public_target_capability": "generic Triton Target-Normal primitives only",
        }
        workspace = _TargetNormalWorkspace.create(
            {
                "semantic-program.intent": semantic.text,
                "frontend.json": json.dumps(frontend_json, indent=2, sort_keys=True) + "\n",
                "TARGET_NORMAL_IR.md": render_target_normal_contract(),
                "TOOL.md": _TOOL_CONTRACT,
            }
        )
        turns: list[CodexCLITurn] = []
        checks: list[dict[str, Any]] = []
        thread_ids: list[str] = []
        thread_id: str | None = None
        elapsed_ms = 0.0
        accepted = False
        checked = None
        last_good: str | None = None
        session: Any | None = None
        bridge: CodexCLIToolBridge | None = None

        def check_current() -> tuple[Any, dict[str, Any]]:
            nonlocal last_good
            try:
                text = workspace.read_target_normal()
                result, receipt = check_target_normal_json(
                    text,
                    semantic.to_json(),
                    frontend=self.frontend.value,
                    entry=function_name,
                )
            except Exception as exc:
                result = None
                receipt = {
                    "protocol": "intentir.replay-target-normal-check.v1",
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "target_normal_unavailable",
                            "path": "F_f",
                            "detail": str(exc),
                        }
                    ],
                }
            checks.append(receipt)
            if result is not None:
                last_good = workspace.read_target_normal()
            return result, receipt

        def write_receipt(receipt: Mapping[str, Any]) -> None:
            _atomic_write(
                workspace.scratch_dir / "target-normal-check.json",
                json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n",
            )

        def target_normal_check_tool() -> Mapping[str, Any]:
            _result, receipt = check_current()
            return receipt

        def submit_target_normal_tool() -> Mapping[str, Any]:
            nonlocal accepted, checked
            checked, receipt = check_current()
            if checked is not None:
                accepted = True
            return {
                "protocol": "intentir.replay-target-normal-submit.v1",
                "status": "pass" if checked is not None else "fail",
                "check": receipt,
                "authority": (
                    "J-only Target-Normal ingress; no pi/C/Source/Gate lookup and no repair"
                ),
            }

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "target_normal_check": target_normal_check_tool,
                "submit_target_normal": submit_target_normal_tool,
            },
        )

        def bridge_receipt() -> dict[str, Any]:
            evidence = bridge.evidence
            receipts = evidence.get("receipts", [])
            return {
                "protocol": evidence.get("protocol"),
                "calls": evidence.get("calls", 0),
                "receipts": [
                    {
                        key: item.get(key)
                        for key in ("tool", "status", "argument_count", "elapsed_ms")
                    }
                    for item in receipts
                    if isinstance(item, Mapping)
                ],
                "authority": evidence.get("authority"),
            }

        def new_session() -> Any:
            return self.session_factory(
                provider=provider,
                input_dir=workspace.input_dir,
                output_dir=workspace.output_dir,
                scratch_dir=workspace.scratch_dir,
                tmp_dir=workspace.tmp_dir,
                model=provider.model,
                effort=effort,
            )

        prompt = _INITIAL_PROMPT
        try:
            bridge.start()
            session = new_session()
            while not accepted:
                remaining = episode_limit - elapsed_ms / 1000.0
                if remaining <= 0:
                    raise RuntimeError("Target-Normal Agent episode wall-time exhausted")
                try:
                    turn = session.run(
                        prompt,
                        timeout_seconds=min(remaining, turn_timeout),
                        effort=effort,
                        stop_requested=self.stop_requested,
                    )
                except CodexCLIError as exc:
                    if "wall-time limit" not in str(exc):
                        raise
                    elapsed_ms += min(remaining, turn_timeout) * 1000.0
                    if last_good is not None:
                        _atomic_write(workspace.target_normal_path, last_good)
                        checked, receipt = check_current()
                        if checked is not None:
                            accepted = True
                            break
                    raise
                turns.append(turn)
                elapsed_ms += turn.elapsed_ms
                thread_id = turn.thread_id
                if thread_id not in thread_ids:
                    thread_ids.append(thread_id)
                if accepted:
                    break
                _result, receipt = check_current()
                write_receipt(receipt)
                prompt = _REPAIR_PROMPT

            if checked is None:
                checked, receipt = check_current()
                if checked is None:
                    raise RuntimeError("Target-Normal submission was not closed")

            composition = compose_triton_target_normal(
                checked,
                packet["pi"],
                packet["c"],
            )
            ingress = replay_source_ingress(
                composition.source,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )
            if ingress.status != "pass" or ingress.code is None:
                raise RuntimeError(
                    "Composer output failed Replay ingress: "
                    + json.dumps(ingress.to_check_response(), sort_keys=True)
                )
            compiler = TritonCompiler(
                timeout=120.0,
                cache_dir=workspace.tmp_dir / "target-normal-compile-cache",
            )
            if masked:
                compile_result = compiler.compile(
                    KernelSource.from_replay(ingress.code, function_name),
                    build_source_blind_replay_compile_case(artifact),
                )
                if not compile_result.ok:
                    raise RuntimeError(
                        "MaskJ bottom did not compile: "
                        + (compile_result.diagnostic or "unknown compiler error")
                    )
                compile_receipt: Mapping[str, Any] = {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "pass",
                    "diagnostics": [],
                    "mode": "masked_bottom_compile_only",
                }
            else:
                compile_check = source_blind_replay_compile_check(
                    ingress.code,
                    artifact,
                    function_name,
                    compiler=compiler,
                    packet=packet,
                )
                compile_receipt = compile_check.to_check_response()
                if compile_check.status != "pass":
                    raise RuntimeError(
                        "Composer output failed source-blind compile contract: "
                        + json.dumps(compile_receipt, sort_keys=True)
                    )

            trajectory = {
                "protocol": CODEX_CLI_AGENT_RUNTIME,
                "backend": CODEX_TARGET_NORMAL_REPLAY_BACKEND,
                "fresh_session": True,
                "thread_id": thread_id,
                "thread_ids": list(thread_ids),
                "turns": [turn.to_evidence() for turn in turns],
                "target_normal_checks": checks,
                "target_normal": checked.to_evidence(),
                "composer": dict(composition.receipt),
                "compile": dict(compile_receipt),
                "isolation": {
                    "source_blind": True,
                    "agent_input": [
                        "semantic-program.intent",
                        "frontend.json",
                        "TARGET_NORMAL_IR.md",
                        "TOOL.md",
                    ],
                    "agent_cannot_read": [
                        "pi",
                        "MaskC",
                        "Source",
                        "Harness",
                        "Gate",
                        "target_source",
                    ],
                    "composer_host_only": True,
                },
                "tool_bridge": bridge_receipt(),
            }
            raw = _trajectory_bytes(trajectory)
            receipt = _context_receipt({}, packet["u_f"])
            receipt.update(
                {
                    "backend_protocol": CODEX_TARGET_NORMAL_REPLAY_BACKEND,
                    "fresh_session": True,
                    "thread_id": thread_id,
                    "thread_ids": list(thread_ids),
                    "target_normal_bytes": len(
                        json.dumps(checked.authored_json(), sort_keys=True).encode("utf-8")
                    ),
                    "target_normal_checker_calls": len(checks),
                    "composer_profile": composition.receipt.get("profile"),
                    "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                }
            )
            generation = ReplayGeneration(
                code=ingress.code,
                function_name=function_name,
                model=provider.model,
                response_id=thread_id,
                context_receipt=receipt,
                usage=_usage(turns),
                response_model=provider.model,
                entry_function_name=ingress.entry_function_name,
                response_text=composition.source,
                raw_response=raw,
                attempts=len(turns),
                elapsed_ms=elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_TARGET_NORMAL_REPLAY_BACKEND,
                physical_provider_calls=sum(turn.model_segments for turn in turns),
                transport_retries=0,
            )
            self._records.append(generation)
            return generation
        except Exception as exc:
            cancelled = exc if isinstance(exc, CodexCLICancelled) else None
            if cancelled is not None:
                elapsed_ms += cancelled.elapsed_ms
            trajectory = {
                "protocol": CODEX_CLI_AGENT_RUNTIME,
                "backend": CODEX_TARGET_NORMAL_REPLAY_BACKEND,
                "fresh_session": True,
                "thread_id": thread_id,
                "thread_ids": list(thread_ids),
                "turns": [turn.to_evidence() for turn in turns],
                "target_normal_checks": checks,
                "failure_class": type(exc).__name__,
                "failure": str(exc),
                "isolation": {
                    "source_blind": True,
                    "composer_host_only": True,
                },
            }
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text="",
                response_id=thread_id,
                response_model=provider.model,
                usage=_usage(turns),
                raw_response=_trajectory_bytes(trajectory),
                attempts=len(turns),
                elapsed_ms=elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_TARGET_NORMAL_REPLAY_BACKEND,
                physical_provider_calls=sum(turn.model_segments for turn in turns)
                + (cancelled.physical_calls if cancelled is not None else 0),
                transport_retries=0,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        finally:
            if bridge is not None:
                bridge.close()
            if session is not None:
                session.close()
            workspace.cleanup()


__all__ = [
    "CODEX_TARGET_NORMAL_REPLAY_BACKEND",
    "CodexTargetNormalReplayWriter",
]
