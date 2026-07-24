"""Fail-closed filesystem sandbox for one IntentIR Lift invocation."""

from __future__ import annotations

import json
import os
import site
import stat
import shutil
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel

from minisweagent.exceptions import Submitted


INTENTIR_LIFT_SUBMIT = "INTENTIR_LIFT_SUBMIT"
INTENTIR_CANDIDATE_CHECK = "intentir_candidate_check"
CANDIDATE_CHECK_PROTOCOL = "intentir.scrc.candidate-check.v1"
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
MAX_ACTION_OUTPUT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 2048
MAX_WORKSPACE_DEPTH = 64
MAX_CANDIDATE_CHECK_FIELD_CHARS = 4096
MAX_CANDIDATE_CHECK_OBSERVED_ITEMS = 20
MAX_CANDIDATE_CHECK_RESULT_BYTES = 128 * 1024
_SANDBOX_START_MARKER = b"__INTENTIR_LIFT_SANDBOX_STARTED__\n"


def _remove_path_iterative(path: Path) -> None:
    """Remove one exact path tree without Python recursion."""

    try:
        root_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_stat.st_mode) or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    stack: list[tuple[Path, bool]] = [(path, False)]
    while stack:
        current, visited = stack.pop()
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(current_stat.st_mode) or current.is_symlink():
            current.unlink(missing_ok=True)
            continue
        os.chmod(current, stat.S_IRWXU)
        if visited:
            current.rmdir()
            continue
        stack.append((current, True))
        with os.scandir(current) as iterator:
            for entry in iterator:
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append((child, False))
                else:
                    child.unlink(missing_ok=True)


@dataclass
class IntentIRLiftWorkspace:
    """Private host directories projected into exactly one agent sandbox."""

    root: Path
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, files: Mapping[str, str | bytes]) -> "IntentIRLiftWorkspace":
        root = Path(tempfile.mkdtemp(prefix="intentir-lift-"))
        workspace = cls(
            root=root,
            input_dir=root / "input",
            output_dir=root / "output",
            scratch_dir=root / "work",
            tmp_dir=root / "tmp",
        )
        workspace.input_dir.mkdir()
        workspace.output_dir.mkdir()
        workspace.scratch_dir.mkdir()
        workspace.tmp_dir.mkdir()
        for relative, content in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                workspace.cleanup()
                raise ValueError(f"unsafe Lift input path: {relative!r}")
            destination = workspace.input_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                destination.write_bytes(content)
            else:
                destination.write_text(content, encoding="utf-8")
        return workspace

    @property
    def candidate_path(self) -> Path:
        return self.output_dir / "candidate.json"

    def read_candidate(self) -> str:
        entries = tuple(self.output_dir.iterdir())
        if entries != (self.candidate_path,):
            names = sorted(path.name for path in entries)
            raise ValueError(
                "Lift output must contain exactly candidate.json; "
                f"found {names}"
            )
        path = self.candidate_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("candidate.json must be a regular file")
        size = path.stat().st_size
        if not 1 <= size <= MAX_CANDIDATE_BYTES:
            raise ValueError(
                f"candidate.json size must be in [1, {MAX_CANDIDATE_BYTES}]"
            )
        raw = path.read_bytes()
        if len(raw) != size:
            raise ValueError("candidate.json changed while being read")
        text = raw.decode("utf-8")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError("candidate.json must contain one JSON object")
        return text

    def cleanup(self) -> None:
        _remove_path_iterative(self.root)


class IntentIRLiftEnvironmentConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path | None = None
    timeout: int = 60
    candidate_check_limit: int = 7
    candidate_submission_check_limit: int = 1
    executable: str = os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
    env: dict[str, str] = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
    }


class IntentIRLiftEnvironment:
    """Execute agent shell actions with only /input, /output, and /work visible."""

    def __init__(
        self,
        *,
        config_class: type[IntentIRLiftEnvironmentConfig] = IntentIRLiftEnvironmentConfig,
        candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        if self.config.candidate_check_limit < 0:
            raise ValueError("candidate_check_limit must be non-negative")
        if self.config.candidate_submission_check_limit < 0:
            raise ValueError(
                "candidate_submission_check_limit must be non-negative"
            )
        self._candidate_checker = candidate_checker
        self._candidate_check_calls = 0
        self._candidate_check_evaluations = 0
        self._candidate_check_calls_by_trigger = {
            "explicit": 0,
            "submission": 0,
        }
        self._candidate_check_evaluations_by_trigger = {
            "explicit": 0,
            "submission": 0,
        }
        self._candidate_check_shared_evaluations = 0
        self._candidate_check_submission_reserved_evaluations = 0
        self._candidate_check_audit: list[dict[str, Any]] = []
        self._candidate_check_passed_bytes: bytes | None = None
        self._validate_paths()
        self._probe()

    def _execute_candidate_check(
        self,
        arguments: list[str],
        *,
        trigger: str = "explicit",
    ) -> dict[str, Any]:
        if trigger not in {"explicit", "submission"}:
            raise ValueError("unsupported CandidateCheck trigger")
        self._candidate_check_calls += 1
        self._candidate_check_calls_by_trigger[trigger] += 1
        candidate_bytes: bytes | None = None
        budget_pool: str | None = None
        checker_evaluated = False
        if len(arguments) != 2 or arguments[1] not in {
            "/work/candidate.json", "/output/candidate.json",
        }:
            result: Mapping[str, Any] = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "invalid_candidate_path",
                        "path": "<candidate>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
        elif self._candidate_checker is None:
            result = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "candidate_check_unavailable",
                        "path": "<candidate>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
        else:
            candidate_path = (
                self.config.scratch_dir / "candidate.json"
                if arguments[1].startswith("/work/")
                else self.config.output_dir / "candidate.json"
            )
            try:
                if candidate_path.is_symlink() or not candidate_path.is_file():
                    raise ValueError("candidate must be a regular file")
                size = candidate_path.stat().st_size
                if not 1 <= size <= MAX_CANDIDATE_BYTES:
                    raise ValueError("candidate size is outside the allowed range")
                candidate_bytes = candidate_path.read_bytes()
                if len(candidate_bytes) != size:
                    raise ValueError("candidate changed while being read")
                candidate_text = candidate_bytes.decode("utf-8")
            except Exception as exc:
                result = {
                    "protocol": CANDIDATE_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "candidate_check_failed",
                            "path": "<candidate>",
                            "observed_refs_or_types": [type(exc).__name__],
                        }
                    ],
                }
            else:
                if (
                    self._candidate_check_shared_evaluations
                    < self.config.candidate_check_limit
                ):
                    self._candidate_check_shared_evaluations += 1
                    budget_pool = "shared"
                elif (
                    trigger == "submission"
                    and self._candidate_check_submission_reserved_evaluations
                    < self.config.candidate_submission_check_limit
                ):
                    self._candidate_check_submission_reserved_evaluations += 1
                    budget_pool = "submission_reserve"
                if budget_pool is None:
                    result = {
                        "protocol": CANDIDATE_CHECK_PROTOCOL,
                        "status": "fail",
                        "diagnostics": [
                            {
                                "rule_code": "check_budget_exhausted",
                                "path": "<candidate>",
                                "observed_refs_or_types": [],
                            }
                        ],
                    }
                else:
                    checker_evaluated = True
                    self._candidate_check_evaluations += 1
                    self._candidate_check_evaluations_by_trigger[trigger] += 1
                    try:
                        result = self._candidate_checker(candidate_text)
                    except Exception as exc:
                        result = {
                            "protocol": CANDIDATE_CHECK_PROTOCOL,
                            "status": "fail",
                            "diagnostics": [
                                {
                                    "rule_code": "candidate_check_failed",
                                    "path": "<candidate>",
                                    "observed_refs_or_types": [
                                        type(exc).__name__
                                    ],
                                }
                            ],
                        }
        protocol = result.get("protocol")
        status = result.get("status")
        diagnostics = result.get("diagnostics")
        if (
            set(result) != {"protocol", "status", "diagnostics"}
            or protocol != CANDIDATE_CHECK_PROTOCOL
            or status not in {"pass", "fail"}
            or not isinstance(diagnostics, list)
            or len(diagnostics) > 20
            or (status == "pass" and bool(diagnostics))
            or (status == "fail" and not diagnostics)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {
                    "rule_code",
                    "path",
                    "observed_refs_or_types",
                }
                or not isinstance(item.get("rule_code"), str)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("observed_refs_or_types"), list)
                or len(item.get("rule_code", "")) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                or len(item.get("path", "")) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                or len(item.get("observed_refs_or_types", []))
                > MAX_CANDIDATE_CHECK_OBSERVED_ITEMS
                or any(
                    not isinstance(observed, str)
                    or len(observed) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                    for observed in item.get("observed_refs_or_types", [])
                )
                for item in diagnostics
            )
            or len(
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
            > MAX_CANDIDATE_CHECK_RESULT_BYTES
        ):
            raise RuntimeError("candidate checker returned an invalid closed result")
        self._candidate_check_audit.append(
            {
                "protocol": protocol,
                "status": status,
                "trigger": trigger,
                "budget_pool": budget_pool,
                "evaluated": checker_evaluated,
                "rule_codes": [
                    item.get("rule_code")
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                ],
                "rule_paths": [
                    {
                        "rule_code": item.get("rule_code"),
                        "path": item.get("path"),
                    }
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                    and isinstance(item.get("path"), str)
                ],
            }
        )
        if status == "pass" and checker_evaluated and candidate_bytes is not None:
            self._candidate_check_passed_bytes = candidate_bytes
        elif (
            checker_evaluated
            and candidate_bytes == self._candidate_check_passed_bytes
        ):
            self._candidate_check_passed_bytes = None
        return {
            "output": json.dumps(result, sort_keys=True) + "\n",
            "returncode": 0 if status == "pass" else 1,
            "exception_info": "",
        }

    def _validate_paths(self) -> None:
        if self.config.tmp_dir is None:
            self.config.tmp_dir = self.config.scratch_dir.parent / "tmp"
        paths = (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        )
        resolved = tuple(path.resolve(strict=True) for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Lift input, output, and work directories must differ")
        if any(not path.is_dir() or path.is_symlink() for path in paths):
            raise ValueError("Lift sandbox paths must be real directories")
        (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ) = resolved

    def _base_command(self) -> list[str]:
        command = [
            self.config.executable,
            "--unshare-all",
            "--unshare-user-try",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            command.extend(["--ro-bind-try", system_path, system_path])
        for package_root in site.getsitepackages():
            intentir_package = (Path(package_root) / "intentir").resolve()
            if intentir_package.is_dir() and str(intentir_package).startswith(
                ("/usr/", "/lib/", "/lib64/")
            ):
                command.extend(
                    [
                        "--tmpfs",
                        str(intentir_package),
                        "--remount-ro",
                        str(intentir_package),
                    ]
                )
        command.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--ro-bind",
                str(self.config.input_dir),
                "/input",
                "--bind",
                str(self.config.output_dir),
                "/output",
                "--bind",
                str(self.config.scratch_dir),
                "/work",
                "--bind",
                str(self.config.tmp_dir),
                "/tmp",
                "--chdir",
                "/work",
                "--remount-ro",
                "/dev",
                "--remount-ro",
                "/",
            ]
        )
        for key, value in self.config.env.items():
            command.extend(["--setenv", key, value])
        return command

    def _probe(self) -> None:
        executable = shutil.which(self.config.executable)
        if executable is None:
            raise RuntimeError("IntentIR Lift requires bubblewrap (bwrap)")
        command = self._base_command() + [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                "python -c 'import importlib.util,sys; "
                "sys.exit(importlib.util.find_spec(\"intentir\") is not None)'"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            raise RuntimeError(f"bubblewrap capability probe failed: {exc}") from exc
        if result.returncode != 0:
            diagnostic = result.stdout.strip()[:500]
            raise RuntimeError(
                "bubblewrap isolation probe failed"
                + (f": {diagnostic}" if diagnostic else "")
            )

    def _writable_usage(self) -> tuple[int, int, int]:
        entries = 0
        size = 0
        max_depth = 0
        stack = [
            (self.config.output_dir, 0),
            (self.config.scratch_dir, 0),
            (self.config.tmp_dir, 0),
        ]
        while stack:
            directory, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if max_depth > MAX_WORKSPACE_DEPTH:
                return entries, size, max_depth
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > MAX_WORKSPACE_ENTRIES:
                        return entries, size, max_depth
                    item_stat = entry.stat(follow_symlinks=False)
                    size += max(item_stat.st_size, item_stat.st_blocks * 512)
                    if size > MAX_WORKSPACE_BYTES:
                        return entries, size, max_depth
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((Path(entry.path), depth + 1))
        return entries, size, max_depth

    def _reset_writable_dirs(self) -> None:
        for directory in (
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ):
            os.chmod(directory, stat.S_IRWXU)
            for child in directory.iterdir():
                _remove_path_iterative(child)

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        if cwd not in {"", "/work"}:
            raise ValueError("IntentIR Lift cwd is fixed at /work")
        command = action.get("command", "")
        if not isinstance(command, str):
            raise TypeError("Lift shell command must be a string")
        try:
            candidate_check_arguments = shlex.split(command)
        except ValueError:
            candidate_check_arguments = []
        if (
            candidate_check_arguments
            and candidate_check_arguments[0] == INTENTIR_CANDIDATE_CHECK
        ):
            return self._execute_candidate_check(candidate_check_arguments)
        bounded_shell = (
            f"printf '{_SANDBOX_START_MARKER.decode()}'; "
            "ulimit -t 60 -f 8192 -n 64 -u 128 -v 524288; "
            "set -o pipefail; "
            f"/bin/bash --noprofile --norc -c \"$1\" 2>&1 | "
            f"/usr/bin/head -c {MAX_ACTION_OUTPUT_BYTES}"
        )
        invocation = self._base_command() + [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            bounded_shell,
            "intentir-lift-action",
            command,
        ]
        action_timeout = timeout or self.config.timeout
        process: subprocess.Popen[bytes] | None = None
        limit_error = ""
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(
                    invocation,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                )
                started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started >= action_timeout:
                        limit_error = (
                            f"action exceeded {action_timeout} second timeout"
                        )
                        process.kill()
                        break
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = (
                            "workspace became unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        process.kill()
                        break
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = (
                            "workspace exceeded limit: "
                            f"entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                            f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                            f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                        )
                        process.kill()
                        break
                    time.sleep(0.05)
                returncode = process.wait(timeout=5)
                if not limit_error:
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = (
                            "workspace became unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        if (
                            entries > MAX_WORKSPACE_ENTRIES
                            or size > MAX_WORKSPACE_BYTES
                            or depth > MAX_WORKSPACE_DEPTH
                        ):
                            limit_error = (
                                "workspace exceeded limit: "
                                f"entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                                f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                                f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                            )
                output_file.seek(0)
                raw_output = output_file.read(
                    len(_SANDBOX_START_MARKER) + MAX_ACTION_OUTPUT_BYTES + 1
                )
            marker_at = raw_output.find(_SANDBOX_START_MARKER)
            if marker_at < 0:
                diagnostic = raw_output[:500].decode("utf-8", errors="replace")
                raise RuntimeError(
                    "bubblewrap action failed before sandbox start"
                    + (f": {diagnostic.strip()}" if diagnostic.strip() else "")
                )
            action_output = raw_output[marker_at + len(_SANDBOX_START_MARKER) :]
            output_text = action_output[:MAX_ACTION_OUTPUT_BYTES].decode(
                "utf-8", errors="replace"
            )
            output: dict[str, Any] = {
                "output": output_text,
                "returncode": returncode,
                "exception_info": limit_error,
            }
            if len(action_output) >= MAX_ACTION_OUTPUT_BYTES:
                output["returncode"] = returncode or 2
                output["exception_info"] = (
                    f"action output reached {MAX_ACTION_OUTPUT_BYTES} byte limit"
                )
            if limit_error:
                output["returncode"] = -1
                self._reset_writable_dirs()
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            raise RuntimeError(
                f"IntentIR Lift sandbox action failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._check_submission(output)
        return output

    def _check_submission(self, output: dict[str, Any]) -> None:
        lines = str(output.get("output", "")).lstrip().splitlines()
        if (
            not lines
            or lines[0].strip() != INTENTIR_LIFT_SUBMIT
            or output.get("returncode") != 0
        ):
            return
        workspace = IntentIRLiftWorkspace(
            root=self.config.input_dir.parent,
            input_dir=self.config.input_dir,
            output_dir=self.config.output_dir,
            scratch_dir=self.config.scratch_dir,
            tmp_dir=self.config.tmp_dir,
        )
        try:
            candidate_text = workspace.read_candidate()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected by the local syntax check: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return
        candidate_bytes = candidate_text.encode("utf-8")
        if self._candidate_checker is not None:
            if candidate_bytes != self._candidate_check_passed_bytes:
                preflight = self._execute_candidate_check(
                    [INTENTIR_CANDIDATE_CHECK, "/output/candidate.json"],
                    trigger="submission",
                )
                if preflight["returncode"] != 0:
                    output["returncode"] = 2
                    output["output"] += (
                        "\nSubmission preflight rejected the current candidate. "
                        "Edit the complete object and submit again.\n"
                        + str(preflight["output"])
                    )
                    return
        try:
            final_candidate_text = workspace.read_candidate()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected while confirming the checked bytes: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return
        if final_candidate_text.encode("utf-8") != candidate_bytes:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected because candidate.json changed after "
                "the exact bytes were checked. Recheck and submit again.\n"
            )
            return
        raise Submitted(
            {
                "role": "exit",
                "content": "candidate.json submitted",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "candidate.json",
                },
            }
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"system": "Linux", **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "sandbox": "intentir-bubblewrap-v1",
                        "network": "unshared",
                        "input": "read-only",
                        "output": "candidate-only",
                        "command_timeout_seconds": self.config.timeout,
                        "candidate_check_shared_limit": (
                            self.config.candidate_check_limit
                        ),
                        "candidate_check_submission_reserve": (
                            self.config.candidate_submission_check_limit
                        ),
                    },
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                },
                "candidate_check": {
                    "calls": self._candidate_check_calls,
                    "attempts": self._candidate_check_calls,
                    "evaluations": self._candidate_check_evaluations,
                    "limit": (
                        self.config.candidate_check_limit
                        + self.config.candidate_submission_check_limit
                    ),
                    "explicit": {
                        "calls": self._candidate_check_calls_by_trigger[
                            "explicit"
                        ],
                        "evaluations": (
                            self._candidate_check_evaluations_by_trigger[
                                "explicit"
                            ]
                        ),
                    },
                    "submission": {
                        "calls": self._candidate_check_calls_by_trigger[
                            "submission"
                        ],
                        "evaluations": (
                            self._candidate_check_evaluations_by_trigger[
                                "submission"
                            ]
                        ),
                    },
                    "shared_budget": {
                        "used": self._candidate_check_shared_evaluations,
                        "limit": self.config.candidate_check_limit,
                    },
                    "submission_reserve": {
                        "used": (
                            self._candidate_check_submission_reserved_evaluations
                        ),
                        "limit": self.config.candidate_submission_check_limit,
                    },
                    "results": list(self._candidate_check_audit),
                },
            }
        }


__all__ = [
    "CANDIDATE_CHECK_PROTOCOL",
    "INTENTIR_LIFT_SUBMIT",
    "INTENTIR_CANDIDATE_CHECK",
    "IntentIRLiftEnvironment",
    "IntentIRLiftEnvironmentConfig",
    "IntentIRLiftWorkspace",
    "MAX_ACTION_OUTPUT_BYTES",
    "MAX_CANDIDATE_BYTES",
    "MAX_WORKSPACE_BYTES",
    "MAX_WORKSPACE_DEPTH",
    "MAX_WORKSPACE_ENTRIES",
]
