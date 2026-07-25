"""Fail-closed sandbox for one source-blind IntentIR Replay episode."""

from __future__ import annotations

import json
import os
import shlex
import stat
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict

from minisweagent.exceptions import Submitted


INTENTIR_REPLAY_SUBMIT = "INTENTIR_REPLAY_SUBMIT"
INTENTIR_REPLAY_CHECK = "intentir_replay_check"
INTENTIR_PACKET_QUERY = "intentir_packet_query"
REPLAY_CHECK_PROTOCOL = "intentir.scrc.replay-check.v1"
REPLAY_INPUT_FILES = frozenset(
    {"README.md", "packet.json", "REPLAY_CONTRACT.md", "TOOL.md"}
)
REPLAY_SANDBOX_PROTOCOL = "intentir.replay-busybox-empty-root.v1"
REPLAY_SANDBOX_HOSTNAME = "intentir-replay"
REPLAY_BWRAP = Path("/usr/bin/bwrap")
REPLAY_BUSYBOX = Path("/usr/bin/busybox")
REPLAY_TOOL_APPLETS = (
    "awk",
    "basename",
    "cat",
    "cmp",
    "cp",
    "cut",
    "dd",
    "diff",
    "dirname",
    "echo",
    "env",
    "expr",
    "false",
    "find",
    "grep",
    "head",
    "hexdump",
    "hostname",
    "ln",
    "ls",
    "mkdir",
    "mktemp",
    "mv",
    "od",
    "printf",
    "pwd",
    "readlink",
    "realpath",
    "rm",
    "rmdir",
    "sed",
    "seq",
    "sh",
    "sha256sum",
    "sleep",
    "sort",
    "stat",
    "tail",
    "tee",
    "test",
    "timeout",
    "touch",
    "tr",
    "true",
    "uniq",
    "wc",
    "which",
    "xargs",
)
MAX_REPLAY_BYTES = 1024 * 1024
MAX_ACTION_OUTPUT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 2048
MAX_WORKSPACE_DEPTH = 64
MAX_CHECK_FIELD_CHARS = 4096
MAX_CHECK_HANDLES = 32
MAX_CHECK_RESULT_BYTES = 128 * 1024
MAX_PACKET_QUERY_OUTPUT_BYTES = 10_000
_SANDBOX_START_MARKER = b"__INTENTIR_REPLAY_SANDBOX_STARTED__\n"


def _packet_json_chars(value: Any) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _packet_path(value: Any, path: str) -> Any:
    current = value
    if path in {"", "."}:
        return current
    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                raise KeyError(
                    f"unknown path component {component!r}; "
                    f"choose {list(current)}"
                )
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                raise IndexError(
                    f"list index {index} out of range 0..{len(current) - 1}"
                )
            current = current[index]
        else:
            raise KeyError(f"cannot descend through {component!r}")
    return current


def _packet_children(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return {}
    result = {
        str(key): {
            "type": type(item).__name__,
            "json_chars": _packet_json_chars(item),
            "query": f"{INTENTIR_PACKET_QUERY} section {path}.{key}".replace(
                "..", "."
            ),
        }
        for key, item in list(items)[:96]
    }
    total = len(value)
    if total > len(result):
        result["<omitted>"] = {"count": total - len(result)}
    return result


def _packet_ops(body: Any, prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(body, list):
        return result
    for index, operation in enumerate(body):
        if not isinstance(operation, Mapping):
            continue
        path = f"{prefix}.{index}"
        result.append(
            {
                "path": path,
                "op": operation.get("op"),
                "result": operation.get("result"),
                "operands": operation.get("operands"),
                "scope_id": operation.get("scope_id"),
            }
        )
        result.extend(_packet_ops(operation.get("body"), f"{path}.body"))
        for arm in ("positive", "negative"):
            branch = operation.get(arm)
            if isinstance(branch, Mapping):
                result.extend(
                    _packet_ops(branch.get("body"), f"{path}.{arm}.body")
                )
    return result


def _packet_overview(packet: Mapping[str, Any]) -> dict[str, Any]:
    j = packet.get("j", {})
    intent = j.get("intent", {}) if isinstance(j, Mapping) else {}
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    boundary = j.get("boundary", {}) if isinstance(j, Mapping) else {}
    pi = packet.get("pi", {})
    c = packet.get("c", {})
    u_f = packet.get("u_f", {})
    signature = j.get("signature", {}) if isinstance(j, Mapping) else {}
    operations = _packet_ops(
        function.get("body", []) if isinstance(function, Mapping) else [],
        "j.intent.intent_function.body",
    )
    return {
        "root_sections": {
            key: {
                "json_chars": _packet_json_chars(value),
                "query": f"{INTENTIR_PACKET_QUERY} section {key}",
            }
            for key, value in packet.items()
        },
        "entry": u_f.get("replay_function") if isinstance(u_f, Mapping) else None,
        "frontend": u_f.get("frontend") if isinstance(u_f, Mapping) else None,
        "j_parameters": [
            {
                key: parameter.get(key)
                for key in ("name", "kind", "mode", "dtype", "rank")
            }
            for parameter in signature.get("parameters", [])
            if isinstance(parameter, Mapping)
        ],
        "operation_count": len(operations),
        "operations": operations[:32],
        "operations_omitted": max(0, len(operations) - 32),
        "boundary_counts": {
            key: len(value)
            for key, value in boundary.items()
            if isinstance(value, list)
        }
        if isinstance(boundary, Mapping)
        else {},
        "pi_counts": {
            key: len(value)
            for key, value in pi.items()
            if isinstance(value, list)
        }
        if isinstance(pi, Mapping)
        else {},
        "c_slots": c.get("slots", []) if isinstance(c, Mapping) else [],
        "next": [
            f"Query `{INTENTIR_PACKET_QUERY} section j`, `section pi`, "
            "`section c`, and `section u_f` separately.",
            "A large section returns child paths; query one of those paths.",
            f"Use `{INTENTIR_PACKET_QUERY} find HANDLE` to follow references.",
        ],
    }


def _packet_find(packet: Mapping[str, Any], needle: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if len(matches) >= 96:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        elif needle.casefold() in str(value).casefold():
            visible = value
            if isinstance(value, str) and len(value) > 512:
                visible = value[:512] + "...[truncated]"
            matches.append({"path": path, "value": visible})

    visit(packet, "")
    return {
        "query": needle,
        "match_count_shown": len(matches),
        "matches": matches,
    }


def _query_packet(packet: Mapping[str, Any], arguments: list[str]) -> Any:
    if arguments == [INTENTIR_PACKET_QUERY, "overview"]:
        return _packet_overview(packet)
    if len(arguments) == 3 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "section",
    ]:
        path = arguments[2]
        value = _packet_path(packet, path)
        rendered = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False
        )
        if len(rendered) <= MAX_PACKET_QUERY_OUTPUT_BYTES:
            return value
        return {
            "path": path,
            "json_chars": _packet_json_chars(value),
            "children": _packet_children(value, path),
        }
    if (
        len(arguments) == 3
        and arguments[:2] == [INTENTIR_PACKET_QUERY, "find"]
        and 0 < len(arguments[2]) <= 128
    ):
        return _packet_find(packet, arguments[2])
    raise ValueError(
        "usage: intentir_packet_query overview | section PATH | find HANDLE"
    )


def _remove_path_iterative(path: Path) -> None:
    """Remove one exact private workspace tree without Python recursion."""

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


def _closed_failure(rule_code: str, *, path: str = "<replay>") -> dict[str, Any]:
    return {
        "protocol": REPLAY_CHECK_PROTOCOL,
        "status": "fail",
        "diagnostics": [
            {
                "rule_code": rule_code,
                "path": path,
                "anonymous_handles": [],
                "line": None,
            }
        ],
    }


def _is_static_elf(path: Path) -> bool:
    """Return true only for an ELF executable without a PT_INTERP header."""

    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 52 or header[:4] != b"\x7fELF":
            return False
        elf_class = header[4]
        byte_order = header[5]
        endian = "<" if byte_order == 1 else ">" if byte_order == 2 else ""
        if not endian:
            return False
        if elf_class == 2:
            program_offset = struct.unpack_from(endian + "Q", header, 32)[0]
            entry_size = struct.unpack_from(endian + "H", header, 54)[0]
            entry_count = struct.unpack_from(endian + "H", header, 56)[0]
        elif elf_class == 1:
            program_offset = struct.unpack_from(endian + "I", header, 28)[0]
            entry_size = struct.unpack_from(endian + "H", header, 42)[0]
            entry_count = struct.unpack_from(endian + "H", header, 44)[0]
        else:
            return False
        if entry_size < 4 or entry_count > 4096:
            return False
        for index in range(entry_count):
            stream.seek(program_offset + index * entry_size)
            entry = stream.read(entry_size)
            if len(entry) != entry_size:
                return False
            if struct.unpack_from(endian + "I", entry)[0] == 3:
                return False
    return True


@dataclass
class IntentIRReplayWorkspace:
    """Fresh host directories projected into exactly one Replay sandbox."""

    root: Path
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, files: Mapping[str, str | bytes]) -> "IntentIRReplayWorkspace":
        if set(files) != REPLAY_INPUT_FILES:
            raise ValueError(
                "Replay input must contain exactly the fixed allowlisted files"
            )
        root = Path(tempfile.mkdtemp(prefix="intentir-replay-"))
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
            for relative, content in files.items():
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError(f"unsafe Replay input path: {relative!r}")
                destination = workspace.input_dir / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    destination.write_bytes(content)
                else:
                    destination.write_text(content, encoding="utf-8")
                destination.chmod(stat.S_IRUSR)
        except Exception:
            workspace.cleanup()
            raise
        return workspace

    @property
    def replay_path(self) -> Path:
        return self.output_dir / "replay.source"

    def read_replay(self) -> str:
        entries = tuple(self.output_dir.iterdir())
        if entries != (self.replay_path,):
            names = sorted(path.name for path in entries)
            raise ValueError(
                "Replay output must contain exactly replay.source; "
                f"found {names}"
            )
        if self.replay_path.is_symlink() or not self.replay_path.is_file():
            raise ValueError("replay.source must be a regular file")
        size = self.replay_path.stat().st_size
        if not 1 <= size <= MAX_REPLAY_BYTES:
            raise ValueError(
                f"replay.source size must be in [1, {MAX_REPLAY_BYTES}]"
            )
        raw = self.replay_path.read_bytes()
        if len(raw) != size:
            raise ValueError("replay.source changed while being read")
        return raw.decode("utf-8")

    def cleanup(self) -> None:
        _remove_path_iterative(self.root)


class IntentIRReplayEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path | None = None
    timeout: int = 60
    replay_check_limit: int = 7
    replay_submission_check_limit: int = 1


class IntentIRReplayEnvironment:
    """Run shell actions without Source, host code, network, or compiler access."""

    def __init__(
        self,
        *,
        config_class: type[IntentIRReplayEnvironmentConfig] = IntentIRReplayEnvironmentConfig,
        replay_checker: Callable[[str], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        if self.config.replay_check_limit < 0:
            raise ValueError("replay_check_limit must be non-negative")
        if self.config.replay_submission_check_limit < 0:
            raise ValueError("replay_submission_check_limit must be non-negative")
        self._replay_checker = replay_checker
        self._check_calls = 0
        self._check_evaluations = 0
        self._calls_by_trigger = {"explicit": 0, "submission": 0}
        self._evaluations_by_trigger = {"explicit": 0, "submission": 0}
        self._shared_evaluations = 0
        self._submission_reserved_evaluations = 0
        self._check_audit: list[dict[str, Any]] = []
        self._passed_bytes: bytes | None = None
        self._validate_paths()
        self._probe()

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
            raise ValueError("Replay input, output, work, and tmp directories must differ")
        if any(not path.is_dir() or path.is_symlink() for path in paths):
            raise ValueError("Replay sandbox paths must be real directories")
        (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ) = resolved

    def _base_command(self) -> list[str]:
        command = [
            str(REPLAY_BWRAP),
            "--unshare-all",
            "--unshare-user-try",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--hostname",
            REPLAY_SANDBOX_HOSTNAME,
            "--tmpfs",
            "/",
            "--dir",
            "/bin",
        ]
        # Only these entrypoint names are mounted. BusyBox remains a multicall
        # binary and could be copied/renamed inside a writable bind; the hard
        # boundary is therefore the empty rootfs, absent host tools/data, clear
        # environment, and new network/PID namespaces—not an applet claim.
        for applet in REPLAY_TOOL_APPLETS:
            command.extend(
                ["--ro-bind", str(REPLAY_BUSYBOX), f"/bin/{applet}"]
            )
        command.extend(
            [
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
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "LANG",
                "C",
                "--setenv",
                "LC_ALL",
                "C",
                "--setenv",
                "PATH",
                "/bin",
                "--setenv",
                "TMPDIR",
                "/tmp",
            ]
        )
        return command

    def _probe(self) -> None:
        if (
            REPLAY_BWRAP.is_symlink()
            or not REPLAY_BWRAP.is_file()
            or REPLAY_BUSYBOX.is_symlink()
            or not REPLAY_BUSYBOX.is_file()
            or not _is_static_elf(REPLAY_BUSYBOX)
        ):
            raise RuntimeError("IntentIR Replay requires static /usr/bin/busybox")
        busybox_applets = set(
            subprocess.run(
                [str(REPLAY_BUSYBOX), "--list"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
                timeout=10,
            ).stdout.splitlines()
        )
        missing_applets = sorted(set(REPLAY_TOOL_APPLETS) - busybox_applets)
        if missing_applets:
            raise RuntimeError(
                "IntentIR Replay BusyBox lacks required entrypoints: "
                + ", ".join(missing_applets)
            )
        command = self._base_command() + [
            "/bin/sh",
            "-c",
            (
                "test \"$(hostname)\" "
                f"= {REPLAY_SANDBOX_HOSTNAME}; "
                "test ! -e /proc; "
                "test ! -e /usr; test ! -e /etc; test ! -e /lib; "
                "test ! -e /lib64; test ! -e /usr/local; "
                "for name in python python3 gcc g++ clang ld as nvcc ptxas "
                "triton torch intentir; do ! command -v \"$name\"; done; "
                "test \"$(find /input -type f | wc -l)\" -eq 4; "
                "test \"$(find /input -type f | sed 's#^/input/##' | sort | "
                "sha256sum | cut -d' ' -f1)\" = "
                "$(printf '%s\\n' README.md REPLAY_CONTRACT.md TOOL.md "
                "packet.json | sort | sha256sum | cut -d' ' -f1)"
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
                "bubblewrap Replay isolation probe failed"
                + (f": {diagnostic}" if diagnostic else "")
            )
        namespace_probe = subprocess.Popen(
            self._base_command()
            + [
                "/bin/sh",
                "-c",
                "printf 'ready\\n'; sleep 2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if namespace_probe.stdout is None:
                raise RuntimeError("Replay network namespace probe has no output")
            if namespace_probe.stdout.readline().strip() != "ready":
                raise RuntimeError("Replay network namespace probe did not start")
            host_namespace = os.readlink("/proc/self/ns/net")
            pending = [namespace_probe.pid]
            observed_namespaces: set[str] = set()
            while pending:
                pid = pending.pop()
                try:
                    observed_namespaces.add(
                        os.readlink(f"/proc/{pid}/ns/net")
                    )
                    children = Path(
                        f"/proc/{pid}/task/{pid}/children"
                    ).read_text(encoding="utf-8")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                pending.extend(int(item) for item in children.split())
            if not (observed_namespaces - {host_namespace}):
                raise RuntimeError("Replay sandbox did not unshare the network")
            self._network_namespace_probed = True
        finally:
            namespace_probe.terminate()
            try:
                namespace_probe.wait(timeout=5)
            except subprocess.TimeoutExpired:
                namespace_probe.kill()
                namespace_probe.wait(timeout=5)

    def _read_check_target(self, path: str) -> tuple[bytes, str]:
        target = (
            self.config.scratch_dir / "replay.source"
            if path == "/work/replay.source"
            else self.config.output_dir / "replay.source"
        )
        if target.is_symlink() or not target.is_file():
            raise ValueError("Replay check target must be a regular file")
        size = target.stat().st_size
        if not 1 <= size <= MAX_REPLAY_BYTES:
            raise ValueError("Replay check target is outside the size bound")
        raw = target.read_bytes()
        if len(raw) != size:
            raise ValueError("Replay check target changed while being read")
        return raw, raw.decode("utf-8")

    @staticmethod
    def _validate_check_result(result: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise RuntimeError("Replay checker returned a non-object result")
        value = dict(result)
        protocol = value.get("protocol")
        status = value.get("status")
        diagnostics = value.get("diagnostics")
        valid = (
            set(value) == {"protocol", "status", "diagnostics"}
            and protocol == REPLAY_CHECK_PROTOCOL
            and status in {"pass", "fail"}
            and isinstance(diagnostics, list)
            and len(diagnostics) <= 20
            and ((status == "pass" and not diagnostics) or (status == "fail" and bool(diagnostics)))
        )
        if valid:
            for item in diagnostics:
                if not isinstance(item, Mapping) or set(item) != {
                    "rule_code", "path", "anonymous_handles", "line"
                }:
                    valid = False
                    break
                handles = item.get("anonymous_handles")
                line = item.get("line")
                if (
                    not isinstance(item.get("rule_code"), str)
                    or not isinstance(item.get("path"), str)
                    or len(item["rule_code"]) > MAX_CHECK_FIELD_CHARS
                    or len(item["path"]) > MAX_CHECK_FIELD_CHARS
                    or not isinstance(handles, list)
                    or len(handles) > MAX_CHECK_HANDLES
                    or any(
                        not isinstance(handle, str)
                        or len(handle) > MAX_CHECK_FIELD_CHARS
                        for handle in handles
                    )
                    or (
                        line is not None
                        and (
                            not isinstance(line, int)
                            or isinstance(line, bool)
                            or line < 1
                        )
                    )
                ):
                    valid = False
                    break
        if valid and len(json.dumps(value, sort_keys=True).encode()) > MAX_CHECK_RESULT_BYTES:
            valid = False
        if not valid:
            raise RuntimeError("Replay checker returned an invalid closed result")
        return value

    def _execute_replay_check(
        self,
        arguments: list[str],
        *,
        trigger: str = "explicit",
    ) -> dict[str, Any]:
        if trigger not in {"explicit", "submission"}:
            raise ValueError("unsupported ReplayCheck trigger")
        self._check_calls += 1
        self._calls_by_trigger[trigger] += 1
        source_bytes: bytes | None = None
        budget_pool: str | None = None
        evaluated = False
        if len(arguments) != 2 or arguments[1] not in {
            "/work/replay.source", "/output/replay.source"
        }:
            result = _closed_failure("invalid_replay_path")
        elif self._replay_checker is None:
            result = _closed_failure("replay_check_unavailable")
        else:
            try:
                source_bytes, source_text = self._read_check_target(arguments[1])
            except Exception:
                result = _closed_failure("replay_check_read_failed")
            else:
                if self._shared_evaluations < self.config.replay_check_limit:
                    self._shared_evaluations += 1
                    budget_pool = "shared"
                elif (
                    trigger == "submission"
                    and self._submission_reserved_evaluations
                    < self.config.replay_submission_check_limit
                ):
                    self._submission_reserved_evaluations += 1
                    budget_pool = "submission_reserve"
                if budget_pool is None:
                    result = _closed_failure("check_budget_exhausted")
                else:
                    evaluated = True
                    self._check_evaluations += 1
                    self._evaluations_by_trigger[trigger] += 1
                    try:
                        result = self._replay_checker(source_text)
                    except Exception:
                        result = _closed_failure("replay_check_internal_failure")
        result = self._validate_check_result(result)
        diagnostics = result["diagnostics"]
        self._check_audit.append(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "trigger": trigger,
                "budget_pool": budget_pool,
                "evaluated": evaluated,
                "rule_paths": [
                    {"rule_code": item["rule_code"], "path": item["path"]}
                    for item in diagnostics
                ],
            }
        )
        if result["status"] == "pass" and evaluated and source_bytes is not None:
            self._passed_bytes = source_bytes
        elif evaluated and source_bytes == self._passed_bytes:
            self._passed_bytes = None
        return {
            "output": json.dumps(result, sort_keys=True) + "\n",
            "returncode": 0 if result["status"] == "pass" else 1,
            "exception_info": "",
        }

    def _execute_packet_query(self, arguments: list[str]) -> dict[str, Any]:
        try:
            packet = json.loads(
                (self.config.input_dir / "packet.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(packet, Mapping) or set(packet) != {
                "j",
                "c",
                "pi",
                "u_f",
            }:
                raise ValueError("Replay packet has an invalid closed root")
            value = _query_packet(packet, arguments)
            rendered = json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False
            )
            if len(rendered.encode("utf-8")) > MAX_PACKET_QUERY_OUTPUT_BYTES:
                raise ValueError("query result is too large; use a narrower path")
        except Exception as exc:
            return {
                "output": f"packet query rejected: {type(exc).__name__}: {exc}\n",
                "returncode": 1,
                "exception_info": "",
            }
        return {
            "output": rendered + "\n",
            "returncode": 0,
            "exception_info": "",
        }

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
        for directory in (self.config.output_dir, self.config.scratch_dir, self.config.tmp_dir):
            os.chmod(directory, stat.S_IRWXU)
            for child in directory.iterdir():
                _remove_path_iterative(child)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        if cwd not in {"", "/work"}:
            raise ValueError("IntentIR Replay cwd is fixed at /work")
        command = action.get("command", "")
        if not isinstance(command, str):
            raise TypeError("Replay shell command must be a string")
        try:
            check_arguments = shlex.split(command)
        except ValueError:
            check_arguments = []
        if check_arguments and check_arguments[0] == INTENTIR_PACKET_QUERY:
            return self._execute_packet_query(check_arguments)
        if check_arguments and check_arguments[0] == INTENTIR_REPLAY_CHECK:
            return self._execute_replay_check(check_arguments)
        bounded_shell = (
            f"printf '{_SANDBOX_START_MARKER.decode()}'; "
            "ulimit -t 60 -f 8192 -n 64 -p 128 -v 524288; "
            "set -o pipefail; "
            f"/bin/sh -c \"$1\" 2>&1 | "
            f"/bin/head -c {MAX_ACTION_OUTPUT_BYTES}"
        )
        invocation = self._base_command() + [
            "/bin/sh",
            "-c",
            bounded_shell,
            "intentir-replay-action",
            command,
        ]
        process: subprocess.Popen[bytes] | None = None
        limit_error = ""
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(invocation, stdout=output_file, stderr=subprocess.STDOUT)
                started = time.monotonic()
                action_timeout = timeout or self.config.timeout
                while process.poll() is None:
                    if time.monotonic() - started >= action_timeout:
                        limit_error = f"action exceeded {action_timeout} second timeout"
                        process.kill()
                        break
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = f"workspace became unreadable: {type(exc).__name__}"
                        process.kill()
                        break
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = (
                            f"workspace exceeded limit: entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                            f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                            f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                        )
                        process.kill()
                        break
                    time.sleep(0.05)
                returncode = process.wait(timeout=5)
                if not limit_error:
                    entries, size, depth = self._writable_usage()
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = "workspace exceeded limit after action"
                output_file.seek(0)
                raw_output = output_file.read(
                    len(_SANDBOX_START_MARKER) + MAX_ACTION_OUTPUT_BYTES + 1
                )
            marker_at = raw_output.find(_SANDBOX_START_MARKER)
            if marker_at < 0:
                raise RuntimeError("bubblewrap action failed before Replay sandbox start")
            action_output = raw_output[marker_at + len(_SANDBOX_START_MARKER):]
            output = {
                "output": action_output[:MAX_ACTION_OUTPUT_BYTES].decode("utf-8", errors="replace"),
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
                f"IntentIR Replay sandbox action failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._check_submission(output)
        return output

    def _check_submission(self, output: dict[str, Any]) -> None:
        lines = str(output.get("output", "")).lstrip().splitlines()
        if not lines or lines[0].strip() != INTENTIR_REPLAY_SUBMIT or output.get("returncode") != 0:
            return
        workspace = IntentIRReplayWorkspace(
            root=self.config.input_dir.parent,
            input_dir=self.config.input_dir,
            output_dir=self.config.output_dir,
            scratch_dir=self.config.scratch_dir,
            tmp_dir=self.config.tmp_dir,
        )
        try:
            source_text = workspace.read_replay()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += f"\nSubmission rejected by output policy: {type(exc).__name__}\n"
            return
        source_bytes = source_text.encode("utf-8")
        if self._replay_checker is not None and source_bytes != self._passed_bytes:
            preflight = self._execute_replay_check(
                [INTENTIR_REPLAY_CHECK, "/output/replay.source"],
                trigger="submission",
            )
            if preflight["returncode"] != 0:
                output["returncode"] = 2
                output["output"] += (
                    "\nSubmission preflight rejected the current Replay source. "
                    "Edit it and submit again.\n" + str(preflight["output"])
                )
                return
        try:
            final_source = workspace.read_replay()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += f"\nSubmission changed after checking: {type(exc).__name__}\n"
            return
        if final_source.encode("utf-8") != source_bytes:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected because replay.source changed after "
                "the exact bytes were checked.\n"
            )
            return
        raise Submitted(
            {
                "role": "exit",
                "content": "replay.source submitted",
                "extra": {"exit_status": "Submitted", "submission": "replay.source"},
            }
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"system": "Linux", **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "sandbox": REPLAY_SANDBOX_PROTOCOL,
                        "hostname": REPLAY_SANDBOX_HOSTNAME,
                        "network": "new-unshared-namespace-host-probed",
                        "proc": "not-mounted",
                        "input": "read-only-allowlist",
                        "output": "replay-source-only",
                        "mounted_busybox_entrypoints": list(
                            REPLAY_TOOL_APPLETS
                        ),
                        "replay_check_shared_limit": self.config.replay_check_limit,
                        "replay_check_submission_reserve": (
                            self.config.replay_submission_check_limit
                        ),
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "replay_check": {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "calls": self._check_calls,
                    "evaluations": self._check_evaluations,
                    "limit": (
                        self.config.replay_check_limit
                        + self.config.replay_submission_check_limit
                    ),
                    "explicit": {
                        "calls": self._calls_by_trigger["explicit"],
                        "evaluations": self._evaluations_by_trigger["explicit"],
                    },
                    "submission": {
                        "calls": self._calls_by_trigger["submission"],
                        "evaluations": self._evaluations_by_trigger["submission"],
                    },
                    "shared_budget": {
                        "used": self._shared_evaluations,
                        "limit": self.config.replay_check_limit,
                    },
                    "submission_reserve": {
                        "used": self._submission_reserved_evaluations,
                        "limit": self.config.replay_submission_check_limit,
                    },
                    "results": list(self._check_audit),
                },
            }
        }


__all__ = [
    "INTENTIR_PACKET_QUERY",
    "INTENTIR_REPLAY_CHECK",
    "INTENTIR_REPLAY_SUBMIT",
    "REPLAY_CHECK_PROTOCOL",
    "REPLAY_INPUT_FILES",
    "IntentIRReplayEnvironment",
    "IntentIRReplayEnvironmentConfig",
    "IntentIRReplayWorkspace",
    "MAX_ACTION_OUTPUT_BYTES",
    "MAX_REPLAY_BYTES",
]
