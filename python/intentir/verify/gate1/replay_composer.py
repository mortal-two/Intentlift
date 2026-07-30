#!/usr/bin/env python3
"""Episode-local, syntax-only source composition for source-blind Replay.

The composer deliberately knows nothing about IntentIR artifacts.  It receives
only model-authored syntax nodes, renders them with a Python-like or C-like
layout profile, and writes the model's own Replay bytes.  It is both importable
by the host runtime and executable as a self-contained helper copied into one
fresh Replay workspace.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPLAY_COMPOSER_PROTOCOL = "intentir.replay.syntax-composer.v2"
REPLAY_COMPOSER_STATE_PROTOCOL = "intentir.replay.syntax-composer-state.v2"
REPLAY_COMPOSER_AUDIT_PROTOCOL = "intentir.replay.syntax-composer-audit.v2"
REPLAY_COMPOSER_TOOL = "replay_composer.py"
REPLAY_COMPOSER_BUILDER = "build_replay.py"
REPLAY_COMPOSER_PLAN = "replay.composer.json"
REPLAY_COMPOSER_NODE_DRAFT = "replay.composer.node.json"
REPLAY_COMPOSER_STATE = "replay.composer.state.json"
REPLAY_COMPOSER_SOURCE_MAP = "replay.composer.source-map.json"
REPLAY_COMPOSER_RENDERED = "replay.composer.rendered"
REPLAY_COMPOSER_AUDIT = "replay.composer.audit.jsonl"
REPLAY_COMPOSER_CHECKPOINT_PLAN = "replay.composer.checkpoint.json"
REPLAY_COMPOSER_CHECKPOINT_SOURCE = "replay.composer.checkpoint.source"
REPLAY_COMPOSER_CHECKPOINT_MAP = "replay.composer.checkpoint.source-map.json"

MAX_COMPOSER_PLAN_BYTES = 2 * 1024 * 1024
MAX_COMPOSER_RENDER_BYTES = 1024 * 1024
MAX_COMPOSER_NODES = 4096
MAX_COMPOSER_DEPTH = 64
MAX_COMPOSER_LINE_CHARS = 16 * 1024
MAX_COMPOSER_RAW_LINES = 65536
MAX_COMPOSER_AUDIT_ROWS = 4096
MAX_COMPOSER_AUDIT_BYTES = 4 * 1024 * 1024

_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
_FRONTEND_PROFILE = {
    "triton": "python",
    "tilelang": "python",
    "cuda": "c_like",
}
REPLAY_COMPOSER_MODES = frozenset({"off", "available", "preferred"})
_EDIT_OPERATIONS = frozenset(
    {
        "insert_before",
        "insert_after",
        "append_child",
        "replace",
        "delete",
        "move_before",
        "move_after",
        "move_child",
    }
)


class ReplayComposerError(ValueError):
    """A closed syntax-plan or episode-state violation."""


@dataclass(frozen=True)
class RenderedReplay:
    source: str
    source_map: Mapping[str, Mapping[str, Any]]
    node_count: int
    node_kinds: Mapping[str, int]
    frontend: str
    layout_profile: str


def line(node_id: str, text: str) -> dict[str, Any]:
    """Create one generic agent-authored source line node."""

    return {"id": node_id, "kind": "line", "text": text}


def raw_block(node_id: str, lines: Sequence[str]) -> dict[str, Any]:
    """Create an exact multi-line syntax escape without interpreting its text."""

    if isinstance(lines, (str, bytes)):
        raise ReplayComposerError("raw_block lines must be a sequence of lines")
    return {"id": node_id, "kind": "raw_block", "lines": list(lines)}


def block(
    node_id: str,
    header: str,
    children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one generic indentation/braced block node."""

    if isinstance(children, (str, bytes)):
        raise ReplayComposerError("block children must be a sequence of nodes")
    return {
        "id": node_id,
        "kind": "block",
        "header": header,
        "children": [dict(child) for child in children],
    }


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReplayComposerError(
            f"{path} keys differ from the closed grammar "
            f"(missing={missing}, extra={extra})"
        )


def _line_text(value: Any, path: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise ReplayComposerError(f"{path} must be a string")
    if (not allow_empty and not value) or len(value) > MAX_COMPOSER_LINE_CHARS:
        raise ReplayComposerError(f"{path} length is outside the closed bound")
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ReplayComposerError(f"{path} must contain exactly one text line")
    return value


class _PlanRenderer:
    def __init__(self, frontend: str) -> None:
        try:
            self.profile = _FRONTEND_PROFILE[frontend]
        except KeyError as exc:
            raise ReplayComposerError(
                f"frontend {frontend!r} has no syntax-only composer profile"
            ) from exc
        self.frontend = frontend
        self.lines: list[str] = []
        self.source_map: dict[str, dict[str, Any]] = {}
        self.ids: set[str] = set()
        self.node_kinds: Counter[str] = Counter()
        self.raw_line_count = 0

    @staticmethod
    def _indented(text: str, depth: int) -> str:
        if not text:
            return ""
        return "    " * depth + text

    def _node_id(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or _ID.fullmatch(value) is None:
            raise ReplayComposerError(f"{path}.id is not a closed syntax-node id")
        if value in self.ids:
            raise ReplayComposerError(f"duplicate syntax-node id {value!r}")
        self.ids.add(value)
        if len(self.ids) > MAX_COMPOSER_NODES:
            raise ReplayComposerError("composer node count exceeds the closed bound")
        return value

    def render_nodes(
        self,
        values: Any,
        *,
        depth: int,
        parent: str | None,
        path: str,
    ) -> None:
        if depth > MAX_COMPOSER_DEPTH:
            raise ReplayComposerError("composer nesting exceeds the closed bound")
        if not isinstance(values, list):
            raise ReplayComposerError(f"{path} must be an ordered array")
        for index, value in enumerate(values):
            self.render_node(
                value,
                depth=depth,
                parent=parent,
                path=f"{path}[{index}]",
            )

    def render_node(
        self,
        value: Any,
        *,
        depth: int,
        parent: str | None,
        path: str,
    ) -> None:
        if not isinstance(value, Mapping):
            raise ReplayComposerError(f"{path} must be an object")
        kind = value.get("kind")
        if kind not in {"line", "block", "raw_block"}:
            raise ReplayComposerError(f"{path}.kind is outside the closed grammar")
        node_id = self._node_id(value.get("id"), path)
        self.node_kinds[str(kind)] += 1
        start = len(self.lines) + 1
        header_line: int | None = None
        footer_line: int | None = None

        if kind == "line":
            _exact_keys(value, {"id", "kind", "text"}, path)
            text = _line_text(value["text"], f"{path}.text", allow_empty=True)
            self.lines.append(self._indented(text, depth))
        elif kind == "raw_block":
            _exact_keys(value, {"id", "kind", "lines"}, path)
            raw_lines = value["lines"]
            if not isinstance(raw_lines, list) or not raw_lines:
                raise ReplayComposerError(f"{path}.lines must be a non-empty array")
            self.raw_line_count += len(raw_lines)
            if self.raw_line_count > MAX_COMPOSER_RAW_LINES:
                raise ReplayComposerError("raw composer line count exceeds the closed bound")
            for raw_index, raw in enumerate(raw_lines):
                text = _line_text(
                    raw,
                    f"{path}.lines[{raw_index}]",
                    allow_empty=True,
                )
                self.lines.append(self._indented(text, depth))
        else:
            _exact_keys(value, {"id", "kind", "header", "children"}, path)
            header = _line_text(
                value["header"], f"{path}.header", allow_empty=False
            )
            header_line = len(self.lines) + 1
            if self.profile == "python":
                if not header.rstrip().endswith(":"):
                    raise ReplayComposerError(
                        f"{path}.header must end in ':' for a Python-like profile"
                    )
                self.lines.append(self._indented(header, depth))
            else:
                if header.rstrip().endswith("{") or header.rstrip().endswith("}"):
                    raise ReplayComposerError(
                        f"{path}.header must omit C-like block braces"
                    )
                self.lines.append(self._indented(f"{header} {{", depth))
            self.render_nodes(
                value["children"],
                depth=depth + 1,
                parent=node_id,
                path=f"{path}.children",
            )
            if self.profile == "c_like":
                footer_line = len(self.lines) + 1
                self.lines.append(self._indented("}", depth))

        end = len(self.lines)
        self.source_map[node_id] = {
            "kind": kind,
            "parent": parent,
            "start_line": start,
            "end_line": end,
            "header_line": header_line,
            "footer_line": footer_line,
        }


def render_replay_plan(
    plan: Mapping[str, Any], *, expected_frontend: str | None = None
) -> RenderedReplay:
    """Render one strict syntax plan without reading J, C, pi, or compiler state."""

    if not isinstance(plan, Mapping):
        raise ReplayComposerError("composer plan root must be an object")
    _exact_keys(plan, {"protocol", "frontend", "nodes"}, "<plan>")
    if plan["protocol"] != REPLAY_COMPOSER_PROTOCOL:
        raise ReplayComposerError("composer plan protocol mismatch")
    frontend = plan["frontend"]
    if not isinstance(frontend, str) or frontend not in _FRONTEND_PROFILE:
        raise ReplayComposerError("composer plan frontend is unsupported")
    if expected_frontend is not None and frontend != expected_frontend:
        raise ReplayComposerError("composer plan frontend differs from the episode")
    renderer = _PlanRenderer(frontend)
    renderer.render_nodes(plan["nodes"], depth=0, parent=None, path="nodes")
    source = "\n".join(renderer.lines) + "\n"
    if not source.strip():
        raise ReplayComposerError("composer plan rendered no source")
    if len(source.encode("utf-8")) > MAX_COMPOSER_RENDER_BYTES:
        raise ReplayComposerError("rendered Replay exceeds the closed byte bound")
    return RenderedReplay(
        source=source,
        source_map=dict(renderer.source_map),
        node_count=len(renderer.ids),
        node_kinds=dict(renderer.node_kinds),
        frontend=frontend,
        layout_profile=renderer.profile,
    )


def _atomic_write_bytes(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"), mode=mode)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _reject_json_constant(value: str) -> None:
    raise ReplayComposerError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_json_loads(value: bytes | str, *, path: str) -> Any:
    try:
        return json.loads(value, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayComposerError(f"composer file {path!r} is not strict JSON") from exc


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ReplayComposerError(f"required composer file {path.name!r} is absent")
    size = path.stat().st_size
    if not 1 <= size <= limit:
        raise ReplayComposerError(f"composer file {path.name!r} has invalid size")
    value = path.read_bytes()
    if len(value) != size:
        raise ReplayComposerError(f"composer file {path.name!r} changed while reading")
    return value


def _read_json(path: Path, limit: int = MAX_COMPOSER_PLAN_BYTES) -> Mapping[str, Any]:
    value = _strict_json_loads(_read_bounded(path, limit), path=path.name)
    if not isinstance(value, Mapping):
        raise ReplayComposerError(f"composer file {path.name!r} must contain an object")
    return value


def _paths(work_dir: Path, output_path: Path | None = None) -> dict[str, Path]:
    raw_work = Path(work_dir)
    if raw_work.is_symlink():
        raise ReplayComposerError("composer work directory may not be a symlink")
    work = raw_work.resolve(strict=True)
    if not work.is_dir():
        raise ReplayComposerError("composer work directory must be a real directory")
    expected_output = (work.parent / "output" / "replay.source").resolve(
        strict=False
    )
    if output_path is not None and Path(output_path).is_symlink():
        raise ReplayComposerError("composer output may not be a symlink")
    output = (
        Path(output_path).resolve(strict=False)
        if output_path is not None
        else expected_output
    )
    if output != expected_output:
        raise ReplayComposerError("composer output must be this episode's replay.source")
    return {
        "work": work,
        "tool": work / REPLAY_COMPOSER_TOOL,
        "builder": work / REPLAY_COMPOSER_BUILDER,
        "plan": work / REPLAY_COMPOSER_PLAN,
        "node_draft": work / REPLAY_COMPOSER_NODE_DRAFT,
        "state": work / REPLAY_COMPOSER_STATE,
        "source_map": work / REPLAY_COMPOSER_SOURCE_MAP,
        "rendered": work / REPLAY_COMPOSER_RENDERED,
        "audit": work / REPLAY_COMPOSER_AUDIT,
        "checkpoint_plan": work / REPLAY_COMPOSER_CHECKPOINT_PLAN,
        "checkpoint_source": work / REPLAY_COMPOSER_CHECKPOINT_SOURCE,
        "checkpoint_map": work / REPLAY_COMPOSER_CHECKPOINT_MAP,
        "output": output,
    }


def _load_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    state = dict(_read_json(paths["state"]))
    if state.get("protocol") != REPLAY_COMPOSER_STATE_PROTOCOL:
        raise ReplayComposerError("composer state protocol mismatch")
    revision = state.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise ReplayComposerError("composer state revision is invalid")
    if state.get("offer_mode") not in REPLAY_COMPOSER_MODES - {"off"}:
        raise ReplayComposerError("composer state offer mode is invalid")
    return state


def _append_audit(paths: Mapping[str, Path], value: Mapping[str, Any]) -> None:
    audit = paths["audit"]
    if audit.exists():
        if not audit.is_file() or audit.is_symlink():
            raise ReplayComposerError("composer audit must be a regular file")
        size = audit.stat().st_size
        if not 0 <= size <= MAX_COMPOSER_AUDIT_BYTES:
            raise ReplayComposerError("composer audit byte count exceeds the closed bound")
        existing = audit.read_bytes()
        if len(existing) != size:
            raise ReplayComposerError("composer audit changed while reading")
    else:
        existing = b""
    if existing.count(b"\n") >= MAX_COMPOSER_AUDIT_ROWS:
        raise ReplayComposerError("composer audit row count exceeds the closed bound")
    row = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    if len(existing) + len(row) > MAX_COMPOSER_AUDIT_BYTES:
        raise ReplayComposerError("composer audit byte count exceeds the closed bound")
    _atomic_write_bytes(audit, existing + row)


def _builder_scaffold(frontend: str) -> str:
    return f'''"""Agent-owned syntax composition for this one fresh Replay episode."""

from replay_composer import block, compose, line, raw_block


# Author target source text from the exact packet yourself.  These constructors
# only manage syntax layout and stable IDs; they do not select an API or algorithm.
nodes = [
    # line("imports", "..."),
    # block("function", "def ...:", [line("body", "...")]),
    # raw_block("escape", ["..."]),
]

compose({frontend!r}, nodes)
'''


def install_episode_composer(
    work_dir: Path, frontend: str, *, offer_mode: str = "available"
) -> Mapping[str, Any]:
    """Install one source-neutral helper and empty plan in a fresh workspace."""

    if frontend not in _FRONTEND_PROFILE:
        raise ReplayComposerError(f"frontend {frontend!r} has no composer profile")
    if offer_mode not in REPLAY_COMPOSER_MODES - {"off"}:
        raise ReplayComposerError("installed composer mode must be available/preferred")
    paths = _paths(work_dir)
    if any(
        paths[name].exists()
        for name in {"tool", "builder", "plan", "state", "audit"}
    ):
        raise ReplayComposerError("fresh Replay composer paths are unexpectedly occupied")
    module_source = Path(__file__).read_text(encoding="utf-8")
    _atomic_write_text(paths["tool"], module_source, mode=0o500)
    _atomic_write_text(paths["builder"], _builder_scaffold(frontend))
    _atomic_write_text(
        paths["plan"],
        _json_text(
            {
                "protocol": REPLAY_COMPOSER_PROTOCOL,
                "frontend": frontend,
                "nodes": [],
            }
        ),
    )
    state = {
        "protocol": REPLAY_COMPOSER_STATE_PROTOCOL,
        "frontend": frontend,
        "layout_profile": _FRONTEND_PROFILE[frontend],
        "offer_mode": offer_mode,
        "revision": 0,
        "last_action": "install",
        "node_count": 0,
        "node_kinds": {},
        "rendered_lines": 0,
        "source_map_nodes": 0,
    }
    _atomic_write_text(paths["state"], _json_text(state))
    _append_audit(
        paths,
        {
            "protocol": REPLAY_COMPOSER_AUDIT_PROTOCOL,
            "status": "pass",
            "action": "install",
            "revision": 0,
            "frontend": frontend,
            "offer_mode": offer_mode,
            "node_count": 0,
        },
    )
    return {
        "protocol": REPLAY_COMPOSER_PROTOCOL,
        "status": "pass",
        "frontend": frontend,
        "layout_profile": _FRONTEND_PROFILE[frontend],
        "offer_mode": offer_mode,
        "paths": {
            "tool": f"work/{REPLAY_COMPOSER_TOOL}",
            "builder": f"work/{REPLAY_COMPOSER_BUILDER}",
            "plan": f"work/{REPLAY_COMPOSER_PLAN}",
            "node_draft": f"work/{REPLAY_COMPOSER_NODE_DRAFT}",
            "state": f"work/{REPLAY_COMPOSER_STATE}",
            "source_map": f"work/{REPLAY_COMPOSER_SOURCE_MAP}",
        },
    }


def _commit_render(
    paths: Mapping[str, Path],
    state: Mapping[str, Any],
    rendered: RenderedReplay,
    *,
    action: str,
) -> Mapping[str, Any]:
    revision = int(state["revision"]) + 1
    _atomic_write_text(paths["output"], rendered.source)
    _atomic_write_text(paths["rendered"], rendered.source)
    source_map = {
        "protocol": REPLAY_COMPOSER_PROTOCOL,
        "frontend": rendered.frontend,
        "revision": revision,
        "nodes": dict(rendered.source_map),
    }
    _atomic_write_text(paths["source_map"], _json_text(source_map))
    next_state = {
        "protocol": REPLAY_COMPOSER_STATE_PROTOCOL,
        "frontend": rendered.frontend,
        "layout_profile": rendered.layout_profile,
        "offer_mode": state["offer_mode"],
        "revision": revision,
        "last_action": action,
        "node_count": rendered.node_count,
        "node_kinds": dict(rendered.node_kinds),
        "rendered_lines": len(rendered.source.splitlines()),
        "source_map_nodes": len(rendered.source_map),
    }
    _atomic_write_text(paths["state"], _json_text(next_state))
    _append_audit(
        paths,
        {
            "protocol": REPLAY_COMPOSER_AUDIT_PROTOCOL,
            "status": "pass",
            "action": action,
            "revision": revision,
            "frontend": rendered.frontend,
            "node_count": rendered.node_count,
            "node_kinds": dict(rendered.node_kinds),
            "rendered_lines": len(rendered.source.splitlines()),
            "rendered_bytes": len(rendered.source.encode("utf-8")),
        },
    )
    return {**next_state, "status": "pass", "source_map": REPLAY_COMPOSER_SOURCE_MAP}


def render_episode_composer(
    work_dir: Path, output_path: Path | None = None
) -> Mapping[str, Any]:
    paths = _paths(work_dir, output_path)
    state = _load_state(paths)
    plan = _read_json(paths["plan"])
    rendered = render_replay_plan(plan, expected_frontend=str(state["frontend"]))
    return _commit_render(paths, state, rendered, action="render")


def compose(
    frontend: str,
    nodes: Sequence[Mapping[str, Any]],
    *,
    work_dir: Path | None = None,
    output_path: Path | None = None,
) -> Mapping[str, Any]:
    """Validate and render an agent-owned builder script's generic syntax nodes."""

    if isinstance(nodes, (str, bytes)):
        raise ReplayComposerError("composer nodes must be a sequence of node objects")
    selected_work_dir = (
        Path(__file__).resolve().parent if work_dir is None else Path(work_dir)
    )
    paths = _paths(selected_work_dir, output_path)
    state = _load_state(paths)
    plan = {
        "protocol": REPLAY_COMPOSER_PROTOCOL,
        "frontend": frontend,
        "nodes": [dict(node) for node in nodes],
    }
    rendered = render_replay_plan(
        plan, expected_frontend=str(state["frontend"])
    )
    _atomic_write_text(paths["plan"], _json_text(plan))
    return _commit_render(paths, state, rendered, action="compose")


def _find_syntax_node(
    nodes: list[Any], node_id: str
) -> tuple[list[Any], int, Mapping[str, Any]] | None:
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        if node.get("id") == node_id:
            return nodes, index, node
        if node.get("kind") == "block":
            children = node.get("children")
            if isinstance(children, list):
                found = _find_syntax_node(children, node_id)
                if found is not None:
                    return found
    return None


def _required_syntax_node(
    nodes: list[Any], node_id: str, *, role: str
) -> tuple[list[Any], int, Mapping[str, Any]]:
    if not isinstance(node_id, str) or _ID.fullmatch(node_id) is None:
        raise ReplayComposerError(f"{role} is not a closed syntax-node id")
    found = _find_syntax_node(nodes, node_id)
    if found is None:
        raise ReplayComposerError(f"{role} {node_id!r} is absent from the plan")
    return found


def _draft_syntax_node(paths: Mapping[str, Path]) -> Mapping[str, Any]:
    draft = _read_json(paths["node_draft"])
    if "id" not in draft or "kind" not in draft:
        raise ReplayComposerError("composer node draft must contain id and kind")
    return copy.deepcopy(draft)


def edit_episode_composer(
    work_dir: Path,
    operation: str,
    node_id: str,
    anchor_id: str | None = None,
    output_path: Path | None = None,
) -> Mapping[str, Any]:
    """Apply one local syntax-tree edit and render the resulting exact source."""

    if operation not in _EDIT_OPERATIONS:
        raise ReplayComposerError(f"unsupported composer edit {operation!r}")
    moving = operation.startswith("move_")
    if moving != (anchor_id is not None):
        raise ReplayComposerError(
            "move edits require an anchor id; non-move edits forbid one"
        )

    paths = _paths(work_dir, output_path)
    state = _load_state(paths)
    original = _read_json(paths["plan"])
    render_replay_plan(original, expected_frontend=str(state["frontend"]))
    plan = copy.deepcopy(dict(original))
    nodes = plan.get("nodes")
    if not isinstance(nodes, list):
        raise ReplayComposerError("composer plan nodes must be an ordered array")

    if operation in {"insert_before", "insert_after", "append_child", "replace"}:
        container, index, target = _required_syntax_node(
            nodes, node_id, role="edit target"
        )
        draft = _draft_syntax_node(paths)
        if operation == "replace":
            if draft.get("id") != node_id:
                raise ReplayComposerError(
                    "replacement node must preserve the target's stable id"
                )
            container[index] = draft
        elif operation == "insert_before":
            container.insert(index, draft)
        elif operation == "insert_after":
            container.insert(index + 1, draft)
        else:
            if target.get("kind") != "block" or not isinstance(
                target.get("children"), list
            ):
                raise ReplayComposerError("append-child target must be a block")
            target["children"].append(draft)
    elif operation == "delete":
        container, index, _target = _required_syntax_node(
            nodes, node_id, role="delete target"
        )
        container.pop(index)
    else:
        assert anchor_id is not None
        if node_id == anchor_id:
            raise ReplayComposerError("a node cannot be moved relative to itself")
        source_container, source_index, _source = _required_syntax_node(
            nodes, node_id, role="moving node"
        )
        moved = source_container.pop(source_index)
        anchor_container, anchor_index, anchor = _required_syntax_node(
            nodes, anchor_id, role="move anchor"
        )
        if operation == "move_before":
            anchor_container.insert(anchor_index, moved)
        elif operation == "move_after":
            anchor_container.insert(anchor_index + 1, moved)
        else:
            if anchor.get("kind") != "block" or not isinstance(
                anchor.get("children"), list
            ):
                raise ReplayComposerError("move-child anchor must be a block")
            anchor["children"].append(moved)

    rendered = render_replay_plan(
        plan, expected_frontend=str(state["frontend"])
    )
    _atomic_write_text(paths["plan"], _json_text(plan))
    return _commit_render(paths, state, rendered, action=operation)


def checkpoint_episode_composer(
    work_dir: Path,
    output_path: Path | None = None,
    *,
    expected_source: str | None = None,
) -> bool:
    """Checkpoint exact composer plan/source only when they still correspond."""

    paths = _paths(work_dir, output_path)
    if any(
        not paths[name].is_file()
        for name in ("output", "rendered", "source_map")
    ):
        return False
    state = _load_state(paths)
    plan_bytes = _read_bounded(paths["plan"], MAX_COMPOSER_PLAN_BYTES)
    plan = _strict_json_loads(plan_bytes, path=paths["plan"].name)
    if not isinstance(plan, Mapping):
        raise ReplayComposerError("composer checkpoint plan must contain an object")
    rendered = render_replay_plan(plan, expected_frontend=str(state["frontend"]))
    current = _read_bounded(paths["output"], MAX_COMPOSER_RENDER_BYTES).decode("utf-8")
    snapshot = _read_bounded(paths["rendered"], MAX_COMPOSER_RENDER_BYTES).decode("utf-8")
    if current != snapshot or current != rendered.source:
        return False
    if expected_source is not None and current != expected_source:
        return False
    source_map_bytes = _read_bounded(paths["source_map"], MAX_COMPOSER_PLAN_BYTES)
    source_map = _strict_json_loads(source_map_bytes, path=paths["source_map"].name)
    if not isinstance(source_map, Mapping):
        raise ReplayComposerError("composer source map must contain an object")
    if (
        source_map.get("protocol") != REPLAY_COMPOSER_PROTOCOL
        or source_map.get("frontend") != rendered.frontend
        or source_map.get("nodes") != rendered.source_map
    ):
        raise ReplayComposerError("composer source map does not match the exact plan")
    map_revision = source_map.get("revision")
    if (
        not isinstance(map_revision, int)
        or isinstance(map_revision, bool)
        or map_revision < 1
    ):
        raise ReplayComposerError("composer source-map revision is invalid")
    if map_revision != state["revision"]:
        raise ReplayComposerError("composer state/source-map revisions differ")
    _atomic_write_bytes(paths["checkpoint_plan"], plan_bytes)
    _atomic_write_text(paths["checkpoint_source"], current)
    revision = int(state["revision"]) + 1
    source_map = {**source_map, "revision": revision}
    source_map_bytes = _json_text(source_map).encode("utf-8")
    _atomic_write_bytes(paths["source_map"], source_map_bytes)
    _atomic_write_bytes(paths["checkpoint_map"], source_map_bytes)
    state.update(
        {
            "frontend": rendered.frontend,
            "layout_profile": rendered.layout_profile,
            "revision": revision,
            "last_action": "checkpoint",
            "node_count": rendered.node_count,
            "node_kinds": dict(rendered.node_kinds),
            "rendered_lines": len(rendered.source.splitlines()),
            "source_map_nodes": len(rendered.source_map),
        }
    )
    _atomic_write_text(paths["state"], _json_text(state))
    _append_audit(
        paths,
        {
            "protocol": REPLAY_COMPOSER_AUDIT_PROTOCOL,
            "status": "pass",
            "action": "checkpoint",
            "revision": revision,
            "frontend": state["frontend"],
            "node_count": rendered.node_count,
        },
    )
    return True


def restore_episode_composer_checkpoint(
    work_dir: Path,
    output_path: Path | None = None,
    *,
    expected_source: str | None = None,
) -> bool:
    paths = _paths(work_dir, output_path)
    required = ("checkpoint_plan", "checkpoint_source", "checkpoint_map")
    if any(not paths[name].is_file() for name in required):
        return False
    state = _load_state(paths)
    source = _read_bounded(
        paths["checkpoint_source"], MAX_COMPOSER_RENDER_BYTES
    ).decode("utf-8")
    if expected_source is not None and source != expected_source:
        return False
    plan_bytes = _read_bounded(paths["checkpoint_plan"], MAX_COMPOSER_PLAN_BYTES)
    plan = _strict_json_loads(plan_bytes, path=paths["checkpoint_plan"].name)
    if not isinstance(plan, Mapping):
        raise ReplayComposerError("composer checkpoint plan must contain an object")
    rendered = render_replay_plan(plan, expected_frontend=str(state["frontend"]))
    if rendered.source != source:
        raise ReplayComposerError("composer checkpoint source does not match its plan")
    source_map_bytes = _read_bounded(
        paths["checkpoint_map"], MAX_COMPOSER_PLAN_BYTES
    )
    source_map = _strict_json_loads(
        source_map_bytes, path=paths["checkpoint_map"].name
    )
    if not isinstance(source_map, Mapping):
        raise ReplayComposerError("composer checkpoint source map must contain an object")
    if (
        source_map.get("protocol") != REPLAY_COMPOSER_PROTOCOL
        or source_map.get("frontend") != rendered.frontend
        or source_map.get("nodes") != rendered.source_map
    ):
        raise ReplayComposerError(
            "composer checkpoint source map does not match its exact plan"
        )
    map_revision = source_map.get("revision")
    if (
        not isinstance(map_revision, int)
        or isinstance(map_revision, bool)
        or map_revision < 1
    ):
        raise ReplayComposerError("composer checkpoint source-map revision is invalid")
    if map_revision > state["revision"]:
        raise ReplayComposerError("composer checkpoint revision is from the future")
    _atomic_write_bytes(paths["plan"], plan_bytes)
    _atomic_write_text(paths["output"], source)
    _atomic_write_text(paths["rendered"], source)
    revision = int(state["revision"]) + 1
    current_source_map = {**source_map, "revision": revision}
    _atomic_write_text(paths["source_map"], _json_text(current_source_map))
    state.update(
        {
            "frontend": rendered.frontend,
            "layout_profile": rendered.layout_profile,
            "revision": revision,
            "last_action": "rollback",
            "node_count": rendered.node_count,
            "node_kinds": dict(rendered.node_kinds),
            "rendered_lines": len(rendered.source.splitlines()),
            "source_map_nodes": len(rendered.source_map),
        }
    )
    _atomic_write_text(paths["state"], _json_text(state))
    _append_audit(
        paths,
        {
            "protocol": REPLAY_COMPOSER_AUDIT_PROTOCOL,
            "status": "pass",
            "action": "rollback",
            "revision": revision,
            "frontend": state["frontend"],
            "node_count": rendered.node_count,
        },
    )
    return True


def composer_status(work_dir: Path, output_path: Path | None = None) -> Mapping[str, Any]:
    paths = _paths(work_dir, output_path)
    state = _load_state(paths)
    output_matches = False
    if paths["output"].is_file() and paths["rendered"].is_file():
        output_matches = _read_bounded(
            paths["output"], MAX_COMPOSER_RENDER_BYTES
        ) == _read_bounded(paths["rendered"], MAX_COMPOSER_RENDER_BYTES)
    source_map_revision_matches: bool | None = None
    if paths["source_map"].is_file():
        source_map = _read_json(paths["source_map"])
        source_map_revision_matches = (
            source_map.get("protocol") == REPLAY_COMPOSER_PROTOCOL
            and source_map.get("frontend") == state.get("frontend")
            and source_map.get("revision") == state.get("revision")
        )
    return {
        **state,
        "status": "pass",
        "output_matches_last_render": output_matches,
        "source_map_revision_matches_state": source_map_revision_matches,
        "checkpoint_available": all(
            paths[name].is_file()
            for name in ("checkpoint_plan", "checkpoint_source", "checkpoint_map")
        ),
    }


def _audit_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise ReplayComposerError("composer audit must be a regular file")
    size = path.stat().st_size
    if not 0 <= size <= MAX_COMPOSER_AUDIT_BYTES:
        raise ReplayComposerError("composer audit byte count exceeds the closed bound")
    raw_text = path.read_text(encoding="utf-8")
    if len(raw_text.encode("utf-8")) != size:
        raise ReplayComposerError("composer audit changed while reading")
    rows: list[Mapping[str, Any]] = []
    for raw in raw_text.splitlines()[:MAX_COMPOSER_AUDIT_ROWS]:
        try:
            value = _strict_json_loads(raw, path=path.name)
        except ReplayComposerError:
            continue
        if isinstance(value, Mapping):
            rows.append(value)
    return rows


def load_episode_composer_evidence(
    work_dir: Path, output_path: Path | None = None
) -> Mapping[str, Any]:
    """Return bounded proof-irrelevant telemetry, never plan/source content."""

    try:
        paths = _paths(work_dir, output_path)
        state = _load_state(paths)
        rows = _audit_rows(paths["audit"])
        actions = Counter(
            str(row.get("action", "unknown"))
            for row in rows
            if row.get("status") == "pass"
        )
        failed_actions = Counter(
            str(row.get("action", "unknown"))
            for row in rows
            if row.get("status") != "pass"
        )
        current_matches = False
        if paths["output"].is_file() and paths["rendered"].is_file():
            current_matches = (
                _read_bounded(paths["output"], MAX_COMPOSER_RENDER_BYTES)
                == _read_bounded(paths["rendered"], MAX_COMPOSER_RENDER_BYTES)
            )
        source_map_revision_matches: bool | None = None
        if paths["source_map"].is_file():
            source_map = _read_json(paths["source_map"])
            source_map_revision_matches = (
                source_map.get("protocol") == REPLAY_COMPOSER_PROTOCOL
                and source_map.get("frontend") == state.get("frontend")
                and source_map.get("revision") == state.get("revision")
            )
        used = any(
            actions.get(action, 0) > 0
            for action in {"compose", "render", *_EDIT_OPERATIONS}
        )
        output_exists = paths["output"].is_file()
        return {
            "protocol": REPLAY_COMPOSER_PROTOCOL,
            "status": "pass",
            "installed": paths["tool"].is_file(),
            "builder_installed": paths["builder"].is_file(),
            "offer_mode": state.get("offer_mode"),
            "used": used,
            "selected_path": (
                "composer"
                if used and current_matches
                else "raw_escape"
                if output_exists
                else "none"
            ),
            "frontend": state.get("frontend"),
            "layout_profile": state.get("layout_profile"),
            "revision": state.get("revision"),
            "last_action": state.get("last_action"),
            "node_count": state.get("node_count"),
            "node_kinds": state.get("node_kinds", {}),
            "source_map_nodes": state.get("source_map_nodes", 0),
            "actions": dict(actions),
            "failed_actions": dict(failed_actions),
            "output_matches_last_render": current_matches,
            "source_map_revision_matches_state": source_map_revision_matches,
            "checkpoint_available": all(
                paths[name].is_file()
                for name in ("checkpoint_plan", "checkpoint_source", "checkpoint_map")
            ),
            "authority": (
                "proof-irrelevant episode-local syntax telemetry only; it is not "
                "trusted for J/pi consumption, Source equivalence, or Gate closure"
            ),
        }
    except Exception as exc:
        return {
            "protocol": REPLAY_COMPOSER_PROTOCOL,
            "status": "unavailable",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }


def composer_diagnostic_map(
    work_dir: Path,
    source: str,
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Map diagnostic lines to the smallest exact rendered syntax-node spans."""

    try:
        paths = _paths(work_dir)
        if _read_bounded(paths["rendered"], MAX_COMPOSER_RENDER_BYTES).decode("utf-8") != source:
            return []
        plan = _read_json(paths["plan"])
        rendered = render_replay_plan(plan)
        if rendered.source != source:
            return []
        source_map = _read_json(paths["source_map"])
        state = _load_state(paths)
        nodes = source_map.get("nodes")
        if (
            source_map.get("protocol") != REPLAY_COMPOSER_PROTOCOL
            or source_map.get("frontend") != rendered.frontend
            or not isinstance(nodes, Mapping)
            or nodes != rendered.source_map
            or source_map.get("revision") != state.get("revision")
        ):
            return []
    except Exception:
        return []
    result: list[Mapping[str, Any]] = []
    for index, diagnostic in enumerate(diagnostics):
        line = diagnostic.get("line")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            continue
        matched: list[tuple[int, str]] = []
        for node_id, span in nodes.items():
            if not isinstance(node_id, str) or not isinstance(span, Mapping):
                continue
            start = span.get("start_line")
            end = span.get("end_line")
            if (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and start <= line <= end
            ):
                matched.append((end - start, node_id))
        if matched:
            matched.sort()
            result.append(
                {
                    "diagnostic_index": index,
                    "line": line,
                    "primary_node": matched[0][1],
                    "containing_nodes": [item[1] for item in matched[:8]],
                }
            )
    return result


def _cli_response(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )


def _main(argv: Sequence[str]) -> int:
    action = argv[0] if argv else "invalid"
    simple_actions = {"render", "status", "checkpoint", "rollback"}
    one_node_edits = {
        "insert-before",
        "insert-after",
        "append-child",
        "replace",
        "delete",
    }
    two_node_edits = {"move-before", "move-after", "move-child"}
    valid = (
        len(argv) == 1 and action in simple_actions
    ) or (
        len(argv) == 2 and action in one_node_edits
    ) or (
        len(argv) == 3 and action in two_node_edits
    )
    if not valid:
        _cli_response(
            {
                "protocol": REPLAY_COMPOSER_PROTOCOL,
                "status": "fail",
                "diagnostic": (
                    "usage: replay_composer.py render|status|checkpoint|rollback; "
                    "insert-before|insert-after|append-child|replace|delete NODE; "
                    "move-before|move-after|move-child NODE ANCHOR"
                ),
            }
        )
        return 2
    work_dir = Path(__file__).resolve().parent
    try:
        if action == "render":
            result = render_episode_composer(work_dir)
        elif action == "status":
            result = composer_status(work_dir)
        elif action == "checkpoint":
            passed = checkpoint_episode_composer(work_dir)
            if not passed:
                raise ReplayComposerError(
                    "checkpoint requires exact current composer-rendered Replay bytes"
                )
            result = composer_status(work_dir)
        elif action == "rollback":
            passed = restore_episode_composer_checkpoint(work_dir)
            if not passed:
                raise ReplayComposerError("no composer checkpoint is available")
            result = composer_status(work_dir)
        else:
            result = edit_episode_composer(
                work_dir,
                action.replace("-", "_"),
                argv[1],
                argv[2] if len(argv) == 3 else None,
            )
    except Exception as exc:
        try:
            paths = _paths(work_dir)
            state = _load_state(paths)
            _append_audit(
                paths,
                {
                    "protocol": REPLAY_COMPOSER_AUDIT_PROTOCOL,
                    "status": "fail",
                    "action": action,
                    "revision": state.get("revision"),
                    "frontend": state.get("frontend"),
                    "diagnostic_class": type(exc).__name__,
                },
            )
        except Exception:
            pass
        _cli_response(
            {
                "protocol": REPLAY_COMPOSER_PROTOCOL,
                "status": "fail",
                "diagnostic": f"{type(exc).__name__}: {exc}",
            }
        )
        return 2
    _cli_response(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "REPLAY_COMPOSER_PROTOCOL",
    "REPLAY_COMPOSER_BUILDER",
    "REPLAY_COMPOSER_MODES",
    "REPLAY_COMPOSER_PLAN",
    "REPLAY_COMPOSER_NODE_DRAFT",
    "REPLAY_COMPOSER_SOURCE_MAP",
    "REPLAY_COMPOSER_STATE",
    "REPLAY_COMPOSER_TOOL",
    "RenderedReplay",
    "ReplayComposerError",
    "block",
    "checkpoint_episode_composer",
    "composer_diagnostic_map",
    "composer_status",
    "compose",
    "edit_episode_composer",
    "install_episode_composer",
    "load_episode_composer_evidence",
    "line",
    "raw_block",
    "render_episode_composer",
    "render_replay_plan",
    "restore_episode_composer_checkpoint",
]
