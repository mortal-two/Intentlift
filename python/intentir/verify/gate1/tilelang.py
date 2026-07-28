"""TileLang Level-2 observer: structural TIR with diagnostic PTX evidence."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from .ptx import (
    PTXFingerprint,
    compare_ptx_fingerprints,
    fingerprint_ptx,
)


TILELANG_FINGERPRINT_SCHEMA = "tilelang-tir-structural-ptx-diagnostic-v2"
_T_OP_RE = re.compile(r"\bT\.([A-Za-z_][A-Za-z0-9_]*)")


class TileLangObservationError(ValueError):
    """TileLang artifacts cannot support the strong observer."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def normalize_tilelang_tir_ops(tir: str) -> tuple[str, ...]:
    """Retain public TileLang/TIR operation topology, not source identifiers."""

    if not isinstance(tir, str) or not tir.strip():
        raise TileLangObservationError("TileLang TIR is empty")
    operations: list[str] = []
    for raw in tir.splitlines():
        line = raw.split("#", 1)[0]
        operations.extend(f"T.{match}" for match in _T_OP_RE.findall(line))
        stripped = line.strip()
        if stripped.startswith("if "):
            operations.append("control.if")
        elif stripped.startswith("else:"):
            operations.append("control.else")
    if "T.prim_func" not in operations:
        raise TileLangObservationError("TileLang artifact lacks a T.prim_func")
    if "T.Kernel" not in operations and "T.launch_thread" not in operations:
        raise TileLangObservationError("TileLang artifact lacks device launch topology")
    return tuple(operations)


def _call_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_path(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


class _TIRAlphaNormalizer(ast.NodeTransformer):
    """Remove authored identifiers while retaining TIR data/control structure."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.next_parameter = 0
        self.next_local = 0

    def _bind(self, name: str, *, parameter: bool = False) -> str:
        existing = self.names.get(name)
        if existing is not None:
            return existing
        if parameter:
            replacement = f"p{self.next_parameter}"
            self.next_parameter += 1
        else:
            replacement = f"v{self.next_local}"
            self.next_local += 1
        self.names[name] = replacement
        return replacement

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "tir_entry"
        node.args = self.visit(node.args)  # type: ignore[assignment]
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.returns = self.visit(node.returns) if node.returns is not None else None
        node.body = [self.visit(item) for item in node.body]
        node.type_comment = None
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._bind(node.arg, parameter=True)
        node.annotation = (
            self.visit(node.annotation) if node.annotation is not None else None
        )
        node.type_comment = None
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "T":
            return node
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            node.id = self._bind(node.id)
        elif node.id in self.names:
            node.id = self.names[node.id]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        path = _call_path(node.func)
        if path == "T.func_attr" and node.args and isinstance(node.args[0], ast.Dict):
            attributes = node.args[0]
            for index, key in enumerate(attributes.keys):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "global_symbol"
                    and index < len(attributes.values)
                ):
                    attributes.values[index] = ast.Constant("<entry>")
        elif path == "T.block" and node.args:
            if isinstance(node.args[0], ast.Constant) and isinstance(
                node.args[0].value, str
            ):
                node.args[0] = ast.Constant("<block>")
        return node


def normalize_tilelang_tir_structure(tir: str) -> str:
    """Build an identifier-free structural AST for the public TIR program.

    Unlike the operation inventory, this retains constants, loop bounds,
    subscripts, operators, buffer declarations, calls and control nesting.  It
    deliberately ignores comments, source locations, entry names and block
    labels.
    """

    if not isinstance(tir, str) or not tir.strip():
        raise TileLangObservationError("TileLang TIR is empty")
    try:
        tree = ast.parse(tir)
    except SyntaxError as exc:
        raise TileLangObservationError(f"TileLang TIR is not parseable: {exc}") from exc
    normalized = _TIRAlphaNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


@dataclass(frozen=True)
class TileLangFingerprint:
    schema_version: str
    tir_ops: tuple[str, ...]
    tir_op_multiset: tuple[tuple[str, int], ...]
    tir_structure: str
    interface: tuple[tuple[str, str], ...]
    ptx: PTXFingerprint
    schedule_view: tuple[tuple[str, str], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tir_ops": list(self.tir_ops),
            "tir_op_multiset": [list(item) for item in self.tir_op_multiset],
            "tir_structure": self.tir_structure,
            "interface": [list(item) for item in self.interface],
            "ptx": self.ptx.to_dict(),
            "schedule_view": [list(item) for item in self.schedule_view],
            "digest": self.digest,
        }


def fingerprint_tilelang(
    tir: str,
    ptx: str,
    *,
    interface: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> TileLangFingerprint:
    tir_ops = normalize_tilelang_tir_ops(tir)
    tir_structure = normalize_tilelang_tir_structure(tir)
    interface_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (interface or {}).items()
        )
    )
    schedule_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (schedule or {}).items()
        )
    )
    ptx_fingerprint = fingerprint_ptx(
        ptx,
        interface=interface,
        target=target,
        schedule=schedule,
    )
    payload = {
        "schema": TILELANG_FINGERPRINT_SCHEMA,
        "tir_ops": tir_ops,
        "tir_structure": tir_structure,
        "interface": interface_view,
        "ptx": ptx_fingerprint.digest,
    }
    return TileLangFingerprint(
        TILELANG_FINGERPRINT_SCHEMA,
        tir_ops,
        tuple(sorted(Counter(tir_ops).items())),
        tir_structure,
        interface_view,
        ptx_fingerprint,
        schedule_view,
        _digest(payload),
    )


@dataclass(frozen=True)
class TileLangDifference:
    category: str
    source: Any
    replay: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "source": self.source,
            "replay": self.replay,
            "message": self.message,
        }


@dataclass(frozen=True)
class TileLangComparison:
    hard_pass: bool
    hard_passes: Mapping[str, bool]
    diagnostic_passes: Mapping[str, bool]
    differences: tuple[TileLangDifference, ...]
    source_digest: str
    replay_digest: str

    @property
    def primary_difference_category(self) -> str | None:
        return next(
            (
                item.category
                for item in self.differences
                if item.category in self.hard_passes
                and not self.hard_passes[item.category]
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "hard_passes": dict(self.hard_passes),
            "diagnostic_passes": dict(self.diagnostic_passes),
            "primary_difference_category": self.primary_difference_category,
            "differences": [item.to_dict() for item in self.differences],
            "source_digest": self.source_digest,
            "replay_digest": self.replay_digest,
        }


def compare_tilelang_fingerprints(
    source: TileLangFingerprint,
    replay: TileLangFingerprint,
) -> TileLangComparison:
    ptx = compare_ptx_fingerprints(source.ptx, replay.ptx)
    base_fields = (
        ("schema", source.schema_version, replay.schema_version),
        ("interface", source.interface, replay.interface),
        ("tir_structure", source.tir_structure, replay.tir_structure),
        ("tir_topology", source.tir_ops, replay.tir_ops),
        ("tir_operations", source.tir_op_multiset, replay.tir_op_multiset),
    )
    passes = {category: expected == actual for category, expected, actual in base_fields}
    # PTX is below TileLang's frontend/compiler boundary.  Keep only its schema,
    # callable ABI and selected target as a hard safety envelope; exact backend
    # instruction spelling, multiplicity and topology remain auditable but
    # cannot veto an equal structural TIR program.
    ptx_hard_fields = {"schema", "interface", "target"}
    passes.update(
        {
            f"ptx.{key}": value
            for key, value in ptx.hard_passes.items()
            if key in ptx_hard_fields
        }
    )
    diagnostic_passes = {
        f"ptx.{key}": value for key, value in ptx.hard_passes.items()
    }
    differences = [
        TileLangDifference(
            category,
            expected,
            actual,
            f"TileLang {category} differs between Source and Replay",
        )
        for category, expected, actual in base_fields
        if expected != actual
    ]
    differences.extend(
        TileLangDifference(
            f"ptx.{item.category}",
            item.source,
            item.replay,
            item.message,
        )
        for item in ptx.differences
    )
    return TileLangComparison(
        all(passes.values()),
        passes,
        diagnostic_passes,
        tuple(differences),
        source.digest,
        replay.digest,
    )


class TileLangComparator:
    def compare(
        self,
        source: TileLangFingerprint,
        replay: TileLangFingerprint,
    ) -> TileLangComparison:
        return compare_tilelang_fingerprints(source, replay)


__all__ = [
    "TILELANG_FINGERPRINT_SCHEMA",
    "TileLangComparator",
    "TileLangComparison",
    "TileLangDifference",
    "TileLangFingerprint",
    "TileLangObservationError",
    "compare_tilelang_fingerprints",
    "fingerprint_tilelang",
    "normalize_tilelang_tir_ops",
    "normalize_tilelang_tir_structure",
]
