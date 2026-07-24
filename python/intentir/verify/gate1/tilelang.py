"""TileLang Level-2 observer: normalized TIR topology plus shared PTX semantics."""

from __future__ import annotations

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


TILELANG_FINGERPRINT_SCHEMA = "tilelang-tir-ptx-topology-v1"
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


@dataclass(frozen=True)
class TileLangFingerprint:
    schema_version: str
    tir_ops: tuple[str, ...]
    tir_op_multiset: tuple[tuple[str, int], ...]
    interface: tuple[tuple[str, str], ...]
    ptx: PTXFingerprint
    schedule_view: tuple[tuple[str, str], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tir_ops": list(self.tir_ops),
            "tir_op_multiset": [list(item) for item in self.tir_op_multiset],
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
        "interface": interface_view,
        "ptx": ptx_fingerprint.digest,
    }
    return TileLangFingerprint(
        TILELANG_FINGERPRINT_SCHEMA,
        tir_ops,
        tuple(sorted(Counter(tir_ops).items())),
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
    differences: tuple[TileLangDifference, ...]
    source_digest: str
    replay_digest: str

    @property
    def primary_difference_category(self) -> str | None:
        return self.differences[0].category if self.differences else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "hard_passes": dict(self.hard_passes),
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
        ("tir_topology", source.tir_ops, replay.tir_ops),
        ("tir_operations", source.tir_op_multiset, replay.tir_op_multiset),
    )
    passes = {category: expected == actual for category, expected, actual in base_fields}
    passes.update({f"ptx.{key}": value for key, value in ptx.hard_passes.items()})
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
]
