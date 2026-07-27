"""Auditable, uncertified development variants of an immutable production J.

The backend sometimes needs a shape or dtype specialization that the accepted
capture did not encode.  Such a specialization is useful for compiler
capability work, but it is not an SCRC certificate.  This module keeps that
boundary explicit: every authored change is an exact JSON-pointer edit, the
accepted snapshot is never mutated, derived Intent value types are rebuilt by
the canonical J elaborator, and the resulting object carries a new digest and
an ``uncertified_backend_dev`` receipt.

This is a transport/provenance mechanism, not a kernel emitter.  The resulting
J still enters the ordinary ``materialize_j -> MLIR -> target`` pipeline.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from intentir.ir.canonical_j import elaborate_j
from intentir.ir.production_snapshot import (
    ProductionSnapshot,
    canonical_json,
    content_digest,
)
from intentir.verify.gate1.model import ImplementationConfig, SemanticArchive


J_DEV_VARIANT_PROTOCOL = "intentir.backend.dev-j-variant.v1"
J_DEV_CERTIFICATION_STATUS = "uncertified_backend_dev"
J_DEV_ALLOWED_CLAIMS = (
    "backend_capability",
    "target_correctness_on_recorded_QD",
    "performance",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]*$")
_DTYPES = frozenset(
    {
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "f16",
        "bf16",
        "f32",
        "f64",
    }
)


class JDevVariantError(ValueError):
    """Raised when a development-J edit or receipt is not exact."""


def _detached_json(value: Any) -> Any:
    """Copy through the strict JSON spelling used by content digests."""

    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise JDevVariantError(f"development edit is not strict JSON: {exc}") from exc


def _decode_pointer(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise JDevVariantError("development edit json_path must be a JSON pointer")
    result: list[str] = []
    for raw in path[1:].split("/"):
        index = 0
        decoded = ""
        while index < len(raw):
            if raw[index] != "~":
                decoded += raw[index]
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise JDevVariantError(
                    f"development edit json_path has invalid escape: {path!r}"
                )
            decoded += "~" if raw[index + 1] == "0" else "/"
            index += 2
        result.append(decoded)
    return tuple(result)


def _is_shape_path(parts: tuple[str, ...]) -> bool:
    return len(parts) >= 3 and parts[-2] == "shape" and parts[-1].isdigit()


def _classify_path(parts: tuple[str, ...]) -> str:
    if not parts or parts[0] not in {"j", "c"}:
        raise JDevVariantError("development edits must target /j or /c")
    if parts[0] == "c":
        if len(parts) == 3 and parts[1] == "values":
            return "c_value"
        raise JDevVariantError("development C edits may only replace /c/values/<slot>")
    if _is_shape_path(parts):
        return "shape"
    if parts[-1] in {"dtype", "to_dtype", "source_dtype", "target_dtype"}:
        return "dtype"
    if len(parts) == 4 and parts[1:3] == ("boundary", "domains"):
        return "domain"
    raise JDevVariantError(
        "development J edits are limited to typed shape/dtype fields and "
        "boundary domain expressions"
    )


def _lookup_parent(payload: Any, parts: tuple[str, ...]) -> tuple[Any, str | int]:
    current = payload
    for part in parts[:-1]:
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise JDevVariantError(f"development edit path component {part!r} is absent")
            current = current[int(part)]
        elif isinstance(current, dict):
            if part not in current:
                raise JDevVariantError(f"development edit path component {part!r} is absent")
            current = current[part]
        else:
            raise JDevVariantError("development edit traverses a scalar JSON value")
    leaf: str | int = parts[-1]
    if isinstance(current, list):
        if not parts[-1].isdigit() or int(parts[-1]) >= len(current):
            raise JDevVariantError(
                f"development edit list index {parts[-1]!r} is absent"
            )
        leaf = int(parts[-1])
    elif isinstance(current, dict):
        if parts[-1] not in current:
            raise JDevVariantError(
                f"development edit field {parts[-1]!r} is absent"
            )
    else:
        raise JDevVariantError("development edit parent is a scalar JSON value")
    return current, leaf


@dataclass(frozen=True)
class JDevEdit:
    """One authored, exact replacement in J or C."""

    json_path: str
    old_value: Any
    new_value: Any
    upstream_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        parts = _decode_pointer(self.json_path)
        kind = _classify_path(parts)
        old_value = _detached_json(self.old_value)
        new_value = _detached_json(self.new_value)
        if old_value == new_value:
            raise JDevVariantError("development edit must change its target value")
        if kind == "shape":
            for value in (old_value, new_value):
                if isinstance(value, bool) or not isinstance(value, (int, str)):
                    raise JDevVariantError(
                        "shape dimensions must be non-negative integers or symbols"
                    )
                if isinstance(value, int) and value < 0:
                    raise JDevVariantError("shape dimensions cannot be negative")
                if isinstance(value, str) and not value:
                    raise JDevVariantError("shape symbols cannot be empty")
        elif kind == "dtype":
            if old_value not in _DTYPES or new_value not in _DTYPES:
                raise JDevVariantError("development dtype edit uses an unknown dtype")
        elif kind == "domain":
            if not isinstance(old_value, str) or not isinstance(new_value, str):
                raise JDevVariantError("boundary domain edits must replace strings")
        if (
            not isinstance(self.upstream_evidence, tuple)
            or not self.upstream_evidence
            or any(not isinstance(item, str) or not item for item in self.upstream_evidence)
        ):
            raise JDevVariantError(
                "every development edit requires non-empty upstream evidence"
            )
        object.__setattr__(self, "old_value", old_value)
        object.__setattr__(self, "new_value", new_value)

    @property
    def kind(self) -> str:
        return _classify_path(_decode_pointer(self.json_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "json_path": self.json_path,
            "kind": self.kind,
            "old_value": _detached_json(self.old_value),
            "new_value": _detached_json(self.new_value),
            "upstream_evidence": list(self.upstream_evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JDevEdit":
        expected = {
            "json_path",
            "kind",
            "old_value",
            "new_value",
            "upstream_evidence",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise JDevVariantError(
                f"development edit must contain exactly {sorted(expected)}"
            )
        evidence = value["upstream_evidence"]
        if not isinstance(evidence, list):
            raise JDevVariantError("development edit upstream_evidence must be a list")
        edit = cls(
            json_path=value["json_path"],
            old_value=value["old_value"],
            new_value=value["new_value"],
            upstream_evidence=tuple(evidence),
        )
        if value["kind"] != edit.kind:
            raise JDevVariantError("development edit kind disagrees with json_path")
        return edit


def _apply_edit(payload: dict[str, Any], edit: JDevEdit) -> None:
    parts = _decode_pointer(edit.json_path)
    parent, leaf = _lookup_parent(payload, parts)
    actual = parent[leaf]
    if type(actual) is not type(edit.old_value) or actual != edit.old_value:
        raise JDevVariantError(
            f"development edit {edit.json_path!r} expected "
            f"{edit.old_value!r}, observed {actual!r}"
        )
    parent[leaf] = _detached_json(edit.new_value)


def _reelaborate_derived_types(j_payload: dict[str, Any]) -> SemanticArchive:
    """Let canonical J typing rebuild all derived op result types."""

    try:
        function = j_payload["intent"]["intent_function"]
        body = function["body"]
    except (KeyError, TypeError) as exc:
        raise JDevVariantError("development J lacks canonical intent_function body") from exc
    if not isinstance(body, list):
        raise JDevVariantError("development J intent body must be a list")
    for node in body:
        if not isinstance(node, dict):
            raise JDevVariantError("development J body node must be an object")
        node.pop("type", None)
    try:
        return elaborate_j(SemanticArchive.from_json(j_payload))
    except (TypeError, ValueError, KeyError) as exc:
        raise JDevVariantError(
            "development J is not self-consistent after canonical type "
            f"re-elaboration: {type(exc).__name__}: {exc}"
        ) from exc


def _validate_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise JDevVariantError(f"{name} must be a non-empty stable identifier")


@dataclass(frozen=True)
class JDevVariant:
    """One fully decoded development J plus its immutable lineage receipt."""

    base_snapshot_digest: str
    base_j_digest: str
    base_c_digest: str | None
    base_certification_status: str
    source_kernel_id: str
    shape_config_id: str
    dtype_config_id: str
    edits: tuple[JDevEdit, ...]
    j: SemanticArchive
    c: ImplementationConfig | None

    def __post_init__(self) -> None:
        for name in ("base_snapshot_digest", "base_j_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise JDevVariantError(f"{name} must be SHA-256")
        if self.base_c_digest is not None and (
            not isinstance(self.base_c_digest, str)
            or _SHA256.fullmatch(self.base_c_digest) is None
        ):
            raise JDevVariantError("base_c_digest must be SHA-256 or null")
        if self.base_certification_status not in {
            "scrc_certified",
            "uncertified_frontend_candidate",
        }:
            raise JDevVariantError("unsupported base certification status")
        for name in ("source_kernel_id", "shape_config_id", "dtype_config_id"):
            _validate_id(name, getattr(self, name))
        if not isinstance(self.edits, tuple) or not self.edits:
            raise JDevVariantError("development J requires at least one exact edit")
        paths = [edit.json_path for edit in self.edits]
        if len(paths) != len(set(paths)):
            raise JDevVariantError("development J contains duplicate edit paths")
        snapshot = ProductionSnapshot(self.j, self.c)
        object.__setattr__(self, "j", snapshot.j)
        object.__setattr__(self, "c", snapshot.c)
        if snapshot.j_digest == self.base_j_digest:
            raise JDevVariantError("development J digest must differ from its base J")

    @property
    def variant_j_digest(self) -> str:
        return content_digest(self.j.to_json())

    @property
    def variant_c_digest(self) -> str | None:
        return content_digest(self.c.to_json()) if self.c is not None else None

    @property
    def variant_snapshot_digest(self) -> str:
        return content_digest(
            {
                "j": self.j.to_json(),
                "c": self.c.to_json() if self.c is not None else None,
            }
        )

    def _content(self) -> dict[str, Any]:
        return {
            "protocol": J_DEV_VARIANT_PROTOCOL,
            "base_snapshot_digest": self.base_snapshot_digest,
            "base_j_digest": self.base_j_digest,
            "base_c_digest": self.base_c_digest,
            "base_certification_status": self.base_certification_status,
            "source_kernel_id": self.source_kernel_id,
            "shape_config_id": self.shape_config_id,
            "dtype_config_id": self.dtype_config_id,
            "edits": [edit.to_dict() for edit in self.edits],
            "variant_j": self.j.to_json(),
            "variant_c": self.c.to_json() if self.c is not None else None,
            "variant_j_digest": self.variant_j_digest,
            "variant_c_digest": self.variant_c_digest,
            "variant_snapshot_digest": self.variant_snapshot_digest,
            "certification_status": J_DEV_CERTIFICATION_STATUS,
            "allowed_claims": list(J_DEV_ALLOWED_CLAIMS),
            "j_unchanged": False,
        }

    @property
    def receipt_digest(self) -> str:
        return content_digest(self._content())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "receipt_digest": self.receipt_digest}

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        ) + "\n"

    def validate_against_base(self, base: ProductionSnapshot) -> None:
        """Replay the authored edits against the selected immutable base.

        Content digests make stale or mismatched inputs visible, but they are not
        signatures.  A consumer that owns the production catalog must also prove
        that ``variant_j``/``variant_c`` are exactly the canonical result of the
        recorded edits, rather than trusting self-consistent receipt metadata.
        """

        if not isinstance(base, ProductionSnapshot):
            raise JDevVariantError("development J validation requires ProductionSnapshot")
        if base.snapshot_digest != self.base_snapshot_digest:
            raise JDevVariantError("development J base snapshot digest mismatch")
        if base.j_digest != self.base_j_digest:
            raise JDevVariantError("development J base J digest mismatch")
        if base.c_digest != self.base_c_digest:
            raise JDevVariantError("development J base C digest mismatch")
        replayed = create_j_dev_variant(
            base,
            source_kernel_id=self.source_kernel_id,
            shape_config_id=self.shape_config_id,
            dtype_config_id=self.dtype_config_id,
            edits=self.edits,
            base_certification_status=self.base_certification_status,
        )
        if replayed.j.to_json() != self.j.to_json():
            raise JDevVariantError(
                "development J payload is not the canonical replay of its edits"
            )
        replayed_c = replayed.c.to_json() if replayed.c is not None else None
        active_c = self.c.to_json() if self.c is not None else None
        if replayed_c != active_c:
            raise JDevVariantError(
                "development C payload is not the canonical replay of its edits"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JDevVariant":
        expected = {
            "protocol",
            "base_snapshot_digest",
            "base_j_digest",
            "base_c_digest",
            "base_certification_status",
            "source_kernel_id",
            "shape_config_id",
            "dtype_config_id",
            "edits",
            "variant_j",
            "variant_c",
            "variant_j_digest",
            "variant_c_digest",
            "variant_snapshot_digest",
            "certification_status",
            "allowed_claims",
            "j_unchanged",
            "receipt_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise JDevVariantError(
                f"development J receipt must contain exactly {sorted(expected)}"
            )
        if value["protocol"] != J_DEV_VARIANT_PROTOCOL:
            raise JDevVariantError("unsupported development J protocol")
        if value["certification_status"] != J_DEV_CERTIFICATION_STATUS:
            raise JDevVariantError("development J cannot claim certification")
        if value["allowed_claims"] != list(J_DEV_ALLOWED_CLAIMS):
            raise JDevVariantError("development J allowed_claims mismatch")
        if value["j_unchanged"] is not False:
            raise JDevVariantError("development J must record j_unchanged=false")
        raw_edits = value["edits"]
        if not isinstance(raw_edits, list):
            raise JDevVariantError("development J edits must be a list")
        try:
            variant = cls(
                base_snapshot_digest=value["base_snapshot_digest"],
                base_j_digest=value["base_j_digest"],
                base_c_digest=value["base_c_digest"],
                base_certification_status=value["base_certification_status"],
                source_kernel_id=value["source_kernel_id"],
                shape_config_id=value["shape_config_id"],
                dtype_config_id=value["dtype_config_id"],
                edits=tuple(JDevEdit.from_dict(item) for item in raw_edits),
                j=SemanticArchive.from_json(value["variant_j"]),
                c=(
                    ImplementationConfig.from_json(value["variant_c"])
                    if value["variant_c"] is not None
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, JDevVariantError):
                raise
            raise JDevVariantError(str(exc)) from exc
        for field, actual in (
            ("variant_j_digest", variant.variant_j_digest),
            ("variant_c_digest", variant.variant_c_digest),
            ("variant_snapshot_digest", variant.variant_snapshot_digest),
            ("receipt_digest", variant.receipt_digest),
        ):
            if value[field] != actual:
                raise JDevVariantError(f"development J {field} mismatch")
        return variant

    @classmethod
    def from_json(cls, payload: str) -> "JDevVariant":
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise JDevVariantError(f"invalid development J JSON: {exc}") from exc
        return cls.from_dict(value)


def create_j_dev_variant(
    base: ProductionSnapshot,
    *,
    source_kernel_id: str,
    shape_config_id: str,
    dtype_config_id: str,
    edits: tuple[JDevEdit, ...],
    base_certification_status: str = "scrc_certified",
) -> JDevVariant:
    """Create a detached, canonical development variant from exact edits."""

    if not isinstance(base, ProductionSnapshot):
        raise JDevVariantError("development J requires a ProductionSnapshot base")
    if not isinstance(edits, tuple) or not edits:
        raise JDevVariantError("development J requires a non-empty tuple of edits")
    paths = [edit.json_path for edit in edits]
    if len(paths) != len(set(paths)):
        raise JDevVariantError("development J contains duplicate edit paths")
    payload = {
        "j": copy.deepcopy(base.j.to_json()),
        "c": copy.deepcopy(base.c.to_json()) if base.c is not None else None,
    }
    for edit in edits:
        if not isinstance(edit, JDevEdit):
            raise JDevVariantError("development edits must use JDevEdit")
        if edit.json_path.startswith("/c/") and payload["c"] is None:
            raise JDevVariantError("development edit targets absent C")
        _apply_edit(payload, edit)
    canonical_j = _reelaborate_derived_types(payload["j"])
    try:
        canonical_c = (
            ImplementationConfig.from_json(payload["c"])
            if payload["c"] is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise JDevVariantError(
            f"development C is not canonical after edits: {type(exc).__name__}: {exc}"
        ) from exc
    return JDevVariant(
        base_snapshot_digest=base.snapshot_digest,
        base_j_digest=base.j_digest,
        base_c_digest=base.c_digest,
        base_certification_status=base_certification_status,
        source_kernel_id=source_kernel_id,
        shape_config_id=shape_config_id,
        dtype_config_id=dtype_config_id,
        edits=edits,
        j=canonical_j,
        c=canonical_c,
    )


def load_j_dev_variant(path: str | Path) -> JDevVariant:
    target = Path(path)
    try:
        return JDevVariant.from_json(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JDevVariantError(f"cannot read development J {target}: {exc}") from exc


__all__ = [
    "J_DEV_ALLOWED_CLAIMS",
    "J_DEV_CERTIFICATION_STATUS",
    "J_DEV_VARIANT_PROTOCOL",
    "JDevEdit",
    "JDevVariant",
    "JDevVariantError",
    "create_j_dev_variant",
    "load_j_dev_variant",
]
