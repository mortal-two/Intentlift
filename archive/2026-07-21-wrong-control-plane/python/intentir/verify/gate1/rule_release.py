"""Versioned, explicitly approved updates for a :class:`RuleRegistry`.

This module deliberately sits outside the deterministic composer.  A release
is inert data until a caller explicitly applies it to the exact parent
registry named by the release.  Runtime discovery, implicit hot reload, and
best-effort replacement are intentionally unsupported.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .model import (
    SchemaError,
    StrictJSONMixin,
    _enum,
    _strict_object,
    _string,
    _string_list,
)
from .rules import RuleCard, RuleRegistry, RuleSelectionMode


RULE_RELEASE_SCHEMA = "scrc-rule-release-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RULE_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_QUALIFIED_CALL_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([A-Za-z_]\w*)\.([A-Za-z_]\w*)"
)
_ALLOWED_API_ROOTS = frozenset({"libdevice", "tl", "triton"})
_MAX_CARD_BYTES = 64 * 1024


class RuleReleaseStatus(str, Enum):
    """Lifecycle state of a rule-library release."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REVOKED = "revoked"


class RuleChangeKind(str, Enum):
    """Supported registry mutations.

    Releases remain intentionally small: a card may be added, or one existing
    card may be replaced after proving the exact digest of the old card.
    """

    ADD = "add"
    REPLACE = "replace"


class RuleReleaseApplyError(SchemaError):
    """Raised when a valid release cannot be applied to a supplied registry."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"rule release contains a non-canonical JSON value: {exc}"
        ) from exc
    return encoded.encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path)
    if _SHA256_RE.fullmatch(result) is None:
        raise SchemaError(f"{path} must be a lowercase SHA-256 digest")
    return result


def rule_card_digest(card: RuleCard | dict[str, Any]) -> str:
    """Return the canonical content digest of one complete rule card."""

    parsed = card if isinstance(card, RuleCard) else RuleCard.from_json(card)
    return _digest_json(parsed.to_json())


def _validate_release_card(card: RuleCard, *, additive: bool) -> None:
    """Reject legacy or structurally unsafe cards at the release boundary."""

    if _RULE_ID_RE.fullmatch(card.id) is None:
        raise RuleReleaseApplyError(
            f"release rule card id must be lower_snake_case: {card.id!r}"
        )
    if len(card.version) > 128:
        raise RuleReleaseApplyError("release rule card version is too long")
    if not card.target_kinds:
        raise RuleReleaseApplyError(
            f"release rule card {card.id!r} requires explicit target_kinds"
        )
    if card.construction_phase is None:
        raise RuleReleaseApplyError(
            f"release rule card {card.id!r} requires construction_phase"
        )
    if additive and card.selection_mode is not RuleSelectionMode.DERIVED:
        raise RuleReleaseApplyError(
            "add releases may introduce only derived cards; new override "
            "families require a code/schema release"
        )
    encoded = _canonical_json_bytes(card.to_json())
    if len(encoded) > _MAX_CARD_BYTES:
        raise RuleReleaseApplyError(
            f"release rule card {card.id!r} exceeds {_MAX_CARD_BYTES} bytes"
        )
    for spelling in card.api_mapping.values():
        qualified = set(_QUALIFIED_CALL_RE.findall(spelling))
        unknown = sorted(
            f"{root}.{attribute}"
            for root, attribute in qualified
            if root not in _ALLOWED_API_ROOTS and attribute != "to"
        )
        if unknown:
            raise RuleReleaseApplyError(
                f"release rule card {card.id!r} uses unreviewed API root(s): "
                + ", ".join(unknown)
            )


def _selection_contract(card: RuleCard) -> tuple[Any, ...]:
    return (
        card.frontend,
        tuple(card.applies_to),
        card.selection_mode,
        card.override_family,
        card.strategy_family,
        card.strategy_choice,
        tuple(card.target_kinds),
        card.construction_phase,
        tuple(card.required_fields),
        tuple(card.provides),
        tuple(card.requires),
    )


@dataclass(frozen=True)
class RuleReleaseChange(StrictJSONMixin):
    """One add or exact-card replacement in a release."""

    kind: RuleChangeKind
    card: RuleCard
    replaced_card_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(RuleChangeKind, self.kind, "RuleReleaseChange.kind"),
        )
        card = (
            self.card
            if isinstance(self.card, RuleCard)
            else RuleCard.from_json(self.card)
        )
        # Keep release contents detached from a caller-owned RuleCard object.
        object.__setattr__(self, "card", RuleCard.from_json(card.to_json()))

        if self.kind is RuleChangeKind.ADD:
            if self.replaced_card_digest is not None:
                raise SchemaError(
                    "add RuleReleaseChange must not declare replaced_card_digest"
                )
        else:
            object.__setattr__(
                self,
                "replaced_card_digest",
                _sha256(
                    self.replaced_card_digest,
                    "RuleReleaseChange.replaced_card_digest",
                ),
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuleReleaseChange":
        data = _strict_object(
            value,
            path="RuleReleaseChange",
            allowed={"kind", "card", "replaced_card_digest"},
            required={"kind", "card"},
        )
        card = data["card"]
        if not isinstance(card, dict):
            raise SchemaError("RuleReleaseChange.card must be an object")
        return cls(
            kind=data["kind"],
            card=RuleCard.from_json(card),
            replaced_card_digest=data.get("replaced_card_digest"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "card": self.card.to_json(),
        }
        if self.replaced_card_digest is not None:
            result["replaced_card_digest"] = self.replaced_card_digest
        return result


@dataclass(frozen=True)
class RuleRelease(StrictJSONMixin):
    """One auditable registry update tied to an exact parent registry."""

    release_id: str
    version: str
    status: RuleReleaseStatus
    parent_registry_digest: str
    changes: tuple[RuleReleaseChange, ...] = field(default_factory=tuple)
    trigger: tuple[str, ...] = field(default_factory=tuple)
    regression: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""
    approval_content_digest: str | None = None
    schema: str = RULE_RELEASE_SCHEMA

    def __post_init__(self) -> None:
        schema = _string(self.schema, "RuleRelease.schema")
        if schema != RULE_RELEASE_SCHEMA:
            raise SchemaError(
                f"RuleRelease.schema must be {RULE_RELEASE_SCHEMA!r}"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self, "release_id", _string(self.release_id, "RuleRelease.release_id")
        )
        object.__setattr__(
            self, "version", _string(self.version, "RuleRelease.version")
        )
        object.__setattr__(
            self,
            "status",
            _enum(RuleReleaseStatus, self.status, "RuleRelease.status"),
        )
        object.__setattr__(
            self,
            "parent_registry_digest",
            _sha256(
                self.parent_registry_digest,
                "RuleRelease.parent_registry_digest",
            ),
        )

        if not isinstance(self.changes, (list, tuple)):
            raise SchemaError("RuleRelease.changes must be an array")
        changes = tuple(
            item
            if isinstance(item, RuleReleaseChange)
            else RuleReleaseChange.from_json(item)
            for item in self.changes
        )
        if not changes:
            raise SchemaError("RuleRelease.changes must not be empty")
        card_ids = [change.card.id for change in changes]
        if len(card_ids) != len(set(card_ids)):
            raise SchemaError(
                "RuleRelease.changes must not target the same card id twice"
            )
        # Changes are independent by construction, so canonicalize their order
        # to make release identity independent of JSON list ordering.
        object.__setattr__(
            self,
            "changes",
            tuple(sorted(changes, key=lambda change: change.card.id)),
        )

        for name in ("trigger", "regression", "evidence"):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise SchemaError(f"RuleRelease.{name} must be an array")
            values = _string_list(
                list(raw), f"RuleRelease.{name}", unique=True
            )
            if not values:
                raise SchemaError(f"RuleRelease.{name} must not be empty")
            object.__setattr__(self, name, tuple(sorted(values)))
        object.__setattr__(
            self, "rationale", _string(self.rationale, "RuleRelease.rationale")
        )

        computed = self.computed_approval_content_digest
        if self.approval_content_digest is None:
            object.__setattr__(self, "approval_content_digest", computed)
        else:
            supplied = _sha256(
                self.approval_content_digest,
                "RuleRelease.approval_content_digest",
            )
            if supplied != computed:
                raise SchemaError(
                    "RuleRelease.approval_content_digest does not match "
                    "canonical release content"
                )
            object.__setattr__(self, "approval_content_digest", supplied)

    def _approval_content_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "release_id": self.release_id,
            "version": self.version,
            "status": self.status.value,
            "parent_registry_digest": self.parent_registry_digest,
            "changes": [change.to_json() for change in self.changes],
            "trigger": list(self.trigger),
            "regression": list(self.regression),
            "evidence": list(self.evidence),
            "rationale": self.rationale,
        }

    @property
    def computed_approval_content_digest(self) -> str:
        """Digest of all release content except the digest field itself."""

        return _digest_json(self._approval_content_json())

    def assert_approval_content_digest(self) -> None:
        """Recheck content identity, including against post-construction drift."""

        if self.approval_content_digest != self.computed_approval_content_digest:
            raise RuleReleaseApplyError(
                "rule release approval content digest no longer matches content"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuleRelease":
        data = _strict_object(
            value,
            path="RuleRelease",
            allowed={
                "schema",
                "release_id",
                "version",
                "status",
                "parent_registry_digest",
                "changes",
                "trigger",
                "regression",
                "evidence",
                "rationale",
                "approval_content_digest",
            },
            required={
                "schema",
                "release_id",
                "version",
                "status",
                "parent_registry_digest",
                "changes",
                "trigger",
                "regression",
                "evidence",
                "rationale",
                "approval_content_digest",
            },
        )
        changes = data["changes"]
        if not isinstance(changes, list):
            raise SchemaError("RuleRelease.changes must be an array")
        return cls(
            schema=data["schema"],
            release_id=data["release_id"],
            version=data["version"],
            status=data["status"],
            parent_registry_digest=data["parent_registry_digest"],
            changes=tuple(
                RuleReleaseChange.from_json(change) for change in changes
            ),
            trigger=tuple(
                _string_list(data["trigger"], "RuleRelease.trigger", unique=True)
            ),
            regression=tuple(
                _string_list(
                    data["regression"], "RuleRelease.regression", unique=True
                )
            ),
            evidence=tuple(
                _string_list(
                    data["evidence"], "RuleRelease.evidence", unique=True
                )
            ),
            rationale=data["rationale"],
            approval_content_digest=data["approval_content_digest"],
        )

    def to_json(self) -> dict[str, Any]:
        result = self._approval_content_json()
        assert self.approval_content_digest is not None
        result["approval_content_digest"] = self.approval_content_digest
        return result

    @classmethod
    def from_path(cls, path: Path) -> "RuleRelease":
        return load_rule_release(path)

    def apply(self, base_registry: RuleRegistry) -> RuleRegistry:
        return apply_rule_release(base_registry, self)


def apply_rule_release(
    base_registry: RuleRegistry,
    release: RuleRelease | dict[str, Any],
) -> RuleRegistry:
    """Explicitly apply one approved release to its exact parent registry."""

    if not isinstance(base_registry, RuleRegistry):
        raise TypeError("base_registry must be a RuleRegistry")
    parsed = (
        release
        if isinstance(release, RuleRelease)
        else RuleRelease.from_json(release)
    )
    parsed.assert_approval_content_digest()
    if parsed.status is not RuleReleaseStatus.APPROVED:
        raise RuleReleaseApplyError(
            "only an approved rule release may be applied"
        )
    if base_registry.digest != parsed.parent_registry_digest:
        raise RuleReleaseApplyError(
            "rule release parent registry digest does not match base registry"
        )

    cards = [RuleCard.from_json(card.to_json()) for card in base_registry.cards]
    index_by_id = {card.id: index for index, card in enumerate(cards)}
    base_frontends = {card.frontend for card in cards}
    for change in parsed.changes:
        card = RuleCard.from_json(change.card.to_json())
        _validate_release_card(
            card,
            additive=change.kind is RuleChangeKind.ADD,
        )
        existing_index = index_by_id.get(card.id)
        if change.kind is RuleChangeKind.ADD:
            if existing_index is not None:
                raise RuleReleaseApplyError(
                    f"add change targets existing rule card id: {card.id}"
                )
            if card.frontend not in base_frontends:
                raise RuleReleaseApplyError(
                    "add release cannot introduce a new frontend"
                )
            index_by_id[card.id] = len(cards)
            cards.append(card)
            continue

        if existing_index is None:
            raise RuleReleaseApplyError(
                f"replace change targets unknown rule card id: {card.id}"
            )
        existing = cards[existing_index]
        actual_digest = rule_card_digest(existing)
        if actual_digest != change.replaced_card_digest:
            raise RuleReleaseApplyError(
                f"replacement digest for rule card {card.id!r} does not "
                "match the base card"
            )
        if rule_card_digest(card) == actual_digest:
            raise RuleReleaseApplyError(
                f"replacement rule card {card.id!r} is identical to the base card"
            )
        if card.version == existing.version:
            raise RuleReleaseApplyError(
                f"replacement rule card {card.id!r} must change version"
            )
        if _selection_contract(card) != _selection_contract(existing):
            raise RuleReleaseApplyError(
                f"replacement rule card {card.id!r} changes composer selection "
                "metadata; publish a code/schema release instead"
            )
        cards[existing_index] = card

    return RuleRegistry(cards)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"rule release JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise SchemaError(f"rule release JSON contains non-finite value {value!r}")


def load_rule_release(path: Path) -> RuleRelease:
    """Load one strict release JSON document from ``path``."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SchemaError(f"cannot read rule release {path}: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid rule release JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError("RuleRelease JSON root must be an object")
    return RuleRelease.from_json(value)


__all__ = [
    "RULE_RELEASE_SCHEMA",
    "RuleChangeKind",
    "RuleRelease",
    "RuleReleaseApplyError",
    "RuleReleaseChange",
    "RuleReleaseStatus",
    "apply_rule_release",
    "load_rule_release",
    "rule_card_digest",
]
