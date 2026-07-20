from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from intentir.verify.gate1.model import Frontend, SchemaError
from intentir.verify.gate1.rule_release import (
    RULE_RELEASE_SCHEMA,
    RuleChangeKind,
    RuleRelease,
    RuleReleaseApplyError,
    RuleReleaseChange,
    RuleReleaseStatus,
    apply_rule_release,
    load_rule_release,
    rule_card_digest,
)
from intentir.verify.gate1.rules import (
    RuleCard,
    RuleSelectionMode,
    default_triton_rule_registry,
)


def _triu_card() -> RuleCard:
    return RuleCard(
        id="realize_triangular_select",
        frontend=Frontend.TRITON,
        version="1",
        applies_to=["triu"],
        reads=["operand", "attrs.k", "rank-2 semantic coordinates"],
        api_mapping={
            "predicate": "column - row >= k",
            "selection": "tl.where",
        },
        example=(
            "Select the input value above the declared diagonal and a "
            "dtype-correct zero elsewhere."
        ),
        target_kinds=["intent_op:triu"],
        construction_phase=60,
        required_fields=["operands", "attrs.k"],
    )


def _release(
    *,
    status: RuleReleaseStatus = RuleReleaseStatus.APPROVED,
    changes: tuple[RuleReleaseChange, ...] | None = None,
    parent_digest: str | None = None,
) -> RuleRelease:
    registry = default_triton_rule_registry()
    return RuleRelease(
        release_id="triton_layout_triangular_v1",
        version="1",
        status=status,
        parent_registry_digest=parent_digest or registry.digest,
        changes=changes
        or (RuleReleaseChange(RuleChangeKind.ADD, _triu_card()),),
        trigger=("fg_trace:no_target_kind_card:intent_op:triu",),
        regression=("gate1_default_registry_regression",),
        evidence=("test_gate1_triu_rule_composition",),
        rationale="Add a reviewed, source-independent triangular selection mechanism.",
    )


def test_approved_add_release_applies_to_exact_parent_without_mutating_base() -> None:
    base = default_triton_rule_registry()
    release = _release(parent_digest=base.digest)

    updated = apply_rule_release(base, release)

    assert "realize_triangular_select" not in base
    assert "realize_triangular_select" in updated
    assert len(updated) == len(base) + 1
    assert updated.get("realize_triangular_select").to_json() == (
        _triu_card().to_json()
    )
    assert updated.digest != base.digest


@pytest.mark.parametrize(
    "status",
    [RuleReleaseStatus.PROPOSED, RuleReleaseStatus.REVOKED],
)
def test_only_approved_release_can_be_applied(status: RuleReleaseStatus) -> None:
    base = default_triton_rule_registry()

    with pytest.raises(RuleReleaseApplyError, match="only an approved"):
        _release(status=status, parent_digest=base.digest).apply(base)


def test_parent_registry_digest_must_match_exactly() -> None:
    base = default_triton_rule_registry()
    release = _release(parent_digest="0" * 64)

    with pytest.raises(RuleReleaseApplyError, match="parent registry digest"):
        release.apply(base)


def test_replace_requires_exact_old_card_digest() -> None:
    base = default_triton_rule_registry()
    old = base.get("realize_elementwise")
    replacement = replace(
        old,
        version="6",
        pitfalls=[*old.pitfalls, "Use only a reviewed semantic dispatch mapping."],
    )
    release = _release(
        parent_digest=base.digest,
        changes=(
            RuleReleaseChange(
                RuleChangeKind.REPLACE,
                replacement,
                replaced_card_digest=rule_card_digest(old),
            ),
        ),
    )

    updated = release.apply(base)

    assert base.get("realize_elementwise").version == old.version
    assert updated.get("realize_elementwise").version == "6"
    assert len(updated) == len(base)


def test_replace_rejects_wrong_or_noop_card_digest() -> None:
    base = default_triton_rule_registry()
    old = base.get("realize_elementwise")
    changed = replace(old, version="6")
    wrong_digest = RuleReleaseChange(
        RuleChangeKind.REPLACE,
        changed,
        replaced_card_digest="0" * 64,
    )
    with pytest.raises(RuleReleaseApplyError, match="does not match"):
        _release(
            parent_digest=base.digest,
            changes=(wrong_digest,),
        ).apply(base)

    noop = RuleReleaseChange(
        RuleChangeKind.REPLACE,
        old,
        replaced_card_digest=rule_card_digest(old),
    )
    with pytest.raises(RuleReleaseApplyError, match="identical"):
        _release(
            parent_digest=base.digest,
            changes=(noop,),
        ).apply(base)


def test_add_rejects_existing_card_and_replace_rejects_unknown_card() -> None:
    base = default_triton_rule_registry()
    existing = base.get("realize_elementwise")
    add_existing = RuleReleaseChange(RuleChangeKind.ADD, existing)
    with pytest.raises(RuleReleaseApplyError, match="existing rule card"):
        _release(
            parent_digest=base.digest,
            changes=(add_existing,),
        ).apply(base)

    unknown = _triu_card()
    replace_unknown = RuleReleaseChange(
        RuleChangeKind.REPLACE,
        unknown,
        replaced_card_digest="0" * 64,
    )
    with pytest.raises(RuleReleaseApplyError, match="unknown rule card"):
        _release(
            parent_digest=base.digest,
            changes=(replace_unknown,),
        ).apply(base)


def test_release_boundary_rejects_legacy_cards_and_dynamic_override_families() -> None:
    base = default_triton_rule_registry()
    legacy = RuleCard("legacy_dynamic", Frontend.TRITON)
    with pytest.raises(RuleReleaseApplyError, match="explicit target_kinds"):
        _release(
            parent_digest=base.digest,
            changes=(RuleReleaseChange("add", legacy),),
        ).apply(base)

    override = RuleCard(
        "dynamic_override",
        Frontend.TRITON,
        selection_mode=RuleSelectionMode.OVERRIDE,
        override_family="dynamic_family",
        strategy_family="dynamic_family",
        strategy_choice="dynamic_choice",
        target_kinds=["intent_op:triu"],
        construction_phase=60,
    )
    with pytest.raises(RuleReleaseApplyError, match="only derived cards"):
        _release(
            parent_digest=base.digest,
            changes=(RuleReleaseChange("add", override),),
        ).apply(base)


def test_replace_cannot_change_composer_selection_metadata() -> None:
    base = default_triton_rule_registry()
    old = base.get("realize_elementwise")
    changed = replace(
        old,
        version="6",
        target_kinds=[*old.target_kinds, "intent_op:triu"],
    )
    release = _release(
        parent_digest=base.digest,
        changes=(
            RuleReleaseChange(
                "replace",
                changed,
                replaced_card_digest=rule_card_digest(old),
            ),
        ),
    )

    with pytest.raises(RuleReleaseApplyError, match="selection metadata"):
        release.apply(base)


def test_release_json_is_strict_and_approval_digest_binds_all_content() -> None:
    release = _release()
    payload = release.to_json()

    assert payload["schema"] == RULE_RELEASE_SCHEMA
    assert RuleRelease.from_json(payload).to_json() == payload

    tampered = json.loads(json.dumps(payload))
    tampered["rationale"] = "Different rationale"
    with pytest.raises(SchemaError, match="approval_content_digest"):
        RuleRelease.from_json(tampered)

    unknown = json.loads(json.dumps(payload))
    unknown["unexpected"] = True
    with pytest.raises(SchemaError, match="unknown key"):
        RuleRelease.from_json(unknown)


def test_change_schema_enforces_add_and_replace_digest_contracts() -> None:
    card = _triu_card()

    with pytest.raises(SchemaError, match="must not declare"):
        RuleReleaseChange(
            RuleChangeKind.ADD,
            card,
            replaced_card_digest="0" * 64,
        )
    with pytest.raises(SchemaError, match="must be a string"):
        RuleReleaseChange(RuleChangeKind.REPLACE, card)

    payload = RuleReleaseChange(RuleChangeKind.ADD, card).to_json()
    payload["unexpected"] = 1
    with pytest.raises(SchemaError, match="unknown key"):
        RuleReleaseChange.from_json(payload)


def test_release_identity_is_independent_of_change_and_evidence_list_order() -> None:
    base = default_triton_rule_registry()
    extra = RuleCard(
        id="realize_flip_indexing",
        frontend=Frontend.TRITON,
        target_kinds=["intent_op:flip"],
        construction_phase=60,
        required_fields=["operands", "attrs.axes"],
    )
    first = RuleRelease(
        release_id="layout_v1",
        version="1",
        status="approved",
        parent_registry_digest=base.digest,
        changes=(
            RuleReleaseChange("add", _triu_card()),
            RuleReleaseChange("add", extra),
        ),
        trigger=("b", "a"),
        regression=("d", "c"),
        evidence=("f", "e"),
        rationale="Canonical ordering test.",
    )
    second = RuleRelease(
        release_id="layout_v1",
        version="1",
        status="approved",
        parent_registry_digest=base.digest,
        changes=tuple(reversed(first.changes)),
        trigger=tuple(reversed(first.trigger)),
        regression=tuple(reversed(first.regression)),
        evidence=tuple(reversed(first.evidence)),
        rationale=first.rationale,
    )

    assert first.approval_content_digest == second.approval_content_digest
    assert first.to_json() == second.to_json()


def test_path_loader_rejects_duplicate_keys_and_loads_valid_release(
    tmp_path: Path,
) -> None:
    release = _release()
    path = tmp_path / "release.json"
    path.write_text(release.to_json_str(), encoding="utf-8")

    assert load_rule_release(path).to_json() == release.to_json()
    assert RuleRelease.from_path(path).to_json() == release.to_json()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"scrc-rule-release-v1","schema":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="duplicate key"):
        load_rule_release(duplicate)


def test_apply_rechecks_digest_after_nested_card_mutation() -> None:
    base = default_triton_rule_registry()
    release = _release(parent_digest=base.digest)
    release.changes[0].card.pitfalls.append("post-approval mutation")

    with pytest.raises(RuleReleaseApplyError, match="no longer matches"):
        release.apply(base)
