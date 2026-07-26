"""Frozen backend experiment arms for causal performance attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from intentir.ir.production_snapshot import content_digest


ABLATION_PROTOCOL = "intentir.backend.ablation-plan.v1"
ABLATION_SCOPE = "J-grounded Q/D-certified concrete lowering-state search"
ABLATION_BUDGET_RULE = (
    "all non-baseline arms receive the same caller-supplied total candidate budget"
)
ABLATION_CORRECTNESS = {
    "immutable_semantic_input": "J=A+B",
    "compiler_inputs": ["J", "optional_C", "target_capabilities"],
    "search_domain_edge": "target_ref_j",
    "certification_domain_edges": [
        "target_source",
        "target_replay",
        "target_ref_j",
    ],
    "domain_rule": "D_search and D_cert have disjoint exact input digests",
    "comparison_rule": "every edge is direct; tolerance transitivity is forbidden",
    "c_role": "candidate ordering only",
}
DEFAULT_ABLATION_SPEC = (
    Path(__file__).resolve().parents[3]
    / "spec"
    / "lowering"
    / "backend-search-ablation-v1.json"
)


class ExperimentArm(str, Enum):
    E0_SOURCE = "E0"
    E1_CANONICAL = "E1"
    E2_SCHEDULE = "E2"
    E3_STATE_SEARCH = "E3"
    E4_ERASE_C = "E4"
    L0_LOCAL_OFF = "L0"
    L1_LOCAL_ON = "L1"
    N0_NUMERIC_OFF = "N0"
    N1_REDUCTION_TREE = "N1"
    N2_FMA = "N2"
    N3_PRECISION_APPROX = "N3"


@dataclass(frozen=True)
class SearchAxes:
    """Mechanically enforce what differs between paired experiment arms."""

    arm: ExperimentArm
    candidate_budget: int
    use_c_prior: bool
    search_structural_states: bool
    enable_local_b_transformations: bool
    search_reduction_topology: bool
    search_fma: bool
    search_precision: bool
    search_approximation: bool
    baseline_kind: str | None = None
    protocol: str = ABLATION_PROTOCOL

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ExperimentArm):
            object.__setattr__(self, "arm", ExperimentArm(self.arm))
        if isinstance(self.candidate_budget, bool) or self.candidate_budget < 0:
            raise ValueError("candidate budget must be a non-negative integer")
        if self.baseline_kind is not None:
            if self.baseline_kind not in {"source", "canonical"}:
                raise ValueError("baseline_kind must be source or canonical")
            if self.candidate_budget != 0:
                raise ValueError("baseline arms do not own a search budget")
        elif self.candidate_budget <= 0:
            raise ValueError("search arms require a positive candidate budget")
        for name in (
            "use_c_prior",
            "search_structural_states",
            "enable_local_b_transformations",
            "search_reduction_topology",
            "search_fma",
            "search_precision",
            "search_approximation",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"ablation axis {name} must be boolean")
        if self.protocol != ABLATION_PROTOCOL:
            raise ValueError("unsupported backend ablation protocol")

    @property
    def numeric_state_level(self) -> str:
        if self.search_precision or self.search_approximation:
            return "precision_approx"
        if self.search_fma:
            return "fma"
        if self.search_reduction_topology:
            return "reduction_tree"
        return "off"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "arm": self.arm.value,
            "candidate_budget": self.candidate_budget,
            "use_c_prior": self.use_c_prior,
            "search_structural_states": self.search_structural_states,
            "enable_local_b_transformations": self.enable_local_b_transformations,
            "search_reduction_topology": self.search_reduction_topology,
            "search_fma": self.search_fma,
            "search_precision": self.search_precision,
            "search_approximation": self.search_approximation,
            "numeric_state_level": self.numeric_state_level,
            "baseline_kind": self.baseline_kind,
        }


def load_ablation_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the machine-consumed experiment contract, failing closed."""

    source = DEFAULT_ABLATION_SPEC if path is None else Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != ABLATION_PROTOCOL:
        raise ValueError("unsupported backend ablation spec")
    if set(value) != {"protocol", "scope", "correctness", "budget_rule", "arms"}:
        raise ValueError("backend ablation spec has missing or unknown root fields")
    if value["scope"] != ABLATION_SCOPE:
        raise ValueError("backend ablation spec has an unsupported scope")
    if value["budget_rule"] != ABLATION_BUDGET_RULE:
        raise ValueError("backend ablation spec changed the equal-budget rule")
    if value["correctness"] != ABLATION_CORRECTNESS:
        raise ValueError("backend ablation spec differs from the Q/D implementation")
    records = value.get("arms")
    if not isinstance(records, list) or not records:
        raise ValueError("backend ablation spec requires a non-empty arms array")
    expected = {item.value for item in ExperimentArm}
    observed = {
        item.get("arm") for item in records if isinstance(item, dict)
    }
    if observed != expected or len(records) != len(expected):
        raise ValueError("backend ablation spec arms are incomplete or duplicated")
    return value


def ablation_spec_digest(path: Path | None = None) -> str:
    return content_digest(load_ablation_spec(path))


def standard_ablation_plan(
    candidate_budget: int,
    *,
    spec_path: Path | None = None,
) -> tuple[SearchAxes, ...]:
    """Instantiate the machine-specified arms with one equal total budget."""

    if isinstance(candidate_budget, bool) or candidate_budget <= 0:
        raise ValueError("standard ablation candidate budget must be positive")
    records = load_ablation_spec(spec_path)["arms"]
    axis_names = (
        "use_c_prior",
        "search_structural_states",
        "enable_local_b_transformations",
        "search_reduction_topology",
        "search_fma",
        "search_precision",
        "search_approximation",
    )
    result = []
    for record in records:
        budget_class = record.get("budget_class")
        if budget_class not in {"baseline", "equal_total"}:
            raise ValueError("unknown ablation budget_class")
        allowed = {
            "arm",
            "budget_class",
            "baseline_kind",
            *axis_names,
        }
        if set(record) != allowed:
            raise ValueError("ablation arm has missing or unknown fields")
        result.append(
            SearchAxes(
                arm=ExperimentArm(record["arm"]),
                candidate_budget=(0 if budget_class == "baseline" else candidate_budget),
                baseline_kind=record["baseline_kind"],
                **{name: record[name] for name in axis_names},
            )
        )
    return tuple(result)


def assert_fair_pair(left: SearchAxes, right: SearchAxes, *, varying: str) -> None:
    """Fail if a claimed one-factor ablation changes any unreported axis."""

    left_record = left.to_dict()
    right_record = right.to_dict()
    ignored = {"arm", varying, "numeric_state_level"}
    differences = {
        key
        for key in left_record
        if key not in ignored and left_record[key] != right_record[key]
    }
    if differences:
        raise ValueError(
            "ablation arms differ outside the declared axis: "
            + ", ".join(sorted(differences))
        )


__all__ = [
    "ABLATION_BUDGET_RULE",
    "ABLATION_CORRECTNESS",
    "ABLATION_PROTOCOL",
    "ABLATION_SCOPE",
    "DEFAULT_ABLATION_SPEC",
    "ExperimentArm",
    "SearchAxes",
    "ablation_spec_digest",
    "assert_fair_pair",
    "load_ablation_spec",
    "standard_ablation_plan",
]
