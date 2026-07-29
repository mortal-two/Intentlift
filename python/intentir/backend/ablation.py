"""Backend experiment arms for causal performance attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ABLATION_PROTOCOL = "intentir.backend.ablation-plan.v3"
ABLATION_SCOPE = "J-grounded Q/D-certified concrete lowering-state search"
ABLATION_BUDGET_RULE = (
    "C-Fixed receives exactly one mechanical legal projection of C; the "
    "caller-supplied equal budget governs bounded C-guided and Erase-C "
    "search-efficiency orders, while formal Schedule-Only and IntentIR "
    "performance envelopes use the complete admitted candidate space"
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
    "domain_rule": "D_search and D_cert use disjoint explicitly named cases",
    "comparison_rule": "every edge is direct; tolerance transitivity is forbidden",
    "c_role": "candidate ordering only",
}
ABLATION_CANDIDATE_SPACE = {
    "search_object": (
        "concrete post-pass MLIR state plus schedule assignment and "
        "transformation trace"
    ),
    "budget_unit": "one explicitly enumerated candidate",
    "failed_attempt_rule": (
        "one distinct exact candidate that fails before state formation consumes "
        "one budget unit"
    ),
    "expanded_arm_nesting_rule": (
        "bounded-search and anytime diagnostics visit the fixed-structure "
        "schedule spine in the same C-conditioned order before visiting other "
        "concrete lowering states; this order does not truncate the "
        "complete-space main performance envelope"
    ),
    "c_fixed_rule": (
        "the canonical structural path plus one mechanical legal projection of "
        "the authored C tile/schedule forms exactly one candidate and performs "
        "no search, workload-derived occupancy and cost heuristics are forbidden"
    ),
    "abstraction_rule": (
        "no implementation-family object, family id, or named emitter participates "
        "in generation, legality, or selection"
    ),
}
ABLATION_MEASUREMENT = {
    "formal_panels": {
        "cuda_external": ["Source", "IntentIR"],
        "rvv_external": ["Native RVV", "IntentIR"],
        "internal_attribution": ["C-Fixed", "Schedule-Only", "IntentIR"],
        "c_prior_table": [
            "best performance at fixed budget",
            "candidates to 95% of final best",
        ],
    },
    "external_match_rule": (
        "operator, shape, dtype, layout, output relation, timed boundary, "
        "cache policy, device and profiler must match exactly"
    ),
    "timed_scope": (
        "prepared launch or prepared graph only; allocation and correctness "
        "synchronization are outside the timed closure"
    ),
    "finalist_rule": (
        "fresh round-robin remeasurement with alternating traversal; initial "
        "ranking samples are excluded"
    ),
    "confirmation_rule": (
        "equal-count fresh matched-baseline and target batches in alternating pair order"
    ),
    "warmup_rule": (
        "before confirmation, Source and target prepared thunks receive the same "
        "declared untimed time budget in round-robin order; cold-data timing then "
        "evicts L2 before every measured launch"
    ),
    "e3_e2_comparison_rule": (
        "the primary marginal E3/E2 gain selects the complete fixed-structure "
        "schedule-spine best and complete global best from one transported "
        "candidate universe and one target timing run; separately launched "
        "bounded E2/E3 searches remain anytime diagnostics"
    ),
    "main_three_arm_rule": (
        "C-Fixed, Schedule-Only and IntentIR are performance envelopes over one "
        "mechanical C projection, the complete canonical-path schedule space "
        "and the complete admitted lowering-state space respectively; Source "
        "remains the mandatory independent GPU external baseline"
    ),
    "aggregation": "median per side with every paired batch retained",
}
DEFAULT_ABLATION_SPEC = (
    Path(__file__).resolve().parents[3]
    / "spec"
    / "lowering"
    / "backend-search-ablation-v3.json"
)


class ExperimentArm(str, Enum):
    E0_SOURCE = "E0"
    E1_CANONICAL = "E1"
    C_FIXED = "CF"
    E2_SCHEDULE = "E2"
    E3_STATE_SEARCH = "E3"
    E4_ERASE_C = "E4"
    L0_LOCAL_OFF = "L0"
    L1_LOCAL_ON = "L1"
    N0_NUMERIC_OFF = "N0"
    N1_REDUCTION_TREE = "N1"
    N2_FMA = "N2"
    N3_PRECISION_APPROX = "N3"

    @property
    def experiment_name(self) -> str:
        return {
            ExperimentArm.E0_SOURCE: "Source",
            ExperimentArm.E1_CANONICAL: "Canonical-Diagnostic",
            ExperimentArm.C_FIXED: "C-Fixed",
            ExperimentArm.E2_SCHEDULE: "Schedule-Only",
            ExperimentArm.E3_STATE_SEARCH: "IntentIR",
            ExperimentArm.E4_ERASE_C: "IntentIR-Erase-C",
            ExperimentArm.L0_LOCAL_OFF: "Local-B-Off-Diagnostic",
            ExperimentArm.L1_LOCAL_ON: "Local-B-On-Diagnostic",
            ExperimentArm.N0_NUMERIC_OFF: "Numeric-Off-Diagnostic",
            ExperimentArm.N1_REDUCTION_TREE: "Reduction-Tree-Diagnostic",
            ExperimentArm.N2_FMA: "FMA-Diagnostic",
            ExperimentArm.N3_PRECISION_APPROX: "Precision-Approx-Diagnostic",
        }[self]


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
            if self.baseline_kind not in {"source", "canonical", "c_fixed"}:
                raise ValueError(
                    "baseline_kind must be source, canonical, or c_fixed"
                )
            expected_budget = 1 if self.baseline_kind == "c_fixed" else 0
            if self.candidate_budget != expected_budget:
                raise ValueError(
                    "source/canonical baselines own no search budget and "
                    "C-Fixed owns exactly one candidate"
                )
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
            "experiment_name": self.arm.experiment_name,
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
    if set(value) != {
        "protocol",
        "scope",
        "correctness",
        "candidate_space",
        "measurement",
        "budget_rule",
        "arms",
    }:
        raise ValueError("backend ablation spec has missing or unknown root fields")
    if value["scope"] != ABLATION_SCOPE:
        raise ValueError("backend ablation spec has an unsupported scope")
    if value["budget_rule"] != ABLATION_BUDGET_RULE:
        raise ValueError("backend ablation spec changed the equal-budget rule")
    if value["correctness"] != ABLATION_CORRECTNESS:
        raise ValueError("backend ablation spec differs from the Q/D implementation")
    if value["candidate_space"] != ABLATION_CANDIDATE_SPACE:
        raise ValueError("backend ablation spec differs from candidate enumeration")
    if value["measurement"] != ABLATION_MEASUREMENT:
        raise ValueError("backend ablation spec differs from measurement protocol")
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
        if budget_class not in {"baseline", "single_candidate", "equal_total"}:
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
                candidate_budget=(
                    0
                    if budget_class == "baseline"
                    else 1
                    if budget_class == "single_candidate"
                    else candidate_budget
                ),
                baseline_kind=record["baseline_kind"],
                **{name: record[name] for name in axis_names},
            )
        )
    return tuple(result)


def assert_fair_pair(left: SearchAxes, right: SearchAxes, *, varying: str) -> None:
    """Fail if a claimed one-factor ablation changes any unreported axis."""

    left_record = left.to_dict()
    right_record = right.to_dict()
    ignored = {"arm", "experiment_name", varying, "numeric_state_level"}
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
    "ABLATION_CANDIDATE_SPACE",
    "ABLATION_MEASUREMENT",
    "ABLATION_PROTOCOL",
    "ABLATION_SCOPE",
    "DEFAULT_ABLATION_SPEC",
    "ExperimentArm",
    "SearchAxes",
    "assert_fair_pair",
    "load_ablation_spec",
    "standard_ablation_plan",
]
