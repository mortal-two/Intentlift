"""Frozen backend experiment arms for causal performance attribution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


ABLATION_PROTOCOL = "intentir.backend.ablation-plan.v1"


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


def standard_ablation_plan(candidate_budget: int) -> tuple[SearchAxes, ...]:
    """Return the paper arms with equal budgets where causal pairing requires it."""

    if isinstance(candidate_budget, bool) or candidate_budget <= 0:
        raise ValueError("standard ablation candidate budget must be positive")
    common = {
        "candidate_budget": candidate_budget,
        "use_c_prior": True,
        "search_structural_states": True,
        "enable_local_b_transformations": True,
        "search_reduction_topology": True,
        "search_fma": True,
        "search_precision": True,
        "search_approximation": True,
    }
    return (
        SearchAxes(
            ExperimentArm.E0_SOURCE,
            0,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            baseline_kind="source",
        ),
        SearchAxes(
            ExperimentArm.E1_CANONICAL,
            0,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            baseline_kind="canonical",
        ),
        SearchAxes(
            ExperimentArm.E2_SCHEDULE,
            candidate_budget,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        SearchAxes(ExperimentArm.E3_STATE_SEARCH, **common),
        SearchAxes(
            ExperimentArm.E4_ERASE_C,
            **{**common, "use_c_prior": False},
        ),
        SearchAxes(
            ExperimentArm.L0_LOCAL_OFF,
            **{**common, "enable_local_b_transformations": False},
        ),
        SearchAxes(ExperimentArm.L1_LOCAL_ON, **common),
        SearchAxes(
            ExperimentArm.N0_NUMERIC_OFF,
            **{
                **common,
                "search_reduction_topology": False,
                "search_fma": False,
                "search_precision": False,
                "search_approximation": False,
            },
        ),
        SearchAxes(
            ExperimentArm.N1_REDUCTION_TREE,
            **{
                **common,
                "search_fma": False,
                "search_precision": False,
                "search_approximation": False,
            },
        ),
        SearchAxes(
            ExperimentArm.N2_FMA,
            **{
                **common,
                "search_precision": False,
                "search_approximation": False,
            },
        ),
        SearchAxes(ExperimentArm.N3_PRECISION_APPROX, **common),
    )


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
    "ABLATION_PROTOCOL",
    "ExperimentArm",
    "SearchAxes",
    "assert_fair_pair",
    "standard_ablation_plan",
]
