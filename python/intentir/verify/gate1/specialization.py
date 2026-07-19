"""Fixed-C multi-specialization Gate1 execution.

One validated J/C artifact and one fresh replay source are shared by every
runtime specialization.  Source and replay compile separately per profile;
the family closes only when every deterministic TTIR comparison hard-passes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .compare import TTIRComparator, TTIRComparison
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .composer import assert_replay_plan_matches
from .diagnose import Diagnosis, Responsibility, sanitize_replay_diagnostic
from .evaluation import FailureCause
from .leakage import LeakagePolicy, scan_static_leakage
from .model import Gate1Artifact, Gate1Status, ParameterKind
from .orchestrate import (
    ReliftGuidance,
    _build_relift_guidance,
    _build_structural_relift_guidance,
    _hard_difference_digest,
    _validate_compiled_interface,
)
from .replay import ReplayGeneration, ReplayWriter
from .rules import RuleRegistry
from .ttir import fingerprint_ttir


GATE1_SPECIALIZATION_POLICY = "fixed_c_family_full_sweep_relift_v1"
GATE1_FAMILY_DIFF_POLICY = "exact_ordered_fixed_c_family_hard_diff_v1"
GATE1_FAMILY_RELIFT_GUIDANCE_POLICY = (
    "deterministic_fixed_c_family_hard_diff_guidance_v2"
)
GATE1_FAMILY_CONTRACT_DIGEST_SCHEMA = (
    "fixed_c_specialization_family_contract_v1"
)


def _dtype_key(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.removeprefix("torch.").casefold()
    return {
        "float": "f32",
        "float32": "f32",
        "float16": "f16",
        "half": "f16",
        "bfloat16": "bf16",
        "int32": "i32",
        "int64": "i64",
        "bool": "i1",
    }.get(value, value)


@dataclass(frozen=True)
class Gate1SpecializationProfile:
    profile_id: str
    case: CompileCase

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("Gate1 specialization profile_id must be non-empty")
        if not isinstance(self.case, CompileCase):
            raise TypeError("Gate1 specialization profile case must be CompileCase")

    def to_json(self) -> dict[str, Any]:
        return {"profile_id": self.profile_id, "case": self.case.to_payload()}


@dataclass(frozen=True)
class Gate1SpecializationFamily:
    family_id: str
    primary_profile_id: str
    profiles: tuple[Gate1SpecializationProfile, ...]
    fixed_c: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("Gate1 specialization family_id must be non-empty")
        profiles = tuple(self.profiles)
        if not profiles:
            raise ValueError("Gate1 specialization family requires profiles")
        if any(not isinstance(item, Gate1SpecializationProfile) for item in profiles):
            raise TypeError("Gate1 specialization family contains an invalid profile")
        ids = [item.profile_id for item in profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Gate1 specialization profile ids must be unique")
        if self.primary_profile_id not in set(ids):
            raise ValueError("primary_profile_id is absent from the family")
        if not self.fixed_c:
            raise ValueError("Gate1 v8 specialization families require fixed_c=true")
        schedules = [dict(item.case.schedule) for item in profiles]
        if any(value != schedules[0] for value in schedules[1:]):
            raise ValueError("fixed-C Gate1 profiles must share identical schedule C")
        object.__setattr__(self, "profiles", profiles)

    def to_json(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "primary_profile_id": self.primary_profile_id,
            "fixed_c": self.fixed_c,
            "profiles": [item.to_json() for item in self.profiles],
        }


def specialization_family_contract_payload(
    family: Gate1SpecializationFamily,
) -> dict[str, Any]:
    """Return the canonical fixed-C family identity used by Gate1 evidence.

    Profile declaration order is deliberately excluded from the identity.
    The complete compile case is retained, however, so a digest cannot be
    reused after changing runtime bindings, grid, target, or frozen schedule.
    """

    if not isinstance(family, Gate1SpecializationFamily):
        raise TypeError("family must be Gate1SpecializationFamily")
    return {
        "digest_schema": GATE1_FAMILY_CONTRACT_DIGEST_SCHEMA,
        "family_id": family.family_id,
        "primary_profile_id": family.primary_profile_id,
        "fixed_c": family.fixed_c,
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "case": profile.case.to_payload(),
            }
            for profile in sorted(
                family.profiles, key=lambda item: item.profile_id
            )
        ],
    }


def specialization_family_contract_digest(
    family: Gate1SpecializationFamily,
) -> str:
    """Hash the complete canonical fixed-C family contract."""

    return hashlib.sha256(
        json.dumps(
            specialization_family_contract_payload(family),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_specialization_profile(
    artifact: Gate1Artifact,
    profile: Gate1SpecializationProfile,
) -> None:
    """Fail fast when a runtime profile is outside the shared artifact."""

    case = profile.case
    expected_schedule = dict(artifact.schedule.values)
    if dict(case.schedule) != expected_schedule:
        raise ValueError(
            f"profile {profile.profile_id!r} schedule differs from frozen C"
        )
    runtime_parameters = [
        item
        for item in artifact.interface.parameters
        if item.kind is not ParameterKind.CONSTEXPR
    ]
    expected_names = [item.name for item in runtime_parameters]
    if list(case.arguments) != expected_names:
        raise ValueError(
            f"profile {profile.profile_id!r} argument order/names differ from "
            f"interface: {list(case.arguments)!r} != {expected_names!r}"
        )
    if len(case.grid) != len(artifact.interface.grid):
        raise ValueError(
            f"profile {profile.profile_id!r} grid rank differs from interface"
        )

    tensors = {
        tensor.name: tensor
        for tensor in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    scalar_values = {
        name: argument.value
        for name, argument in case.arguments.items()
        if argument.kind == "scalar"
    }
    for parameter in runtime_parameters:
        argument = case.arguments[parameter.name]
        expects_tensor = parameter.kind in {
            ParameterKind.POINTER, ParameterKind.TENSOR,
        }
        if expects_tensor != (argument.kind == "tensor"):
            raise ValueError(
                f"profile {profile.profile_id!r} argument {parameter.name!r} "
                "kind differs from interface"
            )
        if _dtype_key(parameter.dtype) != _dtype_key(argument.dtype):
            raise ValueError(
                f"profile {profile.profile_id!r} argument {parameter.name!r} "
                "dtype differs from interface"
            )
        if not expects_tensor:
            continue
        assert argument.shape is not None
        if parameter.rank != len(argument.shape):
            raise ValueError(
                f"profile {profile.profile_id!r} argument {parameter.name!r} "
                "rank differs from interface"
            )
        tensor = tensors.get(parameter.tensor_ref or "")
        if tensor is None:
            raise ValueError(
                f"profile {profile.profile_id!r} pointer {parameter.name!r} "
                "has no J tensor_ref"
            )
        if len(tensor.shape) != len(argument.shape):
            raise ValueError("J tensor rank differs from specialization argument")
        for declared, concrete in zip(tensor.shape, argument.shape, strict=True):
            if isinstance(declared, int):
                if declared != concrete:
                    raise ValueError(
                        f"profile {profile.profile_id!r} changes static J extent "
                        f"{declared} to {concrete}"
                    )
                continue
            bound = scalar_values.get(declared)
            if isinstance(bound, bool) or not isinstance(bound, int):
                raise ValueError(
                    f"profile {profile.profile_id!r} does not bind dynamic J "
                    f"dimension {declared!r} with an integer runtime scalar"
                )
            if bound != concrete:
                raise ValueError(
                    f"profile {profile.profile_id!r} binds {declared!r}={bound}, "
                    f"but tensor extent is {concrete}"
                )


@dataclass(frozen=True)
class Gate1SpecializationRecord:
    profile_id: str
    source_compile: CompileResult
    replay_compile: CompileResult | None = None
    comparison: TTIRComparison | None = None
    diagnostic: str = ""
    source_compile_reused: bool = False
    replay_compile_reused: bool = False

    @property
    def passed(self) -> bool:
        return bool(
            self.source_compile.ok
            and self.replay_compile is not None
            and self.replay_compile.ok
            and self.comparison is not None
            and self.comparison.hard_pass
        )

    def to_json(self) -> dict[str, Any]:
        def compile_json(result: CompileResult | None) -> dict[str, Any] | None:
            if result is None:
                return None
            return {
                "ok": result.ok,
                "diagnostic": result.diagnostic,
                "cache_key": result.cache_key,
                "cache_hit": result.cache_hit,
                "elapsed_ms": result.elapsed_ms,
                "compiler": dict(result.compiler),
                "target": dict(result.target),
            }

        return {
            "profile_id": self.profile_id,
            "passed": self.passed,
            "source_compile": compile_json(self.source_compile),
            "source_compile_reused": self.source_compile_reused,
            "replay_compile": compile_json(self.replay_compile),
            "replay_compile_reused": self.replay_compile_reused,
            "comparison": (
                None if self.comparison is None else self.comparison.to_dict()
            ),
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class Gate1FamilyExecutionResult:
    status: Gate1Status
    family: Gate1SpecializationFamily
    artifact: Gate1Artifact
    generation: ReplayGeneration | None
    records: tuple[Gate1SpecializationRecord, ...]
    diagnostics: tuple[str, ...] = ()

    @property
    def all_passed(self) -> bool:
        return bool(self.records) and all(item.passed for item in self.records)

    @property
    def failed_profile_ids(self) -> tuple[str, ...]:
        return tuple(item.profile_id for item in self.records if not item.passed)

    @property
    def hard_comparison_complete(self) -> bool:
        return bool(self.records) and all(
            item.source_compile.ok
            and item.replay_compile is not None
            and item.replay_compile.ok
            and item.comparison is not None
            for item in self.records
        )

    def relift_evidence_json(self) -> dict[str, Any]:
        """Return source-free family evidence safe for outer Relift.

        The payload deliberately excludes the artifact, replay generation,
        replay source, raw TTIR and compiler diagnostics.  Replay compile
        errors are reduced to bounded summaries by ``_failure_stage``.
        """

        records: list[dict[str, Any]] = []
        for record in self.records:
            stage, diagnostic = _failure_stage(record)
            source_compile = {
                "ok": record.source_compile.ok,
                "cache_key": record.source_compile.cache_key,
                "cache_hit": record.source_compile.cache_hit,
                "elapsed_ms": record.source_compile.elapsed_ms,
            }
            replay_compile = (
                None
                if record.replay_compile is None
                else {
                    "ok": record.replay_compile.ok,
                    "cache_key": record.replay_compile.cache_key,
                    "cache_hit": record.replay_compile.cache_hit,
                    "elapsed_ms": record.replay_compile.elapsed_ms,
                }
            )
            records.append(
                {
                    "profile_id": record.profile_id,
                    "passed": record.passed,
                    "failure_stage": stage,
                    "source_compile_ok": record.source_compile.ok,
                    "source_compile_reused": (
                        record.source_compile_reused
                    ),
                    "replay_compile_ok": bool(
                        record.replay_compile and record.replay_compile.ok
                    ),
                    "replay_compile_reused": (
                        record.replay_compile_reused
                    ),
                    "source_compile": source_compile,
                    "replay_compile": replay_compile,
                    "comparison": (
                        None
                        if record.comparison is None
                        else record.comparison.to_dict()
                    ),
                    "diagnostic": diagnostic,
                }
            )
        return {
            "policy": GATE1_SPECIALIZATION_POLICY,
            "diff_policy": GATE1_FAMILY_DIFF_POLICY,
            "family_id": self.family.family_id,
            "primary_profile_id": self.family.primary_profile_id,
            "fixed_c": self.family.fixed_c,
            "status": self.status.value,
            "all_passed": self.all_passed,
            "hard_comparison_complete": self.hard_comparison_complete,
            "failed_profile_ids": list(self.failed_profile_ids),
            "family_hard_difference_digest": family_hard_difference_digest(
                self
            ),
            "records": records,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": GATE1_SPECIALIZATION_POLICY,
            "diff_policy": GATE1_FAMILY_DIFF_POLICY,
            "status": self.status.value,
            "all_passed": self.all_passed,
            "hard_comparison_complete": self.hard_comparison_complete,
            "failed_profile_ids": list(self.failed_profile_ids),
            "family_hard_difference_digest": family_hard_difference_digest(
                self
            ),
            "family": self.family.to_json(),
            "artifact": self.artifact.to_json(),
            "generation": (
                None
                if self.generation is None
                else {
                    "prompt_sha256": self.generation.prompt_sha256,
                    "requested_model": self.generation.requested_model,
                    "generation_id": self.generation.generation_id,
                    "code_sha256": self.generation.code_sha256,
                    "response_model": self.generation.response_model,
                    "response_id": self.generation.response_id,
                }
            ),
            "records": [item.to_json() for item in self.records],
            "diagnostics": list(self.diagnostics),
        }


def _failure_stage(
    record: Gate1SpecializationRecord,
) -> tuple[str, str]:
    if not record.source_compile.ok:
        return "source_compile", "trusted source specialization did not compile"
    if record.replay_compile is None:
        return "not_run", "replay evaluation was not reached"
    if not record.replay_compile.ok:
        return (
            "replay_compile",
            sanitize_replay_diagnostic(record.replay_compile.diagnostic),
        )
    if record.comparison is None:
        return "comparison_unavailable", "compiled replay has no comparison"
    if record.comparison.hard_pass:
        return "passed", ""
    return "hard_mismatch", "deterministic TTIR hard comparison did not close"


def family_hard_difference_digest(
    result: Gate1FamilyExecutionResult,
) -> str | None:
    """Hash one complete fixed-C family hard-comparison state.

    A digest exists only after every declared profile produced a comparable
    source/replay fingerprint.  Compile/setup failures therefore cannot be
    mistaken for deterministic semantic stagnation.
    """

    if not result.hard_comparison_complete:
        return None
    payload = {
        "policy": GATE1_FAMILY_DIFF_POLICY,
        "family_id": result.family.family_id,
        "profiles": [
            {
                "profile_id": record.profile_id,
                "hard_difference_digest": _hard_difference_digest(
                    record.comparison
                ),
            }
            for record in sorted(result.records, key=lambda item: item.profile_id)
            if record.comparison is not None
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _compress_family_difference_groups(
    profile_differences: Sequence[Mapping[str, Any]],
    *,
    affected_profiles: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate exact difference items across a fixed-C family."""

    grouped: dict[str, dict[str, Any]] = {}
    for profile in profile_differences:
        profile_id = str(profile["profile_id"])
        for difference in profile["grouped_differences"]:
            key = json.dumps(
                {
                    "kind": difference["kind"],
                    "item": difference["item"],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            )
            entry = grouped.setdefault(
                key,
                {
                    "kind": difference["kind"],
                    "item": difference["item"],
                    "counts": {},
                },
            )
            entry["counts"][profile_id] = (
                difference["expected_count"],
                difference["actual_count"],
            )

    all_affected = set(affected_profiles)

    def render(entry: Mapping[str, Any]) -> dict[str, Any]:
        counts = entry["counts"]
        profile_ids = sorted(counts)
        pairs = {tuple(counts[profile_id]) for profile_id in profile_ids}
        result: dict[str, Any] = {
            "kind": entry["kind"],
            "item": entry["item"],
            "profile_ids": profile_ids,
        }
        if len(pairs) == 1:
            expected_count, actual_count = next(iter(pairs))
            result["expected_count"] = expected_count
            result["actual_count"] = actual_count
        else:
            result["counts_by_profile"] = [
                {
                    "profile_id": profile_id,
                    "expected_count": counts[profile_id][0],
                    "actual_count": counts[profile_id][1],
                }
                for profile_id in profile_ids
            ]
        return result

    consensus: list[dict[str, Any]] = []
    profile_specific: list[dict[str, Any]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        target = (
            consensus
            if set(entry["counts"]) == all_affected
            else profile_specific
        )
        target.append(render(entry))
    return consensus, profile_specific


def _family_structural_relift_guidance(
    artifact: Gate1Artifact,
    result: Gate1FamilyExecutionResult,
) -> dict[str, Any]:
    """Group identical source/replay structural deltas across profiles."""

    profile_by_id = {
        profile.profile_id: profile for profile in result.family.profiles
    }
    groups: dict[str, dict[str, Any]] = {}
    for record in sorted(result.records, key=lambda item: item.profile_id):
        if record.passed:
            continue
        profile = profile_by_id[record.profile_id]
        if (
            record.replay_compile is None
            or record.source_compile.ttir is None
            or record.replay_compile.ttir is None
        ):
            continue
        source_fp = fingerprint_ttir(
            record.source_compile.ttir,
            python_interface={
                **dict(record.source_compile.python_interface),
                "grid": list(profile.case.grid),
                "grid_contract": list(artifact.interface.grid),
            },
            schedule=profile.case.schedule,
        )
        replay_fp = fingerprint_ttir(
            record.replay_compile.ttir,
            python_interface={
                **dict(record.replay_compile.python_interface),
                "grid": list(profile.case.grid),
                "grid_contract": list(artifact.interface.grid),
            },
            schedule=profile.case.schedule,
        )
        structural = _build_structural_relift_guidance(
            source_fp, replay_fp
        )
        key = json.dumps(
            structural,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        group = groups.setdefault(
            key,
            {"profile_ids": [], "structural": structural},
        )
        group["profile_ids"].append(record.profile_id)
    return {
        "policy": "deterministic_fixed_c_family_structure_guidance_v1",
        "groups": [
            {
                "profile_ids": sorted(groups[key]["profile_ids"]),
                **groups[key]["structural"],
            }
            for key in sorted(groups)
        ],
    }


def build_family_relift_guidance(
    artifact: Gate1Artifact,
    result: Gate1FamilyExecutionResult,
) -> ReliftGuidance | None:
    """Aggregate a complete family sweep into one deterministic worklist."""

    digest = family_hard_difference_digest(result)
    if digest is None or result.all_passed:
        return None
    comparisons = {
        record.profile_id: record.comparison
        for record in result.records
        if record.comparison is not None
    }
    if len(comparisons) != len(result.records):
        return None

    category_order = tuple(
        component.category
        for component in result.records[0].comparison.components
        if component.hard
    )
    failed_categories = tuple(
        category
        for category in category_order
        if any(
            not next(
                component
                for component in comparison.components
                if component.category == category
            ).passed
            for comparison in comparisons.values()
        )
    )
    confirmed_categories = tuple(
        category
        for category in category_order
        if all(
            next(
                component
                for component in comparison.components
                if component.category == category
            ).passed
            for comparison in comparisons.values()
        )
    )

    per_profile_guidance = {
        record.profile_id: _build_relift_guidance(
            artifact, record.comparison
        )
        for record in result.records
        if record.comparison is not None and not record.comparison.hard_pass
    }
    category_guidance: list[dict[str, Any]] = []
    for category in failed_categories:
        affected_profiles: list[str] = []
        confirmed_profiles: list[str] = []
        profile_differences: list[dict[str, Any]] = []
        template: Mapping[str, Any] | None = None
        for record in sorted(result.records, key=lambda item: item.profile_id):
            assert record.comparison is not None
            component = next(
                item
                for item in record.comparison.components
                if item.category == category
            )
            if component.passed:
                confirmed_profiles.append(record.profile_id)
                continue
            affected_profiles.append(record.profile_id)
            guidance = per_profile_guidance[record.profile_id]
            entry = next(
                item
                for item in guidance.category_guidance
                if item["category"] == category
            )
            template = entry
            profile_differences.append(
                {
                    "profile_id": record.profile_id,
                    "raw_difference_count": entry["raw_difference_count"],
                    "grouped_differences": entry["grouped_differences"],
                }
            )
        assert template is not None
        consensus, profile_specific = _compress_family_difference_groups(
            profile_differences,
            affected_profiles=affected_profiles,
        )
        category_guidance.append(
            {
                "category": category,
                "affected_profiles": affected_profiles,
                "confirmed_profiles": confirmed_profiles,
                "raw_difference_count_by_profile": {
                    item["profile_id"]: item["raw_difference_count"]
                    for item in profile_differences
                },
                "consensus_differences": consensus,
                "profile_specific_differences": profile_specific,
                "inspect_j_surfaces": template["inspect_j_surfaces"],
                "reviewed_strategy_families": template[
                    "reviewed_strategy_families"
                ],
                "semantic_question": template["semantic_question"],
            }
        )

    current_j_summary = dict(
        next(iter(per_profile_guidance.values())).current_j_summary
    )
    current_j_summary["fixed_c_family"] = {
        "family_id": result.family.family_id,
        "primary_profile_id": result.family.primary_profile_id,
        "profiles": sorted(record.profile_id for record in result.records),
        "hard_closed_profiles": [
            record.profile_id
            for record in sorted(
                result.records, key=lambda item: item.profile_id
            )
            if record.passed
        ],
        "failed_profiles": sorted(result.failed_profile_ids),
    }
    return ReliftGuidance(
        hard_difference_digest=digest,
        failed_hard_categories=failed_categories,
        confirmed_hard_categories=confirmed_categories,
        category_guidance=tuple(category_guidance),
        current_j_summary=current_j_summary,
        structural_guidance=_family_structural_relift_guidance(
            artifact, result
        ),
        decision_contract=(
            "rederive_whole_tensor_semantics_from_source",
            "classify_each_failed_category_as_j_gap_or_reviewed_strategy_gap_or_no_expressible_change",
            "edit_only_j_strategy_location_map",
            "preserve_confirmed_hard_categories",
            "preserve_hard_closed_fixed_c_profiles",
            "one_artifact_one_replay_for_the_complete_fixed_c_family",
            "never_patch_or_describe_replay_code",
            "return_unchanged_editable_fields_when_no_semantic_change_is_justified",
        ),
        policy=GATE1_FAMILY_RELIFT_GUIDANCE_POLICY,
    )


def diagnose_family_failure(
    result: Gate1FamilyExecutionResult,
) -> Diagnosis:
    replay_compile_failures = [
        record
        for record in result.records
        if record.source_compile.ok
        and record.replay_compile is not None
        and not record.replay_compile.ok
    ]
    if replay_compile_failures:
        return Diagnosis(
            FailureCause.REPLAY_COMPILE,
            Responsibility.LIFT_ARTIFACT,
            tuple(
                f"profile={record.profile_id}: "
                f"{sanitize_replay_diagnostic(record.replay_compile.diagnostic)}"
                for record in replay_compile_failures
            ),
            tuple(record.profile_id for record in result.records if record.passed),
            ("j", "strategy", "location_map"),
            "同一不可变 replay 在至少一个 fixed-C profile 上未通过同前端编译；"
            "禁止 profile-local replay repair，完整 family 证据只回流一次 Lift。",
        )

    failed = [
        record
        for record in result.records
        if record.comparison is not None and not record.comparison.hard_pass
    ]
    primary_category = next(
        (
            record.comparison.primary_difference_category
            for record in failed
            if record.comparison.primary_difference_category is not None
        ),
        None,
    )
    cause_by_category = {
        "interface": FailureCause.INTERFACE,
        "op_multiset": FailureCause.OPERATOR,
        "dataflow": FailureCause.DATAFLOW,
        "reduction": FailureCause.REDUCTION,
        "predicate_effect": FailureCause.PREDICATE_EFFECT,
    }
    return Diagnosis(
        cause_by_category.get(primary_category, FailureCause.UNKNOWN),
        Responsibility.LIFT_ARTIFACT,
        tuple(
            f"profile={record.profile_id}: failed_hard_categories="
            f"{[component.category for component in record.comparison.components if component.hard and not component.passed]}"
            for record in failed
        ),
        tuple(record.profile_id for record in result.records if record.passed),
        ("j", "strategy", "location_map"),
        "完整 fixed-C family sweep 存在硬差异；将有序的逐 profile 结构差分一次性"
        "回流 source-aware Lift，已闭合 profile 是下一轮不可回归约束。",
    )


class Gate1FamilyRunner:
    """Run one replay generation over an all-pass fixed-C profile family."""

    def __init__(
        self,
        *,
        compiler: TritonCompiler | None = None,
        comparator: TTIRComparator | None = None,
        leakage_policy: LeakagePolicy | None = None,
    ) -> None:
        self.compiler = compiler or TritonCompiler()
        self.comparator = comparator or TTIRComparator()
        self.leakage_policy = leakage_policy or LeakagePolicy()

    def run(
        self,
        source: KernelSource,
        artifact: Gate1Artifact,
        family: Gate1SpecializationFamily,
        registry: RuleRegistry,
        writer: ReplayWriter | None = None,
        *,
        generation: ReplayGeneration | None = None,
        primary_record: Gate1SpecializationRecord | None = None,
    ) -> Gate1FamilyExecutionResult:
        if (writer is None) == (generation is None):
            raise ValueError(
                "Gate1 family run requires exactly one of writer or generation"
            )
        diagnostics: list[str] = []
        assert_replay_plan_matches(artifact, registry)
        for profile in family.profiles:
            validate_specialization_profile(artifact, profile)
        if primary_record is not None and (
            primary_record.profile_id != family.primary_profile_id
        ):
            raise ValueError(
                "primary_record profile_id differs from family primary"
            )
        source_text, _ = source.read_code()
        leakage = scan_static_leakage(
            artifact.plan,
            source=source_text,
            j=artifact.j,
            policy=self.leakage_policy,
            registry=registry,
        )
        if not leakage.passed:
            return Gate1FamilyExecutionResult(
                Gate1Status.LEAKAGE_REJECTED,
                family,
                artifact,
                None,
                (),
                tuple(item.detail for item in leakage.violations),
            )

        source_results: list[
            tuple[
                Gate1SpecializationProfile,
                CompileResult,
                Any,
                Gate1SpecializationRecord | None,
            ]
        ] = []
        for profile in family.profiles:
            if (
                primary_record is not None
                and profile.profile_id == family.primary_profile_id
            ):
                source_results.append(
                    (
                        profile,
                        primary_record.source_compile,
                        None,
                        primary_record,
                    )
                )
                continue
            compiled = self.compiler.compile(source, profile.case)
            if not compiled.ok or not compiled.ttir:
                source_results.append((profile, compiled, None, None))
                continue
            fingerprint = fingerprint_ttir(
                compiled.ttir,
                python_interface={
                    **dict(compiled.python_interface),
                    "grid": list(profile.case.grid),
                    "grid_contract": list(artifact.interface.grid),
                },
                schedule=profile.case.schedule,
            )
            _validate_compiled_interface(compiled, fingerprint, artifact)
            source_results.append((profile, compiled, fingerprint, None))

        if any(
            not compiled.ok
            or (seed is None and fingerprint is None)
            for _, compiled, fingerprint, seed in source_results
        ):
            records = tuple(
                Gate1SpecializationRecord(
                    profile.profile_id,
                    compiled,
                    diagnostic=compiled.diagnostic or "source compile failed",
                )
                for profile, compiled, _, _ in source_results
            )
            return Gate1FamilyExecutionResult(
                Gate1Status.COMPILE_FAILED,
                family,
                artifact,
                None,
                records,
                ("at least one source specialization failed to compile",),
            )

        if generation is None:
            assert writer is not None
            generation = writer.generate(artifact, registry, masked=False)
        replay_source = KernelSource.from_code(
            generation.code, artifact.interface.function_name
        )
        records: list[Gate1SpecializationRecord] = []
        for profile, source_compile, source_fp, seed in source_results:
            if seed is not None:
                records.append(seed)
                continue
            replay_compile = self.compiler.compile(replay_source, profile.case)
            if not replay_compile.ok or not replay_compile.ttir:
                records.append(
                    Gate1SpecializationRecord(
                        profile.profile_id,
                        source_compile,
                        replay_compile,
                        diagnostic=replay_compile.diagnostic,
                    )
                )
                continue
            replay_fp = fingerprint_ttir(
                replay_compile.ttir,
                python_interface={
                    **dict(replay_compile.python_interface),
                    "grid": list(profile.case.grid),
                    "grid_contract": list(artifact.interface.grid),
                },
                schedule=profile.case.schedule,
            )
            _validate_compiled_interface(replay_compile, replay_fp, artifact)
            comparison = self.comparator.compare(source_fp, replay_fp)
            records.append(
                Gate1SpecializationRecord(
                    profile.profile_id,
                    source_compile,
                    replay_compile,
                    comparison,
                    "" if comparison.hard_pass else "hard TTIR mismatch",
                )
            )
        status = (
            Gate1Status.REPLAY_CLOSED
            if records and all(item.passed for item in records)
            else Gate1Status.NO_CLOSURE
        )
        if status is Gate1Status.NO_CLOSURE:
            diagnostics.append(
                "fixed-C specialization family requires all profiles to hard-close"
            )
        return Gate1FamilyExecutionResult(
            status,
            family,
            artifact,
            generation,
            tuple(records),
            tuple(diagnostics),
        )


__all__ = [
    "GATE1_FAMILY_CONTRACT_DIGEST_SCHEMA",
    "GATE1_FAMILY_DIFF_POLICY",
    "GATE1_FAMILY_RELIFT_GUIDANCE_POLICY",
    "GATE1_SPECIALIZATION_POLICY",
    "Gate1FamilyExecutionResult",
    "Gate1FamilyRunner",
    "Gate1SpecializationFamily",
    "Gate1SpecializationProfile",
    "Gate1SpecializationRecord",
    "build_family_relift_guidance",
    "diagnose_family_failure",
    "family_hard_difference_digest",
    "specialization_family_contract_digest",
    "specialization_family_contract_payload",
    "validate_specialization_profile",
]
