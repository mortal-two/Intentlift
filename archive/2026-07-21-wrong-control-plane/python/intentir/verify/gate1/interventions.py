"""Deterministic paired J interventions for Gate1 causality evidence.

The replay writer sees only the ordinary source-isolated artifact.  Mutation
metadata and the expected TTIR delta remain on the judge side, so a writer
cannot optimize its answer against the intervention oracle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .compare import TTIRComparison, compare_fingerprints
from .composer import materialize_replay_plan
from .compiler import CompileResult
from .model import Gate1Artifact, SchemaError
from .replay import ReplayGeneration
from .rules import RuleRegistry, default_triton_rule_registry
from .ttir import TTIRFingerprint


_BODY_PATH = re.compile(
    r"^(?:intent_function\.)?body\[(\d+)\](?:\.body\[(\d+)\])*$"
)


PAIRED_INTERVENTION_FAMILY_POLICY = (
    "fixed_c_family_exact_delta_all_profiles_v1"
)
PAIRED_INTERVENTION_FAMILY_DIGEST_SCHEMA = (
    "paired_intervention_fixed_c_family_evidence_v2"
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


class InterventionKind(str, Enum):
    BINARY_FUNCTION_REPLACEMENT = "binary_function_replacement"
    OUTPUT_CONSTANT_REPLACEMENT = "output_constant_replacement"
    MASKED_LOAD_FILL_REPLACEMENT = "masked_load_fill_replacement"
    REDUCTION_SUM_TO_MAX = "reduction_sum_to_max"


class InterventionStatus(str, Enum):
    PASS = "pass"
    IGNORED_J = "ignored_j"
    WRONG_OPERATOR_DELTA = "wrong_operator_delta"
    WRONG_CONSTANT_DELTA = "wrong_constant_delta"
    WRONG_FILL_DELTA = "wrong_fill_delta"
    OFF_TARGET_DRIFT = "off_target_drift"
    GENERATION_FAILED = "generation_failed"
    COMPILE_FAILED = "compile_failed"
    EVALUATION_FAILED = "evaluation_failed"


INTERVENTION_SEMANTIC_FAILURE_STATUSES = frozenset(
    {
        InterventionStatus.IGNORED_J,
        InterventionStatus.WRONG_OPERATOR_DELTA,
        InterventionStatus.WRONG_CONSTANT_DELTA,
        InterventionStatus.WRONG_FILL_DELTA,
        InterventionStatus.OFF_TARGET_DRIFT,
    }
)


@dataclass(frozen=True)
class SemanticIntervention:
    intervention_id: str
    kind: InterventionKind
    target_ref: str
    semantic_path: str
    before: str
    after: str
    expected_removed_op: str | None
    expected_added_op: str | None
    expected_fill_before: str | None
    expected_fill_after: str | None
    expected_fill_changes: int
    expected_store_before: str | None
    expected_store_after: str | None
    expected_store_changes: int
    expected_changed_categories: tuple[str, ...]
    allowed_changed_category_sets: tuple[tuple[str, ...], ...]
    baseline_j_digest: str
    intervention_j_digest: str
    artifact: Gate1Artifact

    def to_json(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "kind": self.kind.value,
            "target_ref": self.target_ref,
            "semantic_path": self.semantic_path,
            "before": self.before,
            "after": self.after,
            "expected_removed_op": self.expected_removed_op,
            "expected_added_op": self.expected_added_op,
            "expected_fill_before": self.expected_fill_before,
            "expected_fill_after": self.expected_fill_after,
            "expected_fill_changes": self.expected_fill_changes,
            "expected_store_before": self.expected_store_before,
            "expected_store_after": self.expected_store_after,
            "expected_store_changes": self.expected_store_changes,
            "expected_changed_categories": list(
                self.expected_changed_categories
            ),
            "allowed_changed_category_sets": [
                list(categories)
                for categories in self.allowed_changed_category_sets
            ],
            "baseline_j_digest": self.baseline_j_digest,
            "intervention_j_digest": self.intervention_j_digest,
        }


@dataclass(frozen=True)
class InterventionVerdict:
    status: InterventionStatus
    comparison: TTIRComparison
    changed_categories: tuple[str, ...]
    operator_delta: tuple[tuple[str, int], ...]
    memory_fill_delta: tuple[tuple[str, int], ...]
    store_value_delta: tuple[tuple[str, int], ...]
    reduction_delta: tuple[tuple[str, int], ...]
    diagnostic: str

    @property
    def passed(self) -> bool:
        return self.status is InterventionStatus.PASS

    def exact_delta_payload(self) -> dict[str, Any]:
        """Return score-free, prose-free semantic evidence for hashing."""

        hard_components: list[dict[str, Any]] = []
        for component in sorted(
            (item for item in self.comparison.components if item.hard),
            key=lambda item: item.category,
        ):
            differences = [
                {
                    "kind": difference.kind,
                    "item": difference.item,
                    "expected_count": difference.expected_count,
                    "actual_count": difference.actual_count,
                }
                for difference in component.differences
            ]
            differences.sort(key=_stable_json)
            hard_components.append(
                {
                    "category": component.category,
                    "applicable": component.applicable,
                    "passed": component.passed,
                    "differences": differences,
                }
            )
        return {
            "status": self.status.value,
            "changed_categories": list(self.changed_categories),
            "operator_delta": [list(item) for item in self.operator_delta],
            "memory_fill_delta": [
                list(item) for item in self.memory_fill_delta
            ],
            "store_value_delta": [
                list(item) for item in self.store_value_delta
            ],
            "reduction_delta": [
                list(item) for item in self.reduction_delta
            ],
            "hard_pass": self.comparison.hard_pass,
            "hard_components": hard_components,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "changed_categories": list(self.changed_categories),
            "operator_delta": dict(self.operator_delta),
            "memory_fill_delta": dict(self.memory_fill_delta),
            "store_value_delta": dict(self.store_value_delta),
            "reduction_delta": dict(self.reduction_delta),
            "diagnostic": self.diagnostic,
            "comparison": self.comparison.to_dict(),
        }


@dataclass(frozen=True)
class InterventionRecord:
    spec: SemanticIntervention
    status: InterventionStatus
    generation: ReplayGeneration | None = None
    compile: CompileResult | None = None
    verdict: InterventionVerdict | None = None
    diagnostic: str = ""
    profile_records: tuple["InterventionProfileRecord", ...] = ()
    family_policy: str | None = None
    baseline_code_sha256: str | None = None
    family_id: str | None = None
    family_contract_digest: str | None = None
    family_outcome: str | None = None

    @property
    def passed(self) -> bool:
        return bool(
            self.status is InterventionStatus.PASS
            and (
                not self.profile_records
                or all(item.passed for item in self.profile_records)
            )
        )

    @property
    def family_evidence_complete(self) -> bool:
        return bool(
            self.profile_records
            and all(item.verdict is not None for item in self.profile_records)
        )

    @property
    def family_digest(self) -> str | None:
        if (
            not self.profile_records
            or self.generation is None
            or self.baseline_code_sha256 is None
            or self.family_id is None
            or self.family_contract_digest is None
            or self.family_outcome is None
        ):
            return None
        return _digest(
            {
                "digest_schema": PAIRED_INTERVENTION_FAMILY_DIGEST_SCHEMA,
                "policy": self.family_policy,
                "family_id": self.family_id,
                "family_contract_digest": self.family_contract_digest,
                "family_outcome": self.family_outcome,
                "intervention_id": self.spec.intervention_id,
                "baseline_j_digest": self.spec.baseline_j_digest,
                "intervention_j_digest": self.spec.intervention_j_digest,
                "baseline_code_sha256": self.baseline_code_sha256,
                "mutant_code_sha256": self.generation.code_sha256,
                "profiles": [
                    {
                        "profile_id": item.profile_id,
                        "status": item.status.value,
                        "exact_delta": (
                            None
                            if item.verdict is None
                            else item.verdict.exact_delta_payload()
                        ),
                    }
                    for item in sorted(
                        self.profile_records,
                        key=lambda record: record.profile_id,
                    )
                ],
            }
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_json(),
            "status": self.status.value,
            "passed": self.passed,
            "generation": None
            if self.generation is None
            else {
                "prompt_sha256": self.generation.prompt_sha256,
                "requested_model": self.generation.requested_model,
                "generation_id": self.generation.generation_id,
                "code_sha256": self.generation.code_sha256,
                "response_model": self.generation.response_model,
                "response_id": self.generation.response_id,
                "masked": self.generation.masked,
                "ablation_kind": self.generation.ablation_kind,
            },
            "compile": None
            if self.compile is None
            else {
                "ok": self.compile.ok,
                "diagnostic": self.compile.diagnostic,
                "cache_key": self.compile.cache_key,
                "cache_hit": self.compile.cache_hit,
                "elapsed_ms": self.compile.elapsed_ms,
            },
            "verdict": None if self.verdict is None else self.verdict.to_json(),
            "diagnostic": self.diagnostic,
            "family_policy": self.family_policy,
            "baseline_code_sha256": self.baseline_code_sha256,
            "family_id": self.family_id,
            "family_contract_digest": self.family_contract_digest,
            "family_outcome": self.family_outcome,
            "family_evidence_complete": self.family_evidence_complete,
            "family_digest": self.family_digest,
            "profile_records": [
                item.to_json()
                for item in sorted(
                    self.profile_records,
                    key=lambda record: record.profile_id,
                )
            ],
        }


@dataclass(frozen=True)
class InterventionProfileRecord:
    """One fixed-C profile observation for a shared mutant replay.

    The parent :class:`InterventionRecord` owns the single replay generation.
    Per-profile records contain only the two compilations and the exact delta
    verdict, so they cannot be mistaken for profile-local writer invocations.
    """

    profile_id: str
    status: InterventionStatus
    baseline_compile: CompileResult | None = None
    compile: CompileResult | None = None
    verdict: InterventionVerdict | None = None
    diagnostic: str = ""
    baseline_compile_reused: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError(
                "intervention profile_id must be a non-empty string"
            )
        if not isinstance(self.status, InterventionStatus):
            object.__setattr__(
                self, "status", InterventionStatus(self.status)
            )

    @property
    def passed(self) -> bool:
        return self.status is InterventionStatus.PASS

    @staticmethod
    def _compile_json(
        result: CompileResult | None,
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "ok": result.ok,
            "diagnostic": result.diagnostic,
            "cache_key": result.cache_key,
            "cache_hit": result.cache_hit,
            "elapsed_ms": result.elapsed_ms,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "status": self.status.value,
            "passed": self.passed,
            "baseline_compile": self._compile_json(
                self.baseline_compile
            ),
            "baseline_compile_reused": self.baseline_compile_reused,
            "compile": self._compile_json(self.compile),
            "verdict": (
                None if self.verdict is None else self.verdict.to_json()
            ),
            "diagnostic": self.diagnostic,
        }


_FLOAT_BINARY_REPLACEMENTS = {
    "add": ("sub", "arith.addf", "arith.subf"),
    "sub": ("add", "arith.subf", "arith.addf"),
    "mul": ("add", "arith.mulf", "arith.addf"),
    "div": ("mul", "arith.divf", "arith.mulf"),
}


def _body_path_indices(path: str) -> tuple[int, ...]:
    if _BODY_PATH.fullmatch(path) is None:
        raise SchemaError(f"unsupported IntentIR node path for intervention: {path!r}")
    return tuple(int(value) for value in re.findall(r"body\[(\d+)\]", path))


def _operation_at(intent: dict[str, Any], path: str) -> dict[str, Any]:
    indices = _body_path_indices(path)
    operations = intent["intent_function"]["body"]
    operation: dict[str, Any] | None = None
    for depth, index in enumerate(indices):
        if index >= len(operations):
            raise SchemaError(f"IntentIR node path is out of range: {path!r}")
        operation = operations[index]
        if depth + 1 < len(indices):
            operations = operation.get("body", [])
    assert operation is not None
    return operation


def _result_dtype(intent: dict[str, Any], operation: dict[str, Any]) -> str | None:
    type_record = operation.get("type")
    if isinstance(type_record, dict) and isinstance(type_record.get("dtype"), str):
        return type_record["dtype"]
    result = operation.get("result")
    if not isinstance(result, str):
        return None
    function = intent["intent_function"]
    for tensor in [*function.get("inputs", []), *function.get("outputs", [])]:
        if tensor.get("name") == result and isinstance(tensor.get("dtype"), str):
            return tensor["dtype"]
    for candidate in function.get("body", []):
        if candidate.get("result") != result:
            continue
        candidate_type = candidate.get("type")
        if isinstance(candidate_type, dict) and isinstance(
            candidate_type.get("dtype"), str
        ):
            return candidate_type["dtype"]
    return None


def _nodes_upstream_of_reductions(
    intent: dict[str, Any],
    node_ids: dict[str, str],
    reduction_node_refs: set[str],
) -> set[str]:
    """Return node ids whose results can flow into a boundary reduction."""

    path_to_id = {path: node_id for node_id, path in node_ids.items()}

    def canonical_ref(value: str) -> str | None:
        if value in node_ids:
            return value
        return path_to_id.get(value)

    operations = {
        node_id: _operation_at(intent, path)
        for node_id, path in node_ids.items()
    }
    producer_by_value: dict[str, str] = {}
    for node_id, operation in operations.items():
        result = operation.get("result")
        values = [result] if isinstance(result, str) else result
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    producer_by_value[value] = node_id

    consumers_by_node: dict[str, set[str]] = {
        node_id: set() for node_id in operations
    }
    for consumer_id, operation in operations.items():
        for operand in operation.get("operands", []):
            producer_id = producer_by_value.get(operand)
            if producer_id is not None:
                consumers_by_node[producer_id].add(consumer_id)

    upstream = {
        canonical
        for reference in reduction_node_refs
        if (canonical := canonical_ref(reference)) is not None
    }
    changed = True
    while changed:
        changed = False
        for producer_id, consumers in consumers_by_node.items():
            if producer_id not in upstream and consumers & upstream:
                upstream.add(producer_id)
                changed = True
    return upstream


def _floating_dtype(dtype: str | None) -> bool:
    if dtype is None:
        return False
    normalized = dtype.casefold().removeprefix("torch.")
    return normalized.startswith(("f", "bf")) or normalized in {
        "float",
        "half",
        "double",
        "bfloat16",
        "float16",
        "float32",
        "float64",
    }


def _constant_value(signature: str) -> float | None:
    dense = re.search(r"\bdense<([^>]+)>", signature)
    if dense is not None:
        token = dense.group(1).strip()
    else:
        scalar = re.search(r"\barith\.constant\s+([^\s:]+)", signature)
        if scalar is None:
            return None
        token = scalar.group(1).strip()
    try:
        return float(token)
    except ValueError:
        pass
    if not token.casefold().startswith("0x"):
        return None
    try:
        bits = int(token, 16)
        if re.search(r"f32\b", signature):
            return float(struct.unpack("!f", bits.to_bytes(4, "big"))[0])
        if re.search(r"f64\b", signature):
            return float(struct.unpack("!d", bits.to_bytes(8, "big"))[0])
        if re.search(r"f16\b", signature):
            return float(struct.unpack("!e", bits.to_bytes(2, "big"))[0])
    except (OverflowError, ValueError, struct.error):
        return None
    return None


def _constant_type(signature: str) -> str:
    marker = signature.rfind(":")
    return signature[marker + 1 :].strip() if marker >= 0 else ""


def _expected_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"-inf", "-infinity"}:
        return float("-inf")
    if normalized in {"+inf", "inf", "+infinity", "infinity"}:
        return float("inf")
    return float(value)


def _fill_counter(fingerprint: TTIRFingerprint) -> Counter[str]:
    return Counter(
        effect.other
        for effect in fingerprint.memory_effects
        if effect.kind == "load" and effect.masked and effect.other is not None
    )


def _store_value_counter(fingerprint: TTIRFingerprint) -> Counter[str]:
    return Counter(
        effect.other
        for effect in fingerprint.memory_effects
        if effect.kind == "store" and effect.other is not None
    )


def _reduction_counter(fingerprint: TTIRFingerprint) -> Counter[str]:
    return Counter(_stable_json(item.to_dict()) for item in fingerprint.reductions)


def build_semantic_interventions(
    artifact: Gate1Artifact,
    *,
    limit: int = 1,
    registry: RuleRegistry | None = None,
) -> tuple[SemanticIntervention, ...]:
    """Return deterministic, type-preserving J→recompose interventions."""

    if limit < 0:
        raise ValueError("intervention limit must be non-negative")
    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    registry = registry if registry is not None else default_triton_rule_registry()
    if limit == 0:
        return ()

    baseline = artifact.to_json()
    intent = baseline["j"]["intent"]
    function = intent["intent_function"]
    boundary = baseline["j"]["boundary"]
    node_items = sorted(
        baseline["j"]["node_ids"].items(),
        key=lambda item: _body_path_indices(item[1]),
    )
    result: list[SemanticIntervention] = []
    for node_ref, path in node_items:
        operation = _operation_at(intent, path)
        attrs = operation.get("attrs", {})
        binary_fn = attrs.get("fn") if isinstance(attrs, dict) else None
        if (
            operation.get("op") != "map_binary"
            or binary_fn not in _FLOAT_BINARY_REPLACEMENTS
            or len(operation.get("operands", [])) != 2
            or not _floating_dtype(_result_dtype(intent, operation))
        ):
            continue
        replacement, removed_op, added_op = _FLOAT_BINARY_REPLACEMENTS[binary_fn]
        changed = copy.deepcopy(baseline)
        changed_operation = _operation_at(changed["j"]["intent"], path)
        changed_operation["attrs"]["fn"] = replacement

        for frozen in (
            "interface",
            "schedule",
            "strategy",
            "location_map",
            "source",
            "provenance",
        ):
            if changed.get(frozen) != baseline.get(frozen):
                raise AssertionError(f"semantic intervention changed frozen {frozen}")
        if changed["j"]["node_ids"] != baseline["j"]["node_ids"]:
            raise AssertionError("semantic intervention changed J node identities")
        if changed["j"]["boundary"] != baseline["j"]["boundary"]:
            raise AssertionError("binary semantic intervention changed J boundary")

        changed_artifact, _ = materialize_replay_plan(
            Gate1Artifact.from_json(changed), registry
        )
        result.append(
            SemanticIntervention(
                intervention_id=f"binary_fn:{node_ref}:{binary_fn}_to_{replacement}",
                kind=InterventionKind.BINARY_FUNCTION_REPLACEMENT,
                target_ref=node_ref,
                semantic_path=f"j.intent.intent_function.{path}.attrs.fn",
                before=binary_fn,
                after=replacement,
                expected_removed_op=removed_op,
                expected_added_op=added_op,
                expected_fill_before=None,
                expected_fill_after=None,
                expected_fill_changes=0,
                expected_store_before=None,
                expected_store_after=None,
                expected_store_changes=0,
                expected_changed_categories=("dataflow", "op_multiset"),
                allowed_changed_category_sets=(("dataflow", "op_multiset"),),
                baseline_j_digest=_digest(baseline["j"]),
                intervention_j_digest=_digest(changed["j"]),
                artifact=changed_artifact,
            )
        )
        if len(result) >= limit:
            break
    if len(result) >= limit:
        return tuple(result)

    # The first constant coordinate is deliberately narrow: a no-input,
    # single-output kernel whose sole IntentIR node directly defines that output.
    # This makes the TTIR store-value coordinate unique and avoids guessing which
    # of several arithmetic/index constants a global constant bag refers to.
    output_mappings = boundary.get("output_mappings", [])
    accesses = boundary.get("accesses", [])
    for node_ref, path in node_items:
        operation = _operation_at(intent, path)
        attrs = operation.get("attrs", {})
        value = attrs.get("value") if isinstance(attrs, dict) else None
        result_name = operation.get("result")
        matching_outputs = [
            mapping
            for mapping in output_mappings
            if mapping.get("node_ref") == node_ref
            and mapping.get("output") == result_name
            and mapping.get("value") == result_name
        ]
        matching_stores = [
            access
            for access in accesses
            if access.get("kind") == "store"
            and access.get("node_ref") == node_ref
            and access.get("tensor") == result_name
        ]
        if (
            operation.get("op") != "const"
            or len(node_items) != 1
            or len(function.get("body", [])) != 1
            or function.get("inputs")
            or len(function.get("outputs", [])) != 1
            or function["outputs"][0].get("name") != result_name
            or len(matching_outputs) != 1
            or len(matching_stores) != 1
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) != 1.0
            or not _floating_dtype(attrs.get("dtype"))
        ):
            continue
        changed = copy.deepcopy(baseline)
        _operation_at(changed["j"]["intent"], path)["attrs"]["value"] = 0.0
        for frozen in (
            "interface",
            "schedule",
            "strategy",
            "location_map",
            "source",
            "provenance",
        ):
            if changed.get(frozen) != baseline.get(frozen):
                raise AssertionError(f"semantic intervention changed frozen {frozen}")
        if changed["j"]["node_ids"] != baseline["j"]["node_ids"]:
            raise AssertionError("constant intervention changed J node identities")
        if changed["j"]["boundary"] != baseline["j"]["boundary"]:
            raise AssertionError("constant intervention changed J boundary")

        changed_artifact, _ = materialize_replay_plan(
            Gate1Artifact.from_json(changed), registry
        )
        result.append(
            SemanticIntervention(
                intervention_id=f"output_const:{node_ref}:1_to_0",
                kind=InterventionKind.OUTPUT_CONSTANT_REPLACEMENT,
                target_ref=node_ref,
                semantic_path=f"j.intent.intent_function.{path}.attrs.value",
                before="1",
                after="0",
                expected_removed_op=None,
                expected_added_op=None,
                expected_fill_before=None,
                expected_fill_after=None,
                expected_fill_changes=0,
                expected_store_before="1",
                expected_store_after="0",
                expected_store_changes=1,
                expected_changed_categories=("dataflow", "predicate_effect"),
                allowed_changed_category_sets=(("dataflow", "predicate_effect"),),
                baseline_j_digest=_digest(baseline["j"]),
                intervention_j_digest=_digest(changed["j"]),
                artifact=changed_artifact,
            )
        )
        if len(result) >= limit:
            return tuple(result)

    for node_ref, path in node_items:
        operation = _operation_at(intent, path)
        attrs = operation.get("attrs", {})
        if (
            operation.get("op") != "reduce"
            or not isinstance(attrs, dict)
            or attrs.get("fn") != "sum"
            or not _floating_dtype(_result_dtype(intent, operation))
        ):
            continue
        reductions = [
            (index, reduction)
            for index, reduction in enumerate(boundary.get("reductions", []))
            if reduction.get("node_ref") == node_ref
        ]
        matching_loads = [
            (index, access)
            for index, access in enumerate(boundary.get("accesses", []))
            if access.get("kind") == "load"
            and access.get("node_ref") == node_ref
            and access.get("predicate") is not None
            and not isinstance(access.get("fill_value"), bool)
            and isinstance(access.get("fill_value"), (int, float))
            and float(access["fill_value"]) == 0.0
        ]
        if len(reductions) != 1 or len(matching_loads) != 1:
            continue
        reduction_index, reduction = reductions[0]
        if (
            isinstance(reduction.get("identity"), bool)
            or isinstance(reduction.get("masked_contribution"), bool)
            or not isinstance(reduction.get("identity"), (int, float))
            or not isinstance(
                reduction.get("masked_contribution"), (int, float)
            )
            or float(reduction["identity"]) != 0.0
            or float(reduction["masked_contribution"]) != 0.0
        ):
            continue
        access_index, access = matching_loads[0]
        changed = copy.deepcopy(baseline)
        _operation_at(changed["j"]["intent"], path)["attrs"]["fn"] = "max"
        changed_reduction = changed["j"]["boundary"]["reductions"][
            reduction_index
        ]
        changed_reduction["identity"] = "-inf"
        changed_reduction["masked_contribution"] = "-inf"
        changed["j"]["boundary"]["accesses"][access_index][
            "fill_value"
        ] = "-inf"
        for frozen in (
            "interface",
            "schedule",
            "strategy",
            "location_map",
            "source",
            "provenance",
        ):
            if changed.get(frozen) != baseline.get(frozen):
                raise AssertionError(f"semantic intervention changed frozen {frozen}")
        if changed["j"]["node_ids"] != baseline["j"]["node_ids"]:
            raise AssertionError("reduction intervention changed J node identities")

        changed_artifact, _ = materialize_replay_plan(
            Gate1Artifact.from_json(changed), registry
        )
        result.append(
            SemanticIntervention(
                intervention_id=f"reduction:{node_ref}:sum_to_max",
                kind=InterventionKind.REDUCTION_SUM_TO_MAX,
                target_ref=node_ref,
                semantic_path=(
                    f"j.intent.intent_function.{path}.attrs.fn+"
                    f"j.boundary.reductions[{node_ref}]+"
                    f"j.boundary.accesses[{access['access_id']}].fill_value"
                ),
                before="sum/0/0",
                after="max/-inf/-inf",
                expected_removed_op="arith.addf",
                expected_added_op="arith.maxnumf",
                expected_fill_before="0",
                expected_fill_after="-inf",
                expected_fill_changes=1,
                expected_store_before=None,
                expected_store_after=None,
                expected_store_changes=0,
                expected_changed_categories=(
                    "dataflow",
                    "op_multiset",
                    "predicate_effect",
                    "reduction",
                ),
                allowed_changed_category_sets=((
                    "dataflow",
                    "op_multiset",
                    "predicate_effect",
                    "reduction",
                ),),
                baseline_j_digest=_digest(baseline["j"]),
                intervention_j_digest=_digest(changed["j"]),
                artifact=changed_artifact,
            )
        )
        if len(result) >= limit:
            return tuple(result)

    tensor_dtypes = {
        tensor["name"]: tensor.get("dtype")
        for tensor in [*function.get("inputs", []), *function.get("outputs", [])]
    }
    accesses = baseline["j"]["boundary"].get("accesses", [])
    reduction_node_refs = {
        reduction["node_ref"]
        for reduction in baseline["j"]["boundary"].get("reductions", [])
    }
    reduction_upstream_nodes = _nodes_upstream_of_reductions(
        intent,
        baseline["j"]["node_ids"],
        reduction_node_refs,
    )
    path_to_node = {
        path: node_id for node_id, path in baseline["j"]["node_ids"].items()
    }
    for index, access in enumerate(accesses):
        fill_value = access.get("fill_value")
        access_node_ref = access.get("node_ref")
        canonical_access_node = (
            access_node_ref
            if access_node_ref in baseline["j"]["node_ids"]
            else path_to_node.get(access_node_ref)
        )
        if (
            access.get("kind") != "load"
            or access.get("predicate") is None
            or isinstance(fill_value, bool)
            or not isinstance(fill_value, (int, float))
            or float(fill_value) != 0.0
            or not _floating_dtype(tensor_dtypes.get(access.get("tensor")))
            or (
                bool(reduction_node_refs)
                and (
                    canonical_access_node is None
                    or canonical_access_node in reduction_upstream_nodes
                )
            )
        ):
            continue
        changed = copy.deepcopy(baseline)
        changed_access = changed["j"]["boundary"]["accesses"][index]
        changed_access["fill_value"] = 1.0
        for frozen in (
            "interface",
            "schedule",
            "strategy",
            "location_map",
            "source",
            "provenance",
        ):
            if changed.get(frozen) != baseline.get(frozen):
                raise AssertionError(f"semantic intervention changed frozen {frozen}")
        if changed["j"]["intent"] != baseline["j"]["intent"]:
            raise AssertionError("fill intervention changed J intent")
        if changed["j"]["node_ids"] != baseline["j"]["node_ids"]:
            raise AssertionError("fill intervention changed J node identities")

        changed_artifact, _ = materialize_replay_plan(
            Gate1Artifact.from_json(changed), registry
        )
        access_id = access["access_id"]
        result.append(
            SemanticIntervention(
                intervention_id=f"masked_fill:{access_id}:0_to_1",
                kind=InterventionKind.MASKED_LOAD_FILL_REPLACEMENT,
                target_ref=access_id,
                semantic_path=(
                    f"j.boundary.accesses[{access_id}].fill_value"
                ),
                before="0",
                after="1",
                expected_removed_op=None,
                expected_added_op=None,
                expected_fill_before="0",
                expected_fill_after="1",
                expected_fill_changes=1,
                expected_store_before=None,
                expected_store_after=None,
                expected_store_changes=0,
                expected_changed_categories=("dataflow", "predicate_effect"),
                allowed_changed_category_sets=(
                    ("dataflow", "predicate_effect"),
                    ("dataflow", "op_multiset", "predicate_effect"),
                ),
                baseline_j_digest=_digest(baseline["j"]),
                intervention_j_digest=_digest(changed["j"]),
                artifact=changed_artifact,
            )
        )
        if len(result) >= limit:
            break
    return tuple(result)


def evaluate_semantic_intervention(
    spec: SemanticIntervention,
    baseline: TTIRFingerprint,
    intervention: TTIRFingerprint,
) -> InterventionVerdict:
    """Judge an observed replay delta using exact counters and categories."""

    comparison = compare_fingerprints(baseline, intervention)
    changed_categories = tuple(
        sorted(
            component.category
            for component in comparison.components
            if component.hard and not component.passed
        )
    )
    before_ops = Counter(dict(baseline.op_multiset))
    after_ops = Counter(dict(intervention.op_multiset))
    operator_delta = tuple(
        sorted(
            (name, after_ops[name] - before_ops[name])
            for name in before_ops.keys() | after_ops.keys()
            if after_ops[name] != before_ops[name]
        )
    )
    before_fills = _fill_counter(baseline)
    after_fills = _fill_counter(intervention)
    memory_fill_delta = tuple(
        sorted(
            (value, after_fills[value] - before_fills[value])
            for value in before_fills.keys() | after_fills.keys()
            if after_fills[value] != before_fills[value]
        )
    )
    before_stores = _store_value_counter(baseline)
    after_stores = _store_value_counter(intervention)
    store_value_delta = tuple(
        sorted(
            (value, after_stores[value] - before_stores[value])
            for value in before_stores.keys() | after_stores.keys()
            if after_stores[value] != before_stores[value]
        )
    )
    before_reductions = _reduction_counter(baseline)
    after_reductions = _reduction_counter(intervention)
    reduction_delta = tuple(
        sorted(
            (value, after_reductions[value] - before_reductions[value])
            for value in before_reductions.keys() | after_reductions.keys()
            if after_reductions[value] != before_reductions[value]
        )
    )

    def typed_constant_delta_matches(
        delta: tuple[tuple[str, int], ...],
        *,
        before: str | None,
        after: str | None,
        changes: int,
    ) -> bool:
        if changes == 0:
            return not delta
        removed = [
            value
            for value, count in delta
            if count == -changes
        ]
        added = [
            value
            for value, count in delta
            if count == changes
        ]
        return (
            len(removed) == 1
            and len(added) == 1
            and len(delta) == 2
            and _constant_value(removed[0])
            == _expected_float(before)
            and _constant_value(added[0])
            == _expected_float(after)
            and _constant_type(removed[0]) == _constant_type(added[0])
        )

    def fill_delta_matches() -> bool:
        return typed_constant_delta_matches(
            memory_fill_delta,
            before=spec.expected_fill_before,
            after=spec.expected_fill_after,
            changes=spec.expected_fill_changes,
        )

    def store_delta_matches() -> bool:
        return typed_constant_delta_matches(
            store_value_delta,
            before=spec.expected_store_before,
            after=spec.expected_store_after,
            changes=spec.expected_store_changes,
        )
    if comparison.hard_pass:
        status = InterventionStatus.IGNORED_J
        diagnostic = "intervention replay is semantically identical to baseline"
    elif spec.kind is InterventionKind.BINARY_FUNCTION_REPLACEMENT:
        assert spec.expected_removed_op is not None
        assert spec.expected_added_op is not None
        expected_delta = tuple(
            sorted(
                (
                    (spec.expected_removed_op, -1),
                    (spec.expected_added_op, 1),
                )
            )
        )
        target_delta = {
            spec.expected_removed_op: dict(operator_delta).get(
                spec.expected_removed_op, 0
            ),
            spec.expected_added_op: dict(operator_delta).get(
                spec.expected_added_op, 0
            ),
        }
        expected_target_delta = {
            spec.expected_removed_op: -1,
            spec.expected_added_op: 1,
        }
        if target_delta != expected_target_delta:
            status = InterventionStatus.WRONG_OPERATOR_DELTA
            diagnostic = (
                f"target operator delta {target_delta!r} != "
                f"expected {expected_target_delta!r}"
            )
        elif operator_delta != expected_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = (
                f"operator delta contains off-target changes: "
                f"{dict(operator_delta)!r} != {dict(expected_delta)!r}"
            )
        elif memory_fill_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "binary intervention changed a masked-load fill"
        elif store_value_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "binary intervention changed a direct store constant"
        elif reduction_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "binary intervention changed a reduction fingerprint"
        else:
            status = InterventionStatus.PASS
            diagnostic = "target operator delta is exact"
    elif spec.kind is InterventionKind.OUTPUT_CONSTANT_REPLACEMENT:
        if not store_delta_matches():
            status = InterventionStatus.WRONG_CONSTANT_DELTA
            diagnostic = (
                "direct output-store constant delta is not exactly typed 1 -> 0: "
                f"{dict(store_value_delta)!r}"
            )
        elif operator_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = (
                "output constant intervention changed operator counts: "
                f"{dict(operator_delta)!r}"
            )
        elif memory_fill_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "output constant intervention changed a masked-load fill"
        elif reduction_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "output constant intervention changed a reduction fingerprint"
        else:
            status = InterventionStatus.PASS
            diagnostic = "direct output-store constant delta is exactly typed 1 -> 0"
    elif spec.kind is InterventionKind.MASKED_LOAD_FILL_REPLACEMENT:
        if not fill_delta_matches():
            status = InterventionStatus.WRONG_FILL_DELTA
            diagnostic = (
                "masked-load fill delta is not exactly typed 0 -> 1: "
                f"{dict(memory_fill_delta)!r}"
            )
        elif any(
            name != "arith.constant" or abs(count) != 1
            for name, count in operator_delta
        ):
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = (
                "fill intervention changed non-constant operator counts: "
                f"{dict(operator_delta)!r}"
            )
        elif reduction_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "fill intervention changed a reduction fingerprint"
        elif store_value_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "fill intervention changed a direct store constant"
        else:
            status = InterventionStatus.PASS
            diagnostic = "masked-load fill delta is exactly typed 0 -> 1"
    elif spec.kind is InterventionKind.REDUCTION_SUM_TO_MAX:
        assert spec.expected_removed_op is not None
        assert spec.expected_added_op is not None
        target_delta = {
            spec.expected_removed_op: dict(operator_delta).get(
                spec.expected_removed_op, 0
            ),
            spec.expected_added_op: dict(operator_delta).get(
                spec.expected_added_op, 0
            ),
        }
        expected_target_delta = {
            spec.expected_removed_op: -1,
            spec.expected_added_op: 1,
        }
        extra_ops = {
            name: count
            for name, count in operator_delta
            if name not in {
                spec.expected_removed_op,
                spec.expected_added_op,
                "arith.constant",
            }
        }
        removed_reductions = [
            json.loads(value)
            for value, count in reduction_delta
            if count == -1
        ]
        added_reductions = [
            json.loads(value)
            for value, count in reduction_delta
            if count == 1
        ]
        reduction_exact = (
            len(removed_reductions) == 1
            and len(added_reductions) == 1
            and len(reduction_delta) == 2
            and removed_reductions[0]["combiners"] == ["arith.addf"]
            and added_reductions[0]["combiners"] == ["arith.maxnumf"]
            and {
                key: value
                for key, value in removed_reductions[0].items()
                if key != "combiners"
            }
            == {
                key: value
                for key, value in added_reductions[0].items()
                if key != "combiners"
            }
        )
        if target_delta != expected_target_delta:
            status = InterventionStatus.WRONG_OPERATOR_DELTA
            diagnostic = (
                f"target reduction operator delta {target_delta!r} != "
                f"expected {expected_target_delta!r}"
            )
        elif extra_ops or abs(dict(operator_delta).get("arith.constant", 0)) > 1:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = (
                "reduction intervention changed off-target operator counts: "
                f"{dict(operator_delta)!r}"
            )
        elif not reduction_exact:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = (
                "reduction fingerprint is not exactly addf -> maxnumf with "
                "axis/arity preserved"
            )
        elif not fill_delta_matches():
            status = InterventionStatus.WRONG_FILL_DELTA
            diagnostic = (
                "reduction fill delta is not exactly typed 0 -> -inf: "
                f"{dict(memory_fill_delta)!r}"
            )
        elif store_value_delta:
            status = InterventionStatus.OFF_TARGET_DRIFT
            diagnostic = "reduction intervention changed a direct store constant"
        else:
            status = InterventionStatus.PASS
            diagnostic = "sum/0/0 -> max/-inf/-inf delta is exact"
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(f"unsupported intervention kind: {spec.kind}")

    if status is InterventionStatus.PASS and (
        changed_categories not in spec.allowed_changed_category_sets
    ):
        status = InterventionStatus.OFF_TARGET_DRIFT
        diagnostic = (
            f"changed categories {changed_categories!r} not in allowed "
            f"{spec.allowed_changed_category_sets!r}"
        )
    elif status is InterventionStatus.PASS and not comparison.component(
        "schedule-only"
    ).passed:
        status = InterventionStatus.OFF_TARGET_DRIFT
        diagnostic = "semantic intervention changed schedule-only fingerprint"
    elif status is InterventionStatus.PASS:
        diagnostic = "observed TTIR delta exactly matches the registered intervention"
    return InterventionVerdict(
        status,
        comparison,
        changed_categories,
        operator_delta,
        memory_fill_delta,
        store_value_delta,
        reduction_delta,
        diagnostic,
    )


__all__ = [
    "INTERVENTION_SEMANTIC_FAILURE_STATUSES",
    "PAIRED_INTERVENTION_FAMILY_DIGEST_SCHEMA",
    "PAIRED_INTERVENTION_FAMILY_POLICY",
    "InterventionKind",
    "InterventionProfileRecord",
    "InterventionRecord",
    "InterventionStatus",
    "InterventionVerdict",
    "SemanticIntervention",
    "build_semantic_interventions",
    "evaluate_semantic_intervention",
]
