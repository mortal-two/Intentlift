"""Static anti-leakage checks for Gate1 semantic archives and replay plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .model import (
    Gate1Artifact,
    ReplayPlan,
    SchemaError,
    SemanticArchive,
    StrictJSONMixin,
    _enum,
    _integer,
    _strict_object,
    _string,
)
from .rules import RuleRegistry


DEFAULT_API_TOKENS = frozenset(
    {
        # Triton
        "triton.language", "tl.", "tl.load", "tl.store", "tl.max", "tl.sum",
        "tl.min", "tl.dot", "tl.arange", "tl.program_id", "tl.where",
        "tl.exp", "tl.sqrt", "tl.rsqrt", "tl.atomic_add",
        # TileLang / TIR spellings
        "tilelang.", "T.", "T.copy", "T.gemm", "T.reduce", "T.alloc_shared",
        # CUDA C++ spellings
        "__global__", "__device__", "threadIdx", "blockIdx", "blockDim",
        "gridDim", "__syncthreads", "atomicAdd", "cub::",
    }
)

_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|<<|>>|[-+*/%<>&|^~?:]"
)


class LeakageKind(str, Enum):
    SCHEMA = "schema"
    API_TOKEN = "api_token"
    SOURCE_NGRAM = "source_ngram"
    BUDGET = "budget"


@dataclass
class LeakagePolicy(StrictJSONMixin):
    """Fixed, auditable budgets for pi's deliberately tiny channel."""

    max_steps: int = 64
    max_refs_per_step: int = 16
    max_literal_chars: int = 128
    max_total_literal_chars: int = 4096
    max_distinct_literals: int = 256
    source_ngram_size: int = 4
    min_source_token_chars: int = 2

    def __post_init__(self) -> None:
        for name in (
            "max_steps", "max_refs_per_step", "max_literal_chars",
            "max_total_literal_chars", "max_distinct_literals",
            "source_ngram_size", "min_source_token_chars",
        ):
            minimum = 1
            setattr(self, name, _integer(getattr(self, name), f"LeakagePolicy.{name}", minimum=minimum))

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LeakagePolicy":
        keys = {
            "max_steps", "max_refs_per_step", "max_literal_chars",
            "max_total_literal_chars", "max_distinct_literals",
            "source_ngram_size", "min_source_token_chars",
        }
        data = _strict_object(value, path="LeakagePolicy", allowed=keys)
        defaults = cls()
        return cls(**{name: data.get(name, getattr(defaults, name)) for name in keys})

    def to_json(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_refs_per_step": self.max_refs_per_step,
            "max_literal_chars": self.max_literal_chars,
            "max_total_literal_chars": self.max_total_literal_chars,
            "max_distinct_literals": self.max_distinct_literals,
            "source_ngram_size": self.source_ngram_size,
            "min_source_token_chars": self.min_source_token_chars,
        }


@dataclass
class LeakageViolation(StrictJSONMixin):
    kind: LeakageKind
    path: str
    detail: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        self.kind = _enum(LeakageKind, self.kind, "LeakageViolation.kind")
        self.path = _string(self.path, "LeakageViolation.path")
        self.detail = _string(self.detail, "LeakageViolation.detail")
        if self.evidence is not None:
            self.evidence = _string(self.evidence, "LeakageViolation.evidence", nonempty=False)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LeakageViolation":
        data = _strict_object(
            value,
            path="LeakageViolation",
            allowed={"kind", "path", "detail", "evidence"},
            required={"kind", "path", "detail"},
        )
        return cls(data["kind"], data["path"], data["detail"], data.get("evidence"))

    def to_json(self) -> dict[str, Any]:
        result = {"kind": self.kind.value, "path": self.path, "detail": self.detail}
        if self.evidence is not None:
            result["evidence"] = self.evidence
        return result


@dataclass
class LeakageReport(StrictJSONMixin):
    violations: list[LeakageViolation] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations

    def __post_init__(self) -> None:
        self.violations = [
            item if isinstance(item, LeakageViolation) else LeakageViolation.from_json(item)
            for item in self.violations
        ]
        if not isinstance(self.metrics, dict):
            raise SchemaError("LeakageReport.metrics must be an object")
        self.metrics = {
            _string(key, "LeakageReport.metrics key"): _integer(value, f"LeakageReport.metrics.{key}", minimum=0)
            for key, value in self.metrics.items()
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LeakageReport":
        data = _strict_object(
            value,
            path="LeakageReport",
            allowed={"passed", "violations", "metrics"},
            required={"passed", "violations", "metrics"},
        )
        violations = data["violations"]
        if not isinstance(violations, list):
            raise SchemaError("LeakageReport.violations must be an array")
        report = cls(
            [LeakageViolation.from_json(item) for item in violations],
            dict(data["metrics"]) if isinstance(data["metrics"], dict) else data["metrics"],
        )
        if data["passed"] is not report.passed:
            raise SchemaError("LeakageReport.passed contradicts violations")
        return report

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [item.to_json() for item in self.violations],
            "metrics": dict(self.metrics),
        }


def _literal_items(value: Any, path: str) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, list):
        result: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            result.extend(_literal_items(item, f"{path}[{index}]"))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.extend(_literal_items(item, f"{path}.{key}"))
        return result
    return []


def _api_token_occurs(text: str, token: str) -> bool:
    if not token:
        return False
    # Punctuation-bearing API spellings are safe to search literally.  Plain
    # identifiers use lexical boundaries so e.g. "gridDimension" does not
    # trigger the CUDA token "gridDim".
    if any(not (char.isalnum() or char == "_") for char in token):
        return token.casefold() in text.casefold()
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text, re.IGNORECASE) is not None


def _tokens(text: str, minimum_chars: int) -> list[str]:
    return [
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= minimum_chars or not token.isidentifier()
    ]


def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def scan_static_leakage(
    plan: ReplayPlan | dict[str, Any],
    *,
    source: str = "",
    j: SemanticArchive | dict[str, Any] | None = None,
    policy: LeakagePolicy | None = None,
    api_tokens: Iterable[str] = DEFAULT_API_TOKENS,
    registry: RuleRegistry | None = None,
) -> LeakageReport:
    """Scan pi (and optionally J) without executing or asking an LLM.

    Invalid raw pi JSON is returned as a schema violation.  Code that persists
    artifacts should still parse with :class:`ReplayPlan` first so malformed pi
    never enters the normal pipeline.  API spellings are forbidden in both J
    and pi, while source n-grams are checked only in pi: J.B deliberately owns
    source-derived bounds, masks, and index expressions.
    """

    policy = policy or LeakagePolicy()
    violations: list[LeakageViolation] = []
    try:
        parsed_plan = plan if isinstance(plan, ReplayPlan) else ReplayPlan.from_json(plan)
    except (SchemaError, TypeError) as exc:
        return LeakageReport(
            [LeakageViolation(LeakageKind.SCHEMA, "plan", str(exc))],
            {"steps": 0, "literal_chars": 0, "distinct_literals": 0},
        )
    if registry is not None:
        try:
            parsed_plan = registry.validate_strict_plan(parsed_plan)
        except (SchemaError, TypeError) as exc:
            violations.append(
                LeakageViolation(LeakageKind.SCHEMA, "plan", str(exc))
            )

    parsed_j: SemanticArchive | None = None
    if j is not None:
        try:
            parsed_j = j if isinstance(j, SemanticArchive) else SemanticArchive.from_json(j)
        except (SchemaError, TypeError) as exc:
            violations.append(LeakageViolation(LeakageKind.SCHEMA, "j", str(exc)))

    plan_json = parsed_plan.to_json()
    plan_literals = _literal_items(plan_json, "plan")
    all_literals = list(plan_literals)
    j_literals: list[tuple[str, str]] = []
    if parsed_j is not None:
        j_literals = _literal_items(parsed_j.to_json(), "j")
        all_literals.extend(j_literals)

    literal_chars = sum(len(value) for _, value in plan_literals)
    distinct_literals = len({value for _, value in plan_literals})
    if len(parsed_plan.steps) > policy.max_steps:
        violations.append(
            LeakageViolation(
                LeakageKind.BUDGET, "plan.steps",
                f"step count {len(parsed_plan.steps)} exceeds budget {policy.max_steps}",
            )
        )
    for index, step in enumerate(parsed_plan.steps):
        reference_count = len(step.refs) + (1 if step.node_ref is not None else 0)
        if reference_count > policy.max_refs_per_step:
            violations.append(
                LeakageViolation(
                    LeakageKind.BUDGET, f"plan.steps[{index}]",
                    f"reference count {reference_count} exceeds budget {policy.max_refs_per_step}",
                )
            )
    for path, literal in plan_literals:
        if len(literal) > policy.max_literal_chars:
            violations.append(
                LeakageViolation(
                    LeakageKind.BUDGET, path,
                    f"literal length {len(literal)} exceeds budget {policy.max_literal_chars}",
                    literal[:80],
                )
            )
    if literal_chars > policy.max_total_literal_chars:
        violations.append(
            LeakageViolation(
                LeakageKind.BUDGET, "plan",
                f"total literal length {literal_chars} exceeds budget {policy.max_total_literal_chars}",
            )
        )
    if distinct_literals > policy.max_distinct_literals:
        violations.append(
            LeakageViolation(
                LeakageKind.BUDGET, "plan",
                f"distinct literal count {distinct_literals} exceeds budget {policy.max_distinct_literals}",
            )
        )

    concrete_api_tokens = set(api_tokens)
    if registry is not None:
        concrete_api_tokens.update(registry.api_tokens())
    for path, literal in all_literals:
        for token in sorted(concrete_api_tokens):
            if _api_token_occurs(literal, token):
                violations.append(
                    LeakageViolation(
                        LeakageKind.API_TOKEN, path,
                        f"concrete frontend API token {token!r} is forbidden in J/pi",
                        token,
                    )
                )

    if source:
        source_ngrams = _ngrams(
            _tokens(source, policy.min_source_token_chars), policy.source_ngram_size,
        )
        seen_matches: set[tuple[str, tuple[str, ...]]] = set()
        source_checked_literals = list(plan_literals)
        for path, literal in j_literals:
            # Layer-B expressions are expected to preserve exact semantic
            # predicates/indexing from the source.  Other free-form J strings
            # are not an authorized source-text channel and are checked too.
            if path.startswith("j.boundary.domains"):
                continue
            if path.startswith("j.boundary.accesses") and path.endswith(
                (".index_expression", ".predicate")
            ):
                continue
            if path.startswith("j.boundary.output_mappings") and path.endswith(
                ".value"
            ):
                continue
            source_checked_literals.append((path, literal))
        for path, literal in source_checked_literals:
            literal_tokens = _tokens(literal, policy.min_source_token_chars)
            # A short free-form J attribute is still a useful covert channel.
            # Requiring the full plan-sized n-gram would let three source
            # identifiers pass merely because the literal has no fourth token.
            # Keep one/two-token semantic vocabulary out of this heuristic, but
            # scan any longer literal at the strongest size it can support.
            match_size = min(policy.source_ngram_size, len(literal_tokens))
            if path.startswith("j.") and match_size >= 3:
                candidate_ngrams = _ngrams(literal_tokens, match_size)
                candidate_source_ngrams = _ngrams(
                    _tokens(source, policy.min_source_token_chars), match_size,
                )
            else:
                candidate_ngrams = _ngrams(
                    literal_tokens, policy.source_ngram_size,
                )
                candidate_source_ngrams = source_ngrams
            for ngram in candidate_ngrams:
                marker = (path, ngram)
                if ngram in candidate_source_ngrams and marker not in seen_matches:
                    evidence = " ".join(ngram)
                    violations.append(
                        LeakageViolation(
                            LeakageKind.SOURCE_NGRAM, path,
                            f"literal contains a {len(ngram)}-token source n-gram",
                            evidence[:160],
                        )
                    )
                    seen_matches.add(marker)

    return LeakageReport(
        violations=violations,
        metrics={
            "steps": len(parsed_plan.steps),
            "literal_chars": literal_chars,
            "distinct_literals": distinct_literals,
        },
    )


def scan_artifact_leakage(
    artifact: Gate1Artifact,
    *,
    policy: LeakagePolicy | None = None,
    api_tokens: Iterable[str] = DEFAULT_API_TOKENS,
    registry: RuleRegistry | None = None,
) -> LeakageReport:
    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    return scan_static_leakage(
        artifact.plan,
        source=artifact.source or "",
        j=artifact.j,
        policy=policy,
        api_tokens=api_tokens,
        registry=registry,
    )


__all__ = [
    "DEFAULT_API_TOKENS", "LeakageKind", "LeakagePolicy", "LeakageReport",
    "LeakageViolation", "scan_artifact_leakage", "scan_static_leakage",
]
