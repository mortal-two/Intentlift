"""Structured, non-authoritative diagnosis for Gate1 failures.

Only :mod:`.compare` decides pass/fail.  Diagnosis translates deterministic
differences into a repair scope; callers may replace it with an LLM explanation
without granting that model any authority over the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from .compare import TTIRComparison
from .evaluation import FailureCause


class Responsibility(str, Enum):
    INTENT = "intent"
    REPLAY_PLAN = "replay_plan"
    REPLAY_CODE = "replay_code"
    INTERFACE_SCHEDULE = "interface_schedule"
    COMPARATOR = "comparator"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RepairFeedback:
    """Allowlisted replay-side feedback; never contains source fingerprints."""

    cause: FailureCause
    responsibility: Responsibility
    confirmed_fields: tuple[str, ...]
    allowed_modifications: tuple[str, ...]
    instruction: str
    replay_compiler_diagnostic: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cause": self.cause.value,
            "responsibility": self.responsibility.value,
            "confirmed_fields": list(self.confirmed_fields),
            "allowed_modifications": list(self.allowed_modifications),
            "instruction": self.instruction,
        }
        if self.replay_compiler_diagnostic is not None:
            payload["replay_compiler_diagnostic"] = self.replay_compiler_diagnostic
        return payload


_ABSOLUTE_PATH_RE = re.compile(r'(?P<prefix>File\s+["\'])/[^"\']+(?P<suffix>["\'])')


def sanitize_replay_diagnostic(diagnostic: str, *, limit: int = 2000) -> str:
    """Bound replay diagnostics and remove host paths before model feedback."""

    text = str(diagnostic or "unknown replay compile error")
    text = _ABSOLUTE_PATH_RE.sub(r"\g<prefix><path>\g<suffix>", text)
    lines = [line.rstrip() for line in text.splitlines()]
    # Tracebacks repeat framework internals heavily.  Preserve the exception,
    # candidate line reference and nearby compiler message, not the host stack.
    useful = [
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith(("Traceback (most recent call last):",))
    ]
    rendered = "\n".join(useful[-24:])
    return rendered[-limit:]


@dataclass(frozen=True)
class Diagnosis:
    cause: FailureCause
    responsibility: Responsibility
    mismatches: tuple[str, ...]
    confirmed_fields: tuple[str, ...]
    allowed_modifications: tuple[str, ...]
    explanation: str

    def to_json(self) -> dict[str, Any]:
        return {
            "cause": self.cause.value,
            "responsibility": self.responsibility.value,
            "mismatches": list(self.mismatches),
            "confirmed_fields": list(self.confirmed_fields),
            "allowed_modifications": list(self.allowed_modifications),
            "explanation": self.explanation,
        }

    def to_rewriter_feedback(self) -> RepairFeedback:
        """Return the only diagnosis view allowed back to the replay writer.

        Comparator mismatches are derived from the privileged source TTIR.  The
        full :meth:`to_json` payload is therefore for the lifter/diagnostician
        side only: it can contain source-side operation signatures, constants,
        predicates and counts.  A replay repair receives only the mismatch
        class and the already-declared repair scope, and must recover the
        concrete implementation by rereading J, pi and the rule cards.

        Replay compiler diagnostics are safe to include because they describe
        model-generated code rather than the source kernel.
        """

        diagnostic: str | None = None
        if (
            self.cause is FailureCause.REPLAY_COMPILE
            and self.responsibility is Responsibility.REPLAY_CODE
        ):
            diagnostic = sanitize_replay_diagnostic("\n".join(self.mismatches))
        return RepairFeedback(
            cause=self.cause,
            responsibility=self.responsibility,
            confirmed_fields=self.confirmed_fields,
            allowed_modifications=self.allowed_modifications,
            instruction=self.explanation,
            replay_compiler_diagnostic=diagnostic,
        )


_CAUSE_BY_CATEGORY = {
    "interface": FailureCause.INTERFACE,
    "op_multiset": FailureCause.OPERATOR,
    "dataflow": FailureCause.DATAFLOW,
    "reduction": FailureCause.REDUCTION,
    "predicate_effect": FailureCause.PREDICATE_EFFECT,
    "schedule-only": FailureCause.SCHEDULE,
}


def diagnose_compile_failure(diagnostic: str) -> Diagnosis:
    return Diagnosis(
        cause=FailureCause.REPLAY_COMPILE,
        responsibility=Responsibility.REPLAY_CODE,
        mismatches=(sanitize_replay_diagnostic(diagnostic),),
        confirmed_fields=(),
        allowed_modifications=("replay code",),
        explanation="重放代码未通过同前端编译；J 冻结，只允许依据编译诊断小修代码。",
    )


def diagnose_comparison(comparison: TTIRComparison) -> Diagnosis:
    failed = [component for component in comparison.components if not component.passed]
    hard_failed = [component for component in failed if component.hard]
    passed = tuple(
        component.category for component in comparison.components if component.passed
    )
    primary = comparison.primary_difference_category or "dataflow"
    cause = _CAUSE_BY_CATEGORY.get(primary, FailureCause.UNKNOWN)
    messages = tuple(difference.message for difference in comparison.differences)

    if not hard_failed and failed:
        responsibility = Responsibility.INTERFACE_SCHEDULE
        allowed = ("schedule C",)
        explanation = "只有调度视图不同；A/B 闭合，差异只报告、不否决。"
    elif primary == "interface":
        responsibility = Responsibility.INTERFACE_SCHEDULE
        allowed = ("replay signature", "interface contract")
        explanation = "运行时/constexpr 接口不一致；先修重放签名或接口合同，不修改 J.A。"
    else:
        responsibility = Responsibility.REPLAY_PLAN
        allowed = ("pi referenced steps", "replay code")
        explanation = (
            "源与重放的硬指纹不闭合。当前 artifact 冻结时优先归入 π/重放实现；"
            "若后续对照 J 发现源指纹中的语义在 J 中不存在，再升级为重提 J。"
        )
    return Diagnosis(
        cause=cause,
        responsibility=responsibility,
        mismatches=messages,
        confirmed_fields=passed,
        allowed_modifications=allowed,
        explanation=explanation,
    )


__all__ = [
    "Diagnosis",
    "RepairFeedback",
    "Responsibility",
    "diagnose_compile_failure",
    "diagnose_comparison",
    "sanitize_replay_diagnostic",
]
