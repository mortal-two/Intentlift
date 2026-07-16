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
    INTERFACE_SCHEDULE = "interface_schedule"
    COMPARATOR = "comparator"
    UNKNOWN = "unknown"


_ABSOLUTE_PATH_RE = re.compile(r'(?P<prefix>File\s+["\'])/[^"\']+(?P<suffix>["\'])')
_FILE_LOCATION_RE = re.compile(
    r'^\s*File\s+["\'][^"\']+["\'],\s*line\s+\d+(?::\d+)?(?P<detail>:.*)?$'
)
_DIAGNOSTIC_SUMMARY_RE = re.compile(
    r"(?:error|exception|failed|failure|forbidden|invalid|unsupported|"
    r"mismatch|undefined|not defined|requires|expected)",
    re.IGNORECASE,
)


def sanitize_replay_diagnostic(diagnostic: str, *, limit: int = 2000) -> str:
    """Return bounded compiler guidance without replay source or host paths.

    Triton diagnostics commonly contain a traceback followed by replay source
    lines and a final exception summary.  Lift needs the deterministic failure
    class/reason, not the failed implementation.  Keep only summary-like lines
    (plus an inline detail attached to a ``File ... line N:`` record), so the
    next Lift can revise J/pi without seeing a replay body.
    """

    text = str(diagnostic or "unknown replay compile error")
    text = _ABSOLUTE_PATH_RE.sub(r"\g<prefix><path>\g<suffix>", text)
    summaries: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Traceback (most recent call last):"):
            continue
        location = _FILE_LOCATION_RE.match(line)
        if location is not None:
            detail = (location.group("detail") or "").lstrip(": ")
            if detail:
                summaries.append(detail)
            continue
        # Compiler-quoted replay statements and caret context are indented;
        # exception summaries are not.  Never forward the quoted statement.
        if raw[:1].isspace():
            continue
        if set(line) <= {"^", "~", "-", " "}:
            continue
        if _DIAGNOSTIC_SUMMARY_RE.search(line):
            summaries.append(line)
    if not summaries:
        summaries.append(
            "replay frontend rejected candidate; detailed replay text was redacted"
        )
    rendered = "\n".join(summaries[-8:])
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
        responsibility=Responsibility.REPLAY_PLAN,
        mismatches=(sanitize_replay_diagnostic(diagnostic),),
        confirmed_fields=(),
        allowed_modifications=("j", "pi", "location_map"),
        explanation=(
            "重放未通过同前端编译；禁止原地修 replay code。将净化后的编译诊断"
            "回流 Lift，重新检查 J/π 合同并从完整流程重新生成。"
        ),
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
        responsibility = Responsibility.REPLAY_PLAN
        allowed = ("j", "pi", "location_map")
        explanation = (
            "重放接口与冻结 interface 不一致；禁止原地修签名。回流 Lift 检查"
            "J/π 是否给 writer 提供了完整施工约束，然后从头重放。"
        )
    else:
        responsibility = Responsibility.REPLAY_PLAN
        allowed = ("j", "pi", "location_map")
        explanation = (
            "源与重放的硬指纹不闭合。将确定性差分与源码回流 Lift，重新判断"
            "J 是否漏/错以及 π 是否选择了错误施工机制；不修改当前 replay code。"
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
    "Responsibility",
    "diagnose_compile_failure",
    "diagnose_comparison",
    "sanitize_replay_diagnostic",
]
