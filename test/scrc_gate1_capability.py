#!/usr/bin/env python3
"""SCRC Gate1 capability-limit probe for the five feasibility kernels.

This is an experiment driver, not a lit test.  It compiles the trusted source
kernel and one source-blind replay/mutant with the same Triton specialization,
compares their TTIR fingerprints through the production Gate1 comparator, and
then (only as an offline evaluation oracle) executes source and replay on GPU.

The dynamic oracle is intentionally labelled a *two-party proxy*: it compares
``O_src`` with ``O_rep`` and does not pretend to be the full Gate2 three-party
``source / replay / production-lowering`` check described by the SCRC design.

Typical usage from the repository root::

    PYTHONPATH=python python test/scrc_gate1_capability.py --list
    PYTHONPATH=python python test/scrc_gate1_capability.py
    PYTHONPATH=python python test/scrc_gate1_capability.py --gate2-depth full

No repository artifact is written by default.  Triton modules, compiler dumps,
and caches live in temporary directories and are removed after the run.  Use
``--json-output`` only when a persistent machine-readable report is desired.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.verify.gate1.compiler import (  # noqa: E402
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.evaluation import (  # noqa: E402
    CapabilityObservation,
    FailureCause,
    OutcomeClass,
    analyze_failure,
    summarize_capability,
)
from intentir.verify.gate1.leakage import (  # noqa: E402
    LeakageKind,
    scan_static_leakage,
)


SOURCE_MODULE = ROOT / "test" / "scrc_probe_sources.py"
REPLAY_MODULE = ROOT / "test" / "scrc_probe_replays.py"


@dataclass(frozen=True)
class TextMutation:
    """One exact, reviewable mutation of the blind replay module."""

    old: str
    new: str
    expected_occurrences: int = 1

    def apply(self, code: str, case_id: str) -> str:
        occurrences = code.count(self.old)
        if occurrences != self.expected_occurrences:
            raise RuntimeError(
                f"{case_id}: mutation expected {self.expected_occurrences} occurrence(s), "
                f"found {occurrences}: {self.old!r}"
            )
        return code.replace(self.old, self.new)


@dataclass(frozen=True)
class CapabilityCase:
    """A baseline, equivalent rewrite, semantic mutant, or contract mutant."""

    case_id: str
    kernel: str
    title: str
    family: str
    expected_gate1_closed: bool
    expected_gate2_pass: bool
    expected_cause: FailureCause | None
    rationale: str
    mutations: tuple[TextMutation, ...] = ()
    leaked_plan: Mapping[str, Any] | None = None

    @property
    def function_name(self) -> str:
        return {
            "add": "add_kernel",
            "row_sum": "row_sum_kernel",
            "softmax": "softmax_kernel",
            "rmsnorm": "rmsnorm_kernel",
            "matmul": "matmul_kernel",
        }[self.kernel]

    def candidate_code(self, base_code: str | None = None) -> str:
        code = REPLAY_MODULE.read_text(encoding="utf-8") if base_code is None else base_code
        for mutation in self.mutations:
            code = mutation.apply(code, self.case_id)
        return code

    def to_catalog_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kernel": self.kernel,
            "function": self.function_name,
            "title": self.title,
            "family": self.family,
            "expected_gate1_closed": self.expected_gate1_closed,
            "expected_gate2_pass": self.expected_gate2_pass,
            "expected_cause": self.expected_cause.value if self.expected_cause else None,
            "rationale": self.rationale,
            "has_pi_leakage": self.leaked_plan is not None,
        }


def capability_cases() -> tuple[CapabilityCase, ...]:
    """Return the fixed capability matrix used by tests and the CLI.

    ``expected_gate1_closed`` is the design target, not an assertion baked into
    the runner.  A mismatch is itself an experimental result:

    * equivalent + Gate1 rejected -> normalization/coverage ceiling;
    * incorrect + Gate1 closed -> comparator blind spot and precision failure.
    """

    row_sum_body = (
        "    mask = col < N\n\n"
        "    x = tl.load(x_ptr + row * N + col, mask=mask, other=0.0)\n"
        "    out = tl.sum(x, axis=0)"
    )
    return (
        CapabilityCase(
            "add.correct",
            "add",
            "正确盲重放：逐元素加法",
            "baseline",
            True,
            True,
            None,
            "五核基线；应在接口、A/B 指纹和动态行为上全部闭合。",
        ),
        CapabilityCase(
            "row_sum.correct",
            "row_sum",
            "正确盲重放：行归约求和",
            "baseline",
            True,
            True,
            None,
            "五核基线；覆盖带零填充的 masked load 与 sum reduction。",
        ),
        CapabilityCase(
            "softmax.correct",
            "softmax",
            "正确盲重放：数值稳定 softmax",
            "baseline",
            True,
            True,
            None,
            "五核基线；覆盖 max/sum 双归约、-inf fill 和 masked store。",
        ),
        CapabilityCase(
            "rmsnorm.correct",
            "rmsnorm",
            "正确盲重放：RMSNorm",
            "baseline",
            True,
            True,
            None,
            "五核基线；覆盖 rsqrt 与运行时 eps。",
        ),
        CapabilityCase(
            "matmul.correct",
            "matmul",
            "正确盲重放：分块矩阵乘",
            "baseline",
            True,
            True,
            None,
            "五核基线；覆盖 scf.for、tt.dot 和 M/N/K 三向 tail mask。",
        ),
        CapabilityCase(
            "add.commuted",
            "add",
            "等价改写：交换加法操作数",
            "equivalent",
            True,
            True,
            FailureCause.NORMALIZATION_GAP,
            "验证交换律操作数排序；若拒绝，说明规范化损失覆盖率。",
            (TextMutation("    out = a + b", "    out = b + a"),),
        ),
        CapabilityCase(
            "add.redundant_zero",
            "add",
            "等价改写：增加冗余 +0",
            "equivalent",
            True,
            True,
            FailureCause.NORMALIZATION_GAP,
            "验证中性元消除或编译器常量折叠；这是比交换律更刁钻的边界。",
            (TextMutation("    out = a + b", "    out = (a + b) + 0.0"),),
        ),
        CapabilityCase(
            "row_sum.reversed_predicate",
            "row_sum",
            "等价改写：cols < N 改为 N > cols",
            "equivalent",
            True,
            True,
            FailureCause.NORMALIZATION_GAP,
            "验证有界谓词翻转规范化，而非依赖 cmpi 的表面方向。",
            (
                TextMutation(
                    row_sum_body,
                    row_sum_body.replace("mask = col < N", "mask = N > col"),
                ),
            ),
        ),
        CapabilityCase(
            "row_sum.wrong_fill",
            "row_sum",
            "语义变异：归约尾部 fill 从 0 改为 1",
            "semantic_mutant",
            False,
            False,
            FailureCause.PREDICATE_EFFECT,
            "算子多重集不变，仅 fill 错；非整除 N 时每个无效 lane 都污染 sum。",
            (
                TextMutation(
                    row_sum_body,
                    row_sum_body.replace("other=0.0", "other=1.0"),
                ),
            ),
        ),
        CapabilityCase(
            "softmax.missing_stabilization",
            "softmax",
            "语义变异：删除 max-sub 稳定化",
            "semantic_mutant",
            False,
            False,
            FailureCause.OPERATOR,
            "Gate1 应从缺失 reduce_max/sub 硬拒绝；大动态范围输入令 Gate2 失败。",
            (
                TextMutation(
                    "    row_max = tl.max(x, axis=0)\n"
                    "    shifted = x - row_max\n"
                    "    numerator = tl.exp(shifted)",
                    "    numerator = tl.exp(x)",
                ),
            ),
        ),
        CapabilityCase(
            "softmax.wrong_fill",
            "softmax",
            "语义变异：masked load fill 从 -inf 改为 0",
            "semantic_mutant",
            False,
            False,
            FailureCause.PREDICATE_EFFECT,
            "A 图几乎相同，只靠 B 层 fill/effect 才能抓住；全负输入会暴露错误。",
            (
                TextMutation(
                    "    x = tl.load(x_ptr + row * N + col, mask=mask, other=-float(\"inf\"))",
                    "    x = tl.load(x_ptr + row * N + col, mask=mask, other=0.0)",
                ),
            ),
        ),
        CapabilityCase(
            "softmax.reversed_sub",
            "softmax",
            "语义变异：x-max 误写为 max-x",
            "semantic_mutant",
            False,
            False,
            FailureCause.DATAFLOW,
            "算子与归约多重集完全相同，必须保留非交换 sub 的操作数方向。",
            (TextMutation("    shifted = x - row_max", "    shifted = row_max - x"),),
        ),
        CapabilityCase(
            "rmsnorm.sqrt_reciprocal",
            "rmsnorm",
            "等价改写：rsqrt 改为 1/sqrt",
            "equivalent",
            True,
            True,
            FailureCause.NORMALIZATION_GAP,
            "验证已知代数等价模式；若 TTIR 未统一，属于固定规则可消除的假拒绝。",
            (
                TextMutation(
                    "    inv_rms = tl.rsqrt(shifted)",
                    "    inv_rms = 1.0 / tl.sqrt(shifted)",
                ),
            ),
        ),
        CapabilityCase(
            "rmsnorm.wrong_divisor",
            "rmsnorm",
            "语义变异：均方除数 N 改为 BLOCK_N",
            "semantic_mutant",
            False,
            False,
            FailureCause.DATAFLOW,
            "非整除形状下结果错误；要求指纹区分运行时维度与调度常量。",
            (
                TextMutation(
                    "    mean_square = tl.sum(squared, axis=0) / N",
                    "    mean_square = tl.sum(squared, axis=0) / BLOCK_N",
                ),
            ),
        ),
        CapabilityCase(
            "matmul.wrong_dot_operands",
            "matmul",
            "语义变异：dot(a,b) 改为 dot(b,a)",
            "semantic_mutant",
            False,
            False,
            FailureCause.DATAFLOW,
            "方形 block 下仍能编译且算子多重集相同，专门压测 dot 数据流锚点。",
            (
                TextMutation(
                    "        acc += tl.dot(a, b, out_dtype=tl.float32)",
                    "        acc += tl.dot(b, a, out_dtype=tl.float32)",
                ),
            ),
        ),
        CapabilityCase(
            "matmul.wrong_b_address",
            "matmul",
            "语义变异：B 的 K-major 地址改为错误转置式地址",
            "semantic_mutant",
            False,
            False,
            FailureCause.DATAFLOW,
            "load/dot/mask 数量均不变，要求比较器保留符号步长和地址 DAG。",
            (
                TextMutation(
                    "            b_ptr + k[:, None] * N + n[None, :],",
                    "            b_ptr + n[None, :] * K + k[:, None],",
                ),
            ),
        ),
        CapabilityCase(
            "rmsnorm.missing_do_not_specialize",
            "rmsnorm",
            "非语义元数据变化：删除 eps 的 do_not_specialize",
            "metadata_variant",
            True,
            True,
            None,
            "do_not_specialize 只影响 Triton JIT 特化策略，不属于 SCRC 的接口/A/B 硬指纹。",
            (
                TextMutation(
                    '@triton.jit(do_not_specialize=["eps"])\n'
                    "def rmsnorm_kernel",
                    "@triton.jit\ndef rmsnorm_kernel",
                ),
            ),
        ),
        CapabilityCase(
            "add.pi_api_leakage",
            "add",
            "反作弊变异：π.rule 直接泄漏 tl.load",
            "leakage_mutant",
            False,
            True,
            FailureCause.PLAN_LEAKAGE,
            "重放代码本身正确，但 π 出现具体前端 API；应在编译前静态拒绝。",
            leaked_plan={
                "steps": [
                    {"rule": "tl.load", "node_ref": "n0"},
                    {"rule": "elementwise-map", "node_ref": "n1"},
                ]
            },
        ),
    )


def compile_case(kernel: str) -> CompileCase:
    common_schedule: dict[str, int] = {
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
    }
    if kernel == "add":
        return CompileCase(
            arguments={
                "a_ptr": RuntimeArgument.tensor((65,), "float32"),
                "b_ptr": RuntimeArgument.tensor((65,), "float32"),
                "out_ptr": RuntimeArgument.tensor((65,), "float32"),
                "n_elements": RuntimeArgument.scalar(65),
            },
            grid=(2,),
            schedule={"BLOCK": 64, **common_schedule},
        )
    if kernel in {"row_sum", "softmax"}:
        return CompileCase(
            arguments={
                "x_ptr": RuntimeArgument.tensor((3, 31), "float32"),
                "out_ptr": RuntimeArgument.tensor(
                    (3,) if kernel == "row_sum" else (3, 31), "float32"
                ),
                "M": RuntimeArgument.scalar(3),
                "N": RuntimeArgument.scalar(31),
            },
            grid=(3,),
            schedule={"BLOCK_N": 64, **common_schedule},
        )
    if kernel == "rmsnorm":
        return CompileCase(
            arguments={
                "x_ptr": RuntimeArgument.tensor((3, 31), "float32"),
                "w_ptr": RuntimeArgument.tensor((31,), "float32"),
                "out_ptr": RuntimeArgument.tensor((3, 31), "float32"),
                "M": RuntimeArgument.scalar(3),
                "N": RuntimeArgument.scalar(31),
                "eps": RuntimeArgument.scalar(1.0e-5, "float32"),
            },
            grid=(3,),
            schedule={"BLOCK_N": 64, **common_schedule},
        )
    if kernel == "matmul":
        return CompileCase(
            arguments={
                "a_ptr": RuntimeArgument.tensor((17, 19), "float16"),
                "b_ptr": RuntimeArgument.tensor((19, 13), "float16"),
                "out_ptr": RuntimeArgument.tensor((17, 13), "float32"),
                "M": RuntimeArgument.scalar(17),
                "N": RuntimeArgument.scalar(13),
                "K": RuntimeArgument.scalar(19),
            },
            grid=(2, 1),
            schedule={
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "BLOCK_K": 16,
                **common_schedule,
            },
        )
    raise KeyError(f"unknown kernel {kernel!r}")


@dataclass(frozen=True)
class Gate1Evaluation:
    closed: bool
    failure: FailureCause | None
    stage: str
    diagnostic: str = ""
    comparison: Mapping[str, Any] | None = None
    source_compile_ms: float | None = None
    replay_compile_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed": self.closed,
            "failure": self.failure.value if self.failure else None,
            "stage": self.stage,
            "diagnostic": self.diagnostic,
            "source_compile_ms": self.source_compile_ms,
            "replay_compile_ms": self.replay_compile_ms,
            "comparison": dict(self.comparison) if self.comparison is not None else None,
        }


@dataclass(frozen=True)
class Gate2Trial:
    label: str
    dimensions: tuple[int, ...]
    distribution: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Gate2Evaluation:
    passed: bool | None
    trials_run: int
    diagnostic: str = ""
    failed_trial: Gate2Trial | None = None
    max_abs_error: float | None = None
    oracle: str = "two-party-source-vs-replay-gpu-proxy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "trials_run": self.trials_run,
            "diagnostic": self.diagnostic,
            "failed_trial": self.failed_trial.to_dict() if self.failed_trial else None,
            "max_abs_error": self.max_abs_error,
            "oracle": self.oracle,
        }


def _comparison_cause(category: str | None) -> FailureCause:
    return {
        "interface": FailureCause.INTERFACE,
        "op_multiset": FailureCause.OPERATOR,
        "dataflow": FailureCause.DATAFLOW,
        "reduction": FailureCause.REDUCTION,
        "predicate_effect": FailureCause.PREDICATE_EFFECT,
        "schedule-only": FailureCause.SCHEDULE,
    }.get(category, FailureCause.UNKNOWN)


def _comparison_summary(comparison: Any) -> dict[str, Any]:
    if hasattr(comparison, "to_dict"):
        value = comparison.to_dict()
        if isinstance(value, dict):
            return value
    return {
        "hard_pass": bool(
            getattr(comparison, "hard_pass", getattr(comparison, "passed", False))
        ),
        "primary_difference_category": getattr(
            comparison, "primary_difference_category", None
        ),
        "repr": repr(comparison),
    }


def _hard_interface_metadata(
    compiled_interface: Mapping[str, Any],
    specialization: CompileCase,
    function_name: str,
) -> dict[str, Any]:
    """Extend Triton's Python ABI with the hard launch/interface contract.

    Source and replay are compiled with the exact same concrete C values; that
    equality is a hard precondition enforced by sharing one CompileCase.  C is
    merely a soft/report-only component *inside* the TTIR comparator.  The
    selected entry point and launch grid determine which programs cover the
    output domain and therefore also belong to the hard interface.
    """

    return {
        **dict(compiled_interface),
        "function_name": function_name,
        "launch_grid": list(specialization.grid),
    }


def run_gate1_case(
    case: CapabilityCase,
    compiler: TritonCompiler,
) -> Gate1Evaluation:
    """Run static leakage, compile-only, and deterministic TTIR comparison."""

    if case.leaked_plan is not None:
        leakage = scan_static_leakage(case.leaked_plan)
        if not leakage.passed:
            api_evidence = [
                violation.evidence or violation.detail
                for violation in leakage.violations
                if violation.kind is LeakageKind.API_TOKEN
            ]
            return Gate1Evaluation(
                False,
                FailureCause.PLAN_LEAKAGE,
                "static-leakage",
                "π 静态泄漏扫描失败: " + ", ".join(api_evidence),
                comparison={"leakage": leakage.to_json()},
            )

    specialization = compile_case(case.kernel)
    source = compiler.compile(
        KernelSource.from_file(SOURCE_MODULE, case.function_name, trusted=True),
        specialization,
    )
    if not source.ok or not source.ttir:
        return Gate1Evaluation(
            False,
            FailureCause.LOWERING,
            "source-compile",
            source.diagnostic,
            source_compile_ms=source.elapsed_ms,
        )

    replay = compiler.compile(
        KernelSource.from_code(case.candidate_code(), case.function_name),
        specialization,
    )
    if not replay.ok or not replay.ttir:
        return Gate1Evaluation(
            False,
            FailureCause.REPLAY_COMPILE,
            "replay-compile",
            replay.diagnostic,
            source_compile_ms=source.elapsed_ms,
            replay_compile_ms=replay.elapsed_ms,
        )

    try:
        # Delayed import keeps the sample catalog usable while the comparator
        # implementation evolves, without introducing a second Gate1 core.
        from intentir.verify.gate1.compare import compare_ttir

        schedule_view = {
            "grid": list(specialization.grid),
            **dict(specialization.schedule),
        }
        comparison = compare_ttir(
            source.ttir,
            replay.ttir,
            source_python_interface=_hard_interface_metadata(
                source.python_interface,
                specialization,
                case.function_name,
            ),
            replay_python_interface=_hard_interface_metadata(
                replay.python_interface,
                specialization,
                case.function_name,
            ),
            source_schedule=schedule_view,
            replay_schedule=schedule_view,
        )
    except Exception as exc:
        return Gate1Evaluation(
            False,
            FailureCause.LOWERING,
            "fingerprint",
            f"TTIR 指纹提取/比较失败: {type(exc).__name__}: {exc}",
            source_compile_ms=source.elapsed_ms,
            replay_compile_ms=replay.elapsed_ms,
        )

    payload = _comparison_summary(comparison)
    closed = bool(
        getattr(comparison, "hard_pass", getattr(comparison, "passed", False))
    )
    category = getattr(comparison, "primary_difference_category", None)
    diagnostic = ""
    if not closed:
        differences = payload.get("differences", [])
        messages = [
            str(item.get("message", item)) if isinstance(item, dict) else str(item)
            for item in differences[:4]
        ]
        diagnostic = "; ".join(messages) or f"首要不一致类别: {category}"
    return Gate1Evaluation(
        closed,
        None if closed else _comparison_cause(category),
        "closed" if closed else "compare",
        diagnostic,
        comparison=payload,
        source_compile_ms=source.elapsed_ms,
        replay_compile_ms=replay.elapsed_ms,
    )


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Triton module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _quick_trials(kernel: str) -> list[Gate2Trial]:
    seeds = (0, 1, 2)
    if kernel == "add":
        return [Gate2Trial("tail", (65,), "normal", seed) for seed in seeds]
    if kernel == "row_sum":
        return [
            Gate2Trial("tail-fill", (3, 31), "normal", seed) for seed in seeds
        ]
    if kernel == "softmax":
        return [
            Gate2Trial(label, (3, 31), distribution, seed)
            for label, distribution in (
                ("negative-tail", "negative"),
                ("large-dynamic-range", "extreme"),
            )
            for seed in seeds
        ]
    if kernel == "rmsnorm":
        return [
            Gate2Trial("tail-divisor", (3, 31), "normal", seed) for seed in seeds
        ]
    if kernel == "matmul":
        return [
            Gate2Trial("non-square-tail", (33, 31, 53), "normal", seed)
            for seed in seeds
        ]
    raise KeyError(kernel)


def gate2_trials(kernel: str, depth: str) -> list[Gate2Trial]:
    if depth == "quick":
        return _quick_trials(kernel)

    seeds = (0, 1, 2)
    trials: list[Gate2Trial] = []
    if kernel == "add":
        for n in (1, 63, 64, 65, 113):
            for seed in seeds:
                trials.append(Gate2Trial(f"N={n}", (n,), "normal", seed))
        trials.append(Gate2Trial("signed-zero/inf/nan", (65,), "special", 0))
    elif kernel == "row_sum":
        for n in (1, 63, 64, 65):
            for seed in seeds:
                trials.append(Gate2Trial(f"N={n}", (3, n), "normal", seed))
        trials.append(Gate2Trial("cancellation/inf", (3, 63), "special", 0))
    elif kernel == "softmax":
        for n in (1, 63, 64, 65):
            for distribution in ("normal", "negative", "extreme"):
                for seed in seeds:
                    trials.append(
                        Gate2Trial(
                            f"N={n}/{distribution}",
                            (3, n),
                            distribution,
                            seed,
                        )
                    )
    elif kernel == "rmsnorm":
        for n in (1, 63, 64, 65):
            for seed in seeds:
                trials.append(Gate2Trial(f"N={n}", (3, n), "normal", seed))
        trials.append(Gate2Trial("zero/large-range", (3, 63), "special", 0))
    elif kernel == "matmul":
        for dimensions in (
            (1, 1, 1),
            (31, 31, 31),
            (32, 32, 32),
            (33, 33, 33),
            (33, 31, 53),
            (53, 33, 65),
        ):
            for seed in seeds:
                trials.append(
                    Gate2Trial(
                        "M/N/K=" + "/".join(map(str, dimensions)),
                        dimensions,
                        "normal",
                        seed,
                    )
                )
        trials.append(Gate2Trial("zero-matrix", (33, 31, 53), "zero", 0))
    else:
        raise KeyError(kernel)
    return trials


def _torch_generator(torch: Any, seed: int) -> Any:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _launch_trial(
    source_kernel: Any,
    replay_kernel: Any,
    kernel: str,
    trial: Gate2Trial,
) -> tuple[Any, Any]:
    import torch

    generator = _torch_generator(torch, trial.seed)
    common = {"num_warps": 4, "num_stages": 2, "num_ctas": 1}

    if kernel == "add":
        (n,) = trial.dimensions
        if trial.distribution == "special":
            pattern_a = torch.tensor(
                [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
                device="cuda",
                dtype=torch.float32,
            )
            pattern_b = torch.tensor(
                [-0.0, -0.0, 1.0, -1.0, 2.0],
                device="cuda",
                dtype=torch.float32,
            )
            indices = torch.arange(n, device="cuda") % pattern_a.numel()
            a = pattern_a[indices]
            b = pattern_b[indices]
        else:
            a = torch.randn((n,), device="cuda", dtype=torch.float32, generator=generator)
            b = torch.randn((n,), device="cuda", dtype=torch.float32, generator=generator)
        source_out = torch.full((n,), 12345.0, device="cuda", dtype=torch.float32)
        replay_out = torch.full((n,), -12345.0, device="cuda", dtype=torch.float32)
        block = 64
        grid = ((n + block - 1) // block,)
        source_kernel[grid](a, b, source_out, n, BLOCK=block, **common)
        replay_kernel[grid](a, b, replay_out, n, BLOCK=block, **common)
    elif kernel == "row_sum":
        m, n = trial.dimensions
        if trial.distribution == "special":
            x = torch.zeros((m, n), device="cuda", dtype=torch.float32)
            alternating = torch.where(
                torch.arange(n, device="cuda") % 2 == 0,
                torch.tensor(1.0e20, device="cuda"),
                torch.tensor(-1.0e20, device="cuda"),
            )
            x[1] = alternating
            x[2, 0] = float("inf")
            if n > 1:
                x[2, 1] = -float("inf")
        else:
            x = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=generator)
        source_out = torch.full((m,), 12345.0, device="cuda", dtype=torch.float32)
        replay_out = torch.full((m,), -12345.0, device="cuda", dtype=torch.float32)
        # These simple row kernels cover one power-of-two tile.  N=BLOCK+1 is
        # therefore evaluated using the next legal tile, while retaining the
        # requested 63/64/65 boundary shapes in the input matrix.
        block_n = 64 if n <= 64 else 128
        source_kernel[(m,)](x, source_out, m, n, BLOCK_N=block_n, **common)
        replay_kernel[(m,)](x, replay_out, m, n, BLOCK_N=block_n, **common)
    elif kernel == "softmax":
        m, n = trial.dimensions
        if trial.distribution == "negative":
            x = -1.0 - 8.0 * torch.rand(
                (m, n), device="cuda", dtype=torch.float32, generator=generator
            )
        elif trial.distribution == "extreme":
            row = torch.linspace(1000.0, 1040.0, n, device="cuda", dtype=torch.float32)
            x = row[None, :].expand(m, n).contiguous()
            x = x + torch.arange(m, device="cuda", dtype=torch.float32)[:, None]
        else:
            x = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=generator)
        source_out = torch.full((m, n), 12345.0, device="cuda", dtype=torch.float32)
        replay_out = torch.full((m, n), -12345.0, device="cuda", dtype=torch.float32)
        block_n = 64 if n <= 64 else 128
        source_kernel[(m,)](x, source_out, m, n, BLOCK_N=block_n, **common)
        replay_kernel[(m,)](x, replay_out, m, n, BLOCK_N=block_n, **common)
    elif kernel == "rmsnorm":
        m, n = trial.dimensions
        if trial.distribution == "special":
            x = torch.zeros((m, n), device="cuda", dtype=torch.float32)
            x[1] = 1.0e10
            x[2] = torch.linspace(
                -1.0e-10, 1.0e-10, n, device="cuda", dtype=torch.float32
            )
        else:
            x = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=generator)
        weight = torch.randn((n,), device="cuda", dtype=torch.float32, generator=generator)
        source_out = torch.full((m, n), 12345.0, device="cuda", dtype=torch.float32)
        replay_out = torch.full((m, n), -12345.0, device="cuda", dtype=torch.float32)
        args = (x, weight, source_out, m, n, 1.0e-5)
        block_n = 64 if n <= 64 else 128
        source_kernel[(m,)](*args, BLOCK_N=block_n, **common)
        args = (x, weight, replay_out, m, n, 1.0e-5)
        replay_kernel[(m,)](*args, BLOCK_N=block_n, **common)
    elif kernel == "matmul":
        m, n, k = trial.dimensions
        if trial.distribution == "zero":
            a = torch.zeros((m, k), device="cuda", dtype=torch.float16)
            b = torch.zeros((k, n), device="cuda", dtype=torch.float16)
        else:
            a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
            b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=generator)
        source_out = torch.full((m, n), 12345.0, device="cuda", dtype=torch.float32)
        replay_out = torch.full((m, n), -12345.0, device="cuda", dtype=torch.float32)
        meta = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            **common,
        }
        grid = ((m + 31) // 32, (n + 31) // 32)
        source_kernel[grid](a, b, source_out, m, n, k, **meta)
        replay_kernel[grid](a, b, replay_out, m, n, k, **meta)
    else:
        raise KeyError(kernel)

    torch.cuda.synchronize()
    return source_out, replay_out


def _max_abs_error(source: Any, replay: Any) -> float:
    import torch

    source_nan = torch.isnan(source)
    replay_nan = torch.isnan(replay)
    if not torch.equal(source_nan, replay_nan):
        return float("inf")
    finite = torch.isfinite(source) & torch.isfinite(replay)
    if not bool(finite.any()):
        return 0.0
    return float(torch.max(torch.abs(source[finite] - replay[finite])).item())


def run_gate2_case(
    case: CapabilityCase,
    source_module: ModuleType,
    module_dir: Path,
    *,
    depth: str,
) -> Gate2Evaluation:
    """Offline two-party source/replay GPU oracle, including rejected cases."""

    try:
        import torch
        import triton  # noqa: F401
    except Exception as exc:
        return Gate2Evaluation(None, 0, f"CUDA/Triton 不可导入: {exc}")
    if not torch.cuda.is_available():
        return Gate2Evaluation(None, 0, "CUDA 不可用；Gate2 两方代理未执行")

    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", case.case_id)
    digest = hashlib.sha256(case.candidate_code().encode("utf-8")).hexdigest()[:12]
    candidate_path = module_dir / f"{safe_id}_{digest}.py"
    candidate_path.write_text(case.candidate_code(), encoding="utf-8")
    try:
        replay_module = _load_module(
            candidate_path, f"_scrc_gate1_candidate_{safe_id}_{digest}"
        )
        source_kernel = getattr(source_module, case.function_name)
        replay_kernel = getattr(replay_module, case.function_name)
    except Exception as exc:
        return Gate2Evaluation(
            False,
            0,
            f"候选模块加载失败: {type(exc).__name__}: {exc}",
        )

    rtol, atol = ((2.0e-3, 2.0e-3) if case.kernel == "matmul" else (2.0e-5, 2.0e-5))
    trials_run = 0
    for trial in gate2_trials(case.kernel, depth):
        try:
            source_out, replay_out = _launch_trial(
                source_kernel, replay_kernel, case.kernel, trial
            )
            trials_run += 1
            torch.testing.assert_close(
                source_out,
                replay_out,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        except Exception as exc:
            error = None
            if "source_out" in locals() and "replay_out" in locals():
                try:
                    error = _max_abs_error(source_out, replay_out)
                except Exception:
                    error = None
            return Gate2Evaluation(
                False,
                trials_run,
                f"{type(exc).__name__}: {exc}",
                failed_trial=trial,
                max_abs_error=error,
            )
    return Gate2Evaluation(True, trials_run)


def expected_observations(
    cases: Iterable[CapabilityCase] | None = None,
) -> list[CapabilityObservation]:
    return [
        CapabilityObservation(
            case.case_id,
            case.expected_gate1_closed,
            case.expected_gate2_pass,
            None if case.expected_gate1_closed else case.expected_cause,
            case.expected_cause,
        )
        for case in (capability_cases() if cases is None else cases)
    ]


def _gate2_pass_given_rejected(observations: Sequence[CapabilityObservation]) -> float | None:
    rejected = [
        observation
        for observation in observations
        if not observation.gate1_closed and observation.gate2_pass is not None
    ]
    if not rejected:
        return None
    return sum(bool(observation.gate2_pass) for observation in rejected) / len(rejected)


def _format_probability(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _sample_specific_analysis(
    case: CapabilityCase,
    observation: CapabilityObservation,
) -> str:
    if observation.outcome is OutcomeClass.FALSE_ACCEPT:
        return (
            "语义错误通过静态闭合却被 GPU 行为否定，属于比较器盲区；"
            "必须提升对应硬指纹，而不能用 Gate2 长期掩盖。"
        )
    if case.family == "equivalent" and observation.outcome is OutcomeClass.FALSE_REJECT:
        caveat = (
            "；但浮点 +0 会影响 signed-zero，只有在语义契约明确忽略零符号时"
            "才能加入中性元规则"
            if case.case_id == "add.redundant_zero"
            else "；rsqrt 与 1/sqrt 存在舍入差异，规则必须绑定 dtype 与容差策略"
        )
        return (
            "候选按当前 tolerance/equal_nan Gate2 oracle 通过，但 TTIR 仍保留"
            "额外/替代算子；根因是有限规范化规则未覆盖该模式"
            + caveat
            + "。"
        )
    if case.family == "interface_mutant":
        return (
            "固定测试值上的行为一致不等于接口契约一致。do_not_specialize 改变"
            "特化与缓存语义，因此这是有意的保守拒绝，不应为追求覆盖率放宽。"
        )
    if case.family == "leakage_mutant":
        return (
            "候选行为正确，但 π 已泄漏具体前端 API；这是反作弊边界的有意拒绝，"
            "否则无法证明语义主要来自 J。"
        )
    if case.family == "semantic_mutant" and observation.outcome is OutcomeClass.TRUE_REJECT:
        return (
            f"静态首个硬失败归入 {observation.gate1_failure.value if observation.gate1_failure else 'unknown'}；"
            "对抗/边界 GPU 输入也确认行为错误，说明该指纹项具有实际区分力。"
        )
    return case.rationale


def _print_report(report: Mapping[str, Any]) -> None:
    print("SCRC Gate1 能力上限探针")
    print("Gate2 oracle: source-vs-replay GPU two-party proxy（非完整三方 Gate2）")
    print()
    print("case_id                              G1/期望   G2/期望   outcome")
    print("-" * 82)
    for item in report["cases"]:
        gate1 = item["gate1"]["closed"]
        gate2 = item["gate2"]["passed"]
        expected_gate1 = item["catalog"]["expected_gate1_closed"]
        expected_gate2 = item["catalog"]["expected_gate2_pass"]
        outcome = item["outcome"]
        g1_text = f"{str(gate1):5}/{str(expected_gate1):5}"
        g2_text = f"{str(gate2):5}/{str(expected_gate2):5}"
        print(f"{item['catalog']['case_id']:<36} {g1_text:<11} {g2_text:<11} {outcome}")

    summary = report["summary"]
    print()
    print("关键指标")
    print(
        "  P(Gate2 pass | Gate1 closed) = "
        + _format_probability(summary["gate2_precision_given_closed"])
    )
    print(
        "  Gate1 coverage of Gate2-pass = "
        + _format_probability(summary["gate1_coverage_of_gate2_pass"])
    )
    print(
        "  P(Gate2 pass | Gate1 rejected) = "
        + _format_probability(report["gate2_pass_given_rejected"])
    )
    print(
        f"  true_accept={summary['true_accept']} false_accept={summary['false_accept']} "
        f"false_reject={summary['false_reject']} true_reject={summary['true_reject']}"
    )

    analyses = report["failure_analyses"]
    if analyses:
        print()
        print("失败/边界样例根因")
        for analysis in analyses:
            print(
                f"  - [{analysis['priority']}] {analysis['case_id']}: "
                f"{analysis['outcome']} / {analysis['primary_cause']}"
            )
            print(f"    {analysis['explanation']}")
            print(f"    样例分析：{analysis['sample_specific_analysis']}")


def run_suite(
    cases: Sequence[CapabilityCase],
    *,
    cache_dir: Path,
    run_gate2: bool,
    gate2_depth: str,
) -> dict[str, Any]:
    compiler = TritonCompiler(cache_dir=cache_dir, timeout=240.0)
    observations: list[CapabilityObservation] = []
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="intentir-gate1-capability-modules-") as temporary:
        module_dir = Path(temporary)
        try:
            source_module = _load_module(
                SOURCE_MODULE, "_scrc_gate1_capability_source"
            )
        except Exception as exc:
            source_module = None
            source_load_diagnostic = f"{type(exc).__name__}: {exc}"
        else:
            source_load_diagnostic = ""

        for case in cases:
            gate1 = run_gate1_case(case, compiler)
            if not run_gate2:
                gate2 = Gate2Evaluation(None, 0, "由 --skip-gate2 跳过")
            elif source_module is None:
                gate2 = Gate2Evaluation(
                    None,
                    0,
                    "源模块加载失败: " + source_load_diagnostic,
                )
            else:
                gate2 = run_gate2_case(
                    case,
                    source_module,
                    module_dir,
                    depth=gate2_depth,
                )

            observation = CapabilityObservation(
                case.case_id,
                gate1.closed,
                gate2.passed,
                gate1.failure,
                case.expected_cause,
                {
                    "family": case.family,
                    "expected_gate1_closed": case.expected_gate1_closed,
                    "expected_gate2_pass": case.expected_gate2_pass,
                },
            )
            observations.append(observation)
            results.append(
                {
                    "catalog": case.to_catalog_dict(),
                    "gate1": gate1.to_dict(),
                    "gate2": gate2.to_dict(),
                    "outcome": observation.outcome.value,
                    "matches_expected_gate1": gate1.closed == case.expected_gate1_closed,
                    "matches_expected_gate2": (
                        gate2.passed is None or gate2.passed == case.expected_gate2_pass
                    ),
                }
            )

    summary = summarize_capability(observations)
    analyses = []
    for case, observation in zip(cases, observations):
        analysis = analyze_failure(observation)
        if analysis is None:
            continue
        analyses.append(
            {
                "case_id": analysis.case_id,
                "outcome": analysis.outcome.value,
                "primary_cause": analysis.primary_cause.value,
                "explanation": analysis.explanation,
                "sample_specific_analysis": _sample_specific_analysis(
                    case, observation
                ),
                "priority": analysis.priority,
            }
        )

    expected_summary = summarize_capability(expected_observations(cases))
    return {
        "schema": "scrc-gate1-capability-v1",
        "gate2_oracle": "two-party-source-vs-replay-gpu-proxy",
        "gate2_depth": gate2_depth if run_gate2 else "skipped",
        "cases": results,
        "summary": summary.to_json(),
        "gate2_pass_given_rejected": _gate2_pass_given_rejected(observations),
        "failure_analyses": analyses,
        "design_target_summary": {
            **expected_summary.to_json(),
            "gate2_pass_given_rejected": _gate2_pass_given_rejected(
                expected_observations(cases)
            ),
        },
        "expectation_mismatches": [
            item["catalog"]["case_id"]
            for item in results
            if not item["matches_expected_gate1"] or not item["matches_expected_gate2"]
        ],
    }


def _select_cases(case_ids: Sequence[str]) -> tuple[CapabilityCase, ...]:
    cases = capability_cases()
    if not case_ids:
        return cases
    by_id = {case.case_id: case for case in cases}
    unknown = sorted(set(case_ids) - set(by_id))
    if unknown:
        raise SystemExit("unknown case id(s): " + ", ".join(unknown))
    return tuple(by_id[case_id] for case_id in case_ids)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list cases without CUDA")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="ID",
        help="run only one case; repeatable",
    )
    parser.add_argument(
        "--skip-gate2",
        action="store_true",
        help="compile and compare only; do not execute kernels",
    )
    parser.add_argument(
        "--gate2-depth",
        choices=("quick", "full"),
        default="full",
        help="quick targets each mutation; full spans boundary shapes/distributions/seeds",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="optional persistent compiler cache (default: temporary and deleted)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="write the full machine-readable report to this explicit path",
    )
    parser.add_argument(
        "--strict-expectations",
        action="store_true",
        help="exit nonzero when observed G1/G2 differs from the design target",
    )
    parser.add_argument(
        "--fail-on-false-accept",
        action="store_true",
        help="exit nonzero on any Gate1-closed/Gate2-failed case",
    )
    args = parser.parse_args(argv)

    cases = _select_cases(args.case)
    if args.list:
        print(json.dumps([case.to_catalog_dict() for case in cases], indent=2, ensure_ascii=False))
        return 0

    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="intentir-gate1-capability-cache-") as temporary:
            report = run_suite(
                cases,
                cache_dir=Path(temporary),
                run_gate2=not args.skip_gate2,
                gate2_depth=args.gate2_depth,
            )
    else:
        report = run_suite(
            cases,
            cache_dir=args.cache_dir,
            run_gate2=not args.skip_gate2,
            gate2_depth=args.gate2_depth,
        )

    _print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json_output}")

    false_accepts = report["summary"]["false_accept"]
    if args.fail_on_false_accept and false_accepts:
        return 2
    if args.strict_expectations and report["expectation_mismatches"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
