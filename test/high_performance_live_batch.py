"""Official 28-kernel fresh-LLM -> Gate1 -> Gate2 campaign.

The lowering registry intentionally remains verification-neutral.  This
campaign joins its frozen 28 direct-kernel membership to the authentic launch
captures produced from the upstream Liger, FlashAttention and xFormers
wrappers.  Every counted run starts from the captured source snapshot and makes
new provider calls; historical candidate JSON and handwritten MLIR are never
accepted as Lift input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
)
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    GATE2_EXECUTION_PROTOCOL,
    InputDistribution,
    Tolerance,
)

import scrc_live_e2e as live


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MODEL = "gpt-5.6-sol"
OFFICIAL_EFFORT = "xhigh"
OFFICIAL_GATE1_PROFILE = "formal"
OFFICIAL_SEMANTIC_INTERVENTIONS = 0
OFFICIAL_PI_ABLATIONS = False
CAMPAIGN_SCHEMA = "intentir.high-performance.live-gate.v3"
DEFAULT_REGISTRY = ROOT / "spec/lowering/high-performance-kernels.json"
DEFAULT_CAPTURE_MANIFEST = Path(
    "/home/kingdom/phdworks/TianchenIR-rebuttal/artifacts/gate_paper/"
    "provider29_capture_2026-07-16_complete/manifest.json"
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _scenario_name(kernel_id: str) -> str:
    prefix = "high_performance:"
    if not kernel_id.startswith(prefix):
        raise ValueError(f"not a high-performance kernel id: {kernel_id!r}")
    return kernel_id[len(prefix):]


def _dtype_tolerance(arguments: tuple[SCRCCompilerArgument, ...]) -> Tolerance:
    dtypes = {argument.dtype.casefold() for argument in arguments}
    if dtypes & {"f16", "float16", "bf16", "bfloat16"}:
        return Tolerance(rtol=2.0e-2, atol=2.0e-2)
    return Tolerance(rtol=1.0e-4, atol=1.0e-5)


@dataclass(frozen=True)
class CapturedLiveKernelCase:
    """The subset of ``LiveKernelCase`` backed by one authentic P0 launch."""

    name: str
    function_name: str
    source_path: Path
    arguments: tuple[SCRCCompilerArgument, ...]
    grid: tuple[int, ...]
    schedule_values: dict[str, int | float | bool | str]
    compiler_slots: tuple[str, ...]
    grid_contract: tuple[str, ...]
    parameter_order: tuple[str, ...]
    constexpr_parameters: tuple[str, ...]
    origin: Mapping[str, Any] = field(default_factory=dict)
    input_scale: float = 1.0
    tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(rtol=1.0e-4, atol=1.0e-5)
    )
    input_distributions: Mapping[str, InputDistribution] = field(
        default_factory=dict
    )
    input_scales: Mapping[str, float] = field(default_factory=dict)
    gate2_shape_profiles: tuple[Any, ...] = ()
    gate2_value_profiles: tuple[Any, ...] = ()
    gate1_specialization_profiles: tuple[Any, ...] = ()
    gate1_primary_profile_id: str | None = None

    @property
    def source_text(self) -> str:
        return self.source_path.read_text(encoding="utf-8")

    @property
    def kernel_source(self) -> KernelSource:
        return KernelSource.from_file(
            self.source_path, self.function_name, trusted=True
        )

    def compiler_view(self) -> SCRCCompilerView:
        return SCRCCompilerView(
            self.schedule_values,
            self.compiler_slots,
            (),
            self.arguments,
            self.grid,
            self.grid_contract,
            self.parameter_order,
            self.constexpr_parameters,
        )


def _load_membership(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kernels = payload.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 28:
        raise ValueError("high-performance registry must contain exactly 28 kernels")
    ids = [str(kernel["kernel_id"]) for kernel in kernels]
    if len(ids) != len(set(ids)):
        raise ValueError("high-performance registry contains duplicate kernel ids")
    return [dict(kernel) for kernel in kernels]


def _capture_by_name(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("specs")
    if not isinstance(specs, list):
        raise ValueError("capture manifest has no specs array")
    result = {str(spec["name"]): dict(spec) for spec in specs}
    if len(result) != len(specs):
        raise ValueError("capture manifest contains duplicate spec names")
    return result


def _captured_case(
    kernel: Mapping[str, Any], capture: Mapping[str, Any]
) -> CapturedLiveKernelCase:
    kernel_id = str(kernel["kernel_id"])
    expected_symbol = str(kernel["symbol"])
    contract = capture.get("provider29_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{kernel_id} capture has no provider29 contract")
    if contract.get("source_symbol") != expected_symbol:
        raise ValueError(
            f"{kernel_id} capture symbol {contract.get('source_symbol')!r} "
            f"differs from registry symbol {expected_symbol!r}"
        )
    matches = [
        launch
        for launch in capture.get("launches", [])
        if launch.get("target_symbol_match")
        and launch.get("conversion_status") == "converted"
        and isinstance(launch.get("case"), Mapping)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{kernel_id} requires exactly one converted target launch, "
            f"found {len(matches)}"
        )
    launch = matches[0]
    if launch.get("function") != expected_symbol:
        raise ValueError(f"{kernel_id} target launch function mismatch")
    if not launch.get("preflight", {}).get("ok"):
        raise ValueError(f"{kernel_id} target launch did not pass P0 preflight")
    case = launch["case"]
    compiler = case.get("compiler_view")
    if not isinstance(compiler, Mapping):
        raise ValueError(f"{kernel_id} capture has no compiler view")
    arguments = tuple(
        SCRCCompilerArgument(
            str(argument["name"]),
            str(argument["kind"]),
            str(argument["dtype"]),
            tuple(argument.get("shape", ())),
            argument.get("value"),
        )
        for argument in compiler.get("arguments", [])
    )
    source_path = Path(str(case["source_path"])).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"captured source snapshot is missing: {source_path}")
    return CapturedLiveKernelCase(
        name=kernel_id.replace(":", "_").replace(".", "_"),
        function_name=expected_symbol,
        source_path=source_path,
        arguments=arguments,
        grid=tuple(int(value) for value in compiler["grid"]),
        schedule_values=dict(compiler["schedule_values"]),
        compiler_slots=tuple(compiler.get("compiler_slots", ())),
        grid_contract=tuple(compiler.get("grid_contract", ())),
        parameter_order=tuple(compiler.get("parameter_order", ())),
        constexpr_parameters=tuple(
            compiler.get("constexpr_parameters", ())
        ),
        origin={
            "kernel_id": kernel_id,
            "suite": kernel.get("suite"),
            "family": capture.get("family"),
            "capture_schema": launch.get("schema"),
            "capture_manifest_name": capture.get("name"),
            "launch_ordinal": launch.get("ordinal"),
            "runtime_source_path": launch.get("runtime_source_path"),
            "runtime_source_sha256": launch.get("runtime_source_sha256"),
            "source_snapshot": case.get("origin", {}).get("snapshot"),
        },
        tolerance=_dtype_tolerance(arguments),
    )


def load_cases(
    registry_path: Path = DEFAULT_REGISTRY,
    capture_manifest: Path = DEFAULT_CAPTURE_MANIFEST,
) -> list[CapturedLiveKernelCase]:
    membership = _load_membership(registry_path)
    captures = _capture_by_name(capture_manifest)
    cases: list[CapturedLiveKernelCase] = []
    for kernel in membership:
        name = _scenario_name(str(kernel["kernel_id"]))
        capture = captures.get(name)
        if capture is None:
            raise ValueError(f"missing authentic capture for {name!r}")
        cases.append(_captured_case(kernel, capture))
    if len({case.function_name for case in cases}) != 28:
        raise ValueError("official campaign must resolve 28 unique JIT symbols")
    return cases


def _official_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=OFFICIAL_MODEL,
        effort=OFFICIAL_EFFORT,
        seed=args.seed,
        lift_repairs=args.lift_repairs,
        relift_rounds=args.relift_rounds,
        semantic_interventions=OFFICIAL_SEMANTIC_INTERVENTIONS,
        pi_ablations=OFFICIAL_PI_ABLATIONS,
        pi_ablation_limit=args.pi_ablation_limit,
        pi_ablation_kinds=None,
        diagnose_rejected=False,
        gate2_matrix=args.gate2_matrix,
        gate2_seeds=args.gate2_seeds,
        gate1_profile=OFFICIAL_GATE1_PROFILE,
    )


def _fresh_contract(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    lift = report.get("lift")
    if not isinstance(lift, Mapping) or not lift.get("ok"):
        failures.append("fresh_lift_failed")
    else:
        if lift.get("reused"):
            failures.append("lift_was_reused")
        if lift.get("response_model") != OFFICIAL_MODEL:
            failures.append("response_model_mismatch")
        response_ids = lift.get("response_ids")
        if not isinstance(response_ids, list) or not response_ids:
            failures.append("missing_provider_response_id")
        if not isinstance(lift.get("usage"), Mapping):
            failures.append("missing_provider_usage")
    if report.get("model") != OFFICIAL_MODEL:
        failures.append("requested_model_mismatch")
    if report.get("effort") != OFFICIAL_EFFORT:
        failures.append("requested_effort_mismatch")
    provider_request = report.get("provider_request")
    if isinstance(provider_request, Mapping):
        if provider_request.get("model") != OFFICIAL_MODEL:
            failures.append("provider_requested_model_mismatch")
        if provider_request.get("reasoning_effort") != OFFICIAL_EFFORT:
            failures.append("provider_requested_effort_mismatch")
        if provider_request.get("api_key_recorded") is not False:
            failures.append("provider_api_key_recording_policy_mismatch")
    gate1 = report.get("gate1")
    if not isinstance(gate1, Mapping) or gate1.get("status") != "replay_closed":
        failures.append("gate1_not_closed")
    else:
        interventions = gate1.get("interventions")
        if not isinstance(interventions, list) or interventions:
            failures.append("paired_interventions_present")
        masked_generation = gate1.get("masked_generation")
        if not isinstance(masked_generation, Mapping):
            failures.append("missing_masked_j_generation")
        else:
            if masked_generation.get("masked") is not True:
                failures.append("masked_j_generation_not_masked")
            if not masked_generation.get("response_id"):
                failures.append("missing_masked_j_response_id")
            if masked_generation.get("requested_model") != OFFICIAL_MODEL:
                failures.append("masked_j_requested_model_mismatch")
            if masked_generation.get("response_model") != OFFICIAL_MODEL:
                failures.append("masked_j_response_model_mismatch")
        if gate1.get("masked_outcome") not in {
            "hard_mismatch",
            "compile_nonclosure",
        }:
            failures.append("masked_j_did_not_nonclose")
    anti_leakage = report.get("gate1_anti_leakage_policy")
    if not isinstance(anti_leakage, Mapping):
        failures.append("missing_anti_leakage_policy")
    else:
        if anti_leakage.get("paired_semantic_interventions") is not False:
            failures.append("paired_interventions_not_disabled")
        if anti_leakage.get("maximum_interventions") != 0:
            failures.append("paired_intervention_budget_not_zero")
        pi_policy = anti_leakage.get("pi_ablations")
        if (
            not isinstance(pi_policy, Mapping)
            or pi_policy.get("enabled") is not False
        ):
            failures.append("pi_ablations_not_disabled")
    pi_ablations = report.get("pi_ablations")
    if pi_ablations not in (None, []):
        failures.append("pi_ablation_evidence_present")
    gate2 = report.get("gate2")
    if not isinstance(gate2, Mapping):
        suite = report.get("gate2_suite")
        gate2 = suite if isinstance(suite, Mapping) else None
        gate2_passed = bool(gate2 is not None and gate2.get("passed") is True)
    else:
        gate2_passed = gate2.get("status") == "gate2_pass"
    if gate2 is None or not gate2_passed:
        failures.append("gate2_not_passed")
    elif gate2.get("execution_protocol") != GATE2_EXECUTION_PROTOCOL:
        failures.append("gate2_evidence_protocol_mismatch")
    gate2_policy = report.get("gate2_execution_policy")
    if not isinstance(gate2_policy, Mapping):
        failures.append("missing_gate2_execution_policy")
    else:
        if gate2_policy.get("protocol") != GATE2_EXECUTION_PROTOCOL:
            failures.append("gate2_execution_protocol_mismatch")
        if gate2_policy.get("legs") != ["source", "replay"]:
            failures.append("gate2_execution_legs_mismatch")
        if gate2_policy.get("production_lowering_in_verdict") is not False:
            failures.append("production_lowering_entered_gate2_verdict")
    return not failures, failures


def _performance_state(report: Mapping[str, Any]) -> dict[str, Any]:
    performance = report.get("gpu_performance")
    if not isinstance(performance, Mapping) or performance.get("status") != "measured":
        return {
            "measured": False,
            "speedup_source_over_generated": None,
            "non_regression_2pct_pass": False,
            "extension_3pct_speedup_pass": False,
        }
    comparison = performance.get("comparison")
    if not isinstance(comparison, Mapping):
        return {
            "measured": False,
            "speedup_source_over_generated": None,
            "non_regression_2pct_pass": False,
            "extension_3pct_speedup_pass": False,
        }
    speedup = comparison.get("speedup_source_over_generated")
    measured = (
        isinstance(speedup, (int, float))
        and not isinstance(speedup, bool)
        and math.isfinite(float(speedup))
        and float(speedup) > 0.0
        and performance.get("correctness_pass") is True
    )
    return {
        "measured": measured,
        "speedup_source_over_generated": float(speedup) if measured else None,
        "non_regression_2pct_pass": bool(
            measured and comparison.get("non_regression_2pct_pass") is True
        ),
        "extension_3pct_speedup_pass": bool(
            measured and comparison.get("extension_3pct_speedup_pass") is True
        ),
    }


def _campaign_row(
    kernel_id: str,
    report_path: Path,
    report: Mapping[str, Any],
    *,
    resumed: bool,
    exit_code: int | None = None,
) -> dict[str, Any]:
    passed, failures = _fresh_contract(report)
    performance = _performance_state(report)
    row: dict[str, Any] = {
        "kernel_id": kernel_id,
        "status": "complete" if passed else "failed",
        "fresh_full_pipeline": passed,
        "report": str(report_path),
        "resumed": resumed,
        "failures": failures,
        **performance,
    }
    if exit_code is not None:
        row["exit_code"] = exit_code
    return row


def _select_cases(
    cases: list[CapturedLiveKernelCase], requested: list[str]
) -> list[CapturedLiveKernelCase]:
    if not requested:
        return cases
    by_id = {str(case.origin["kernel_id"]): case for case in cases}
    by_name = {_scenario_name(kernel_id): case for kernel_id, case in by_id.items()}
    selected: list[CapturedLiveKernelCase] = []
    for name in requested:
        case = by_id.get(name) or by_name.get(name)
        if case is None:
            raise SystemExit(f"unknown high-performance kernel {name!r}")
        if case not in selected:
            selected.append(case)
    return selected


def run_campaign(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    cases = _select_cases(
        load_cases(args.registry, args.capture_manifest),
        list(args.case or ()),
    )
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    official_args = _official_args(args)
    for index, case in enumerate(cases, start=1):
        kernel_id = str(case.origin["kernel_id"])
        report_path = args.out_dir / f"{case.name}.json"
        if args.resume and report_path.is_file():
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            passed, failures = _fresh_contract(saved)
            if passed:
                print(
                    json.dumps(
                        {
                            "stage": "campaign_reused_complete",
                            "kernel": kernel_id,
                            "index": index,
                            "total": len(cases),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                rows.append(
                    _campaign_row(
                        kernel_id,
                        report_path,
                        saved,
                        resumed=True,
                    )
                )
                continue
        print(
            json.dumps(
                {
                    "stage": "campaign_case_start",
                    "kernel": kernel_id,
                    "index": index,
                    "total": len(cases),
                    "model": OFFICIAL_MODEL,
                    "effort": OFFICIAL_EFFORT,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        code, report = live.run_case(case, official_args)
        report["campaign"] = {
            "schema": CAMPAIGN_SCHEMA,
            "kernel_id": kernel_id,
            "official_model": OFFICIAL_MODEL,
            "official_effort": OFFICIAL_EFFORT,
            "first_lift_cache_policy": "forbid",
            "historical_candidate_json_as_input": "forbidden",
            "gate1_profile": OFFICIAL_GATE1_PROFILE,
            "gate2_execution_protocol": GATE2_EXECUTION_PROTOCOL,
            "semantic_interventions": OFFICIAL_SEMANTIC_INTERVENTIONS,
            "pi_ablations": OFFICIAL_PI_ABLATIONS,
            "required_path": [
                "fresh_lift",
                "normal_replay",
                "masked_j",
                "source_replay_gate2",
            ],
            "origin": dict(case.origin),
        }
        passed, failures = _fresh_contract(report)
        report["campaign"]["fresh_full_pipeline"] = passed
        report["campaign"]["contract_failures"] = failures
        report["gpu_performance"] = {
            "status": "not_measured",
            "baseline": "original_captured_triton",
            "counted_as_performance_pass": False,
        }
        _atomic_json(report_path, report)
        rows.append(
            _campaign_row(
                kernel_id,
                report_path,
                report,
                resumed=False,
                exit_code=code,
            )
        )
    measured_speedups = [
        float(row["speedup_source_over_generated"])
        for row in rows
        if row["measured"]
    ]
    suite_geomean = (
        math.exp(sum(math.log(value) for value in measured_speedups) / 28)
        if len(cases) == 28 and len(measured_speedups) == 28
        else None
    )
    summary = {
        "schema": CAMPAIGN_SCHEMA,
        "model": OFFICIAL_MODEL,
        "effort": OFFICIAL_EFFORT,
        "gate1_profile": OFFICIAL_GATE1_PROFILE,
        "gate2_execution_protocol": GATE2_EXECUTION_PROTOCOL,
        "semantic_interventions": OFFICIAL_SEMANTIC_INTERVENTIONS,
        "pi_ablations": OFFICIAL_PI_ABLATIONS,
        "selected": len(cases),
        "fresh_full_pipeline_passed": sum(
            bool(row["fresh_full_pipeline"]) for row in rows
        ),
        "performance_measured": sum(bool(row["measured"]) for row in rows),
        "non_regression_2pct_passed": sum(
            bool(row["non_regression_2pct_pass"]) for row in rows
        ),
        "extension_3pct_passed": sum(
            bool(row["extension_3pct_speedup_pass"]) for row in rows
        ),
        "performance_passed": sum(
            bool(row["extension_3pct_speedup_pass"]) for row in rows
        ),
        "performance_pass_definition": "per-kernel extension 1.03x + CI lower > 1",
        "paper_suite_geomean_speedup": suite_geomean,
        "paper_suite_target_speedup": 1.35,
        "paper_suite_target_pass": (
            suite_geomean >= 1.35 if suite_geomean is not None else None
        ),
        "rows": rows,
    }
    _atomic_json(args.out_dir / "summary.json", summary)
    return (
        0 if summary["fresh_full_pipeline_passed"] == len(cases) else 1,
        summary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--capture-manifest", type=Path, default=DEFAULT_CAPTURE_MANIFEST
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/intentir_hp28_live_gate"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument("--relift-rounds", type=int, default=3)
    parser.add_argument("--semantic-interventions", type=int, default=0)
    parser.add_argument(
        "--pi-ablations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pi-ablation-limit", type=int, default=3)
    parser.add_argument("--gate2-matrix", action="store_true")
    parser.add_argument(
        "--gate2-seed", dest="gate2_seeds", action="append", type=int
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases = load_cases(args.registry, args.capture_manifest)
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "kernel_id": case.origin["kernel_id"],
                        "function": case.function_name,
                        "source": str(case.source_path),
                        "grid": list(case.grid),
                    }
                    for case in cases
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    code, summary = run_campaign(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
