"""Official 28-kernel fresh-LLM -> Gate1 -> Gate2 CUDA campaign.

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
from intentir.verify.gate2 import InputDistribution, Tolerance

import scrc_live_e2e as live


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MODEL = "gpt-5.6-sol"
OFFICIAL_EFFORT = "xhigh"
OFFICIAL_GATE1_PROFILE = "formal"
OFFICIAL_LOWERING_BACKEND = "cuda"
CAMPAIGN_SCHEMA = "intentir.high-performance.live-gate-cuda.v1"
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
        semantic_interventions=args.semantic_interventions,
        pi_ablations=args.pi_ablations,
        pi_ablation_limit=args.pi_ablation_limit,
        pi_ablation_kinds=None,
        diagnose_rejected=False,
        gate2_matrix=args.gate2_matrix,
        gate2_seeds=args.gate2_seeds,
        gate1_profile=OFFICIAL_GATE1_PROFILE,
        lowering_backend=OFFICIAL_LOWERING_BACKEND,
        cuda_only_fast_path=None,
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
    if report.get("gate1", {}).get("status") != "replay_closed":
        failures.append("gate1_not_closed")
    if report.get("gate2", {}).get("status") != "gate2_pass":
        failures.append("gate2_cuda_not_passed")
    if report.get("gate2_lowering_policy", {}).get("backend") != "cuda":
        failures.append("not_cuda_lowering")
    return not failures, failures


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
                    {
                        "kernel_id": kernel_id,
                        "status": "complete",
                        "fresh_full_pipeline": True,
                        "report": str(report_path),
                        "resumed": True,
                        "failures": failures,
                    }
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
            "lowering_backend": OFFICIAL_LOWERING_BACKEND,
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
            {
                "kernel_id": kernel_id,
                "status": "complete" if passed else "failed",
                "fresh_full_pipeline": passed,
                "exit_code": code,
                "report": str(report_path),
                "resumed": False,
                "failures": failures,
            }
        )
    summary = {
        "schema": CAMPAIGN_SCHEMA,
        "model": OFFICIAL_MODEL,
        "effort": OFFICIAL_EFFORT,
        "gate1_profile": OFFICIAL_GATE1_PROFILE,
        "lowering_backend": OFFICIAL_LOWERING_BACKEND,
        "selected": len(cases),
        "fresh_full_pipeline_passed": sum(
            bool(row["fresh_full_pipeline"]) for row in rows
        ),
        "performance_passed": 0,
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
        default=Path("/tmp/intentir_hp28_live_gate_cuda"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument("--relift-rounds", type=int, default=3)
    parser.add_argument("--semantic-interventions", type=int, default=3)
    parser.add_argument(
        "--pi-ablations",
        action=argparse.BooleanOptionalAction,
        default=True,
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
