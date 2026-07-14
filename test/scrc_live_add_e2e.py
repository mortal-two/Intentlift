"""Run one real SCRC add lift -> Gate1 -> Gate2 closure.

This is intentionally one thin integration entry, not a second verifier.  It
uses the production lifter, source-blind replay writer, Gate1 compiler/
comparator, and Gate2 three-way runners already implemented by the project.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from intentir.extraction.lifters.scrc import (
    SCRCCompilerView,
    SCRCLiftError,
    extract_scrc_artifact,
)
from intentir.extraction.providers.provider import get_provider
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.model import Gate1Status, ParameterKind, ParameterMode
from intentir.verify.gate1.orchestrate import Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import ResponsesReplayWriter
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate2 import (
    ArgumentRole,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Lower2CSpec,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
)

from scrc_probe_sources import add_kernel


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/scrc_probe_sources.py"


def _dtype(value: str | None) -> str:
    aliases = {"f16": "float16", "f32": "float32", "f64": "float64"}
    return aliases.get(value or "f32", value or "float32")


def _compile_case(artifact, n: int) -> CompileCase:
    arguments: dict[str, RuntimeArgument] = {}
    for parameter in artifact.interface.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            continue
        if parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            arguments[parameter.name] = RuntimeArgument.tensor(
                (n,), _dtype(parameter.dtype)
            )
        elif parameter.name == "n_elements":
            arguments[parameter.name] = RuntimeArgument.scalar(n, "int32")
        else:
            raise ValueError(
                f"live add does not know scalar binding for {parameter.name!r}"
            )
    block = int(artifact.schedule.values["BLOCK"])
    return CompileCase(arguments, ((n + block - 1) // block,))


def _symbolic_dims(artifact, n: int) -> dict[str, int]:
    symbols = {
        dim
        for tensor in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
        for dim in tensor.shape
        if isinstance(dim, str)
    }
    return {symbol: n for symbol in symbols}


def _gate2_case(artifact, n: int) -> Gate2Case:
    rng = np.random.default_rng(20260714)
    values: dict[str, Any] = {"n_elements": n}
    arguments: list[TritonArgument] = []
    output_keys: list[str] = []

    for parameter in artifact.interface.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            continue
        if parameter.kind is ParameterKind.SCALAR:
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value_key=parameter.name,
                )
            )
            continue
        tensor_ref = parameter.tensor_ref
        if tensor_ref is None:
            raise ValueError(f"pointer {parameter.name!r} has no tensor_ref")
        if parameter.mode is ParameterMode.INPUT:
            values[tensor_ref] = rng.standard_normal(n).astype(np.float32)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INPUT,
                    value_key=tensor_ref,
                    dtype=parameter.dtype or "f32",
                )
            )
        elif parameter.mode is ParameterMode.OUTPUT:
            output_keys.append(tensor_ref)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.OUTPUT,
                    dtype=parameter.dtype or "f32",
                    shape=(n,),
                    output_key=tensor_ref,
                )
            )
        else:
            raise ValueError(
                f"live add currently expects input/output pointers, got "
                f"{parameter.name}:{parameter.mode.value}"
            )

    if len(output_keys) != 1:
        raise ValueError("live add requires exactly one output")
    block = int(artifact.schedule.values["BLOCK"])
    triton = TritonExecutionSpec(
        tuple(arguments),
        ((n + block - 1) // block,),
        dict(artifact.schedule.values),
    )
    lowering = Lower2CSpec(
        artifact.j.intent,
        _symbolic_dims(artifact, n),
        {tensor.name: tensor.name for tensor in artifact.j.intent.inputs},
        output_keys[0],
    )
    return Gate2Case(
        f"add.live.tail{n}",
        values,
        triton,
        lowering,
        tuple(output_keys),
        {output_keys[0]: Tolerance(rtol=1.0e-5, atol=1.0e-6)},
    )


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if path:
        Path(path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _execution_json(result) -> dict[str, Any]:
    return {
        "leg": result.leg,
        "ok": result.ok,
        "error": result.error,
        "unsupported": result.unsupported,
        "metadata": dict(result.metadata),
        "outputs": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for name, value in result.outputs.items()
        },
    }


def _progress(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    provider = replace(
        get_provider(), model=args.model, reasoning_effort=args.effort
    )
    compiler_view = SCRCCompilerView(
        {
            "BLOCK": args.block,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        ("num_warps", "num_stages"),
        (),
    )
    source_text = inspect.getsource(add_kernel.fn)
    registry = default_triton_rule_registry()
    report: dict[str, Any] = {
        "case": "add",
        "model": args.model,
        "effort": args.effort,
        "n": args.n,
        "compiler_view": compiler_view.to_json(),
    }

    _progress("lift_start", model=args.model, effort=args.effort)
    try:
        extraction = extract_scrc_artifact(
            source_text,
            "add_kernel",
            registry=registry,
            provider=provider,
            compiler_view=compiler_view,
            max_repairs=args.lift_repairs,
            effort=args.effort,
        )
    except SCRCLiftError as exc:
        report["lift"] = {
            "ok": False,
            "attempts": [
                {
                    "attempt": item.attempt,
                    "model": item.model,
                    "response_id": item.response_id,
                    "diagnostic": item.diagnostic,
                    "response_text": item.response_text,
                }
                for item in exc.attempts
            ],
        }
        return 2, report

    artifact = extraction.artifact
    _progress(
        "lift_complete",
        attempts=extraction.provenance.attempts,
        response_id=extraction.provenance.response_id,
    )
    report["lift"] = {
        "ok": True,
        "model": extraction.provenance.model,
        "attempts": extraction.provenance.attempts,
        "repairs": list(extraction.provenance.repairs),
        "response_id": extraction.provenance.response_id,
        "artifact": artifact.to_json(),
    }

    writer = ResponsesReplayWriter(
        provider=provider,
        model_candidates=(args.model,),
        effort=args.effort,
        max_retries=1,
    )
    _progress("gate1_start")
    gate1 = Gate1Runner(
        config=Gate1Config(
            max_repairs=args.rewrite_repairs,
            max_relifts=1,
            stable_closures=1,
            run_masked_ablation=True,
        )
    ).run(
        KernelSource.from_file(SOURCE, "add_kernel", trusted=True),
        artifact,
        _compile_case(artifact, args.n),
        registry,
        writer,
    )
    report["gate1"] = gate1.to_json()
    _progress("gate1_complete", status=gate1.status.value)
    if not gate1.replay_closed:
        return 3, report

    closed = next(
        (
            candidate
            for candidate in gate1.candidates
            if candidate.comparison is not None
            and candidate.comparison.hard_pass
            and candidate.generation is not None
        ),
        None,
    )
    if closed is None or closed.generation is None:
        report["gate1"]["diagnostics"].append(
            "Gate1 closed without a recoverable replay candidate"
        )
        return 3, report
    replay = KernelSource.from_code(
        closed.generation.code, artifact.interface.function_name
    )
    report["replay"] = {
        "response_id": closed.generation.response_id,
        "model": closed.generation.response_model,
        "code_sha256": closed.generation.code_sha256,
        "code": closed.generation.code,
    }

    _progress("gate2_start")
    gate2 = Gate2Runner().run(
        KernelSource.from_file(SOURCE, "add_kernel", trusted=True),
        replay,
        _gate2_case(artifact, args.n),
    )
    report["gate2"] = {
        "case_id": gate2.case_id,
        "status": gate2.status.value,
        "source": _execution_json(gate2.source),
        "replay": _execution_json(gate2.replay),
        "lowering": _execution_json(gate2.lowering),
        "source_replay": None
        if gate2.source_replay is None
        else asdict(gate2.source_replay),
        "source_lowering": None
        if gate2.source_lowering is None
        else asdict(gate2.source_lowering),
        "replay_lowering": None
        if gate2.replay_lowering is None
        else asdict(gate2.replay_lowering),
    }
    _progress("gate2_complete", status=gate2.status.value)
    return (0 if gate2.status is Gate2Status.PASS else 4), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument("--rewrite-repairs", type=int, default=2)
    parser.add_argument("--report")
    args = parser.parse_args()

    code, report = run(args)
    _write_report(args.report, report)
    summary = {
        "exit_code": code,
        "lift": report.get("lift", {}).get("ok"),
        "gate1": report.get("gate1", {}).get("status"),
        "gate2": report.get("gate2", {}).get("status"),
        "report": args.report,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
