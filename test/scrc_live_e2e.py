"""Reusable real SCRC lift -> Gate1 -> Gate2 integration harness.

The harness is data-driven by one concrete compile/execution specialization.
It does not add a verifier or an alternate correctness path: all decisions are
still made by the production SCRC lifter, Gate1, and Gate2 implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
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

from scrc_probe_sources import (
    add_kernel,
    rmsnorm_kernel,
    row_sum_kernel,
    softmax_kernel,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/scrc_probe_sources.py"


@dataclass(frozen=True)
class LiveKernelCase:
    name: str
    function: Any
    arguments: tuple[SCRCCompilerArgument, ...]
    grid: tuple[int, ...]
    schedule_values: dict[str, int | float | bool]
    compiler_slots: tuple[str, ...]
    grid_contract: tuple[str, ...]
    constexpr_parameters: tuple[str, ...]
    input_scale: float
    tolerance: Tolerance

    @property
    def function_name(self) -> str:
        return self.function.fn.__name__

    def compiler_view(self) -> SCRCCompilerView:
        return SCRCCompilerView(
            self.schedule_values,
            self.compiler_slots,
            (),
            self.arguments,
            self.grid,
            self.grid_contract,
            tuple(
                [argument.name for argument in self.arguments]
                + list(self.constexpr_parameters)
            ),
            self.constexpr_parameters,
        )


def build_case(args: argparse.Namespace) -> LiveKernelCase:
    n = int(args.n)
    if n <= 0:
        raise ValueError("N must be positive")
    if args.kernel == "add":
        block = int(args.block if args.block is not None else 256)
        arguments = (
            SCRCCompilerArgument("a_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument("b_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument(
                "n_elements", "scalar", "i32", value=n
            ),
        )
        return LiveKernelCase(
            "add",
            add_kernel,
            arguments,
            ((n + block - 1) // block,),
            {
                "BLOCK": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(n_elements, BLOCK)",),
            ("BLOCK",),
            1.0,
            Tolerance(rtol=1.0e-5, atol=1.0e-6),
        )
    if args.kernel == "row_sum":
        m = int(args.m)
        if m <= 0:
            raise ValueError("M must be positive")
        block = int(args.block if args.block is not None else 1024)
        if block < n:
            raise ValueError(
                "row_sum first closure requires one BLOCK_N tile covering N"
            )
        arguments = (
            SCRCCompilerArgument("x_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m,)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
        )
        return LiveKernelCase(
            "row_sum",
            row_sum_kernel,
            arguments,
            (m,),
            {
                "BLOCK_N": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("M",),
            ("BLOCK_N",),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )
    if args.kernel == "softmax":
        m = int(args.m)
        if m <= 0:
            raise ValueError("M must be positive")
        block = int(args.block if args.block is not None else 1024)
        if block < n:
            raise ValueError(
                "softmax first closure requires one BLOCK_N tile covering N"
            )
        arguments = (
            SCRCCompilerArgument("x_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
        )
        return LiveKernelCase(
            "softmax",
            softmax_kernel,
            arguments,
            (m,),
            {
                "BLOCK_N": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("M",),
            ("BLOCK_N",),
            2.0,
            Tolerance(rtol=2.0e-4, atol=2.0e-5),
        )
    if args.kernel == "rmsnorm":
        m = int(args.m)
        if m <= 0:
            raise ValueError("M must be positive")
        block = int(args.block if args.block is not None else 1024)
        if block < n:
            raise ValueError(
                "rmsnorm first closure requires one BLOCK_N tile covering N"
            )
        eps = float(args.eps)
        arguments = (
            SCRCCompilerArgument("x_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("w_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
            SCRCCompilerArgument("eps", "scalar", "f32", value=eps),
        )
        return LiveKernelCase(
            "rmsnorm",
            rmsnorm_kernel,
            arguments,
            (m,),
            {
                "BLOCK_N": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("M",),
            ("BLOCK_N",),
            0.25,
            Tolerance(rtol=3.0e-4, atol=3.0e-5),
        )
    raise KeyError(args.kernel)


def _numpy_dtype(value: str) -> np.dtype:
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    return np.dtype(aliases.get(value.removeprefix("torch."), value))


def _runtime_dtype(value: str) -> str:
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "i1": "bool",
    }
    return aliases.get(value.removeprefix("torch."), value)


def _compile_case(case: LiveKernelCase) -> CompileCase:
    arguments: dict[str, RuntimeArgument] = {}
    for argument in case.arguments:
        if argument.kind == "tensor":
            arguments[argument.name] = RuntimeArgument.tensor(
                argument.shape, _runtime_dtype(argument.dtype)
            )
        else:
            assert argument.value is not None
            arguments[argument.name] = RuntimeArgument.scalar(
                argument.value, _runtime_dtype(argument.dtype)
            )
    return CompileCase(arguments, case.grid)


def _runtime_by_name(case: LiveKernelCase) -> dict[str, SCRCCompilerArgument]:
    return {argument.name: argument for argument in case.arguments}


def _symbolic_dims(artifact, case: LiveKernelCase) -> dict[str, int]:
    runtime = _runtime_by_name(case)
    tensors = {
        tensor.name: tensor
        for tensor in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    dims: dict[str, int] = {}
    for parameter in artifact.interface.parameters:
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            continue
        argument = runtime[parameter.name]
        tensor = tensors[parameter.tensor_ref]
        if len(tensor.shape) != len(argument.shape):
            raise ValueError(
                f"J tensor {tensor.name!r} rank differs from compiler view"
            )
        for declared, concrete in zip(
            tensor.shape, argument.shape, strict=True
        ):
            if isinstance(declared, str):
                previous = dims.setdefault(declared, concrete)
                if previous != concrete:
                    raise ValueError(
                        f"symbolic dim {declared!r} has inconsistent bindings"
                    )
            elif int(declared) != concrete:
                raise ValueError(
                    f"static J extent {declared} differs from {concrete}"
                )
    return dims


def _gate2_case(
    artifact, case: LiveKernelCase, seed: int
) -> Gate2Case:
    rng = np.random.default_rng(seed)
    runtime = _runtime_by_name(case)
    values: dict[str, Any] = {}
    arguments: list[TritonArgument] = []
    output_keys: list[str] = []

    for parameter in artifact.interface.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            continue
        concrete = runtime[parameter.name]
        if parameter.kind is ParameterKind.SCALAR:
            value_key = parameter.tensor_ref or parameter.name
            scalar_value: Any = concrete.value
            if parameter.tensor_ref is not None:
                dtype = _numpy_dtype(parameter.dtype or concrete.dtype)
                scalar_value = dtype.type(concrete.value)
            values[value_key] = scalar_value
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value_key=value_key,
                )
            )
            continue
        tensor_ref = parameter.tensor_ref
        if tensor_ref is None:
            raise ValueError(f"pointer {parameter.name!r} has no tensor_ref")
        if parameter.mode is ParameterMode.INPUT:
            dtype = _numpy_dtype(parameter.dtype or concrete.dtype)
            values[tensor_ref] = (
                rng.standard_normal(concrete.shape) * case.input_scale
            ).astype(dtype)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INPUT,
                    value_key=tensor_ref,
                    dtype=parameter.dtype or concrete.dtype,
                )
            )
        elif parameter.mode is ParameterMode.OUTPUT:
            output_keys.append(tensor_ref)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.OUTPUT,
                    dtype=parameter.dtype or concrete.dtype,
                    shape=concrete.shape,
                    output_key=tensor_ref,
                )
            )
        else:
            raise ValueError(
                f"live harness does not yet have a case binding for "
                f"{parameter.name}:{parameter.mode.value}"
            )

    if len(output_keys) != 1:
        raise ValueError("current lower2c leg requires exactly one output")
    triton = TritonExecutionSpec(
        tuple(arguments), case.grid, dict(artifact.schedule.values)
    )
    lowering = Lower2CSpec(
        artifact.j.intent,
        _symbolic_dims(artifact, case),
        {tensor.name: tensor.name for tensor in artifact.j.intent.inputs},
        output_keys[0],
    )
    return Gate2Case(
        f"{case.name}.live",
        values,
        triton,
        lowering,
        tuple(output_keys),
        {output_keys[0]: case.tolerance},
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


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _redact_artifact_source(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    source = result.pop("source", None)
    if isinstance(source, str):
        encoded = source.encode("utf-8")
        result["source_sha256"] = hashlib.sha256(encoded).hexdigest()
        result["source_bytes"] = len(encoded)
    return result


def _artifact_evidence_json(artifact) -> dict[str, Any]:
    return _redact_artifact_source(artifact.to_json())


def _generation_evidence(generation) -> dict[str, Any]:
    encoded = generation.code.encode("utf-8")
    return {
        "code": generation.code,
        "code_bytes": len(encoded),
    }


def _gate1_evidence_json(gate1) -> dict[str, Any]:
    payload = gate1.to_json()
    payload["artifact"] = _redact_artifact_source(payload["artifact"])
    for serialized, candidate in zip(
        payload["candidates"], gate1.candidates, strict=True
    ):
        if candidate.generation is not None:
            serialized["generation"].update(
                _generation_evidence(candidate.generation)
            )
    if gate1.masked_generation is not None:
        payload["masked_generation"].update(
            _generation_evidence(gate1.masked_generation)
        )
    return payload


def _lift_attempt_evidence(item) -> dict[str, Any]:
    encoded = item.response_text.encode("utf-8")
    return {
        "attempt": item.attempt,
        "model": item.model,
        "response_id": item.response_id,
        "diagnostic": item.diagnostic,
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "response_bytes": len(encoded),
    }


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    case = build_case(args)
    provider = replace(
        get_provider(), model=args.model, reasoning_effort=args.effort
    )
    compiler_view = case.compiler_view()
    source_text = inspect.getsource(case.function.fn)
    registry = default_triton_rule_registry()
    report: dict[str, Any] = {
        "case": case.name,
        "model": args.model,
        "effort": args.effort,
        "seed": args.seed,
        "compiler_view": compiler_view.to_json(),
    }

    _progress(
        "lift_start", kernel=case.name, model=args.model, effort=args.effort
    )
    try:
        extraction = extract_scrc_artifact(
            source_text,
            case.function_name,
            registry=registry,
            provider=provider,
            compiler_view=compiler_view,
            max_repairs=args.lift_repairs,
            effort=args.effort,
        )
    except SCRCLiftError as exc:
        report["lift"] = {
            "ok": False,
            "attempts": [_lift_attempt_evidence(item) for item in exc.attempts],
        }
        _progress("lift_failed", kernel=case.name, attempts=len(exc.attempts))
        return 2, report

    artifact = extraction.artifact
    _progress(
        "lift_complete",
        kernel=case.name,
        attempts=extraction.provenance.attempts,
        response_id=extraction.provenance.response_id,
    )
    report["lift"] = {
        "ok": True,
        "model": extraction.provenance.model,
        "attempts": extraction.provenance.attempts,
        "repairs": list(extraction.provenance.repairs),
        "response_id": extraction.provenance.response_id,
        "artifact": _artifact_evidence_json(artifact),
    }

    writer = ResponsesReplayWriter(
        provider=provider,
        model_candidates=(args.model,),
        effort=args.effort,
        max_retries=1,
    )
    _progress("gate1_start", kernel=case.name)
    gate1 = Gate1Runner(
        config=Gate1Config(
            max_repairs=args.rewrite_repairs,
            max_relifts=1,
            stable_closures=1,
            run_masked_ablation=True,
        )
    ).run(
        KernelSource.from_file(SOURCE, case.function_name, trusted=True),
        artifact,
        _compile_case(case),
        registry,
        writer,
    )
    report["gate1"] = _gate1_evidence_json(gate1)
    _progress("gate1_complete", kernel=case.name, status=gate1.status.value)
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

    _progress("gate2_start", kernel=case.name)
    gate2 = Gate2Runner().run(
        KernelSource.from_file(SOURCE, case.function_name, trusted=True),
        replay,
        _gate2_case(artifact, case, args.seed),
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
    _progress("gate2_complete", kernel=case.name, status=gate2.status.value)
    return (0 if gate2.status is Gate2Status.PASS else 4), report


def _parser(default_kernel: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        choices=("add", "row_sum", "softmax", "rmsnorm"),
        default=default_kernel,
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--block", type=int)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument("--rewrite-repairs", type=int, default=2)
    parser.add_argument("--report")
    return parser


def main(default_kernel: str = "add") -> int:
    args = _parser(default_kernel).parse_args()
    code, report = run(args)
    _write_report(args.report, report)
    summary = {
        "exit_code": code,
        "kernel": report.get("case"),
        "lift": report.get("lift", {}).get("ok"),
        "gate1": report.get("gate1", {}).get("status"),
        "gate2": report.get("gate2", {}).get("status"),
        "report": args.report,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
