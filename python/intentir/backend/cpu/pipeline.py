"""Typed IntentIR -> scalar MLIR -> LLVM CPU reference execution.

This is the scalar leg of source-anchored Gate2.  After Gate1 closes, it lowers
the typed ``IntentFunction`` derived from that final J through the
backend-neutral Intent-to-Linalg path, compiles one value-independent shared
object, and invokes one ranked-memref entry for all logical outputs.  It
contains no source-runtime, JSON-interpreter, CUDA/ROCm, or production-target
fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...frontend.emit_mlir import IndexBoundsProjection, emit_module
from ...ir.model import IntentFunction
from .abi import (
    CPU_ENTRY_SYMBOL,
    CPUABIValidationError,
    CPUBackendUnsupportedError,
    mlir_element_type,
    numpy_dtype,
    validate_array,
)
from .compiler import compile_cpu_module
from .runtime import execute_isolated


CPU_COMPUTE_SYMBOL = "intentir_cpu_compute_v1"

# This is an executor conformance boundary, not a generation rule catalog.  It
# mirrors the operations actually eliminated by ConvertIntentToLinalg plus
# constants emitted directly as arith.constant.  Anything else is rejected
# before an external compiler can accidentally turn "unsupported" into an
# opaque tool failure.
CPU_LOWERABLE_OPS = frozenset(
    {
        "cast",
        "concat",
        "const",
        "contraction",
        "gather",
        "iota",
        "map_binary",
        "map_unary",
        "pad",
        "reduce",
        "select",
        "transpose",
    }
)


@dataclass(frozen=True)
class CPUExecutionResult:
    """All logical outputs and non-verdict execution provenance for one call."""

    outputs: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def _walk_ops(body):
    for op in body:
        yield op
        yield from _walk_ops(op.body)


def validate_cpu_capability(fn: IntentFunction) -> None:
    """Fail closed when ``E_J`` has no complete scalar lowering."""

    operations = tuple(_walk_ops(fn.body))
    if any(op.op == "loop" for op in operations):
        raise CPUBackendUnsupportedError(
            "scalar CPU execution does not yet lower structured IntentIR loops "
            "for semantic recurrence"
        )
    unsupported = sorted({op.op for op in operations} - CPU_LOWERABLE_OPS)
    if unsupported:
        if "scatter" in unsupported:
            raise CPUBackendUnsupportedError(
                "scalar CPU execution rejects logical update until duplicate, "
                "conflict, order, and result relations are typed in J"
            )
        raise CPUBackendUnsupportedError(
            "scalar CPU execution has no complete lowering for IntentOp(s): "
            + ", ".join(unsupported)
        )


def _tensor_type(shape: Sequence[int], element_type: str) -> str:
    if shape:
        return f"tensor<{'x'.join(str(extent) for extent in shape)}x{element_type}>"
    return f"tensor<{element_type}>"


def _memref_type(shape: Sequence[int], element_type: str) -> str:
    if shape:
        return f"memref<{'x'.join(str(extent) for extent in shape)}x{element_type}>"
    return f"memref<{element_type}>"


def _checked_dims(dims: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(dims, Mapping) or not all(
        isinstance(name, str)
        and name
        and isinstance(extent, int)
        and not isinstance(extent, bool)
        and extent >= 0
        for name, extent in dims.items()
    ):
        raise CPUABIValidationError(
            "CPU execution dims must map names to non-negative integers"
        )
    return dict(dims)


def _infer_contracts(
    fn: IntentFunction,
    dims: Mapping[str, int],
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None,
) -> tuple[
    list[tuple[int, ...]],
    list[str],
    list[tuple[int, ...]],
    list[str],
]:
    from ...frontend.emit_mlir import _infer_body, _resolve

    extents = dict(dims)
    input_shapes = [tuple(_resolve(tensor.shape, extents)) for tensor in fn.inputs]
    input_element_types = [mlir_element_type(tensor.dtype) for tensor in fn.inputs]

    env = {
        tensor.name: list(shape)
        for tensor, shape in zip(fn.inputs, input_shapes, strict=True)
    }
    env.update(
        {
            tensor.name: _resolve(tensor.shape, extents)
            for tensor in fn.outputs
        }
    )
    _infer_body(fn.body, env, 0, extents, broadcast_axes)
    output_shapes = [tuple(env[tensor.name]) for tensor in fn.outputs]
    output_element_types = [
        mlir_element_type(tensor.dtype) for tensor in fn.outputs
    ]
    return (
        input_shapes,
        input_element_types,
        output_shapes,
        output_element_types,
    )


def _append_cpu_entry(
    kernel_mlir: str,
    *,
    input_shapes: Sequence[Sequence[int]],
    input_element_types: Sequence[str],
    output_shapes: Sequence[Sequence[int]],
    output_element_types: Sequence[str],
) -> str:
    module = kernel_mlir.rstrip()
    if not module.endswith("}"):
        raise ValueError("CPU compute emitter did not produce a complete module")

    compute_declaration = f"func.func @{CPU_COMPUTE_SYMBOL}("
    if module.count(compute_declaration) != 1:
        raise ValueError("CPU compute module has an ambiguous entry declaration")
    module = module.replace(
        compute_declaration,
        f"func.func private @{CPU_COMPUTE_SYMBOL}(",
        1,
    )

    input_types = [
        _tensor_type(shape, element_type)
        for shape, element_type in zip(
            input_shapes, input_element_types, strict=True
        )
    ]
    output_tensor_types = [
        _tensor_type(shape, element_type)
        for shape, element_type in zip(
            output_shapes, output_element_types, strict=True
        )
    ]
    output_memref_types = [
        _memref_type(shape, element_type)
        for shape, element_type in zip(
            output_shapes, output_element_types, strict=True
        )
    ]

    arguments = [
        *(f"%input{index}: {ty}" for index, ty in enumerate(input_types)),
        *(
            f"%output{index}: {ty}"
            for index, ty in enumerate(output_memref_types)
        ),
    ]
    call_arguments = ", ".join(
        f"%input{index}" for index in range(len(input_types))
    )
    call_argument_types = ", ".join(input_types)
    if len(output_tensor_types) == 1:
        result_binding = "%result"
        result_type = output_tensor_types[0]
    else:
        result_binding = f"%result:{len(output_tensor_types)}"
        result_type = f"({', '.join(output_tensor_types)})"

    lines = [module[:-1].rstrip()]
    lines.append(
        f"  func.func @{CPU_ENTRY_SYMBOL}({', '.join(arguments)}) "
        "attributes {llvm.emit_c_interface} {"
    )
    lines.append(
        f"    {result_binding} = call @{CPU_COMPUTE_SYMBOL}({call_arguments}) "
        f": ({call_argument_types}) -> {result_type}"
    )
    for index, (tensor_type, memref_type) in enumerate(
        zip(output_tensor_types, output_memref_types, strict=True)
    ):
        value = "%result" if len(output_tensor_types) == 1 else f"%result#{index}"
        lines.append(
            "    bufferization.materialize_in_destination "
            f"{value} in writable %output{index} : "
            f"({tensor_type}, {memref_type}) -> ()"
        )
    lines.extend(("    return", "  }", "}"))
    return "\n".join(lines) + "\n"


def emit_cpu_module(
    fn: IntentFunction,
    dims: Mapping[str, int],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> str:
    """Emit the input-value-independent scalar module with one public entry."""

    if not fn.outputs:
        raise ValueError("CPU execution requires at least one logical output")
    validate_cpu_capability(fn)
    checked_dims = _checked_dims(dims)
    (
        input_shapes,
        input_element_types,
        output_shapes,
        output_element_types,
    ) = _infer_contracts(fn, checked_dims, broadcast_axes)
    kernel = emit_module(
        fn,
        checked_dims,
        func_name=CPU_COMPUTE_SYMBOL,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    )
    return _append_cpu_entry(
        kernel,
        input_shapes=input_shapes,
        input_element_types=input_element_types,
        output_shapes=output_shapes,
        output_element_types=output_element_types,
    )


def execute_on_cpu_outputs(
    fn: IntentFunction,
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> CPUExecutionResult:
    """Compile once and execute all logical outputs with one native call."""

    if len(inputs) != len(fn.inputs):
        raise CPUABIValidationError(
            "CPU execution input arity mismatch: "
            f"expected {len(fn.inputs)}, got {len(inputs)}"
        )
    if not fn.outputs:
        raise CPUABIValidationError(
            "CPU execution requires at least one logical output"
        )
    validate_cpu_capability(fn)
    checked_dims = _checked_dims(dims)
    (
        input_shapes,
        _input_element_types,
        output_shapes,
        _output_element_types,
    ) = _infer_contracts(fn, checked_dims, broadcast_axes)

    checked_inputs = [
        validate_array(
            value,
            shape=shape,
            dtype=tensor.dtype,
            context=f"CPU input {tensor.name!r}",
        )
        for tensor, shape, value in zip(
            fn.inputs, input_shapes, inputs, strict=True
        )
    ]
    output_storage = [
        np.empty(shape, dtype=numpy_dtype(tensor.dtype), order="C")
        for tensor, shape in zip(fn.outputs, output_shapes, strict=True)
    ]
    for tensor, shape, value in zip(
        fn.outputs, output_shapes, output_storage, strict=True
    ):
        validate_array(
            value,
            shape=shape,
            dtype=tensor.dtype,
            context=f"CPU output {tensor.name!r}",
            writable=True,
        )

    module = emit_cpu_module(
        fn,
        checked_dims,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    )
    compiled = compile_cpu_module(module)
    runtime = execute_isolated(compiled, checked_inputs, output_storage)
    logical_outputs = {
        tensor.name: value
        for tensor, value in zip(fn.outputs, runtime.outputs, strict=True)
    }
    metadata = {
        **compiled.evidence,
        **runtime.metadata,
        "compute_symbol": CPU_COMPUTE_SYMBOL,
        "input_count": len(checked_inputs),
        "output_count": len(logical_outputs),
        "module_depends_on_runtime_values": False,
    }
    return CPUExecutionResult(logical_outputs, metadata)


def run_on_cpu_outputs(
    fn: IntentFunction,
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> dict[str, np.ndarray]:
    """Compatibility-shaped result API backed only by the memref ABI path."""

    return dict(
        execute_on_cpu_outputs(
            fn,
            dims,
            inputs,
            broadcast_axes=broadcast_axes,
        ).outputs
    )


def run_on_cpu(
    fn: IntentFunction,
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> np.ndarray:
    """Execute a single-output IntentFunction on the scalar CPU path."""

    if len(fn.outputs) != 1:
        raise ValueError(
            "run_on_cpu requires exactly one output; use run_on_cpu_outputs"
        )
    return run_on_cpu_outputs(
        fn, dims, inputs, broadcast_axes=broadcast_axes
    )[fn.outputs[0].name]


__all__ = [
    "CPU_LOWERABLE_OPS",
    "CPU_COMPUTE_SYMBOL",
    "CPUExecutionResult",
    "emit_cpu_module",
    "execute_on_cpu_outputs",
    "run_on_cpu",
    "run_on_cpu_outputs",
    "validate_cpu_capability",
]
