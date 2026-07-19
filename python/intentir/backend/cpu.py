"""IntentIR -> linalg -> LLVM -> CPU execution via mlir-runner.

The point is architectural, not performance: the SAME intent dialect that the GPU
path lowers reaches a completely different backend through standard MLIR, with no
GPU-specific machinery. intent->linalg is backend-neutral; only the device tail
(gpu/nvvm here, an RVV/vector tail later) is specialized. This module proves that
neutrality by actually executing a lifted kernel on the CPU and reading the result
back.  SCRC Gate2 uses this as its production-lowering leg; it is not a JSON
interpreter. See spec/lowering/index.md ("多后端余量").
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..frontend.emit_mlir import emit_module
from ..ir.model import IntentFunction
from .compile import MLIR_OPT  # reuse the resolved llvm-20 mlir-opt path

LLVM_BIN = Path(MLIR_OPT).parent
LLVM_LIB = Path("/usr/lib/llvm-20/lib")
INTENT_OPT = Path(__file__).resolve().parents[3] / "build" / "tools" / "intent-opt" / "intent-opt"

_CPU_PIPELINE = [
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
    "-convert-linalg-to-loops",
    "-convert-math-to-libm",
    "-convert-scf-to-cf",
    "-convert-cf-to-llvm",
    "-expand-strided-metadata",  # lower memref.expand_shape (reduce keepdims)
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm",
    "-convert-arith-to-llvm",
    "-reconcile-unrealized-casts",
]


def _dense(arr: np.ndarray, element_type: str = "f32") -> str:
    """numpy array -> MLIR dense<...> attribute literal."""
    if element_type == "i1":
        a = np.asarray(arr, dtype=np.bool_)

        def scalar(value) -> str:
            return "true" if bool(value) else "false"

    elif element_type.startswith("i"):
        a = np.asarray(arr, dtype=np.int64)

        def scalar(value) -> str:
            return str(int(value))

    elif element_type == "f16":
        a = np.asarray(arr, dtype=np.float16)

        def scalar(value: float) -> str:
            # MLIR 20's dense-elements parser does not accept the `0xHhhhh`
            # FloatAttr spelling inside a nested list. Decimal conversion from
            # an already-quantized f16 value round-trips exactly to f16.
            text = repr(float(np.asarray(value, dtype=np.float16)))
            return text if any(c in text for c in ".eE") else text + ".0"

    elif element_type == "bf16":
        # NumPy has no portable bfloat16 dtype. Round an f32 value to the
        # nearest-even BF16 payload and use MLIR's `0xRhhhh` spelling.
        a = np.asarray(arr, dtype=np.float32)

        def scalar(value: float) -> str:
            bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
            rounded = bits + 0x7FFF + ((bits >> 16) & 1)
            bf16_value = np.asarray(
                np.uint32(((rounded >> 16) & 0xFFFF) << 16), dtype=np.uint32
            ).view(np.float32).item()
            text = repr(float(bf16_value))
            return text if any(c in text for c in ".eE") else text + ".0"

    elif element_type == "f64":
        a = np.asarray(arr, dtype=np.float64)

        def scalar(value: float) -> str:
            text = repr(float(np.asarray(value, dtype=np.float64)))
            return text if any(c in text for c in ".eE") else text + ".0"

    else:
        a = np.asarray(arr, dtype=np.float32)

        def scalar(value: float) -> str:
            # Hex float attributes preserve every f32 input bit exactly and
            # are also the MLIR spelling accepted for +/-inf, NaN and -0.0.
            bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
            return f"0x{bits:08X}"

    if a.ndim == 0:
        return scalar(a.item())

    def rec(x):
        if x.ndim == 1:
            return "[" + ", ".join(scalar(v) for v in x) + "]"
        return "[" + ", ".join(rec(x[i]) for i in range(x.shape[0])) + "]"

    return rec(a)


def _tt(shape, element_type: str = "f32") -> str:
    return (
        f"tensor<{'x'.join(str(s) for s in shape)}x{element_type}>"
        if shape
        else f"tensor<{element_type}>"
    )


_INTEGER_OUTPUT_DTYPES = {
    "i1": np.bool_,
    "bool": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
}


def _execution_element_type(dtype: str) -> str:
    """Return the element type emitted by the CPU execution frontend."""

    spelling = dtype.casefold().removeprefix("torch.")
    aliases = {
        "half": "f16",
        "float16": "f16",
        "bfloat16": "bf16",
        "float": "f32",
        "float32": "f32",
        "double": "f64",
        "float64": "f64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
    }
    spelling = aliases.get(spelling, spelling)
    if spelling == "bool":
        return "i1"
    if spelling in _INTEGER_OUTPUT_DTYPES or spelling in {
        "f16", "bf16", "f32", "f64",
    }:
        return spelling
    raise ValueError(f"unsupported CPU execution dtype {dtype!r}")


def _build_main(fn: IntentFunction, kernel_mlir: str,
                inputs: list[np.ndarray], out_shape,
                out_element_type: str) -> str:
    """Wrap the emitted kernel with a @main that feeds constant inputs, calls it,
    and prints the (memref) result for read-back."""
    body = kernel_mlir[kernel_mlir.index("{") + 1 : kernel_mlir.rindex("}")]
    lines = ["module {", body.rstrip()]
    lines.append("  func.func @main() {")
    call_args, call_tys = [], []
    for i, (t, arr) in enumerate(zip(fn.inputs, inputs)):
        element_type = _execution_element_type(t.dtype)
        ty = _tt(np.shape(arr), element_type)
        lines.append(
            f"    %in{i} = arith.constant dense<{_dense(arr, element_type)}> : {ty}"
        )
        call_args.append(f"%in{i}")
        call_tys.append(ty)
    oty = _tt(out_shape, out_element_type)
    from ..frontend.emit_mlir import _sanitize_mlir_symbol

    lines.append(
        f"    %y = call @{_sanitize_mlir_symbol(fn.name)}({', '.join(call_args)}) : "
        f"({', '.join(call_tys)}) -> {oty}")
    rank = len(out_shape)
    dims = ", ".join(f"d{i}" for i in range(rank))
    identity_map = f"affine_map<({dims}) -> ({dims})>"
    iterators = ", ".join('"parallel"' for _ in range(rank))
    if out_element_type in {"f16", "bf16", "f32"}:
        # RunnerUtils prints floats with limited decimal precision. Extend
        # half/BF16 values to f32, then bitcast to i32 so Gate2 reads exact
        # produced values without textual rounding.
        print_element_type = "i32"
        print_tensor_type = _tt(out_shape, print_element_type)
        lines.append(
            f"    %print_init = tensor.empty() : {print_tensor_type}"
        )
        lines.append("    %print_values = linalg.generic {")
        lines.append(
            f"      indexing_maps = [{identity_map}, {identity_map}],"
        )
        lines.append(f"      iterator_types = [{iterators}]")
        lines.append(
            f"    }} ins(%y : {oty}) outs(%print_init : {print_tensor_type}) {{"
        )
        lines.append(
            f"    ^bb0(%value: {out_element_type}, %unused: i32):"
        )
        print_source = "%value"
        if out_element_type != "f32":
            lines.append(
                f"      %print_f32 = arith.extf %value : "
                f"{out_element_type} to f32"
            )
            print_source = "%print_f32"
        lines.append(
            f"      %print_value = arith.bitcast {print_source} : f32 to i32"
        )
        lines.append("      linalg.yield %print_value : i32")
        lines.append(f"    }} -> {print_tensor_type}")
        print_value = "%print_values"
        printer = "printMemrefI32"
    elif out_element_type == "f64":
        print_element_type = "i64"
        print_tensor_type = _tt(out_shape, print_element_type)
        lines.append(f"    %print_init = tensor.empty() : {print_tensor_type}")
        lines.append("    %print_values = linalg.generic {")
        lines.append(f"      indexing_maps = [{identity_map}, {identity_map}],")
        lines.append(f"      iterator_types = [{iterators}]")
        lines.append(
            f"    }} ins(%y : {oty}) outs(%print_init : {print_tensor_type}) {{"
        )
        lines.append("    ^bb0(%value: f64, %unused: i64):")
        lines.append("      %print_value = arith.bitcast %value : f64 to i64")
        lines.append("      linalg.yield %print_value : i64")
        lines.append(f"    }} -> {print_tensor_type}")
        print_value = "%print_values"
        printer = "printMemrefI64"
    else:
        # Normalize every signed integer width to i64 for RunnerUtils printing,
        # then cast the parsed values back to the declared IntentIR dtype.
        print_element_type = "i64"
        print_tensor_type = _tt(out_shape, print_element_type)
        if out_element_type == "i64":
            print_value = "%y"
        else:
            lines.append(
                f"    %print_init = tensor.empty() : {print_tensor_type}"
            )
            lines.append("    %print_values = linalg.generic {")
            lines.append(
                f"      indexing_maps = [{identity_map}, {identity_map}],"
            )
            lines.append(f"      iterator_types = [{iterators}]")
            lines.append(
                f"    }} ins(%y : {oty}) outs(%print_init : {print_tensor_type}) {{"
            )
            lines.append(
                f"    ^bb0(%value: {out_element_type}, %unused: i64):"
            )
            extension = "extui" if out_element_type == "i1" else "extsi"
            lines.append(
                f"      %print_value = arith.{extension} %value : "
                f"{out_element_type} to i64"
            )
            lines.append("      linalg.yield %print_value : i64")
            lines.append(f"    }} -> {print_tensor_type}")
            print_value = "%print_values"
        printer = "printMemrefI64"

    mref = (
        f"memref<{'x'.join(str(s) for s in out_shape)}x{print_element_type}>"
        if out_shape
        else f"memref<{print_element_type}>"
    )
    lines.append(
        f"    %m = bufferization.to_memref {print_value} : "
        f"{print_tensor_type} to {mref}"
    )
    lines.append(
        f"    %u = memref.cast %m : {mref} to "
        f"memref<*x{print_element_type}>"
    )
    lines.append(
        f"    call @{printer}(%u) : "
        f"(memref<*x{print_element_type}>) -> ()"
    )
    lines.append("    return")
    lines.append("  }")
    lines.append(
        f"  func.func private @{printer}(%a: "
        f"memref<*x{print_element_type}>)"
    )
    lines.append("}")
    return "\n".join(lines)


def _parse_memref(stdout: str, shape, element_type: str) -> np.ndarray:
    idx = stdout.index("data =")
    nums = re.findall(r"-?\d+", stdout[idx + 6:])
    raw = [int(x) for x in nums]
    if element_type in {"f16", "bf16", "f32"}:
        vals = np.asarray(raw, dtype=np.int32).view(np.float32)
        if element_type == "f16":
            vals = vals.astype(np.float16)
    elif element_type == "f64":
        vals = np.asarray(raw, dtype=np.int64).view(np.float64)
    else:
        dtype = _INTEGER_OUTPUT_DTYPES[element_type]
        vals = np.asarray(raw, dtype=np.int64).astype(dtype, copy=False)
    return vals.reshape(shape) if shape else vals.reshape(())


def run_on_cpu(
    fn: IntentFunction,
    dims: dict[str, int],
    inputs: list[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> np.ndarray:
    """Lower a flat IntentFunction to CPU LLVM and JIT-execute it; return the
    output array. Inputs are positional, matching fn.inputs."""
    kernel = emit_module(fn, dims, broadcast_axes=broadcast_axes)
    # infer output shape from the emitter's own shape inference via a dims-resolve
    from ..frontend.emit_mlir import _resolve, _infer_body
    env = {t.name: _resolve(t.shape, dims) for t in fn.inputs}
    env.update({t.name: _resolve(t.shape, dims) for t in fn.outputs})
    _infer_body(fn.body, env, 0, dict(dims), broadcast_axes)
    for op in fn.body:
        if op.out is not None:
            env[op.out] = env.get(op.result, env.get(op.out))
    out_shape = tuple(env[fn.outputs[0].name])
    out_element_type = _execution_element_type(fn.outputs[0].dtype)

    src = _build_main(fn, kernel, inputs, out_shape, out_element_type)
    p1 = subprocess.run([str(INTENT_OPT), "-convert-intent-to-linalg"],
                        input=src, capture_output=True, text=True)
    if p1.returncode:
        raise RuntimeError(f"intent-opt failed: {p1.stderr[:300]}")
    p2 = subprocess.run([str(LLVM_BIN / "mlir-opt"), *_CPU_PIPELINE],
                        input=p1.stdout, capture_output=True, text=True)
    if p2.returncode:
        raise RuntimeError(f"mlir-opt failed: {p2.stderr[:300]}")
    p3 = subprocess.run(
        [str(LLVM_BIN / "mlir-runner"), "-e", "main",
         "-entry-point-result=void",
         f"--shared-libs={LLVM_LIB / 'libmlir_runner_utils.so'}",
         f"--shared-libs={LLVM_LIB / 'libmlir_c_runner_utils.so'}"],
        input=p2.stdout, capture_output=True, text=True)
    if p3.returncode or "data =" not in p3.stdout:
        raise RuntimeError(f"mlir-runner failed: {(p3.stderr or p3.stdout)[:300]}")
    return _parse_memref(p3.stdout, out_shape, out_element_type)
