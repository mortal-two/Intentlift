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


def _build_main(fn: IntentFunction, kernel_mlir: str,
                inputs: list[np.ndarray], out_shape) -> str:
    """Wrap the emitted kernel with a @main that feeds constant inputs, calls it,
    and prints the (memref) result for read-back."""
    body = kernel_mlir[kernel_mlir.index("{") + 1 : kernel_mlir.rindex("}")]
    lines = ["module {", body.rstrip()]
    lines.append("  func.func @main() {")
    call_args, call_tys = [], []
    for i, (t, arr) in enumerate(zip(fn.inputs, inputs)):
        element_type = (
            t.dtype
            if t.dtype in {"i1", "i8", "i16", "i32", "i64"}
            else "f32"
        )
        ty = _tt(np.shape(arr), element_type)
        lines.append(
            f"    %in{i} = arith.constant dense<{_dense(arr, element_type)}> : {ty}"
        )
        call_args.append(f"%in{i}")
        call_tys.append(ty)
    oty = _tt(out_shape)
    lines.append(
        f"    %y = call @{fn.name}({', '.join(call_args)}) : "
        f"({', '.join(call_tys)}) -> {oty}")
    # RunnerUtils prints f32 values with limited decimal precision.  Bitcast
    # the result to i32 before printing so Gate2 reads back the exact produced
    # IEEE-754 bits instead of comparing against a rounded text rendering.
    ity = _tt(out_shape, "i32")
    rank = len(out_shape)
    dims = ", ".join(f"d{i}" for i in range(rank))
    identity_map = f"affine_map<({dims}) -> ({dims})>"
    iterators = ", ".join('"parallel"' for _ in range(rank))
    lines.append(f"    %bits_init = tensor.empty() : {ity}")
    lines.append("    %bits = linalg.generic {")
    lines.append(
        f"      indexing_maps = [{identity_map}, {identity_map}],"
    )
    lines.append(f"      iterator_types = [{iterators}]")
    lines.append(
        f"    }} ins(%y : {oty}) outs(%bits_init : {ity}) {{"
    )
    lines.append("    ^bb0(%value: f32, %unused: i32):")
    lines.append("      %bit = arith.bitcast %value : f32 to i32")
    lines.append("      linalg.yield %bit : i32")
    lines.append(f"    }} -> {ity}")
    mref = (
        f"memref<{'x'.join(str(s) for s in out_shape)}xi32>"
        if out_shape
        else "memref<i32>"
    )
    lines.append(
        f"    %m = bufferization.to_memref %bits : {ity} to {mref}"
    )
    lines.append(f"    %u = memref.cast %m : {mref} to memref<*xi32>")
    lines.append("    call @printMemrefI32(%u) : (memref<*xi32>) -> ()")
    lines.append("    return")
    lines.append("  }")
    lines.append("  func.func private @printMemrefI32(%a: memref<*xi32>)")
    lines.append("}")
    return "\n".join(lines)


def _parse_memref(stdout: str, shape) -> np.ndarray:
    idx = stdout.index("data =")
    nums = re.findall(r"-?\d+", stdout[idx + 6:])
    vals = np.array([int(x) for x in nums], dtype=np.int32).view(np.float32)
    return vals.reshape(shape) if shape else vals.reshape(())


def run_on_cpu(fn: IntentFunction, dims: dict[str, int],
               inputs: list[np.ndarray]) -> np.ndarray:
    """Lower a flat IntentFunction to CPU LLVM and JIT-execute it; return the
    output array. Inputs are positional, matching fn.inputs."""
    kernel = emit_module(fn, dims)
    # infer output shape from the emitter's own shape inference via a dims-resolve
    from ..frontend.emit_mlir import _resolve, _infer_body
    env = {t.name: _resolve(t.shape, dims) for t in fn.inputs}
    env.update({t.name: _resolve(t.shape, dims) for t in fn.outputs})
    _infer_body(fn.body, env, 0, dict(dims))
    for op in fn.body:
        if op.out is not None:
            env[op.out] = env.get(op.result, env.get(op.out))
    out_shape = tuple(env[fn.outputs[0].name])

    src = _build_main(fn, kernel, inputs, out_shape)
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
    return _parse_memref(p3.stdout, out_shape)
