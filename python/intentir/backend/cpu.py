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


def _dense(arr: np.ndarray) -> str:
    """numpy array -> MLIR dense<...> attribute literal."""
    a = np.asarray(arr, dtype=np.float32)

    def scalar(value: float) -> str:
        text = f"{float(value):.8g}"
        if "e" in text or "E" in text:
            marker = "e" if "e" in text else "E"
            mantissa, exponent = text.split(marker, 1)
            if "." not in mantissa:
                mantissa += ".0"
            return f"{mantissa}e{exponent}"
        if "." not in text and text not in {"inf", "-inf", "nan"}:
            text += ".0"
        return text

    if a.ndim == 0:
        return scalar(float(a))

    def rec(x):
        if x.ndim == 1:
            return "[" + ", ".join(scalar(float(v)) for v in x) + "]"
        return "[" + ", ".join(rec(x[i]) for i in range(x.shape[0])) + "]"

    return rec(a)


def _tt(shape) -> str:
    return f"tensor<{'x'.join(str(s) for s in shape)}xf32>" if shape else "tensor<f32>"


def _build_main(fn: IntentFunction, kernel_mlir: str,
                inputs: list[np.ndarray], out_shape) -> str:
    """Wrap the emitted kernel with a @main that feeds constant inputs, calls it,
    and prints the (memref) result for read-back."""
    body = kernel_mlir[kernel_mlir.index("{") + 1 : kernel_mlir.rindex("}")]
    lines = ["module {", body.rstrip()]
    lines.append("  func.func @main() {")
    call_args, call_tys = [], []
    for i, (t, arr) in enumerate(zip(fn.inputs, inputs)):
        ty = _tt(np.shape(arr))
        lines.append(f"    %in{i} = arith.constant dense<{_dense(arr)}> : {ty}")
        call_args.append(f"%in{i}")
        call_tys.append(ty)
    oty = _tt(out_shape)
    lines.append(
        f"    %y = call @{fn.name}({', '.join(call_args)}) : "
        f"({', '.join(call_tys)}) -> {oty}")
    mref = f"memref<{'x'.join(str(s) for s in out_shape)}xf32>" if out_shape else "memref<f32>"
    lines.append(f"    %m = bufferization.to_memref %y : {oty} to {mref}")
    lines.append(f"    %u = memref.cast %m : {mref} to memref<*xf32>")
    lines.append("    call @printMemrefF32(%u) : (memref<*xf32>) -> ()")
    lines.append("    return")
    lines.append("  }")
    lines.append("  func.func private @printMemrefF32(%a: memref<*xf32>)")
    lines.append("}")
    return "\n".join(lines)


def _parse_memref(stdout: str, shape) -> np.ndarray:
    idx = stdout.index("data =")
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", stdout[idx + 6:])
    vals = np.array([float(x) for x in nums], dtype=np.float32)
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
    _infer_body(fn.body, env, 0)
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
