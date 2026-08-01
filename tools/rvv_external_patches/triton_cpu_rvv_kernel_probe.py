#!/usr/bin/env python3
"""Compile and execute accepted Triton Sources through Triton-CPU on RVV."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
from pathlib import Path

import triton
from triton.compiler import ASTSource


def _load(path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"intentir_{symbol}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Triton Source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def _block_size(size: int) -> int:
    return 1 << (size - 1).bit_length()


def _compiled(source: ASTSource):
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cpu" or target.arch != "riscv64":
        raise RuntimeError(f"expected Triton CPU/riscv64 target, got {target}")
    result = triton.compile(src=source, target=target)
    vector_instructions = sum(
        1
        for line in result.asm["asm"].splitlines()
        if line.strip() and line.strip().split()[0].startswith("v")
    )
    if vector_instructions == 0:
        raise RuntimeError("Triton-CPU produced no RVV vector instruction")
    return result, target, vector_instructions


def _emit(result, path: Path | None) -> None:
    if path is not None:
        path.write_bytes(result.asm["so"])


def _softmax(args) -> dict:
    symbol = "_softmax_single_block_forward_kernel"
    block = _block_size(args.columns)
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "Y_ptr": "*fp32",
            "Y_row_stride": "i64",
            "X_ptr": "*fp32",
            "X_row_stride": "i64",
            "n_cols": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": block},
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.rows * args.columns
        array = ctypes.c_float * count
        values = array(*(0.25 * math.sin(index * 0.01) for index in range(count)))
        output = array()
        compiled[(args.rows, 1, 1)](
            ctypes.addressof(output),
            args.columns,
            ctypes.addressof(values),
            args.columns,
            args.columns,
            block,
        )
        maximum = 0.0
        for row in range(args.rows):
            base = row * args.columns
            row_values = [float(values[base + col]) for col in range(args.columns)]
            peak = max(row_values)
            exponentials = [math.exp(value - peak) for value in row_values]
            denominator = sum(exponentials)
            for col, exponential in enumerate(exponentials):
                maximum = max(
                    maximum,
                    abs(float(output[base + col]) - exponential / denominator),
                )
        if maximum > 1.0e-5:
            raise RuntimeError(f"Triton-CPU/RVV Softmax mismatch: {maximum}")
    return {
        "kernel": "softmax",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.rows, args.columns],
        "dtype": "f32",
        "block_size": block,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _kldiv(args) -> dict:
    symbol = "_kldiv_kernel_forward"
    block = args.block_size
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "y_ptr": "*fp32",
            "y_stride": "i64",
            "gt_ptr": "*fp32",
            "gt_stride": "i64",
            "loss_ptr": "*fp32",
            "loss_stride": "i64",
            "n_cols": "i32",
            "eps": "fp32",
            "BLOCK_SIZE": "constexpr",
            "log_target": "constexpr",
            "reduction": "constexpr",
        },
        constexprs={
            "BLOCK_SIZE": block,
            "log_target": args.log_target,
            "reduction": 3,
        },
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.rows * args.columns
        array = ctypes.c_float * count
        losses = ctypes.c_float * args.rows
        predicted = array(*(-0.1 - 0.001 * (index % args.columns) for index in range(count)))
        target_values = array(*(0.1 + 0.001 * (index % 17) for index in range(count)))
        output = losses()
        compiled[(args.rows, 1, 1)](
            ctypes.addressof(predicted),
            args.columns,
            ctypes.addressof(target_values),
            args.columns,
            ctypes.addressof(output),
            1,
            args.columns,
            args.eps,
            block,
            args.log_target,
            3,
        )
        maximum = 0.0
        for row in range(args.rows):
            expected = 0.0
            for col in range(args.columns):
                index = row * args.columns + col
                truth = float(target_values[index])
                expected += (
                    math.exp(truth) * (truth - float(predicted[index]))
                    if args.log_target
                    else truth
                    * (math.log(max(truth, args.eps)) - float(predicted[index]))
                )
            maximum = max(maximum, abs(float(output[row]) - expected))
        if maximum > 1.0e-4:
            raise RuntimeError(f"Triton-CPU/RVV KLDiv mismatch: {maximum}")
    return {
        "kernel": "kl_div",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.rows, args.columns],
        "dtype": "f32",
        "block_size": block,
        "log_target": args.log_target,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _flash_preprocess(args) -> dict:
    symbol = "_bwd_preprocess_do_o_dot"
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "Out": "*fp16",
            "DO": "*fp16",
            "Delta": "*fp32",
            "stride_ob": "i64",
            "stride_oh": "i64",
            "stride_om": "i64",
            "stride_dob": "i64",
            "stride_doh": "i64",
            "stride_dom": "i64",
            "nheads": "i32",
            "seqlen_q": "i32",
            "seqlen_q_rounded": "i32",
            "headdim": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_HEADDIM": "constexpr",
        },
        constexprs={"BLOCK_M": args.block_m, "BLOCK_HEADDIM": args.block_d},
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.batch * args.heads * args.sequence * args.head_dim
        half_array = ctypes.c_uint16 * count
        out_values = [0.25 * math.sin(index * 0.01) for index in range(count)]
        do_values = [0.5 * math.cos(index * 0.02) for index in range(count)]
        import struct

        out = half_array(*(struct.unpack("<H", struct.pack("<e", value))[0] for value in out_values))
        dout = half_array(*(struct.unpack("<H", struct.pack("<e", value))[0] for value in do_values))
        delta_count = args.batch * args.heads * args.sequence_rounded
        delta_array = ctypes.c_float * delta_count
        delta = delta_array()
        stride_m = args.head_dim
        stride_h = args.sequence * stride_m
        stride_b = args.heads * stride_h
        grid_m = (args.sequence + args.block_m - 1) // args.block_m
        compiled[(grid_m, args.batch * args.heads, 1)](
            ctypes.addressof(out),
            ctypes.addressof(dout),
            ctypes.addressof(delta),
            stride_b,
            stride_h,
            stride_m,
            stride_b,
            stride_h,
            stride_m,
            args.heads,
            args.sequence,
            args.sequence_rounded,
            args.head_dim,
            args.block_m,
            args.block_d,
        )
        maximum = 0.0
        for batch in range(args.batch):
            for head in range(args.heads):
                for row in range(args.sequence):
                    expected = 0.0
                    base = ((batch * args.heads + head) * args.sequence + row) * args.head_dim
                    for col in range(args.head_dim):
                        lhs = struct.unpack("<e", struct.pack("<H", out[base + col]))[0]
                        rhs = struct.unpack("<e", struct.pack("<H", dout[base + col]))[0]
                        expected += float(lhs) * float(rhs)
                    actual = delta[(batch * args.heads + head) * args.sequence_rounded + row]
                    maximum = max(maximum, abs(float(actual) - expected))
        if maximum > 1.0e-4:
            raise RuntimeError(
                f"Triton-CPU/RVV Flash preprocess mismatch: {maximum}"
            )
    return {
        "kernel": "flash_bwd_preprocess",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.batch, args.heads, args.sequence, args.head_dim],
        "dtype": "f16->f32",
        "block_m": args.block_m,
        "block_d": args.block_d,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _geglu(args) -> dict:
    symbol = "_geglu_tanh_forward_kernel"
    block = _block_size(args.columns)
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "a": "*fp32",
            "b": "*fp32",
            "c": "*fp32",
            "stride": "i64",
            "n_cols": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"n_cols": args.columns, "BLOCK_SIZE": block},
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.rows * args.columns
        array = ctypes.c_float * count
        lhs = array(*(0.25 * math.sin(index * 0.01) for index in range(count)))
        rhs = array(*(0.5 + 0.25 * math.cos(index * 0.02) for index in range(count)))
        output = array()
        compiled[(args.rows, 1, 1)](
            ctypes.addressof(lhs),
            ctypes.addressof(rhs),
            ctypes.addressof(output),
            args.columns,
            args.columns,
            block,
        )
        maximum = 0.0
        coefficient = 0.7978845608028654
        for index in range(count):
            value = float(lhs[index])
            gate = 0.5 * value * (
                1.0
                + math.tanh(coefficient * (value + 0.044715 * value * value * value))
            )
            maximum = max(maximum, abs(float(output[index]) - gate * float(rhs[index])))
        if maximum > 1.0e-5:
            raise RuntimeError(f"Triton-CPU/RVV GeGLU mismatch: {maximum}")
    return {
        "kernel": "geglu",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.rows, args.columns],
        "dtype": "f32",
        "block_size": block,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _dyt(args) -> dict:
    symbol = "_dyt_fwd_kernel"
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "X": "*fp32",
            "Y": "*fp32",
            "Alpha": "*fp32",
            "Gamma": "*fp32",
            "Beta": "*fp32",
            "HAVE_BETA": "constexpr",
            "N": "constexpr",
            "BLOCK_N": "constexpr",
        },
        constexprs={"HAVE_BETA": True, "N": args.columns, "BLOCK_N": args.block_size},
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.rows * args.columns
        array = ctypes.c_float * count
        vector = ctypes.c_float * args.columns
        scalar = ctypes.c_float * 1
        values = array(*(0.25 * math.sin(index * 0.01) for index in range(count)))
        output = array()
        alpha = scalar(args.alpha)
        gamma = vector(*(0.5 + 0.25 * math.cos(index * 0.02) for index in range(args.columns)))
        beta = vector(*(0.1 * math.sin(index * 0.03) for index in range(args.columns)))
        grid_x = (args.columns + args.block_size - 1) // args.block_size
        compiled[(grid_x, args.rows, 1)](
            ctypes.addressof(values),
            ctypes.addressof(output),
            ctypes.addressof(alpha),
            ctypes.addressof(gamma),
            ctypes.addressof(beta),
            True,
            args.columns,
            args.block_size,
        )
        maximum = 0.0
        for index in range(count):
            col = index % args.columns
            expected = (
                math.tanh(args.alpha * float(values[index])) * float(gamma[col])
                + float(beta[col])
            )
            maximum = max(maximum, abs(float(output[index]) - expected))
        if maximum > 1.0e-5:
            raise RuntimeError(f"Triton-CPU/RVV DYT mismatch: {maximum}")
    return {
        "kernel": "dyt",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.rows, args.columns],
        "dtype": "f32",
        "block_size": args.block_size,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _rms_norm(args) -> dict:
    symbol = "_rms_norm_forward_kernel"
    block = _block_size(args.columns)
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "Y_ptr": "*fp32",
            "Y_row_stride": "i64",
            "X_ptr": "*fp32",
            "X_row_stride": "i64",
            "W_ptr": "*fp32",
            "W_row_stride": "i64",
            "RSTD_ptr": "*fp32",
            "RSTD_row_stride": "i64",
            "n_cols": "i32",
            "eps": "fp32",
            "offset": "fp32",
            "casting_mode": "constexpr",
            "elementwise_affine": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={
            "casting_mode": 0,
            "elementwise_affine": True,
            "BLOCK_SIZE": block,
        },
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.rows * args.columns
        array = ctypes.c_float * count
        vector = ctypes.c_float * args.columns
        stats = ctypes.c_float * args.rows
        values = array(*(0.25 * math.sin(index * 0.01) for index in range(count)))
        weight = vector(*(0.5 + 0.25 * math.cos(index * 0.02) for index in range(args.columns)))
        output = array()
        rstd = stats()
        compiled[(args.rows, 1, 1)](
            ctypes.addressof(output), args.columns,
            ctypes.addressof(values), args.columns,
            ctypes.addressof(weight), args.columns,
            ctypes.addressof(rstd), 1,
            args.columns, args.eps, 0.0, 0, True, block,
        )
        maximum = 0.0
        for row in range(args.rows):
            base = row * args.columns
            mean_square = sum(float(values[base + col]) ** 2 for col in range(args.columns)) / args.columns
            expected_rstd = 1.0 / math.sqrt(mean_square + args.eps)
            maximum = max(maximum, abs(float(rstd[row]) - expected_rstd))
            for col in range(args.columns):
                expected = float(values[base + col]) * expected_rstd * float(weight[col])
                maximum = max(maximum, abs(float(output[base + col]) - expected))
        if maximum > 1.0e-5:
            raise RuntimeError(f"Triton-CPU/RVV RMSNorm mismatch: {maximum}")
    return {
        "kernel": "liger_rms_norm",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.rows, args.columns],
        "dtype": "f32",
        "block_size": block,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _layer_norm(args) -> dict:
    symbol = "_layer_norm_forward_kernel"
    block = _block_size(args.columns)
    source = ASTSource(
        fn=_load(args.source, symbol),
        signature={
            "Y_ptr": "*fp32", "Y_row_stride": "i64",
            "X_ptr": "*fp32", "X_row_stride": "i64",
            "W_ptr": "*fp32", "W_row_stride": "i64",
            "B_ptr": "*fp32", "B_row_stride": "i64",
            "Mean_ptr": "*fp32", "Mean_row_stride": "i64",
            "RSTD_ptr": "*fp32", "RSTD_row_stride": "i64",
            "n_cols": "i32", "eps": "fp32", "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": block},
    )
    compiled, target, vectors = _compiled(source)
    _emit(compiled, args.emit_shared_object)
    if args.compile_only:
        maximum = None
    else:
        count = args.rows * args.columns
        array = ctypes.c_float * count
        vector = ctypes.c_float * args.columns
        stats = ctypes.c_float * args.rows
        values = array(*(0.25 * math.sin(index * 0.01) for index in range(count)))
        weight = vector(*(0.5 + 0.25 * math.cos(index * 0.02) for index in range(args.columns)))
        bias = vector(*(0.1 * math.sin(index * 0.03) for index in range(args.columns)))
        output, mean, rstd = array(), stats(), stats()
        compiled[(args.rows, 1, 1)](
            ctypes.addressof(output), args.columns,
            ctypes.addressof(values), args.columns,
            ctypes.addressof(weight), args.columns,
            ctypes.addressof(bias), args.columns,
            ctypes.addressof(mean), 1,
            ctypes.addressof(rstd), 1,
            args.columns, args.eps, block,
        )
        maximum = 0.0
        for row in range(args.rows):
            base = row * args.columns
            expected_mean = sum(float(values[base + col]) for col in range(args.columns)) / args.columns
            variance = sum((float(values[base + col]) - expected_mean) ** 2 for col in range(args.columns)) / args.columns
            expected_rstd = 1.0 / math.sqrt(variance + args.eps)
            maximum = max(maximum, abs(float(mean[row]) - expected_mean), abs(float(rstd[row]) - expected_rstd))
            for col in range(args.columns):
                expected = (float(values[base + col]) - expected_mean) * expected_rstd * float(weight[col]) + float(bias[col])
                maximum = max(maximum, abs(float(output[base + col]) - expected))
        if maximum > 1.0e-5:
            raise RuntimeError(f"Triton-CPU/RVV LayerNorm mismatch: {maximum}")
    return {
        "kernel": "layer_norm",
        "status": "compiled" if args.compile_only else "passed",
        "source": str(args.source),
        "symbol": compiled.name,
        "target": {"backend": target.backend, "arch": target.arch},
        "shape": [args.rows, args.columns],
        "dtype": "f32",
        "block_size": block,
        "maximum_absolute_error": maximum,
        "rvv_vector_instruction_count": vectors,
        "shared_object_bytes": len(compiled.asm["so"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kernel", required=True)
    for name in (
        "softmax", "kldiv", "flash-preprocess", "geglu", "dyt",
        "rms-norm", "layer-norm",
    ):
        child = subparsers.add_parser(name)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--emit-shared-object", type=Path)
        child.add_argument("--compile-only", action="store_true")
    softmax = subparsers.choices["softmax"]
    softmax.add_argument("--rows", type=int, default=2)
    softmax.add_argument("--columns", type=int, default=128)
    kldiv = subparsers.choices["kldiv"]
    kldiv.add_argument("--rows", type=int, default=2)
    kldiv.add_argument("--columns", type=int, default=127)
    kldiv.add_argument("--block-size", type=int, default=64)
    kldiv.add_argument("--eps", type=float, default=1.0e-8)
    kldiv.add_argument("--log-target", action="store_true")
    flash = subparsers.choices["flash-preprocess"]
    flash.add_argument("--batch", type=int, default=1)
    flash.add_argument("--heads", type=int, default=2)
    flash.add_argument("--sequence", type=int, default=33)
    flash.add_argument("--sequence-rounded", type=int, default=128)
    flash.add_argument("--head-dim", type=int, default=16)
    flash.add_argument("--block-m", type=int, default=128)
    flash.add_argument("--block-d", type=int, default=128)
    geglu = subparsers.choices["geglu"]
    geglu.add_argument("--rows", type=int, default=2)
    geglu.add_argument("--columns", type=int, default=127)
    dyt = subparsers.choices["dyt"]
    dyt.add_argument("--rows", type=int, default=2)
    dyt.add_argument("--columns", type=int, default=127)
    dyt.add_argument("--block-size", type=int, default=128)
    dyt.add_argument("--alpha", type=float, default=0.5)
    for name in ("rms-norm", "layer-norm"):
        norm = subparsers.choices[name]
        norm.add_argument("--rows", type=int, default=2)
        norm.add_argument("--columns", type=int, default=127)
        norm.add_argument("--eps", type=float, default=1.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compile_only and args.emit_shared_object is None:
        raise ValueError("--compile-only requires --emit-shared-object")
    handlers = {
        "softmax": _softmax,
        "kldiv": _kldiv,
        "flash-preprocess": _flash_preprocess,
        "geglu": _geglu,
        "dyt": _dyt,
        "rms-norm": _rms_norm,
        "layer-norm": _layer_norm,
    }
    result = handlers[args.kernel](args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
