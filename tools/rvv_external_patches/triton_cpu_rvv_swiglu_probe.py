#!/usr/bin/env python3
"""Execute the accepted SwiGLU Triton Source through Triton-CPU on RVV.

The kernel is imported from the original frontend source file instead of being
rewritten here.  ctypes supplies a torch-independent boundary for the probe.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
from pathlib import Path

import triton
from triton.compiler import ASTSource


def _load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("intentir_original_swiglu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Triton Source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._swiglu_forward_kernel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--columns", type=int, default=128)
    parser.add_argument("--gate-multiplier", type=float, default=1.0)
    parser.add_argument("--emit-shared-object", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rows <= 0 or args.columns <= 0:
        raise ValueError("SwiGLU dimensions must be positive")
    block_size = 1 << (args.columns - 1).bit_length()
    kernel = _load_kernel(args.source)
    source = ASTSource(
        fn=kernel,
        signature={
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "c_ptr": "*fp32",
            "stride": "i64",
            "gate_multiplier": "fp32",
            "n_cols": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"n_cols": args.columns, "BLOCK_SIZE": block_size},
    )
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cpu":
        raise RuntimeError(f"Triton portability probe selected {target.backend!r}, not CPU")
    compiled = triton.compile(src=source, target=target)
    assembly = compiled.asm["asm"]
    vector_instruction_count = sum(
        1
        for line in assembly.splitlines()
        if line.strip() and line.strip().split()[0].startswith("v")
    )
    if vector_instruction_count == 0:
        raise RuntimeError("Triton-CPU produced no RVV vector instruction")
    if args.emit_shared_object is not None:
        args.emit_shared_object.write_bytes(compiled.asm["so"])
    if args.compile_only:
        if args.emit_shared_object is None:
            raise ValueError("--compile-only requires --emit-shared-object")
        print(
            json.dumps(
                {
                    "status": "compiled",
                    "same_source": str(args.source),
                    "target": {"backend": target.backend, "arch": target.arch},
                    "columns": args.columns,
                    "dtype": "f32",
                    "block_size": block_size,
                    "kernel_symbol": compiled.name,
                    "rvv_vector_instruction_count": vector_instruction_count,
                    "shared_object_bytes": len(compiled.asm["so"]),
                    "emitted_shared_object": str(args.emit_shared_object),
                },
                sort_keys=True,
            )
        )
        return 0

    element_count = args.rows * args.columns
    array_type = ctypes.c_float * element_count
    lhs = array_type(
        *(0.25 * math.sin(float(index)) for index in range(element_count))
    )
    rhs = array_type(
        *(0.5 + 0.25 * math.cos(float(index)) for index in range(element_count))
    )
    out = array_type()
    compiled[(args.rows, 1, 1)](
        ctypes.addressof(lhs),
        ctypes.addressof(rhs),
        ctypes.addressof(out),
        args.columns,
        args.gate_multiplier,
        args.columns,
        block_size,
    )

    maximum_error = 0.0
    for index in range(element_count):
        scaled = float(lhs[index]) * args.gate_multiplier
        expected = (scaled / (1.0 + math.exp(-scaled))) * float(rhs[index])
        maximum_error = max(maximum_error, abs(float(out[index]) - expected))
    if maximum_error > 1e-6:
        raise RuntimeError(f"Triton-CPU/RVV SwiGLU mismatch: {maximum_error}")
    print(
        json.dumps(
            {
                "status": "passed",
                "same_source": str(args.source),
                "target": {"backend": target.backend, "arch": target.arch},
                "shape": [args.rows, args.columns],
                "dtype": "f32",
                "block_size": block_size,
                "maximum_absolute_error": maximum_error,
                "kernel_symbol": compiled.name,
                "rvv_vector_instruction_count": vector_instruction_count,
                "assembly_products": sorted(compiled.asm),
                "shared_object_bytes": len(compiled.asm["so"]),
                "emitted_shared_object": (
                    str(args.emit_shared_object)
                    if args.emit_shared_object is not None
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
