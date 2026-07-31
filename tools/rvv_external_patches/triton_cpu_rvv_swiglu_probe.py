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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rows <= 0 or args.columns <= 0:
        raise ValueError("SwiGLU dimensions must be positive")
    block_size = 1 << (args.columns - 1).bit_length()
    element_count = args.rows * args.columns
    array_type = ctypes.c_float * element_count
    lhs = array_type(
        *(0.25 * math.sin(float(index)) for index in range(element_count))
    )
    rhs = array_type(
        *(0.5 + 0.25 * math.cos(float(index)) for index in range(element_count))
    )
    out = array_type()

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
    compiled[(args.rows, 1, 1)](
        ctypes.addressof(lhs),
        ctypes.addressof(rhs),
        ctypes.addressof(out),
        args.columns,
        args.gate_multiplier,
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
                "assembly_products": sorted(compiled.asm),
                "shared_object_bytes": len(compiled.asm["so"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
