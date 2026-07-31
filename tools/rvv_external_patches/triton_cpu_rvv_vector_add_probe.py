#!/usr/bin/env python3
"""Compile and execute one unmodified Triton Source through Triton-CPU on RVV.

This is a portability probe, not an IntentIR benchmark or a performance number.
It intentionally uses ctypes buffers so success does not depend on torch.
"""

from __future__ import annotations

import ctypes
import json

import triton
import triton.language as tl
from triton.compiler import ASTSource


BLOCK = 64


@triton.jit
def vector_add(a, b, out, n: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    lhs = tl.load(a + offsets, mask=mask)
    rhs = tl.load(b + offsets, mask=mask)
    tl.store(out + offsets, lhs + rhs, mask=mask)


def main() -> int:
    count = BLOCK
    array_type = ctypes.c_float * count
    lhs = array_type(*(float(index) for index in range(count)))
    rhs = array_type(*(float(2 * index + 1) for index in range(count)))
    out = array_type()

    source = ASTSource(
        fn=vector_add,
        signature={"a": "*fp32", "b": "*fp32", "out": "*fp32", "n": "constexpr"},
        constexprs={"n": count},
    )
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cpu":
        raise RuntimeError(f"Triton portability probe selected {target.backend!r}, not CPU")
    compiled = triton.compile(src=source, target=target)
    compiled[(1, 1, 1)](
        ctypes.addressof(lhs),
        ctypes.addressof(rhs),
        ctypes.addressof(out),
    )

    maximum_error = max(
        abs(float(out[index]) - (float(lhs[index]) + float(rhs[index])))
        for index in range(count)
    )
    if maximum_error != 0.0:
        raise RuntimeError(f"Triton-CPU/RVV output mismatch: {maximum_error}")
    print(
        json.dumps(
            {
                "status": "passed",
                "target": {
                    "backend": target.backend,
                    "arch": target.arch,
                    "warp_size": target.warp_size,
                },
                "elements": count,
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
