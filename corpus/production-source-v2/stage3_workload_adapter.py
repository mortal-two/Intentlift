"""Triton-shaped ABI adapters used only to construct Stage-3 workload values.

The measured external baseline is the paired TileLang Source in
``tilelang_stage3.py``.  These functions give the existing target-neutral
workload builder a compiler ABI with the same tensor order and C coordinate.
"""

import triton
import triton.language as tl


@triton.jit
def gather2d_workload_adapter(p0, p1, p2, p3, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < 256
    row = tl.load(p1 + offset, mask=active, other=0)
    column = tl.load(p2 + offset, mask=active, other=0)
    value = tl.load(p0 + row * 64 + column, mask=active, other=0.0)
    tl.store(p3 + offset, value, mask=active)


@triton.jit
def rowmask_where2d_workload_adapter(p0, p1, p2, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < 64
    value = tl.load(p0 + row * 64 + column, mask=active, other=0.0)
    mask = tl.load(p1 + row)
    tl.store(p2 + row * 64 + column, value * mask.to(tl.float32), mask=active)


__all__ = ["gather2d_workload_adapter", "rowmask_where2d_workload_adapter"]
