"""Minimal real Triton source for an unordered duplicate overwrite contract.

Different program instances may target the same destination.  The logical
contract admits any active contender at that destination; this source exists
to pin the one-of-updates mechanism, not to prescribe an atomic realization.
"""

import triton
import triton.language as tl


@triton.jit
def racing_overwrite_kernel(base, indices, values, n: tl.constexpr):
    program = tl.program_id(0)
    active = program < n
    index = tl.load(indices + program, mask=active, other=0)
    value = tl.load(values + program, mask=active, other=0.0)
    tl.store(base + index, value, mask=active)
