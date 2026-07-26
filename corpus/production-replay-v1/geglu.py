import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4: tl.constexpr, c0: tl.constexpr):
    v14 = tl.program_id(0).to(tl.int64)

    v14_p0 = v14 * p3
    p0_base = p0 + v14_p0

    v14_p1 = v14 * p3
    p1_base = p1 + v14_p1

    v14_p2 = v14 * p3
    p2_base = p2 + v14_p2

    v15 = tl.arange(0, c0)

    p0_lanes = p0_base + v15
    t0_loaded = tl.load(p0_lanes, mask=v15 < p4, other=0.0)
    t0 = t0_loaded

    p1_lanes = p1_base + v15
    t1 = tl.load(p1_lanes, mask=v15 < p4, other=0.0)

    p2_lanes = p2_base + v15

    v0 = t0 * t0
    v1 = v0 * t0
    v2 = 0.044715
    v3 = v2 * v1
    v4 = t0 + v3
    v5 = 0.7978845608028654
    v6 = v5 * v4
    v7 = libdevice.tanh(v6)
    v8 = 1.0
    v9 = v8 + v7
    v10 = 0.5
    v11 = v10 * t0
    v12 = v11 * v9
    v12_materialized = v12
    t2 = v12_materialized * t1

    tl.store(p2_lanes, t2, mask=v15 < p4)
