import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, p5: tl.constexpr, c0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = v0.to(tl.int64)

    v3 = v1 * p3
    v5 = p0 + v3

    v6 = v1 * p3
    v8 = p1 + v6

    v9 = v1 * p3
    v11 = p2 + v9

    v13 = tl.arange(0, c0)
    v15 = v13 < p5

    v16 = v5 + v13
    v17 = tl.load(v16, mask=v15, other=0.0)

    v18 = v8 + v13
    v19 = tl.load(v18, mask=v15, other=0.0)

    v20 = v17.to(tl.float32)
    v21 = v20 * p4
    v22 = tl.sigmoid(v21)
    v23 = v21 * v22
    v24 = v23.to(tl.float32)
    v25 = v24 * v19

    v26 = v11 + v13
    tl.store(v26, v25, mask=v15)
