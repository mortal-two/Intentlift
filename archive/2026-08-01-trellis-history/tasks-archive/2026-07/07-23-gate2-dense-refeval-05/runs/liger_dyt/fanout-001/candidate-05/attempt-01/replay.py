import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def scrc_k0(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5: tl.constexpr,
    p6: tl.constexpr,
    c0: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_n_i64 = pid_n.to(tl.int64)
    n_base = pid_n_i64 * c0
    n_offsets = n_base + tl.arange(0, c0)

    pid_m = tl.program_id(1)
    pid_m_i64 = pid_m.to(tl.int64)
    m_base = pid_m_i64 * p6

    mask = n_offsets < p6

    scale = tl.load(p2 + 0)
    multiplier = tl.load(p3 + n_offsets, mask=mask, other=0.0)
    source = tl.load(
        p0 + (m_base + n_offsets),
        mask=mask,
        other=0.0,
    )

    product = source * scale
    activated = libdevice.tanh(product)
    scaled = activated * multiplier

    addend = tl.load(
        p4 + n_offsets,
        mask=p5 & mask,
        other=0.0,
    )
    added = scaled + addend
    result = tl.where(p5, added, scaled)

    tl.store(
        p1 + (m_base + n_offsets),
        result,
        mask=mask,
    )
