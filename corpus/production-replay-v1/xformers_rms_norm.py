import triton
import triton.language as tl


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
    v8_program = tl.program_id(0).to(tl.int64)
    v8_base = v8_program * p4

    input_base = p0 + v8_base
    output_base = p1 + v8_base
    v9 = tl.arange(0, c0)
    active = v9 < p5

    input_address0 = input_base + v9
    input_value0 = tl.load(
        input_address0,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    )
    squared = input_value0 * input_value0
    square_sum = tl.sum(squared, axis=0)
    mean_square = square_sum / p5
    shifted = mean_square + p3
    reciprocal_root = tl.rsqrt(shifted)

    input_address1 = input_base + v9
    input_value1 = tl.load(
        input_address1,
        mask=active,
        other=0.0,
        eviction_policy="evict_last",
    )
    normalized = input_value1 * reciprocal_root

    if p6:
        scale_address = p2 + v9
        scale = tl.load(scale_address, mask=active)
        selected = normalized * scale
    else:
        selected = normalized

    output_address = output_base + v9
    tl.store(output_address, selected, mask=active)
