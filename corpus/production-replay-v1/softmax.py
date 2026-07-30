import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, c0: tl.constexpr):
    v5 = tl.program_id(0)
    v6 = tl.arange(0, c0)
    lane_mask = v6 < p4

    input_row_offset = v5 * p3
    input_base = p2 + input_row_offset
    input_ptrs = input_base + v6
    input_values = tl.load(input_ptrs, mask=lane_mask, other=-float("inf"))

    row_max = tl.max(input_values, axis=0)
    shifted = input_values - row_max
    numerators = tl.exp(shifted)
    denominator = tl.sum(numerators, axis=0)
    output_values = numerators / denominator

    output_row_offset = v5 * p1
    output_base = p0 + output_row_offset
    output_ptrs = output_base + v6
    tl.store(output_ptrs, output_values, mask=lane_mask)
