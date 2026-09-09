import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, c0: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, c0)
    mask = lane < p4

    input_offset = row * p3 + lane
    values = tl.load(p2 + input_offset, mask=mask, other=-float("inf"))

    row_max = tl.max(values)
    exponentials = tl.exp(values - row_max)
    normalization = tl.sum(exponentials)
    result = exponentials / normalization

    output_offset = row * p1 + lane
    tl.store(p0 + output_offset, result, mask=mask)
