import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    d0 = v0 < p2

    input_base = p0
    input_addresses = input_base + v0
    values = tl.load(input_addresses, mask=d0, other=0.0)

    output_base = p1
    output_addresses = output_base + v0
    tl.store(output_addresses, values, mask=d0)
