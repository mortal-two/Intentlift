import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    row_index = tl.program_id(0)
    column_count = p2
    column_offsets = tl.arange(0, c0)

    load_mask = column_offsets < column_count
    row_offset = row_index * p2

    input_base = p0 + row_offset
    materialized_input_base = input_base
    input_addresses = materialized_input_base + column_offsets

    fill_value = 0.0
    values = tl.load(input_addresses, mask=load_mask, other=fill_value)

    squared_values = values * values
    row_sum = tl.sum(squared_values, axis=0)

    output_address = p1 + row_index
    tl.store(output_address, row_sum)
