import tilelang.language as T

@T.prim_func
def main(input_ptr_handle: T.handle, output_ptr_handle: T.handle, row_max_ptr_handle: T.handle, row_sum_ptr_handle: T.handle):
    M = T.int32()
    input_ptr = T.match_buffer(input_ptr_handle, (M, 16), strides=(16, 1))
    output_ptr = T.match_buffer(output_ptr_handle, (M, 16), strides=(16, 1))
    row_max_ptr = T.match_buffer(row_max_ptr_handle, (M,), strides=(1,))
    row_sum_ptr = T.match_buffer(row_sum_ptr_handle, (M,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(input_ptr[bx, 0:16])
        T.writes(row_max_ptr[bx], row_sum_ptr[bx], output_ptr[bx, 0:16])
        row = T.alloc_buffer((16,), scope="local.fragment")
        mx = T.alloc_buffer((1,), scope="local.fragment")
        exp_row = T.alloc_buffer((16,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        out_row = T.alloc_buffer((16,), scope="local.fragment")
        for r in range(16):
            row[r] = input_ptr[bx, r]
        T.reduce(row[0:16], mx[0], "max", 0, T.bool(True))
        row_max_ptr[bx] = mx[0]
        for r in range(16):
            exp_row[r] = T.exp(row[r] - mx[0])
        T.reduce(exp_row[0:16], sm[0], "sum", 0, T.bool(True))
        row_sum_ptr[bx] = sm[0]
        for r in range(16):
            out_row[r] = exp_row[r] / sm[0]
        for r in range(16):
            output_ptr[bx, r] = out_row[r]
