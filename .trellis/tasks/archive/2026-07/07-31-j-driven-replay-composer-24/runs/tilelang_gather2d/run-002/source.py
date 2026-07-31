import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, row_idx_handle: T.handle, col_idx_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 64), strides=(64, 1))
    row_idx = T.match_buffer(row_idx_handle, (256,), "int32", strides=(1,))
    col_idx = T.match_buffer(col_idx_handle, (256,), "int32", strides=(1,))
    out = T.match_buffer(out_handle, (256,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 256)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(row_idx[bx], col_idx[bx], inp[row_idx[bx], col_idx[bx]])
        T.writes(out[bx])
        r: T.int32 = row_idx[bx]
        c: T.int32 = col_idx[bx]
        out[bx] = inp[r, c]
