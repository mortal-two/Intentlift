import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, row_mask_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 64), strides=(64, 1))
    row_mask = T.match_buffer(row_mask_handle, (M,), "bool", strides=(1,))
    out = T.match_buffer(out_handle, (M, 64), strides=(64, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 64)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(row_mask[bx], inp[bx, 0:64])
        T.writes(out[bx, 0:64])
        m: T.bool = row_mask[bx]
        mf: T.float32 = T.Cast("float32", m)
        for col in T.parallel(64):
            out[bx, col] = inp[bx, col] * mf
