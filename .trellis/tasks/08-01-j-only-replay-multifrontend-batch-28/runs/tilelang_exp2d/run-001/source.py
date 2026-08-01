import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 16), strides=(16, 1))
    out = T.match_buffer(out_handle, (M, 16), strides=(16, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 16)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[bx, 0:16])
        T.writes(out[bx, 0:16])
        for col in T.parallel(16):
            out[bx, col] = T.exp(inp[bx, col])
