import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, lo_handle: T.handle, hi_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 16), strides=(16, 1))
    lo = T.match_buffer(lo_handle, (1,), strides=(1,))
    hi = T.match_buffer(hi_handle, (1,), strides=(1,))
    out = T.match_buffer(out_handle, (M, 16), strides=(16, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 16)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(lo[0], hi[0], inp[bx, 0:16])
        T.writes(out[bx, 0:16])
        lo_s: T.float32 = lo[0]
        hi_s: T.float32 = hi[0]
        for col in T.parallel(16):
            y0: T.float32 = T.max(inp[bx, col], lo_s)
            y1: T.float32 = T.min(y0, hi_s)
            out[bx, col] = y1
