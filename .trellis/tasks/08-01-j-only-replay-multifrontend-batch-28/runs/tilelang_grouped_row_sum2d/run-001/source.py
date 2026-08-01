import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 64), strides=(64, 1))
    out = T.match_buffer(out_handle, (M, 16), strides=(16, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 16)
    by = T.launch_thread("blockIdx.y", M)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[by, bx * 4])
        T.writes(out[by, bx])
        vals = T.alloc_buffer((4,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        T.copy(T.region(inp[by, bx * 4], 1, 1, 4), T.region(vals[0], 2, 4))
        T.reduce(vals[0:4], sm[0], "sum", 0, T.bool(True))
        out[by, bx] = sm[0]
