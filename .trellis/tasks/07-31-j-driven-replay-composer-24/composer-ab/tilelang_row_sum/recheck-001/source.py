import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 16), strides=(16, 1))
    out = T.match_buffer(out_handle, (M,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[bx, 0])
        T.writes(out[bx])
        row = T.alloc_buffer((16,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        T.copy(T.region(inp[bx, 0], 1, 1, 16), T.region(row[0], 2, 16))
        T.reduce(row[0:16], sm[0], "sum", 0, T.bool(True))
        out[bx] = sm[0]
