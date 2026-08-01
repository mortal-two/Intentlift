import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 16), strides=(16, 1))
    out = T.match_buffer(out_handle, (M, 16), strides=(16, 1))
    # with T.block("root"):
    pid_m = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[pid_m, 0], out[pid_m, 0])
        T.writes()
        frag = T.alloc_buffer((16,), scope="local.fragment")
        T.copy(T.region(inp[pid_m, 0], 1, 1, 16), T.region(frag[0], 2, 16))
        T.copy(T.region(frag[0], 1, 16), T.region(out[pid_m, 0], 2, 1, 16))
