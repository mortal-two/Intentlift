import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, weight_handle: T.handle, out_handle: T.handle, rstd_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 64), strides=(64, 1))
    weight = T.match_buffer(weight_handle, (64,), strides=(1,))
    out = T.match_buffer(out_handle, (M, 64), strides=(64, 1))
    rstd = T.match_buffer(rstd_handle, (M,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 64)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[bx, 0], weight[0], out[bx, 0])
        T.writes(rstd[bx])
        row = T.alloc_buffer((64,), scope="local.fragment")
        w = T.alloc_buffer((64,), scope="local.fragment")
        sq = T.alloc_buffer((64,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        out_row = T.alloc_buffer((64,), scope="local.fragment")
        T.copy(T.region(inp[bx, 0], 1, 1, 64), T.region(row[0], 2, 64))
        T.copy(T.region(weight[0], 1, 64), T.region(w[0], 2, 64))
        for r0 in range(64):
            sq[r0] = row[r0] * row[r0]
        T.reduce(sq[0:64], sm[0], "sum", 0, T.bool(True))
        mean: T.float32 = sm[0] * T.float32(0.015625)
        r0: T.float32 = T.rsqrt(mean + T.float32(1.0000000000000001e-05))
        rstd[bx] = r0
        for r1 in range(64):
            out_row[r1] = row[r1] * r0 * w[r1]
        T.copy(T.region(out_row[0], 1, 64), T.region(out[bx, 0], 2, 1, 64))
