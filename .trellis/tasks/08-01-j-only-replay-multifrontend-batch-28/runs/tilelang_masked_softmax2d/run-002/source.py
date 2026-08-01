import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, mask_handle: T.handle, out_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 64), strides=(64, 1))
    mask = T.match_buffer(mask_handle, (64,), "bool", strides=(1,))
    out = T.match_buffer(out_handle, (M, 64), strides=(64, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 64)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[bx, 0], mask[0:64], out[bx, 0])
        T.writes()
        row = T.alloc_buffer((64,), scope="local.fragment")
        masked_row = T.alloc_buffer((64,), scope="local.fragment")
        mx = T.alloc_buffer((1,), scope="local.fragment")
        exp_row = T.alloc_buffer((64,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        out_row = T.alloc_buffer((64,), scope="local.fragment")
        T.copy(T.region(inp[bx, 0], 1, 1, 64), T.region(row[0], 2, 64))
        neg: T.float32 = T.float32(-1000000000.0)
        one: T.float32 = T.float32(1.0)
        for r0 in range(64):
            mf: T.float32 = T.Cast("float32", mask[r0])
            masked_row[r0] = row[r0] * mf + neg * (one - mf)
        T.reduce(masked_row[0:64], mx[0], "max", 0, T.bool(True))
        for r1 in range(64):
            exp_row[r1] = T.exp(masked_row[r1] - mx[0])
        T.reduce(exp_row[0:64], sm[0], "sum", 0, T.bool(True))
        for r2 in range(64):
            out_row[r2] = exp_row[r2] / sm[0]
        T.copy(T.region(out_row[0], 1, 64), T.region(out[bx, 0], 2, 1, 64))
