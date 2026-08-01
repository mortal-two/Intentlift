import tilelang.language as T

@T.prim_func
def main(in_ptr_handle: T.handle, out_ptr_handle: T.handle, weight_ptr_handle: T.handle, bias_ptr_handle: T.handle, out_mean_ptr_handle: T.handle, out_rstd_ptr_handle: T.handle):
    M = T.int32()
    in_ptr = T.match_buffer(in_ptr_handle, (M, 16), strides=(16, 1))
    out_ptr = T.match_buffer(out_ptr_handle, (M, 16), strides=(16, 1))
    weight_ptr = T.match_buffer(weight_ptr_handle, (16,), strides=(1,))
    bias_ptr = T.match_buffer(bias_ptr_handle, (16,), strides=(1,))
    out_mean_ptr = T.match_buffer(out_mean_ptr_handle, (M,), strides=(1,))
    out_rstd_ptr = T.match_buffer(out_rstd_ptr_handle, (M,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(in_ptr[bx, 0:16], weight_ptr[0:16], bias_ptr[0:16])
        T.writes(out_mean_ptr[bx], out_rstd_ptr[bx], out_ptr[bx, 0:16])
        row = T.alloc_buffer((16,), scope="local.fragment")
        w = T.alloc_buffer((16,), scope="local.fragment")
        b = T.alloc_buffer((16,), scope="local.fragment")
        s = T.alloc_buffer((1,), scope="local.fragment")
        diff = T.alloc_buffer((16,), scope="local.fragment")
        sq = T.alloc_buffer((16,), scope="local.fragment")
        ss = T.alloc_buffer((1,), scope="local.fragment")
        out_row = T.alloc_buffer((16,), scope="local.fragment")
        for r in range(16):
            row[r] = in_ptr[bx, r]
        for r in range(16):
            w[r] = weight_ptr[r]
        for r in range(16):
            b[r] = bias_ptr[r]
        T.reduce(row[0:16], s[0], "sum", 0, T.bool(True))
        mean: T.float32 = s[0] / T.float32(16.0)
        out_mean_ptr[bx] = mean
        for r in range(16):
            d: T.float32 = row[r] - mean
            diff[r] = d
            sq[r] = d * d
        T.reduce(sq[0:16], ss[0], "sum", 0, T.bool(True))
        var: T.float32 = ss[0] / T.float32(16.0)
        rstd: T.float32 = T.rsqrt(var + T.float32(1.0000000000000001e-05))
        out_rstd_ptr[bx] = rstd
        for r in range(16):
            out_row[r] = diff[r] * rstd * w[r] + b[r]
        for r in range(16):
            out_ptr[bx, r] = out_row[r]
