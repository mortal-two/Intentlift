import tilelang.language as T

@T.prim_func
def main(inp_handle: T.handle, residual_handle: T.handle, weight_handle: T.handle, bias_handle: T.handle, out_handle: T.handle, mean_handle: T.handle, rstd_handle: T.handle):
    M = T.int32()
    inp = T.match_buffer(inp_handle, (M, 64), strides=(64, 1))
    residual = T.match_buffer(residual_handle, (M, 64), strides=(64, 1))
    weight = T.match_buffer(weight_handle, (64,), strides=(1,))
    bias = T.match_buffer(bias_handle, (64,), strides=(1,))
    out = T.match_buffer(out_handle, (M, 64), strides=(64, 1))
    mean = T.match_buffer(mean_handle, (M,), strides=(1,))
    rstd = T.match_buffer(rstd_handle, (M,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 64)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(inp[bx, 0], residual[bx, 0], weight[0], bias[0])
        T.writes(mean[bx], rstd[bx], out[bx, 0:64])
        x = T.alloc_buffer((64,), scope="local.fragment")
        r = T.alloc_buffer((64,), scope="local.fragment")
        w = T.alloc_buffer((64,), scope="local.fragment")
        b = T.alloc_buffer((64,), scope="local.fragment")
        z = T.alloc_buffer((64,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        sq = T.alloc_buffer((64,), scope="local.fragment")
        sm2 = T.alloc_buffer((1,), scope="local.fragment")
        T.copy(T.region(inp[bx, 0], 1, 1, 64), T.region(x[0], 2, 64))
        T.copy(T.region(residual[bx, 0], 1, 1, 64), T.region(r[0], 2, 64))
        T.copy(T.region(weight[0], 1, 64), T.region(w[0], 2, 64))
        T.copy(T.region(bias[0], 1, 64), T.region(b[0], 2, 64))
        for i0 in range(64):
            z[i0] = x[i0] + r[i0]
        T.reduce(z[0:64], sm[0], "sum", 0, T.bool(True))
        m0: T.float32 = sm[0] * T.float32(0.015625)
        mean[bx] = m0
        for i1 in range(64):
            d: T.float32 = z[i1] - m0
            sq[i1] = d * d
        T.reduce(sq[0:64], sm2[0], "sum", 0, T.bool(True))
        var: T.float32 = sm2[0] * T.float32(0.015625)
        r0: T.float32 = T.rsqrt(var + T.float32(1.0000000000000001e-05))
        rstd[bx] = r0
        for i2 in range(64):
            out[bx, i2] = (z[i2] - m0) * r0 * w[i2] + b[i2]
