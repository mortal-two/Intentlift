import tilelang.language as T

@T.prim_func
def main(X_handle: T.handle, Y_handle: T.handle, W_handle: T.handle, B_handle: T.handle, Mean_handle: T.handle, Rstd_handle: T.handle):
    N = T.int32()
    X = T.match_buffer(X_handle, (N, 64, 16), strides=(1024, 16, 1))
    Y = T.match_buffer(Y_handle, (N, 64, 16), strides=(1024, 16, 1))
    W = T.match_buffer(W_handle, (64,), strides=(1,))
    B = T.match_buffer(B_handle, (64,), strides=(1,))
    Mean = T.match_buffer(Mean_handle, (N, 4), strides=(4, 1))
    Rstd = T.match_buffer(Rstd_handle, (N, 4), strides=(4, 1))
    # with T.block("root"):
    n = T.launch_thread("blockIdx.x", N)
    g = T.launch_thread("blockIdx.y", 4)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(X[n, g * 16, 0], W[g * 16], B[g * 16])
        T.writes(Mean[n, g], Rstd[n, g], Y[n, g * 16:g * 16 + 16, 0:16])
        x_tile = T.alloc_buffer((16, 16), scope="local.fragment")
        w_tile = T.alloc_buffer((16,), scope="local.fragment")
        b_tile = T.alloc_buffer((16,), scope="local.fragment")
        sum_hw = T.alloc_buffer((16,), scope="local.fragment")
        sum_all = T.alloc_buffer((1,), scope="local.fragment")
        x2_tile = T.alloc_buffer((16, 16), scope="local.fragment")
        sum2_hw = T.alloc_buffer((16,), scope="local.fragment")
        sum2_all = T.alloc_buffer((1,), scope="local.fragment")
        c0: T.int32 = g * 16
        T.copy(T.region(X[n, c0, 0], 1, 1, 16, 16), T.region(x_tile[0, 0], 2, 16, 16))
        T.copy(T.region(W[c0], 1, 16), T.region(w_tile[0], 2, 16))
        T.copy(T.region(B[c0], 1, 16), T.region(b_tile[0], 2, 16))
        T.reduce(x_tile[0:16, 0:16], sum_hw[0:16], "sum", 1, T.bool(True))
        T.reduce(sum_hw[0:16], sum_all[0], "sum", 0, T.bool(True))
        mean: T.float32 = sum_all[0] / T.float32(256.0)
        for ci0, hi0 in T.grid(16, 16):
            v: T.float32 = x_tile[ci0, hi0]
            x2_tile[ci0, hi0] = v * v
        T.reduce(x2_tile[0:16, 0:16], sum2_hw[0:16], "sum", 1, T.bool(True))
        T.reduce(sum2_hw[0:16], sum2_all[0], "sum", 0, T.bool(True))
        mean2: T.float32 = sum2_all[0] / T.float32(256.0)
        var: T.float32 = mean2 - mean * mean
        rstd: T.float32 = T.rsqrt(var + T.float32(1.0000000000000001e-05))
        Mean[n, g] = mean
        Rstd[n, g] = rstd
        for ci1, hi1 in T.grid(16, 16):
            Y[n, c0 + ci1, hi1] = (x_tile[ci1, hi1] - mean) * rstd * w_tile[ci1] + b_tile[ci1]
