import tilelang.language as T

@T.prim_func
def main(A_handle: T.handle, W1_handle: T.handle, b1_handle: T.handle, W2_handle: T.handle, b2_handle: T.handle, C_handle: T.handle):
    M, K = T.int32(), T.int32()
    A = T.match_buffer(A_handle, (M, K), strides=(K, 1))
    H = T.int32()
    W1 = T.match_buffer(W1_handle, (K, H), strides=(H, 1))
    b1 = T.match_buffer(b1_handle, (H,), strides=(1,))
    N = T.int32()
    W2 = T.match_buffer(W2_handle, (H, N), strides=(N, 1))
    b2 = T.match_buffer(b2_handle, (N,), strides=(1,))
    C = T.match_buffer(C_handle, (M, N), strides=(N, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", (N + 64 - 1) // 64)
    by = T.launch_thread("blockIdx.y", (M + 64 - 1) // 64)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 64, 0:(K + 31) // 32 * 32 - 31], W1[0:(K + 31) // 32 * 32 - 31, 0:(H + 63) // 64 * 64 - 63], b1[0:(H + 63) // 64 * 64 - 63], W2[0:(H + 63) // 64 * 64 - 63, bx * 64], b2[bx * 64], C[by * 64, bx * 64])
        T.writes()
        C_local = T.alloc_buffer((64, 64), scope="local.fragment")
        A_shared = T.alloc_buffer((64, 32), scope="shared.dyn")
        W1_shared = T.alloc_buffer((32, 64), scope="shared.dyn")
        H_shared = T.alloc_buffer((64, 64), scope="shared.dyn")
        W2_shared = T.alloc_buffer((64, 64), scope="shared.dyn")
        H_local = T.alloc_buffer((64, 64), scope="local.fragment")
        b1_local = T.alloc_buffer((64,), scope="local.fragment")
        b2_local = T.alloc_buffer((64,), scope="local.fragment")
        T.fill(T.region(C_local[0, 0], 2, 64, 64), 0)
        zero: T.float32 = T.float32(0.0)
        for ho in range((H + 63) // 64):
            T.fill(T.region(H_local[0, 0], 2, 64, 64), 0)
            for ko in T.serial((K + 32 - 1) // 32, annotations={"num_stages": 2}):
                T.copy(T.region(A[by * 64, ko * 32], 1, 64, 32), T.region(A_shared[0, 0], 2, 64, 32))
                T.copy(T.region(W1[ko * 32, ho * 64], 1, 32, 64), T.region(W1_shared[0, 0], 2, 32, 64))
                T.gemm_py(T.region(A_shared[0, 0], 1, 64, 32), T.region(W1_shared[0, 0], 1, 32, 64), T.region(H_local[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 32, 0, T.bool(False), 32, 64, 0, 0, 1, 0, 0, 0, 0)
            T.copy(T.region(b1[ho * 64], 1, 64), T.region(b1_local[0], 2, 64))
            for i in T.parallel(64):
                for j in T.parallel(64):
                    H_local[i, j] = T.max(H_local[i, j] + b1_local[j], zero)
            T.copy(T.region(H_local[0, 0], 1, 64, 64), T.region(H_shared[0, 0], 2, 64, 64))
            T.copy(T.region(W2[ho * 64, bx * 64], 1, 64, 64), T.region(W2_shared[0, 0], 2, 64, 64))
            T.gemm_py(T.region(H_shared[0, 0], 1, 64, 64), T.region(W2_shared[0, 0], 1, 64, 64), T.region(C_local[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 0, T.bool(False), 64, 64, 0, 0, 1, 0, 0, 0, 0)
        T.copy(T.region(b2[bx * 64], 1, 64), T.region(b2_local[0], 2, 64))
        for i in T.parallel(64):
            for j in T.parallel(64):
                C_local[i, j] = C_local[i, j] + b2_local[j]
        T.copy(T.region(C_local[0, 0], 1, 64, 64), T.region(C[by * 64, bx * 64], 2, 64, 64))
