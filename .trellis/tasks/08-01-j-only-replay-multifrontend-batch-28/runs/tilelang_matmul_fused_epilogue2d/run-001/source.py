import tilelang.language as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, bias_handle: T.handle, row_mask_handle: T.handle, col_mask_handle: T.handle, C_handle: T.handle):
    M, K = T.int32(), T.int32()
    A = T.match_buffer(A_handle, (M, K), strides=(K, 1))
    N = T.int32()
    B = T.match_buffer(B_handle, (K, N), strides=(N, 1))
    bias = T.match_buffer(bias_handle, (N,), strides=(1,))
    row_mask = T.match_buffer(row_mask_handle, (M,), "bool", strides=(1,))
    col_mask = T.match_buffer(col_mask_handle, (N,), "bool", strides=(1,))
    C = T.match_buffer(C_handle, (M, N), strides=(N, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", (N + 64 - 1) // 64)
    by = T.launch_thread("blockIdx.y", (M + 64 - 1) // 64)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 64, 0:(K + 31) // 32 * 32 - 31], B[0:(K + 31) // 32 * 32 - 31, bx * 64], bias[bx * 64], row_mask[by * 64], col_mask[bx * 64], C[by * 64, bx * 64])
        T.writes()
        A_shared = T.alloc_buffer((64, 32), scope="shared.dyn")
        B_shared = T.alloc_buffer((32, 64), scope="shared.dyn")
        C_local = T.alloc_buffer((64, 64), scope="local.fragment")
        bias_local = T.alloc_buffer((64,), scope="local.fragment")
        row_local = T.alloc_buffer((64,), "bool", scope="local.fragment")
        col_local = T.alloc_buffer((64,), "bool", scope="local.fragment")
        T.fill(T.region(C_local[0, 0], 2, 64, 64), 0)
        for ko in T.serial((K + 32 - 1) // 32, annotations={"num_stages": 2}):
            T.copy(T.region(A[by * 64, ko * 32], 1, 64, 32), T.region(A_shared[0, 0], 2, 64, 32))
            T.copy(T.region(B[ko * 32, bx * 64], 1, 32, 64), T.region(B_shared[0, 0], 2, 32, 64))
            T.gemm_py(T.region(A_shared[0, 0], 1, 64, 32), T.region(B_shared[0, 0], 1, 32, 64), T.region(C_local[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 32, 0, T.bool(False), 32, 64, 0, 0, 1, 0, 0, 0, 0)
        T.copy(T.region(bias[bx * 64], 1, 64), T.region(bias_local[0], 2, 64))
        T.copy(T.region(row_mask[by * 64], 1, 64), T.region(row_local[0], 2, 64))
        T.copy(T.region(col_mask[bx * 64], 1, 64), T.region(col_local[0], 2, 64))
        for i in T.parallel(64):
            for j in T.parallel(64):
                cond: T.bool = T.bitwise_and(row_local[i], col_local[j])
                mf: T.float32 = T.Cast("float32", cond)
                C_local[i, j] = (C_local[i, j] + bias_local[j]) * mf
        T.copy(T.region(C_local[0, 0], 1, 64, 64), T.region(C[by * 64, bx * 64], 2, 64, 64))
