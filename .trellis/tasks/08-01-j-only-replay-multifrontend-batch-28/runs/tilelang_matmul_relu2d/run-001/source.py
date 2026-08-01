import tilelang.language as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    M, K = T.int32(), T.int32()
    A = T.match_buffer(A_handle, (M, K), strides=(K, 1))
    N = T.int32()
    B = T.match_buffer(B_handle, (K, N), strides=(N, 1))
    C = T.match_buffer(C_handle, (M, N), strides=(N, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", (N + 64 - 1) // 64)
    by = T.launch_thread("blockIdx.y", (M + 64 - 1) // 64)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 64, 0:(K + 31) // 32 * 32 - 31], B[0:(K + 31) // 32 * 32 - 31, bx * 64], C[by * 64, bx * 64])
        T.writes()
        A_shared = T.alloc_buffer((64, 32), scope="shared.dyn")
        B_shared = T.alloc_buffer((32, 64), scope="shared.dyn")
        C_local = T.alloc_buffer((64, 64), scope="local.fragment")
        T.fill(T.region(C_local[0, 0], 2, 64, 64), 0)
        for ko in T.serial((K + 32 - 1) // 32, annotations={"num_stages": 2}):
            T.copy(T.region(A[by * 64, ko * 32], 1, 64, 32), T.region(A_shared[0, 0], 2, 64, 32))
            T.copy(T.region(B[ko * 32, bx * 64], 1, 32, 64), T.region(B_shared[0, 0], 2, 32, 64))
            T.gemm_py(T.region(A_shared[0, 0], 1, 64, 32), T.region(B_shared[0, 0], 1, 32, 64), T.region(C_local[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 32, 0, T.bool(False), 32, 64, 0, 0, 1, 0, 0, 0, 0)
        zero: T.float32 = T.float32(0.0)
        for i in T.parallel(64):
            for j in T.parallel(64):
                C_local[i, j] = T.max(C_local[i, j], zero)
        T.copy(T.region(C_local[0, 0], 1, 64, 64), T.region(C[by * 64, bx * 64], 2, 64, 64))
