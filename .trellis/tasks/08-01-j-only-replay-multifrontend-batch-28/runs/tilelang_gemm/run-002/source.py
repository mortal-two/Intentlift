import tilelang.language as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    M, K = T.int32(), T.int32()
    A = T.match_buffer(A_handle, (M, K), "float16", strides=(K, 1))
    N = T.int32()
    B = T.match_buffer(B_handle, (K, N), "float16", strides=(N, 1))
    C = T.match_buffer(C_handle, (M, N), "float16", strides=(N, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", (N + 128 - 1) // 128)
    by = T.launch_thread("blockIdx.y", (M + 128 - 1) // 128)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 128, 0:(K + 31) // 32 * 32 - 31], B[0:(K + 31) // 32 * 32 - 31, bx * 128], C[by * 128, bx * 128])
        T.writes()
        A_shared = T.alloc_buffer((128, 32), "float16", scope="shared.dyn")
        B_shared = T.alloc_buffer((32, 128), "float16", scope="shared.dyn")
        C_local = T.alloc_buffer((128, 128), scope="local.fragment")
        T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
        for ko in T.serial((K + 32 - 1) // 32, annotations={"num_stages": 3}):
            T.copy(T.region(A[by * 128, ko * 32], 1, 128, 32), T.region(A_shared[0, 0], 2, 128, 32))
            T.copy(T.region(B[ko * 32, bx * 128], 1, 32, 128), T.region(B_shared[0, 0], 2, 32, 128))
            T.gemm_py(T.region(A_shared[0, 0], 1, 128, 32), T.region(B_shared[0, 0], 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, 0, 0, 0)
        for i in T.parallel(128):
            for j in T.parallel(128):
                C_local[i, j] = T.max(C_local[i, j], T.float32(0.0))
        T.copy(T.region(C_local[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128))
