import tilelang.language as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    M = T.int32()
    A = T.match_buffer(A_handle, (M, 16), strides=(16, 1))
    B = T.match_buffer(B_handle, (M, 16), strides=(16, 1))
    C = T.match_buffer(C_handle, (M, 16), strides=(16, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", M)
    tx = T.launch_thread("threadIdx.x", 16)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[bx, 0:16], B[bx, 0:16])
        T.writes(C[bx, 0:16])
        for col in T.parallel(16):
            C[bx, col] = A[bx, col] + B[bx, col]
