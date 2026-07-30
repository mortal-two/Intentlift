import tilelang.language as T

M = T.dynamic("M")
N = T.dynamic("N")

@T.prim_func
def scrc_k0(p0: T.Tensor((M, N), "float32"), p1: T.Tensor((M,), "float32")):
    with T.Kernel(M, threads=c0) as bx:
        t0_fragment = T.alloc_fragment((1,), "float32")
        t0_fragment[0] = p0[bx, T.get_thread_binding()]
        n0 = T.alloc_fragment((1,), "float32")
        T.reduce_sum(t0_fragment, n0)
        p1[bx] = n0[0]
