import tilelang.language as T

X0 = T.dynamic("X0")


@T.prim_func
def scrc_k0(
    p0: T.Tensor((X0, 16), "float32"),
    p1: T.Tensor((X0,), "float32"),
):
    with T.Kernel(X0, threads=c0) as d0:
        t0 = T.alloc_fragment((16,), "float32")
        for d1 in T.Parallel(16):
            t0[d1] = p0[d0, d1]
        n0 = T.alloc_fragment((1,), "float32")
        T.reduce_sum(t0, n0, dim=0)
        t1 = n0[0]
        p1[d0] = t1
