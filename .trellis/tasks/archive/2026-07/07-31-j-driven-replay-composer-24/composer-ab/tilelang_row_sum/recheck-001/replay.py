import tilelang.language as T

X0 = T.dynamic("x0")


@T.prim_func
def scrc_k0(
    p0: T.Tensor((X0, 16), "float32"),
    p1: T.Tensor((X0,), "float32"),
):
    with T.Kernel(X0, threads=c0) as d0:
        t0 = T.alloc_fragment((16,), "float32")
        T.copy(p0[d0, 0:16], t0)
        n0 = T.alloc_fragment((1,), "float32")
        T.reduce_sum(t0, n0)
        p1[d0] = n0[0]
