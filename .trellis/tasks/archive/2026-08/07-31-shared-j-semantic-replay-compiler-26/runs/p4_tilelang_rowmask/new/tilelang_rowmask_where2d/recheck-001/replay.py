import tilelang.language as T

X0 = T.dynamic("x0")


@T.prim_func
def scrc_k0(
    p0: T.Tensor((X0, 64), "float32"),
    p1: T.Tensor((X0,), "bool"),
    p2: T.Tensor((X0, 64), "float32"),
):
    with T.Kernel(X0, threads=c0) as v3:
        t1_fragment = T.alloc_fragment((1,), "bool")
        T.copy(p1[v3], t1_fragment[0])
        v0 = T.cast(t1_fragment[0], "float32")
        for v2 in T.Parallel(64):
            p2[v3, v2] = p0[v3, v2] * v0
