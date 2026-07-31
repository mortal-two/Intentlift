import tilelang.language as T
X0 = T.dynamic("x0")
@T.prim_func
def scrc_k0(p0: T.Tensor((X0, 16), "float32"), p1: T.Tensor((16, X0), "float32")):
    with T.Kernel(X0, threads=c0) as v1:
        fragment = T.alloc_fragment((16,), "float32")
        for v2 in T.Parallel(16):
            fragment[v2] = p0[v1, v2]
        for v2 in T.Parallel(16):
            p1[v2, v1] = fragment[v2]
