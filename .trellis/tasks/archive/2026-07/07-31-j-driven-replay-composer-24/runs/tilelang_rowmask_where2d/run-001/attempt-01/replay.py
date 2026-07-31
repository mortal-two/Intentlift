import tilelang.language as T
X0 = T.dynamic("x0")
@T.prim_func
def scrc_k0(p0: T.Tensor((X0, 64), "float32"), p1: T.Tensor((X0,), "bool"), p2: T.Tensor((X0, 64), "float32")):
    with T.Kernel(X0, threads=c0) as v3:
        v0 = T.Cast("float32", p1[v3])
        for v2 in T.Parallel(64):
            v1 = p0[v3, v2]
            p2[v3, v2] = v1 * v0
