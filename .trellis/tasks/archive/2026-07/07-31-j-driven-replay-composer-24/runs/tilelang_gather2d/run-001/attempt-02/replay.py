import tilelang.language as T
X0 = T.dynamic("x0")
@T.prim_func
def scrc_k0(p0: T.Tensor((X0, 64), "float32"), p1: T.Tensor((256,), "int32"), p2: T.Tensor((256,), "int32"), p3: T.Tensor((256,), "float32")):
    with T.Kernel(256, threads=c0) as v2:
        v5 = p1[v2]
        v6 = p2[v2]
        v7 = p0[v5, v6]
        p3[v2] = v7
