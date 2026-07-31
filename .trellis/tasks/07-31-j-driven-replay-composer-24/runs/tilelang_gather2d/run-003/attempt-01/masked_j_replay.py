import tilelang.language as T

X0 = T.dynamic("x0")
X1 = T.dynamic("x1")
X2 = T.dynamic("x2")

@T.prim_func
def scrc_k0(p0: T.Tensor((X1, X2), "float32"), p1: T.Tensor((X0,), "int32"), p2: T.Tensor((X0,), "int32"), p3: T.Tensor((X0,), "float32")):
    with T.Kernel(X0, threads=c0) as bx:
        index0 = p1[bx]
        index1 = p2[bx]
        value = p0[index0, index1]
        p3[bx] = value
