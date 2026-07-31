import tilelang.language as T
X0 = T.dynamic("x0")
X1 = T.dynamic("x1")
@T.prim_func
def scrc_k0(p0: T.Tensor((X0, X1), "float32"), p1: T.Tensor((X0, X1), "float32")):
    with T.Kernel(X0, threads=c0) as (bx,):
        T.copy(p0[bx : bx + 1, 0:X1], p1[bx : bx + 1, 0:X1])
