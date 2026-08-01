import tilelang.language as T


@T.prim_func
def scrc_k0(
    p0: T.Tensor((1, 1), "float32"),
    p1: T.Tensor((1,), "int32"),
    p2: T.Tensor((1,), "int32"),
    p3: T.Tensor((1,), "float32"),
):
    with T.Kernel(1, threads=c0) as bx:
        p3[bx] = p0[p1[bx], p2[bx]]
