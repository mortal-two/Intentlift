import tilelang.language as T

D0 = T.dynamic("d0")
D1 = T.dynamic("d1")

@T.prim_func
def scrc_k0(
    p0: T.Tensor((D0, D1), "float32"),
    p1: T.Tensor((D0, D1), "float32"),
    p2: T.Tensor((D0, D1), "float32"),
):
    with T.Kernel(D0, threads=c0) as i0:
        for i1 in T.Parallel(D1):
            p2[i0, i1] = p0[i0, i1] + p1[i0, i1]
