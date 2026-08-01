import tilelang.language as T

X0 = T.dynamic("X0")


@T.prim_func
def scrc_k0(
    p0: T.Tensor((X0, 64), "float32"),
    p1: T.Tensor((X0,), "bool"),
    p2: T.Tensor((X0, 64), "float32"),
):
    with T.Kernel(X0, threads=c0) as v3:
        with T.LetStmt(p1[v3]) as t1_identity:
            with T.LetStmt(T.cast(t1_identity, "float32")) as v0:
                for v2 in T.Parallel(64):
                    with T.LetStmt(v0) as v0_broadcast:
                        p2[v3, v2] = p0[v3, v2] * v0_broadcast
