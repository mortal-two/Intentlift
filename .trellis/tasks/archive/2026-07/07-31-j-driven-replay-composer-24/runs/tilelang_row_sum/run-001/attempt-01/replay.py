import tilelang.language as T

X0 = T.dynamic("x0")


@T.prim_func
def scrc_k0(
    p0: T.Tensor((X0, 16), "float32"),
    p1: T.Tensor((X0,), "float32"),
):
    with T.Kernel(X0, threads=c0) as d0:
        input_fragment = T.alloc_fragment((16,), "float32")
        output_fragment = T.alloc_fragment((1,), "float32")
        for d1 in T.Parallel(16):
            input_fragment[d1] = p0[d0, d1]
        T.reduce_sum(input_fragment, output_fragment, dim=0)
        p1[d0] = output_fragment[0]
