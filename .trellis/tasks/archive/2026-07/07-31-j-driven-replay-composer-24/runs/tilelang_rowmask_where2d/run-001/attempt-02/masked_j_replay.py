import tilelang.language as T
DIM = T.dynamic("int32")
@T.prim_func
def scrc_k0(p0: T.Tensor((DIM, DIM), "float32"), p1: T.Tensor((DIM,), "bool"), p2: T.Tensor((DIM, DIM), "float32")):
    with T.Kernel(DIM, threads=c0) as d1:
        t1_fragment = T.alloc_fragment((1,), "bool")
        T.copy(p1[d1], t1_fragment)
        for d0 in T.Parallel(DIM):
            a1_base = d1
            r_input = p0[a1_base, d0]
            a2_base = d1
            r0 = t1_fragment[0]
            r1 = T.if_then_else(r0, r_input, T.float32(0))
            p2[a2_base, d0] = r1
