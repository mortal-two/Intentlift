"""TileLang Source kernels for the two Stage-3 production workloads.

The computation is the accepted Source computation.  ``m`` is supplied by the
benchmark harness so TileLang receives the same concrete workload valuation as
the MLIR target; N/L and the authored launch size remain Source constants.
"""


def make_gather2d_prim_func(
    *, m: int, n: int = 64, l: int = 256, threads: int = 128
):
    import tilelang.language as T

    M = int(m)
    N = int(n)
    L = int(l)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        row_idx: T.Tensor((L,), "int32"),
        col_idx: T.Tensor((L,), "int32"),
        out: T.Tensor((L,), "float32"),
    ):
        with T.Kernel(L, threads=threads) as (pid,):
            r = row_idx[pid]
            c = col_idx[pid]
            out[pid] = inp[r, c]

    return main


def make_rowmask_where2d_prim_func(
    *, m: int, n: int = 64, threads: int = 128
):
    import tilelang.language as T

    M = int(m)
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        row_mask: T.Tensor((M,), "bool"),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            mask = row_mask[pid_m]
            mask_f32 = T.cast(mask, "float32")
            for column in T.Parallel(N):
                out[pid_m, column] = inp[pid_m, column] * mask_f32

    return main


__all__ = ["make_gather2d_prim_func", "make_rowmask_where2d_prim_func"]
