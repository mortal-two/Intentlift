"""Isolated flag_gems reference timer (subprocess).

flag_gems' triton `libentry` launcher throws an ABI mismatch ("function takes
exactly 32 arguments (31 given)") under CUDA-graph capture on this sm_120a +
triton-3.6 env, and a failed launch POISONS the CUDA context (hangs any
subsequent timing) -- so flag_gems cannot be co-resident in the same process as
our raw-driver cuLaunchKernel benchmarking. This tiny script runs ONE flag_gems
op in a fresh process, times it with plain ``do_bench`` (launch amortized, L2
flushed -- the task's accepted "or triton do_bench" alternative to graph
capture), and prints ``US <median_us>``. bench_clusters.py subprocess-calls it.

Usage: python3 fg_ref.py '<json-spec>'
"""

from __future__ import annotations

import json
import sys

import torch
import triton


def _bench(fn):
    fn(); torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median") * 1e3


def main():
    spec = json.loads(sys.argv[1])
    op = spec["op"]
    torch.manual_seed(spec.get("seed", 0))
    import flag_gems as fg

    def randn(*s):
        return torch.randn(*s, device="cuda")

    if op == "gather":
        M, N = spec["M"], spec["N"]
        d = randn(M, N); i = torch.randint(0, N, (M, N), device="cuda", dtype=torch.int64)
        fn = lambda: fg.gather(d, 1, i)
    elif op == "embedding":
        V, D, Ni = spec["V"], spec["D"], spec["Ni"]
        w = randn(V, D); i = torch.randint(0, V, (Ni,), device="cuda", dtype=torch.int64)
        fn = lambda: fg.embedding(w, i)
    elif op == "scatter_add":
        M, N, K = spec["M"], spec["N"], spec["K"]
        base = randn(M, N); i = torch.randint(0, N, (M, K), device="cuda", dtype=torch.int64); s = randn(M, K)
        fn = lambda: fg.scatter_add_(base.clone(), 1, i, s)
    elif op == "cat":
        M, N = spec["M"], spec["N"]; a = randn(M, N); b = randn(M, N)
        fn = lambda: fg.cat([a, b], dim=1)
    elif op == "pad":
        M, N = spec["M"], spec["N"]; a = randn(M, N)
        fn = lambda: fg.constant_pad_nd(a, [2, 2, 2, 2], 0.0)
    elif op == "tile":
        M, N = spec["M"], spec["N"]; a = randn(M, N)
        fn = lambda: fg.tile(a, [2, 2])
    elif op == "flip":
        M, N = spec["M"], spec["N"]; a = randn(M, N)
        fn = lambda: fg.flip(a, [1])
    elif op == "triu":
        M, N = spec["M"], spec["N"]; a = randn(M, N)
        fn = lambda: fg.triu(a, 0)
    elif op == "dropout":
        M, N = spec["M"], spec["N"]; a = randn(M, N)
        fn = lambda: fg.dropout(a, p=0.25, training=True)
    elif op == "argmax":
        M, N = spec["M"], spec["N"]; x = randn(M, N)
        fn = lambda: fg.argmax(x, dim=1)
    elif op == "sort":
        M, N = spec["M"], spec["N"]; x = randn(M, N)
        fn = lambda: fg.sort(x, dim=1)[0]
    elif op == "avg_pool2d":
        N, C, H, W = spec["N"], spec["C"], spec["H"], spec["W"]; a = randn(N, C, H, W)
        fn = lambda: fg.avg_pool2d(a, 2, stride=2, padding=0)
    elif op == "max_pool2d":
        N, C, H, W = spec["N"], spec["C"], spec["H"], spec["W"]; a = randn(N, C, H, W)
        fn = lambda: fg.max_pool2d_with_indices(a, 2, stride=2, padding=0)
    elif op == "upsample":
        N, C, H, W = spec["N"], spec["C"], spec["H"], spec["W"]; a = randn(N, C, H, W)
        fn = lambda: fg.upsample_nearest2d(a, (2 * H, 2 * W), None, None)
    elif op == "polar":
        M, N = spec["M"], spec["N"]; absv = torch.rand(M, N, device="cuda") + 0.1; ang = torch.rand(M, N, device="cuda") * 6 - 3
        fn = lambda: fg.polar(absv, ang)
    elif op == "conv2d":
        N, Cin, Cout, H, W = spec["N"], spec["Cin"], spec["Cout"], spec["H"], spec["W"]
        x = randn(N, Cin, H, W); w = randn(Cout, Cin, 3, 3)
        fn = lambda: fg.conv2d(x, w, None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    elif op == "masked_select":
        R, C = spec["R"], spec["C"]; data = randn(R, C); mask = torch.rand(R, C, device="cuda") < 0.5
        fn = lambda: fg.masked_select(data, mask)
    else:
        print("SKIP unknown-op"); return

    try:
        us = _bench(fn)
        print(f"US {us:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
