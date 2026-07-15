"""Legacy multi-frontend demo: the old extractor is language-agnostic.

The same LLM extractor that lifts Triton kernels lifts hand-written CUDA C++ to
the SAME IntentIR. This file holds faithful, optimization-bearing CUDA kernels
(grid-stride loops, shared-memory tiles, warp reductions -- the stuff a real
library ships) and lifts them.

nvcc is not installed in this environment, so the dynamic check here compares
interpret(JSON) against the torch reference op (each kernel computes a named op).
That verifies the FRONTEND-LANGUAGE handling -- the extractor recovered the right
tensor math from CUDA source; it is a language-pluggability demo, distinct from
the Triton corpus where we run the actual kernel.  It is retained as historical
extraction evidence and is not the current SCRC CUDA verification path.
"""

from __future__ import annotations

import json

from intentir.extraction.data.corpus import CORPUS_DIR, _relerr
from .extract import extract_intent

# --- faithful CUDA kernels (the kind a kernel library ships) ----------------- #

CUDA_SOFTMAX = r"""
// Row-wise softmax over a [M, N] matrix. One block per row; block-stride load,
// warp+shared reduction for max then sum, numerically stabilized.
__global__ void softmax_rows(const float* __restrict__ X, float* __restrict__ Y,
                             int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    extern __shared__ float smem[];
    const float* xr = X + (size_t)row * N;
    float* yr = Y + (size_t)row * N;

    // 1) row max
    float m = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, xr[j]);
    smem[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // 2) sum of exp(x - max)
    float s = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) s += __expf(xr[j] - row_max);
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) smem[threadIdx.x] += smem[threadIdx.x + t];
        __syncthreads();
    }
    float denom = smem[0];

    // 3) normalize
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        yr[j] = __expf(xr[j] - row_max) / denom;
}
"""

CUDA_SILU = r"""
// SiLU / swish activation, grid-stride elementwise: y = x * sigmoid(x).
__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ y,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = x[i];
        y[i] = v / (1.0f + __expf(-v));
    }
}
"""

CUDA_RMSNORM = r"""
// RMSNorm over the last dim of a [M, N] tensor: y = x * rsqrt(mean(x^2)+eps) * w.
// One block per row; block-stride sum of squares, shared reduction.
__global__ void rmsnorm_rows(const float* __restrict__ X, const float* __restrict__ W,
                             float* __restrict__ Y, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;
    extern __shared__ float smem[];
    const float* xr = X + (size_t)row * N;
    float* yr = Y + (size_t)row * N;
    float ss = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) ss += xr[j] * xr[j];
    smem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / N + eps);
    for (int j = threadIdx.x; j < N; j += blockDim.x) yr[j] = xr[j] * inv * W[j];
}
"""


def cuda_specs():
    """(name, cuda source, shape_hint, torch reference) for each CUDA kernel."""
    import torch

    M, N = 48, 192
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    xf = torch.randn(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    eps = 1e-5
    return [
        ("cuda_softmax", CUDA_SOFTMAX,
         "single input X shape [M, N]; row softmax over the last axis (1).",
         [x], lambda: torch.softmax(x, dim=1)),
        ("cuda_silu", CUDA_SILU,
         "single input X shape [M, N]; SiLU x*sigmoid(x) elementwise.",
         [xf], lambda: torch.nn.functional.silu(xf)),
        ("cuda_rmsnorm", CUDA_RMSNORM,
         "inputs IN THIS ORDER: X shape [M, N]; W shape [N]; eps scalar shape []. "
         "RMSNorm over the last axis: X * rsqrt(mean(X^2, axis=1)+eps) * W.",
         [x, w, torch.tensor(eps, device="cuda")],
         lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w),
    ]


def run_cuda_frontend(save: bool = True):
    """Lift each CUDA kernel and check interpret(JSON) vs the torch reference."""
    from intentir.verify.back.legacy_v0.interpret import interpret

    rows = []
    for name, src, hint, ti, ref in cuda_specs():
        rec = {"name": name, "extracted": False, "verified": False,
               "rel_err": None, "error": ""}
        try:
            ex = extract_intent(src, name, shape_hint=hint)
            rec["extracted"] = True
            rec["ops"] = [o.op + (":" + o.attrs.get("fn", "") if o.attrs.get("fn") else "")
                          for o in ex.fn.body]
            arrays = [t.detach().cpu().numpy() for t in ti]
            interp_in = {t.name: a for t, a in zip(ex.fn.inputs, arrays)}
            y_int = interpret(ex.fn, interp_in)[ex.fn.outputs[0].name]
            y_ref = ref().detach().cpu().numpy()
            rec["rel_err"] = _relerr(y_int, y_ref)
            rec["verified"] = rec["rel_err"] < 1e-3
            if save:
                d = CORPUS_DIR / "cuda" / name
                d.mkdir(parents=True, exist_ok=True)
                (d / "intent.json").write_text(json.dumps(ex.json_obj, indent=2))
                (d / "kernel.cu").write_text(src)
                (d / "provenance.json").write_text(json.dumps({
                    "name": name, "language": "CUDA C++", "model": ex.provenance.model,
                    "prompt_sha256": ex.provenance.prompt_sha256,
                    "verified_rel_err": rec["rel_err"],
                    "verified_against": "torch reference op (nvcc absent: language demo)",
                    "generated_by": "intentir.extraction (live provider)",
                }, indent=2))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        rows.append(rec)
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("=== Multi-frontend: lifting CUDA C++ kernels (live gpt-5.5) ===\n")
    for r in run_cuda_frontend():
        rel = f"{r['rel_err']:.2e}" if r["rel_err"] is not None else "—"
        st = "PASS" if r["verified"] else ("ERR " + r["error"] if r["error"] else "FAIL")
        print(f"  {r['name']:16s} extract={'PASS' if r['extracted'] else '----'} "
              f"verify(vs torch)={st} rel={rel}")
        if r.get("ops"):
            print(f"      lifted ops: {r['ops']}")
