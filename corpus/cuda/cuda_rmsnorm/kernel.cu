
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
