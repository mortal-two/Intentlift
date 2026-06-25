
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
