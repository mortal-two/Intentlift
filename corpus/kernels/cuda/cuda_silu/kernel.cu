
// SiLU / swish activation, grid-stride elementwise: y = x * sigmoid(x).
__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ y,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = x[i];
        y[i] = v / (1.0f + __expf(-v));
    }
}
