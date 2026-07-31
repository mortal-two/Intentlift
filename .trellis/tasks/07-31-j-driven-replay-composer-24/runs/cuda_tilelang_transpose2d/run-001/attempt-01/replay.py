extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    const int v1 = static_cast<int>(blockIdx.x);
    const int v2 = static_cast<int>(threadIdx.x);
    const float value = p0[v1 * 16 + v2];
    p1[v2 * p2 + v1] = value;
}
