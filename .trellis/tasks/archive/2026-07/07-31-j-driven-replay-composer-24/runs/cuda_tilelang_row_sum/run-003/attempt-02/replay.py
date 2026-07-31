extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    (void)p2;
    const int v1 = static_cast<int>(blockIdx.x);
    const int v2 = static_cast<int>(threadIdx.x) % 16;
    const float a0 = p0[v1 * 16 + v2];
    float t0 = 0.0f + a0;
    const unsigned int participants = __activemask();
    t0 += __shfl_xor_sync(participants, t0, 8, 16);
    t0 += __shfl_xor_sync(participants, t0, 4, 16);
    t0 += __shfl_xor_sync(participants, t0, 2, 16);
    t0 += __shfl_xor_sync(participants, t0, 1, 16);
    p1[v1] = t0;
}
