extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    int v1_load = static_cast<int>(blockIdx.x);
    const float* t0 = p0;
    const float* a0_base = t0 + v1_load * 16;
    int v2 = static_cast<int>(threadIdx.x);
    long long v2_wide = static_cast<long long>(v2);
    const float* a0_lane = a0_base + v2_wide;
    float v0 = *a0_lane;
    unsigned int participants = __activemask();
    v0 += __shfl_xor_sync(participants, v0, 8, 16);
    v0 += __shfl_xor_sync(participants, v0, 4, 16);
    v0 += __shfl_xor_sync(participants, v0, 2, 16);
    v0 += __shfl_xor_sync(participants, v0, 1, 16);
    int v1_store = static_cast<int>(blockIdx.x);
    float* t1 = p1;
    float* a1_base = t1 + v1_store;
    *a1_base = v0;
}
