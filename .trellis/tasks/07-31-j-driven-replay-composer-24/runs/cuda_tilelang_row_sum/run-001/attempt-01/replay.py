extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    (void)p2;
    const int v1 = static_cast<int>(blockIdx.x);
    const int v2 = static_cast<int>(threadIdx.x);
    float reduced = p0[v1 * 16 + v2];
    const unsigned int participants = __activemask();
    reduced += __shfl_xor_sync(participants, reduced, 8, 16);
    reduced += __shfl_xor_sync(participants, reduced, 4, 16);
    reduced += __shfl_xor_sync(participants, reduced, 2, 16);
    reduced += __shfl_xor_sync(participants, reduced, 1, 16);
    p1[v1] = reduced;
}
