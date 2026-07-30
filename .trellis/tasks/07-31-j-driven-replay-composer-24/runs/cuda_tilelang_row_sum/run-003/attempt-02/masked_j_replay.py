extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    const int d0 = static_cast<int>(blockIdx.x);
    const int d1 = static_cast<int>(threadIdx.x);
    const float transferred_a0 = p0[d0 * p2 + d1];
    const float materialized_t0 = 0.0f + transferred_a0;
    const unsigned participant_mask = __activemask();
    float n0 = materialized_t0;
    n0 += __shfl_down_sync(participant_mask, n0, 16);
    n0 += __shfl_down_sync(participant_mask, n0, 8);
    n0 += __shfl_down_sync(participant_mask, n0, 4);
    n0 += __shfl_down_sync(participant_mask, n0, 2);
    n0 += __shfl_down_sync(participant_mask, n0, 1);
    n0 = __shfl_sync(participant_mask, n0, 0);
    p1[d0] = n0;
}
