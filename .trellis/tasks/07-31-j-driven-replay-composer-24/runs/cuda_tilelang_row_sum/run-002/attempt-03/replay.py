extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    const int input_row = static_cast<int>(blockIdx.x);
    const float* input_base = p0 + static_cast<long long>(input_row) * 16LL;
    const int reduction_lane = static_cast<int>(threadIdx.x);
    const long long reduction_lane_wide = static_cast<long long>(reduction_lane);
    float reduced = input_base[reduction_lane_wide];
    const unsigned int participants = __activemask();
    reduced += __shfl_xor_sync(participants, reduced, 8, 16);
    reduced += __shfl_xor_sync(participants, reduced, 4, 16);
    reduced += __shfl_xor_sync(participants, reduced, 2, 16);
    reduced += __shfl_xor_sync(participants, reduced, 1, 16);
    __syncthreads();
    const int output_row = static_cast<int>(blockIdx.x);
    p1[output_row] = reduced;
}
