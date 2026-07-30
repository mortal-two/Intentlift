extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    const unsigned int v1 = blockIdx.x;
    const unsigned int v2_realized = threadIdx.x;
    const unsigned int v2 = v2_realized;
    const unsigned long long a0_base = static_cast<unsigned long long>(v1) * 16ULL;
    const float* a0_address = p0 + (a0_base + static_cast<unsigned long long>(v2));
    float reduced = *a0_address;
    const unsigned int participants = __activemask();
    reduced += __shfl_xor_sync(participants, reduced, 8, 16);
    reduced += __shfl_xor_sync(participants, reduced, 4, 16);
    reduced += __shfl_xor_sync(participants, reduced, 2, 16);
    reduced += __shfl_xor_sync(participants, reduced, 1, 16);
    __syncthreads();
    float* a1_address = p1 + static_cast<unsigned long long>(v1);
    *a1_address = reduced;
}
