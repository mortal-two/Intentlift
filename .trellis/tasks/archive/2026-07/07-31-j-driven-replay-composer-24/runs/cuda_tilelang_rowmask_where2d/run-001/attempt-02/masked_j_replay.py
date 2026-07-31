extern "C" __global__ void scrc_k0(const float* p0, float* p1, const signed char* p2, int p3) {
    const unsigned int d0 = blockIdx.x;
    const unsigned int d1 = threadIdx.x;
    const signed char* a0_base = p2 + d0;
    const signed char a0_transfer = *a0_base;
    const signed char n0 = a0_transfer;
    const float r0 = static_cast<float>(n0);
    const unsigned long long a1_base = static_cast<unsigned long long>(d0);
    const unsigned long long a1_lane = a1_base + static_cast<unsigned long long>(d1);
    const float* a1_address = p0 + a1_lane;
    const float a1_transfer = *a1_address;
    const float r0_at_n2 = r0;
    const float n2 = a1_transfer + r0_at_n2;
    const unsigned long long a2_base = static_cast<unsigned long long>(d0);
    const unsigned long long a2_lane = a2_base + static_cast<unsigned long long>(d1);
    float* a2_address = p1 + a2_lane;
    *a2_address = n2;
}
