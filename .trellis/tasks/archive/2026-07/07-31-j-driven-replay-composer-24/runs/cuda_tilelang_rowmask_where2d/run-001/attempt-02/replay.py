extern "C" __global__ void scrc_k0(const float* p0, float* p1, const signed char* p2, int p3) {
    const unsigned int v3 = blockIdx.x;
    const unsigned int v4 = threadIdx.x;
    const signed char* a0_base = p2 + v3;
    const signed char a0_value = *a0_base;
    const bool v0 = static_cast<bool>(a0_value);
    const float v1 = static_cast<float>(v0);
    const float* a1_base = p0 + v3 * 64;
    const float* a1_lane = a1_base + v4;
    const float a1_value = *a1_lane;
    const float v1_axis1 = v1;
    const float v2 = a1_value * v1_axis1;
    float* a2_base = p1 + v3 * 64;
    float* a2_lane = a2_base + v4;
    *a2_lane = v2;
}
