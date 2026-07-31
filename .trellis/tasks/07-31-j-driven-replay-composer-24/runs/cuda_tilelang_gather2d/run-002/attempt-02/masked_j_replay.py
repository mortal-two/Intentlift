extern "C" __global__ void scrc_k0(const int *p0, const float *p1, float *p2, const int *p3, int p4) {
    const int program_index = static_cast<int>(blockIdx.x);
    const int *a0_address = p3 + program_index;
    const int a0_value = *a0_address;
    const int *a1_address = p0 + program_index;
    const int a1_value = *a1_address;
    const long long a2_offset = static_cast<long long>(a0_value) * static_cast<long long>(p4) + static_cast<long long>(a1_value);
    const float *a2_address = p1 + a2_offset;
    const float a2_value = *a2_address;
    float *a3_address = p2 + program_index;
    *a3_address = a2_value;
}
