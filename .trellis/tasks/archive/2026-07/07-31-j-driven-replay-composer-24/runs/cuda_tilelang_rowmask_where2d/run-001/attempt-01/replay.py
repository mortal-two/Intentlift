extern "C" __global__ void scrc_k0(const float* p0, float* p1, const signed char* p2, int p3) {
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x % 64u;
    const signed char* mask_address_base = p2 + row;
    const signed char mask_element = *mask_address_base;
    const bool mask_i1 = static_cast<bool>(mask_element);
    const float mask_f32 = static_cast<float>(mask_i1);
    const unsigned long long input_address_base = static_cast<unsigned long long>(row) * 64ull;
    const unsigned long long input_offset = input_address_base + static_cast<unsigned long long>(lane);
    const float* input_address_lane = p0 + input_offset;
    const float input_element = *input_address_lane;
    const float mask_broadcast_axis1 = mask_f32;
    const float result_element = input_element * mask_broadcast_axis1;
    const unsigned long long output_address_base = static_cast<unsigned long long>(row) * 64ull;
    const unsigned long long output_offset = output_address_base + static_cast<unsigned long long>(lane);
    float* output_address_lane = p1 + output_offset;
    *output_address_lane = result_element;
}
