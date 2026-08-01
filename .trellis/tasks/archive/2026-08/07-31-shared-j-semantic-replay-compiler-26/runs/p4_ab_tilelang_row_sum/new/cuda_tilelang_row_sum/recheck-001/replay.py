extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
  const int input_v1 = static_cast<int>(blockIdx.x);
  const int input_base_offset = input_v1 * 16;
  const int input_v2 = static_cast<int>(threadIdx.x) % 16;
  const float* input_lane_address = p0 + input_base_offset + input_v2;
  const float transferred_t0 = *input_lane_address;
  const float realized_t0 = 0.0f + transferred_t0;
  const unsigned int participants = __activemask();
  float reduced_t1 = realized_t0;
  reduced_t1 += __shfl_xor_sync(participants, reduced_t1, 8, 16);
  reduced_t1 += __shfl_xor_sync(participants, reduced_t1, 4, 16);
  reduced_t1 += __shfl_xor_sync(participants, reduced_t1, 2, 16);
  reduced_t1 += __shfl_xor_sync(participants, reduced_t1, 1, 16);
  const int output_v1 = static_cast<int>(blockIdx.x);
  float* output_base_address = p1 + output_v1;
  *output_base_address = reduced_t1;
}
