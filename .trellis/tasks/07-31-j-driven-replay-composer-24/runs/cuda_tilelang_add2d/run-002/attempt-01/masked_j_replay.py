extern "C" __global__ void scrc_k0(float *p0, float *p1, float *p2, int p3) {
  const unsigned int d0 = blockIdx.x;
  const unsigned int address_base = d0 * blockDim.x;
  const unsigned int d1 = threadIdx.x;

  const unsigned int lane_index_a0 = address_base + d1;
  float *address_a0 = p0 + lane_index_a0;
  const float value_a0 = *address_a0;

  const unsigned int lane_index_a1 = address_base + d1;
  float *address_a1 = p1 + lane_index_a1;
  const float value_a1 = *address_a1;

  const float n0 = value_a0 + value_a1;

  const unsigned int lane_index_a2 = address_base + d1;
  float *address_a2 = p2 + lane_index_a2;
  *address_a2 = n0;
}
