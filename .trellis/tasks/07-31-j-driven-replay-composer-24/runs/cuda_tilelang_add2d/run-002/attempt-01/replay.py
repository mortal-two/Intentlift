extern "C" __global__ void scrc_k0(
    const float* p0,
    const float* p1,
    float* p2,
    int p3) {
  const int v1 = static_cast<int>(blockIdx.x);
  const int v2 = static_cast<int>(threadIdx.x);
  const int base = v1 * 16;

  const float* a0 = p0 + base + v2;
  const float x = *a0;

  const float* a1 = p1 + base + v2;
  const float y = *a1;

  const float z = x + y;

  float* a2 = p2 + base + v2;
  *a2 = z;
}
