extern "C" __global__ void scrc_k0(const int* p0,
                                    const float* p1,
                                    float* p2,
                                    const int* p3,
                                    int p4) {
  const int index = static_cast<int>(blockIdx.x);
  const int row = p3[index];
  const int column = p0[index];

  float gathered = 0.0f;
  if (column >= 0 && column < 64 && row >= 0 && row < p4) {
    gathered = p1[row * 64 + column];
  }

  p2[index] = gathered;
}
