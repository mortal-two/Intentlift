extern "C" __global__ void scrc_k0(const int* p0, const float* p1, float* p2, const int* p3, int p4) {
    const int d0 = (int)blockIdx.x;
    const int a0 = p3[d0];
    const int a1 = p0[d0];
    const float a2 = (a1 >= 0 && a1 < p4) ? p1[a0 * p4 + a1] : 0.0f;
    p2[d0] = a2;
}
