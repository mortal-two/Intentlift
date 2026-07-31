extern "C" __global__ void scrc_k0(const int* p0, const float* p1, float* p2, const int* p3, int p4) {
    const int v15 = static_cast<int>(blockIdx.x);
    const int v0 = 0;
    const int t2_v15 = p3[v15];
    const int t0_v15 = p0[v15];
    const bool v1 = t0_v15 >= v0;
    const int v2 = 64;
    const bool v3 = t0_v15 < v2;
    const bool v4 = v1 && v3;
    const bool v5 = t2_v15 >= v0;
    const bool v6 = v4 && v5;
    const bool v7 = t2_v15 < p4;
    const bool v8 = v6 && v7;
    const int v9 = v8 ? t2_v15 : v0;
    const int v10 = v8 ? t0_v15 : v0;
    const float v12 = p1[v9 * 64 + v10];
    const float v13 = 0.0f;
    const float v14 = v8 ? v12 : v13;
    p2[v15] = v14;
}
