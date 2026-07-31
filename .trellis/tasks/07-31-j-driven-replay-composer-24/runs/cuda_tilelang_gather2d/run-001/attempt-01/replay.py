extern "C" __global__ void scrc_k0(const int* p0, const float* p1, float* p2, const int* p3, int p4) {
    const int v13 = static_cast<int>(blockIdx.x);
    const int v0 = 0;
    const int v1 = 64;
    const int t2_at_v13 = p3[v13];
    const int t0_at_v13 = p0[v13];
    const bool v2 = t0_at_v13 >= v0;
    const bool v3 = t0_at_v13 < v1;
    const bool v4 = t2_at_v13 >= v0;
    const bool v5 = t2_at_v13 < p4;
    const bool v6 = v2 && v3;
    const bool v7 = v4 && v5;
    const bool v8 = v6 && v7;
    float v10 = 0.0f;
    if (v8) {
        v10 = p1[t2_at_v13 * v1 + t0_at_v13];
    }
    const float v11 = 0.0f;
    const float t4_at_v13 = v8 ? v10 : v11;
    p2[v13] = t4_at_v13;
}
