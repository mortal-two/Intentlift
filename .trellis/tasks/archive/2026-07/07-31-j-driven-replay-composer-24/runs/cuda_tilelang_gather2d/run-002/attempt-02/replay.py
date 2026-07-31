extern "C" __global__ void scrc_k0(const int* p0, const float* p1, float* p2, const int* p3, int p4) {
    const int v13 = static_cast<int>(blockIdx.x);
    const int a0_value = p3[v13];
    const int a1_value = p0[v13];
    const int v0 = 0;
    const bool v1 = a1_value >= v0;
    const int v2 = 64;
    const bool v3 = a1_value < v2;
    const bool v4 = v1 && v3;
    const bool v5 = a0_value >= v0;
    const bool v6 = v4 && v5;
    const bool v7 = a0_value < p4;
    const bool v8 = v6 && v7;
    float v10;
    if ((0 <= a1_value) && (a1_value < 64) && (0 <= a0_value) && (a0_value < p4)) {
        v10 = p1[a0_value * 64 + a1_value];
    }
    else {
        v10 = 0.0f;
    }
    const float v11 = 0.0f;
    const float v12 = v8 ? v10 : v11;
    p2[v13] = v12;
}
