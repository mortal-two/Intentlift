extern "C" __global__ void scrc_k0(float* p0, float* p1, int p2) {
    const unsigned int program_v1 = blockIdx.x;
    const unsigned int load_v1 = program_v1;
    const unsigned int store_v1 = program_v1;
    const unsigned int v2 = threadIdx.x;
    (void)p2;
    const float loaded = p0[load_v1 * 16u + v2];
    float reduced = loaded + 0.0f;
    const unsigned int mask = __activemask();
    reduced += __shfl_xor_sync(mask, reduced, 1, 16);
    reduced += __shfl_xor_sync(mask, reduced, 2, 16);
    reduced += __shfl_xor_sync(mask, reduced, 4, 16);
    reduced += __shfl_xor_sync(mask, reduced, 8, 16);
    p1[store_v1] = reduced;
}
