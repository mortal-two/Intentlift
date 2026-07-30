#include <cuda_runtime.h>
extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    const unsigned int v1 = blockIdx.x;
    const unsigned int v2 = threadIdx.x;
    const float lane_value = p0[v1 * 16u + v2];
    float reduced = lane_value;
    const unsigned int active_lanes = __activemask();
    reduced += __shfl_xor_sync(active_lanes, reduced, 8);
    reduced += __shfl_xor_sync(active_lanes, reduced, 4);
    reduced += __shfl_xor_sync(active_lanes, reduced, 2);
    reduced += __shfl_xor_sync(active_lanes, reduced, 1);
    p1[v1] = reduced;
}
