#include <cuda_runtime.h>

extern "C" __global__ void scrc_k0(float* p0, float* p1, int p2) {
    const int input_row = static_cast<int>(blockIdx.x);
    float* const input = p0;
    const long long input_base = static_cast<long long>(input_row) * 16;
    const int lane = static_cast<int>(threadIdx.x % 16);
    float* const input_address = input + input_base + lane;
    const float loaded = *input_address;
    const unsigned int participants = __activemask();
    float reduced = loaded;
    reduced += __shfl_xor_sync(participants, reduced, 1, 16);
    reduced += __shfl_xor_sync(participants, reduced, 2, 16);
    reduced += __shfl_xor_sync(participants, reduced, 4, 16);
    reduced += __shfl_xor_sync(participants, reduced, 8, 16);
    const int output_row = static_cast<int>(blockIdx.x);
    float* const output = p1;
    float* const output_address = output + output_row;
    *output_address = reduced;
}
