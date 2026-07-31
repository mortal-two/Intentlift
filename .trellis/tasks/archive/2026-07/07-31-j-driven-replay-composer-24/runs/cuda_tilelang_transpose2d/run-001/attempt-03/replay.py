extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    const int v1 = static_cast<int>(blockIdx.x);
    const float4* in_base = reinterpret_cast<const float4*>(p0 + v1 * 16);
    const float4 in0 = *in_base;
    const float4* in_addr1 = in_base + 1;
    const float4 in1 = *in_addr1;
    const float4* in_addr2 = in_base + 2;
    const float4 in2 = *in_addr2;
    const float4* in_addr3 = in_base + 3;
    const float4 in3 = *in_addr3;
    const float materialized[16] = {
        in0.x, in0.y, in0.z, in0.w,
        in1.x, in1.y, in1.z, in1.w,
        in2.x, in2.y, in2.z, in2.w,
        in3.x, in3.y, in3.z, in3.w
    };
    float* out_addr = p1 + v1;
    *out_addr = materialized[0];
    out_addr = out_addr + p2;
    *out_addr = materialized[1];
    out_addr = out_addr + p2;
    *out_addr = materialized[2];
    out_addr = out_addr + p2;
    *out_addr = materialized[3];
    out_addr = out_addr + p2;
    *out_addr = materialized[4];
    out_addr = out_addr + p2;
    *out_addr = materialized[5];
    out_addr = out_addr + p2;
    *out_addr = materialized[6];
    out_addr = out_addr + p2;
    *out_addr = materialized[7];
    out_addr = out_addr + p2;
    *out_addr = materialized[8];
    out_addr = out_addr + p2;
    *out_addr = materialized[9];
    out_addr = out_addr + p2;
    *out_addr = materialized[10];
    out_addr = out_addr + p2;
    *out_addr = materialized[11];
    out_addr = out_addr + p2;
    *out_addr = materialized[12];
    out_addr = out_addr + p2;
    *out_addr = materialized[13];
    out_addr = out_addr + p2;
    *out_addr = materialized[14];
    out_addr = out_addr + p2;
    *out_addr = materialized[15];
}
