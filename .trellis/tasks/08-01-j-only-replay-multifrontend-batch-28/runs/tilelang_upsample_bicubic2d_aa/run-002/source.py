import tilelang.language as T

@T.prim_func
def main(I_handle: T.handle, O_handle: T.handle):
    I_1 = T.match_buffer(I_handle, (1, 1, 4, 4), strides=(16, 16, 4, 1))
    O = T.match_buffer(O_handle, (1, 1, 8, 8), strides=(64, 64, 8, 1))
    # with T.block("root"):
    oh = T.launch_thread("blockIdx.x", 8)
    ow = T.launch_thread("blockIdx.y", 8)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(I_1[0, 0, oh // 2, ow // 2])
        T.writes(O[0, 0, oh, ow])
        ih: T.int32 = oh // 2
        iw: T.int32 = ow // 2
        O[0, 0, oh, ow] = I_1[0, 0, ih, iw]
