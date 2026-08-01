import tilelang.language as T

@T.prim_func
def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, sm_scale_handle: T.handle, Out_handle: T.handle):
    Q = T.match_buffer(Q_handle, (64, 64), strides=(64, 1))
    K = T.match_buffer(K_handle, (64, 64), strides=(64, 1))
    V = T.match_buffer(V_handle, (64, 64), strides=(64, 1))
    sm_scale = T.match_buffer(sm_scale_handle, (1,), strides=(1,))
    Out = T.match_buffer(Out_handle, (64, 64), strides=(64, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 64)
    tx = T.launch_thread("threadIdx.x", 1)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(sm_scale[0], Q[bx, 0:64], K[0:64, 0:64], V[0:64, 0:64])
        T.writes(Out[bx, 0:64])
        m_i = T.alloc_buffer((1,), scope="local.fragment")
        l_i = T.alloc_buffer((1,), scope="local.fragment")
        acc = T.alloc_buffer((64,), scope="local.fragment")
        scores = T.alloc_buffer((16,), scope="local.fragment")
        exp_scores = T.alloc_buffer((16,), scope="local.fragment")
        mx = T.alloc_buffer((1,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        dot_acc = T.alloc_buffer((1,), scope="local.fragment")
        value_acc = T.alloc_buffer((1,), scope="local.fragment")
        neg: T.float32 = T.float32(-1000000000.0)
        one: T.float32 = T.float32(1.0)
        scale: T.float32 = sm_scale[0]
        m_i[0] = neg
        l_i[0] = T.float32(0.0)
        for d0 in range(64):
            acc[d0] = T.float32(0.0)
        for blk0 in range(4):
            base: T.int32 = blk0 * 16
            for j0 in range(16):
                kv: T.int32 = base + j0
                dot_acc[0] = T.float32(0.0)
                for d1 in range(64):
                    dot_acc[0] = dot_acc[0] + Q[bx, d1] * K[kv, d1]
                s: T.float32 = dot_acc[0] * scale
                mf: T.float32 = T.Cast("float32", bx >= kv)
                scores[j0] = s * mf + neg * (one - mf)
            mx[0] = neg
            for j1 in range(16):
                mx[0] = T.max(mx[0], scores[j1])
            m_new: T.float32 = T.max(m_i[0], mx[0])
            alpha: T.float32 = T.exp(m_i[0] - m_new)
            sm[0] = T.float32(0.0)
            for j2 in range(16):
                exp_scores[j2] = T.exp(scores[j2] - m_new)
                sm[0] = sm[0] + exp_scores[j2]
            l_i[0] = l_i[0] * alpha + sm[0]
            for d2 in range(64):
                value_acc[0] = T.float32(0.0)
                for j3 in range(16):
                    kv: T.int32 = base + j3
                    value_acc[0] = value_acc[0] + exp_scores[j3] * V[kv, d2]
                acc[d2] = acc[d2] * alpha + value_acc[0]
            m_i[0] = m_new
        for d3 in range(64):
            Out[bx, d3] = acc[d3] / l_i[0]
