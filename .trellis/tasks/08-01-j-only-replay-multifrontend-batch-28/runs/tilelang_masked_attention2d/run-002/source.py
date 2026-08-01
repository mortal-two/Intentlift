import tilelang.language as T

@T.prim_func
def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, sm_scale_handle: T.handle, Out_handle: T.handle):
    Q = T.match_buffer(Q_handle, (16, 16), strides=(16, 1))
    K = T.match_buffer(K_handle, (16, 16), strides=(16, 1))
    V = T.match_buffer(V_handle, (16, 16), strides=(16, 1))
    sm_scale = T.match_buffer(sm_scale_handle, (1,), strides=(1,))
    Out = T.match_buffer(Out_handle, (16, 16), strides=(16, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 16)
    tx = T.launch_thread("threadIdx.x", 1)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(sm_scale[0], Q[bx, 0:16], K[0:16, 0:16], V[0:16, 0:16])
        T.writes(Out[bx, 0:16])
        scores = T.alloc_buffer((16,), scope="local.fragment")
        dot_acc = T.alloc_buffer((1,), scope="local.fragment")
        mx = T.alloc_buffer((1,), scope="local.fragment")
        exp_scores = T.alloc_buffer((16,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        out_acc = T.alloc_buffer((1,), scope="local.fragment")
        neg: T.float32 = T.float32(-1000000000.0)
        scale: T.float32 = sm_scale[0]
        for j0 in range(16):
            dot_acc[0] = T.float32(0.0)
            for k0 in range(16):
                dot_acc[0] = dot_acc[0] + Q[bx, k0] * K[j0, k0]
            s: T.float32 = dot_acc[0] * scale
            scores[j0] = T.if_then_else(j0 > bx, neg, s)
        mx[0] = neg
        for j1 in range(16):
            mx[0] = T.max(mx[0], scores[j1])
        sm[0] = T.float32(0.0)
        for j2 in range(16):
            exp_scores[j2] = T.exp(scores[j2] - mx[0])
            sm[0] = sm[0] + exp_scores[j2]
        for d0 in range(16):
            out_acc[0] = T.float32(0.0)
            for j3 in range(16):
                prob: T.float32 = exp_scores[j3] / sm[0]
                out_acc[0] = out_acc[0] + prob * V[j3, d0]
            Out[bx, d0] = out_acc[0]
