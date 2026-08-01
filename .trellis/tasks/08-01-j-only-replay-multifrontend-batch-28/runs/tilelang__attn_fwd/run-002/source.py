import tilelang.language as T

@T.prim_func
def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, attn_mask_handle: T.handle, sm_scale_handle: T.handle, Out_handle: T.handle):
    Q = T.match_buffer(Q_handle, (1, 1, 16, 16), strides=(256, 256, 16, 1))
    K = T.match_buffer(K_handle, (1, 1, 16, 16), strides=(256, 256, 16, 1))
    V = T.match_buffer(V_handle, (1, 1, 16, 16), strides=(256, 256, 16, 1))
    attn_mask = T.match_buffer(attn_mask_handle, (1, 1, 16, 16), strides=(256, 256, 16, 1))
    sm_scale = T.match_buffer(sm_scale_handle, (1,), strides=(1,))
    Out = T.match_buffer(Out_handle, (1, 1, 16, 16), strides=(256, 256, 16, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 16)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(Q[bx // 16, 0, bx % 16, 0:16], sm_scale[0], K[bx // 16, 0, 0:16, 0:16], attn_mask[bx // 16, 0, bx % 16, 0:16], V[bx // 16, 0, 0:16, 0:16])
        T.writes(Out[bx // 16, 0, bx % 16, 0:16])
        q_row = T.alloc_buffer((16,), scope="local.fragment")
        scores = T.alloc_buffer((16,), scope="local.fragment")
        tmp = T.alloc_buffer((16,), scope="local.fragment")
        acc = T.alloc_buffer((1,), scope="local.fragment")
        mx = T.alloc_buffer((1,), scope="local.fragment")
        exp_scores = T.alloc_buffer((16,), scope="local.fragment")
        sm = T.alloc_buffer((1,), scope="local.fragment")
        probs = T.alloc_buffer((16,), scope="local.fragment")
        out_row = T.alloc_buffer((16,), scope="local.fragment")
        tmp2 = T.alloc_buffer((16,), scope="local.fragment")
        acc2 = T.alloc_buffer((1,), scope="local.fragment")
        pid_q: T.int32 = bx % 16
        pid_z: T.int32 = bx // 16
        for k in range(16):
            q_row[k] = Q[pid_z, 0, pid_q, k]
        scale: T.float32 = sm_scale[0]
        for j in range(16):
            for k in range(16):
                tmp[k] = q_row[k] * K[pid_z, 0, j, k]
            T.reduce(tmp[0:16], acc[0], "sum", 0, T.bool(True))
            scores[j] = (acc[0] + attn_mask[pid_z, 0, pid_q, j]) * scale
        T.reduce(scores[0:16], mx[0], "max", 0, T.bool(True))
        for j in range(16):
            exp_scores[j] = T.exp(scores[j] - mx[0])
        T.reduce(exp_scores[0:16], sm[0], "sum", 0, T.bool(True))
        for j in range(16):
            probs[j] = exp_scores[j] / sm[0]
        for d in range(16):
            for j in range(16):
                tmp2[j] = probs[j] * V[pid_z, 0, j, d]
            T.reduce(tmp2[0:16], acc2[0], "sum", 0, T.bool(True))
            out_row[d] = acc2[0]
        for d in range(16):
            Out[pid_z, 0, pid_q, d] = out_row[d]
