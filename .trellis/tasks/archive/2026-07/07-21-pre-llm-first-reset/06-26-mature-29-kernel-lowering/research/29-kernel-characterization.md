# 29-kernel characterization → lowering milestones (sm_120a / RTX 5090D)

来源:`characterize-29-kernels` workflow(30 agents,每内核读 Forge optguide 契约 + Dataset 源)。
锚点能力(已验证 2026-06-26):intent op-graph → linalg → gpu → nvvm → sm_120a cubin,bit-exact;
row-local reduce → block-cooperative GPU + broadcast-affine 已 PK 真实 liger(rms_norm/softmax)。

> **去重**:flash_attn 反向"栈"是同一算法的三个入口——`_flash_attn_backward`(host wrapper)/
> `_bwd_kernel`(grid entry)/ `_bwd_kernel_one_col_block`(inner body)。算 **1 个交付物**(M9)。
> 故 **29 个 kernel ≈ 27 个 lowering 交付物**。

---

## 性能标准(修正版 —— 取代早先"parity 即胜"的过保守表述)

**目标:每个 kernel 实测 ≥ 它自己那个库 kernel(同算子、同 input contract、head-to-head)。**

- **"≥" 的语义**:对比对象 = 库**自己**那个 kernel 在**同一算子、同一输入契约**上;**不**借跨算子融合的便宜(那是已退役的 off-target 方向)。
- **beat 的杠杆 = 调度自由**:主要是**少搬 HBM 字节**(更激进融合 / read-once / 少落中间量),其次是 occupancy / 向量化 / launch 几何。**每一个"beat"必须点名机制**(省了哪些字节、补了哪个 occupancy 缺口)——没有机制就没有 headline 数字。
- **Forge 的 1.58–1.75×(rms_norm local_ratio 1.753 / remote 1.582)是存在性证明**:beat 这些 kernel 可达,主要靠库 kernel 没做满的融合/passes。
- **两种基线状态要分清**:
  - 基线**已在最小流量前沿**(如 liger rms_norm 本身就是单 pass read-once)→ 没有字节可省 → **parity 是诚实结果**,如实标注,不粉饰成"胜"。本会话 rms_norm-vs-liger 平手正是此情形。
  - 基线**留有 headroom**(多 pass / 落中间量 / 几何不佳)→ **必须 beat**(这才是编译器价值)。
- **诚实门**:kernel-only 中位数、~1s sustained-clock 暖机、与基线 interleaved;胜必可解释(更少字节/更高 occupancy),绝不靠削弱基线或时钟噪声。**被撤回的 0.77× 是噪声 artifact —— 这是测量纪律的教训,不是"beat 不可能"的教训**。
- **统一标准(用户定:无分级门)**:这是研究编译器、不进真实 LLM 生产 → compute-bound(M7–M9)也按 memory-bound 同一把尺 **实测 ≥ / 尽力 beat**,不给"在 `tl.dot` 的 X% 内"软门。parity 作为诚实地板仅在基线被证明贴硬件天花板时成立(memory=最小流量 / compute=tensor-core 打满),且必须实测论证。当前 fused-flash 慢 ~570–980×(register-spill cliff)**不是合格线而是"未完成"**:M8 done 的前提 = 消除 spill cliff 并达 ≥。

---

## 1. 每族分组

| 族 | kernels | perf_regime | difficulty |
|---|---|---|---|
| **normalization** | liger rms_norm, fused_add_rms_norm, group_norm, layer_norm, poly_norm;flash_attn layer_norm_fwd_1pass;xformers rms_norm, rms_norm_add | memory-bound | 2/2/3/3/3/3/2/3 |
| **softmax** | liger softmax, sparsemax | memory-bound | 1/1 |
| **activation** | dyt, geglu, swiglu | memory-bound | 1/1/1 |
| **position_embedding** | rope, qwen2vl_mrope, llama4_rope | memory-bound | 2/1/2 |
| **loss_divergence** | cross_entropy, jsd, kl_div, tvd | memory-bound | 3/2/2/1 |
| **linear** | flash_attn linear.kernel_fwd | compute-bound | 4 |
| **attention_fwd / splitk** | flash_attn _fwd_kernel;xformers _fwd_kernel_splitK, _splitK_reduce_varargs | mixed / compute / memory | 5/5/3 |
| **attention_bwd** | flash bwd stack(3 入口=1 交付),_bwd_preprocess_do_o_dot,xformers _splitK_reduce_varargs_backward | compute/mixed/memory | 5/2/3 |

---

## 2. 难度排序里程碑 M1 → M9(按"逼出的新 lowering 能力"排序)

每个 Mi 的 perf-bar 一律按上面修正版:**aim to beat via named mechanism;基线已最小流量时 parity 是诚实结果(标注)**。

- **M1 行归约 + broadcast-affine(锚点,已交付)** — rms_norm, softmax。能力:reduce→`gpu.all_reduce` + broadcast normalize + affine epilogue;read-once 两次归约 softmax(maxnumf/add,非 pow2 tail)。Bar:liger 这两个已 read-once → parity 是诚实结果。
- **M2 双归约统计 + fp32 纪律 + 多输出** — layer_norm(liger + flash_attn 1pass)。能力:mean&var 两相关归约、fp32 累加、masked-lane 不计入统计、RSTD/MEAN 多输出、ε-stable rsqrt。Bar:若基线单 pass 则 parity;查 HBM passes 是否有 headroom。
- **M3 纯 elementwise:in-place 别名 / constexpr DCE / fp32-cast 边界** — dyt, geglu, swiglu, rope×3, tvd, bwd_preprocess(便宜 1-reduce)。能力:in-place(rope 改写 q/k)、未选分支必须 DCE(否则寄存器压力破坏 parity)、fp32 链下转到 gate dtype 的精确边界、qwen2vl 三平面加性融合(免 per-plane 临时量 → 这里有真实 beat 机会)。
- **M4 max-shift exp 稳定 + 多公式分支 + 块内 scan** — kl_div, jsd, sparsemax。能力:max-shift exp 补偿(可复用稳定原语)、constexpr 公式分派、**`tl.cumsum` 块内 scan(新原语,未验证)**;sparsemax 是门。
- **M5 多 pass tiled 归约 + 跨 pass 标量持久 + residual 写回** — group_norm, poly_norm, fused_add_rms_norm, xformers rms_norm×2。能力:特征维 tiled 归约(超一个 block)、stats→normalize 跨 pass 标量寄存器持久、in-place residual 写回。**这里基线常有 headroom(多 pass)→ beat 机会大**。
- **M6 online-softmax scan loop(无 matmul)+ 两阶段 in-place 梯度** — cross_entropy。能力:`scf.for`+iter_args 完整 online max-sum-exp 递推、stats/gradient 两阶段、in-place 梯度别名、`debug_barrier`→真实 device sync、7 个 constexpr flag DCE。Bar:loss 是带宽受限。
- **M7 首个真 tiled matmul + 融合 epilogue(perf-bar 转折)** — linear.kernel_fwd。能力:BLOCK_M×N×K tiled-K + `tl.dot`→Blackwell tensor-core、寄存器常驻累加、严格序 epilogue、autotune。**C++ reduce 路径不再适用 → 需 contraction codegen**。Bar:分级(`tl.dot` 的 X% 内)。
- **M8 融合 online-softmax + matmul(flash fwd)+ split-K + LSE reduce** — flash _fwd_kernel, xformers splitK fwd + reduce。能力:每轮双 contraction(QKᵀ then PV)融合 M6 softmax carry、(m,lse,acc)in-flight rescale、TMP scratch、split-K 分解、两 pass LSE reduce。**这是真正的性能缺口(当前 ~570–980× 慢)——交付 = 消除 register-spill cliff**。
- **M9 多 matmul 反向 + 双模归约 + varargs** — flash bwd 栈(1 交付),xformers splitK bwd。能力:5 个交错 contraction、从 LSE recompute、DQ 双模(serial vs atomic_add 等价)、per-(B,H) 指针重定基、varargs split。Bar:两种归约模式正确性是硬门。

---

## 3. RISKS

**A. 测量阻塞(Triton 基线可能在 sm_120a 跑不了 → 对照失效):**
- xformers `_fwd_kernel_splitK` 的 **FP8 路径**(`tl.float8e4nv`)在 sm_120a codegen 不确定 → 基线**限 FP16/BF16/int4/default**,M8 不以 FP8 变体为门。
- flash bwd 栈的 **`tl.debug_barrier()` race 绕过**跨 Triton 版本脆弱 → **基线必须在 5090D 实测,不信 H100 文档数字**。
- **FlagGems scaled/persistent 变体 + Triton tutorial softmax launcher** 命中 Triton-3.6 launcher 参数数 bug → 用**裸 `@triton.jit` / 干净 inline 基线**,不用 wrapper。
- **orchestration 开销掩盖 kernel 时间**(`run_program` 重解析 ~140ms)→ 一律报 **kernel-only 中位数 + 两侧预分配**。

**B. 三个最难 lowering 问题:**
1. **online-softmax + 双 matmul 融合在寄存器预算内(M8)**:正确性已证(scf.for+iter_args,rel 1.1e-7);未解 = (m,lse,acc) carry **加** QK/PV tile 寄存器常驻不 spill(当前 cliff 之源)。**项目核心性能赌注**。
2. **多 matmul 反向 + 双模(atomic vs serial)归约(M9)**:5 交错 contraction + race barrier + varargs;**两侧 sm_120a 基线行为未知 → 目标随实测移动**。
3. **constexpr-flag DCE at scale**(M6 7 flag;M5 3 cast mode;M3 fp32-cast 边界):未选分支若被物化 → 寄存器压力**连 memory-bound 的 parity 都破坏**;cast 边界错则正确性静默失败。

**验证路径目前缺的新原语**:(i) tiled-K `tl.dot`/MMA contraction(M7);(ii) 块内 `tl.cumsum` scan(M4 sparsemax);(iii) `atomic_add` + `debug_barrier`→device-sync(M6/M9);(iv) varargs split-tensor lowering(M8/M9 xformers)。
