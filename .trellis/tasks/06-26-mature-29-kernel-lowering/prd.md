# Mature 29-kernel lowering:IntentIR → GPU,实测 perf ≥ Triton

> 状态:**已定稿**(用户决策:M9 纳入本任务;性能无分级门,compute/memory 同一把尺 实测 ≥)。
> 每内核证据 → [`research/29-kernel-characterization.md`](research/29-kernel-characterization.md)。下一步:M1 开工。

## Goal(目标)

把高性能库的 **29 个**真实 Triton 算子(liger 17 + flash_attn 7 + xformers 5),从原子 IntentIR 出发,经**真正的 C++ MLIR lowering**(`OpRewritePattern` + `OpBuilder` 建 IR,非字符串匹配)下降到 GPU,并在 RTX 5090D(sm_120a)上**实测性能 ≥ 原始 Triton 优化 kernel**。这是把 IntentIR 从"玩具/原型"做成"成熟编译器"的核心工程。

**为什么**:闭合认证语义提升的环——从被优化得面目全非的 kernel 恢复可移植原子 IntentIR,再生成的代码能和库自己的高性能 kernel 同台 PK 且不输。

## 已经为真(地基,本会话已完成)

- **C++ 三分布局**:`lib/Intent`(dialect)/ `lib/Intent/Conversion`(convert-intent-to-linalg)/ `lib/Target`(后端 peephole)。新 lowering pass 落点 = `lib/Intent/Conversion/`。
- **验证过的 lowering 主干**(探针已通):`intent.reduce`(带 `assoc=float_nondeterministic` 许可)+ `map_*` → C++ `OpRewritePattern` 建 `scf.parallel`(行)×`scf.parallel`(特征)+`scf.reduce` → stock MLIR(`-gpu-map-parallel-loops -convert-parallel-loops-to-gpu`)→ `gpu.launch`+`gpu.all_reduce` → warp 蝶式 `shfl` + shared-mem → sm_120a PTX/cubin。**block 协作归约能下到金属且正确**;已 PK 真实 liger rms_norm/softmax bit-exact。
- **dialect 已有算子**:map_unary / map_binary / reduce(assoc) / contraction(subscripts) / loop(scan_axes, tile_hint) / yield。
- **测量纪律**:kernel-only 主指标(两侧预分配、~1s 暖机、interleaved do_bench、中位数);被撤回的 0.77× 教会的是测量纪律,不是"beat 不可能"。

## Scope:29 个 kernel(8 个语义族)

> 每内核原子分解 / 难度 / regime / baseline 可跑性见 [`research/29-kernel-characterization.md`](research/29-kernel-characterization.md)。

| 族 | kernels | perf 性质 |
|---|---|---|
| normalization | liger rms_norm, fused_add_rms_norm, group_norm, layer_norm, poly_norm;flash_attn layer_norm_fwd_1pass;xformers rms_norm, rms_norm_add | memory-bound |
| softmax | liger softmax, sparsemax | memory-bound |
| activation | liger dyt, swiglu, geglu | memory-bound |
| position_embedding | liger rope, qwen2vl_mrope, llama4_rope | memory-bound |
| loss_divergence | liger cross_entropy, kl_div, jsd, tvd | memory-bound |
| linear | flash_attn linear.kernel_fwd | compute-bound |
| attention_forward / splitk | flash_attn _fwd_kernel;xformers _fwd_kernel_splitK, _splitK_reduce_varargs | mixed / compute |
| attention_backward | flash_attn bwd 栈(`_flash_attn_backward`+`_bwd_kernel`+`_bwd_kernel_one_col_block`=1 交付), _bwd_preprocess_do_o_dot;xformers _splitK_reduce_varargs_backward | compute / mixed |

> **去重说明**:flash 反向 3 个入口是同一算法(host wrapper / grid entry / inner body),算 **1 个 lowering 交付物**。故 **29 个 kernel ≈ 27 个交付物**——"29"是算子计数,"27"是交付计数。

## 成熟编译器的硬标准("非玩具"的定义)

1. lowering = C++ `OpRewritePattern`,用 `OpBuilder` 构造真实 IR;**禁止** Python 字符串拼接 MLIR、正则解析 module、per-kernel recognizer。
2. **通用算子驱动**:lower `intent.reduce`/`map_*`/`contraction`/`loop` 一次,各 kernel 作为组合自然下降。判据:**没有任何 per-kernel 的"认出这是 X"分支**。
3. 在 `Passes.td` 注册、`intent-opt` 驱动、可被 lit 测试;复用 MLIR 经验证的转换(gpu-mapping / all_reduce)。

## 性能标准(≥ Triton 的精确语义 —— 已按"Forge 都快过"修正)

**目标:每个 kernel 实测 ≥ 它自己那个库 kernel,且尽力 beat;parity 只在基线已最小流量时作为诚实地板。**

- **"≥" 的对象**:库**自己**那个 kernel,**同算子、同 input contract**,head-to-head。**不**借跨算子融合的便宜(那是已退役的 off-target 方向)。
- **beat 的杠杆 = 调度自由**(这正是"我们 lower 的结构有自己的优化"):① **少搬 HBM 字节**——更激进融合 / read-once / 少落中间量 / 少 pass;② occupancy / 向量化 / launch 几何。**Forge 已实测 1.58–1.75×(rms_norm local_ratio 1.753)→ beat 这些 kernel 可达**。
- **每个"beat"必须点名机制**:省了哪几趟 HBM、补了哪个 occupancy 缺口;没有机制就没有 headline 数字。
- **统一标准,memory-bound 与 compute-bound 同一把尺(无分级门、无放水)**——这是研究编译器、不进真实 LLM 生产,没有交付截止压力,所以 compute-bound 也按实测 ≥ / 尽力 beat 要求,不给"在 `tl.dot` 的 X% 内"软门。
- **parity 作为诚实地板,仅在基线被证明已贴硬件天花板时成立**:① memory-bound = 基线已最小流量(如 liger rms_norm 单 pass read-once,无字节可省;本会话 rms_norm 平手即此);② compute-bound = 基线已把 tensor-core 打满(matmul 利用率贴 roofline)。**两种地板都必须实测论证,不能假设**;基线留 headroom 一律必须 beat。
- **"未达标"不是更软的验收门,是"还没做完"**:当前 fused-flash 慢 ~570–980×(register-spill cliff)→ 这不是 M8 的合格线,而是 M8 **未完成**;M8 done 的前提是消除 spill cliff 并达到 ≥ / 贴顶 parity。
- **诚实门**:kernel-only 中位数、~1s 暖机、interleaved;绝不靠削弱 baseline 或时钟噪声;bit-faithful(rel < 1e-3,通常 ~1e-7,vs fp32 ref + 真实库 kernel)**优先于**性能。

## 里程碑路线 M1 → M9(按"逼出的新 lowering 能力"排序)

证据见 research 文件;每 Mi 的 perf-bar 按上面修正版。

| M | 新 lowering 能力 | kernels | perf-bar |
|---|---|---|---|
| **M1** | 行归约→block-cooperative GPU + broadcast-affine epilogue(**✅ 已交付 `b74b0ab`**:成熟 C++ `convert-intent-row-reduction-to-gpu`,register-tile,通用 DAG 无 per-kernel 分支) | rms_norm, softmax | **实测 1.00× / 0.97×** vs 真实 liger,1507 GB/s(peak HBM),bit-exact rel 4-8e-8 — parity 是诚实天花板(liger 已 read-once) |

> **M1 范围边界(诚实标注)**:当前要求 `N % T == 0`(两个 PK shape 1024/4096 是 pow2,满足);非 pow2 N 时 pass 干净 `notifyMatchFailure` 回退到慢路径。**非 pow2 tail-mask(masked load→reduce identity / -inf lane / predicated store)是 M1 的明确后续**,且该 masked-identity 模式同时是 M4 softmax -inf lane 的模板——并入 M4 前置或 M1.1。验收行的"含非 pow2 tail-mask"挂在此。
| **M2** | 双相关归约(mean&var)+ fp32 累加纪律 + 多输出(RSTD/MEAN)+ masked-lane | layer_norm(liger + flash 1pass) | 查 HBM passes headroom;否则 parity |
| **M3** | 纯 elementwise:in-place 别名、constexpr 分支 DCE、fp32-cast 边界、三平面加性融合 | dyt, geglu, swiglu, rope, qwen2vl_mrope, llama4_rope, tvd, bwd_preprocess | qwen2vl 免临时量 → beat 机会 |
| **M4** | max-shift exp 稳定 + constexpr 公式分派 + **块内 `tl.cumsum` scan(新原语)** | kl_div, jsd, sparsemax | sparsemax 是门 |
| **M5** | 多 pass tiled 归约 + 跨 pass 标量寄存器持久 + residual 写回 | group_norm, poly_norm, fused_add_rms_norm, xformers rms_norm, rms_norm_add | **基线多 pass → beat 机会大** |
| **M6** | online-softmax scan(`scf.for`+iter_args,无 matmul)+ 两阶段 in-place 梯度 + device-sync | cross_entropy | 带宽受限 |
| **M7** | **首个真 tiled matmul**(BLOCK_M×N×K + `tl.dot`→tensor-core)+ 融合 epilogue + autotune(**perf-bar 转折**) | linear.kernel_fwd | 实测 ≥;parity 仅在 tensor-core 已打满时 |
| **M8** | **融合 online-softmax + 双 matmul**(QKᵀ→PV)+ in-flight rescale + split-K + LSE reduce | flash _fwd_kernel, xformers splitK fwd + reduce | 先消 spill cliff,再达实测 ≥(核心赌注) |
| **M9** | 多 matmul 反向(5 交错 contraction)+ DQ 双模(serial vs atomic_add)+ varargs split | flash bwd 栈, xformers splitK bwd | 两种归约模式正确性为硬门 + 实测 ≥ |

> 交付节奏:逐族交付,每 Mi = 一组 PR(新 pattern + lit 回归 + 1 个 vs-真实库 PK 测试)。M1–M6 走已验证的 reduce/map/scan 主干;**M7 是范式转折**(contraction codegen 取代 reduce 路径);M8–M9 是最高峰。

## Acceptance Criteria

- [ ] 27/27 交付物(29 算子)通过 intent → C++ lowering → sm_120a cubin,**零残留 `intent.*` 算子**。
- [ ] 29/29 bit-faithful(rel < 1e-3 vs fp32 ref + 真实库 kernel),含非 pow2 tail-mask。
- [ ] **全 29 算子 kernel-only 实测 ≥**(同一把尺,memory/compute 不分级):beat 须附机制;parity 仅在基线被证明贴硬件天花板时成立并标注(memory=最小流量 / compute=tensor-core 打满);M8 spill cliff 必须先消除。
- [ ] **零 per-kernel Python recognizer / 零字符串拼接 MLIR**;lowering 全在 C++ `OpRewritePattern`;lit + pytest 全绿。
- [ ] 每族有 lit 回归 + 一个 PK 测试(vs 真实库 kernel,在 5090D 实测基线)。

## Risks / Open Questions

- **baseline 可跑性(对照失效风险)**:xformers splitK **FP8 路径**在 sm_120a codegen 不确定 → 限 FP16/BF16,M8 不以 FP8 为门;flash bwd **`debug_barrier` race 绕过**跨版本脆弱 → 基线必须 5090D 实测;**FlagGems scaled/persistent + Triton tutorial softmax** 命中 Triton-3.6 launcher arg-count bug → 用裸 `@triton.jit` 干净基线。某 baseline 跑不了 → PK 退化为 vs fp32 ref + roofline,**显式标注**。
- **三个最难 lowering 问题**:① **online-softmax + 双 matmul 在寄存器预算内不 spill(M8)**——正确性已证(rel 1.1e-7),未解的是 (m,lse,acc) carry **加** QK/PV tile 常驻不 spill(当前 cliff 之源),**项目核心性能赌注**;② **多 matmul 反向 + 双模归约(M9)**——5 交错 contraction + race barrier + varargs,两侧 sm_120a 基线行为未知;③ **constexpr-flag DCE at scale**——未选分支若物化则寄存器压力连 memory-bound parity 都破坏。
- **缺的新原语**:tiled-K `tl.dot`/MMA(M7)、块内 `tl.cumsum`(M4)、`atomic_add`+device-sync(M6/M9)、varargs split-tensor(M8/M9)。
- **contraction→GPU 决策**:复用 MLIR 现成 `linalg.matmul`→gpu tiling 还是自建(类比 reduce 主干)——M7 起评估。

## Out of Scope

- 非 GPU 后端(CPU 已有参照)、多 GPU。
- 训练循环 / autograd 编排(只 lower 单 kernel)。
- 走 Forge 的 OptGuide 搜索(kernel 只作参考/输入)。
- 新的 LLM 提取(提取侧已 real;本任务聚焦 lowering 侧)。

## Definition of Done

27 交付物(29 算子)全部 lower + bit-faithful + **全部实测 ≥**(同一把尺,beat 附机制,parity 仅在贴硬件天花板时并标注);C++ lowering 成熟(无玩具机制);lit+pytest 全绿;每族有回归与 PK;本 PRD acceptance 全勾。

## Technical Notes

- 验证过的探针 / recipe 见 memory(`tianchenir-backend-recipe`, `tianchenir-workspace-layout`)。
- 29 契约:`refs/TianchenForge/artifacts/provider29_optguide/`;源:`Dataset/{Liger-Kernel,flash-attention,xformers}`。
- 既有 C++ lowering 范例:`lib/Intent/Conversion/ConvertIntentToLinalg.cpp`。

## Research References

* [`research/29-kernel-characterization.md`](research/29-kernel-characterization.md) — 30-agent workflow 产出:每内核原子分解 + 难度/regime + baseline 可跑性 + M1→M9 难度排序 + 风险 + 缺失原语。
