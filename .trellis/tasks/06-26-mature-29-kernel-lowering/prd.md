# Mature 29-kernel lowering:IntentIR → GPU,实测 perf ≥ Triton

> 状态:**已定稿 + M2–M9 详细执行规格已展开**(本轮:9-agent workflow 逐 family 读真实 liger/flash/xformers 源 + M1 锚点为 C++ 模板,critic 已验 27-覆盖/依赖无环/perf-bar 一致/可执行性,verdict = "可驱动到 finishable DONE")。
> 用户决策:M9 纳入本任务;性能无分级门,compute/memory 同一把尺 实测 ≥。
> 每内核证据 → [`research/29-kernel-characterization.md`](research/29-kernel-characterization.md)。M1 已交付;M2–M9 规格见下方"详细执行规格"。

## Goal(目标)

把高性能库的 **29 个**真实 Triton 算子(liger 17 + flash_attn 7 + xformers 5),从原子 IntentIR 出发,经**真正的 C++ MLIR lowering**(`OpRewritePattern` + `OpBuilder` 建 IR,非字符串匹配)下降到 GPU,并在 RTX 5090D(sm_120a)上**实测性能 ≥ 原始 Triton 优化 kernel**。这是把 IntentIR 从"玩具/原型"做成"成熟编译器"的核心工程。

**为什么**:闭合认证语义提升的环——从被优化得面目全非的 kernel 恢复可移植原子 IntentIR,再生成的代码能和库自己的高性能 kernel 同台 PK 且不输。

## 已经为真(地基)

- **C++ 三分布局**:`lib/Intent`(dialect)/ `lib/Intent/Conversion`(convert-intent-to-linalg)/ `lib/Target`(后端 peephole)。新 lowering pass 落点 = `lib/Intent/Conversion/`。
- **M1 已交付(`b74b0ab`)**:成熟 C++ `convert-intent-row-reduction-to-gpu`(440 行,module-level `Pass<ModuleOp>`,`RowReductionLowering`)。识别直线 map/reduce DAG(args 按 shape 分 Row[M,N]/Col[N]/Scalar[]),原地重写成**一个 read-once block-cooperative `gpu.launch`**:register-tile(静态 K=N/T 全展开,feature tile 持为 K 个 SSA 寄存器,**非** `memref.alloca`)、uniform per-row 标量 hoist 一次、每 reduce 一个 `gpu.all_reduce`(→ `shfl.sync.bfly`+shared-mem)、epilogue 从寄存器写 Y。**通用 DAG 无 per-kernel 分支**:rms_norm / softmax 同代码路径。实测 1.00× / 0.97× vs 真实 liger,1507 GB/s(peak HBM),bit-exact rel 4-8e-8 —— parity 是诚实天花板(liger 已 read-once)。
- **验证过的下游主干**:`gpu.all_reduce`(add/maxnumf)→ sm_120a 正确;`math.exp`→ex2.approx、`math.sqrt/rsqrt`→`llvm.nvvm.{sqrt,rsqrt}.approx.f` peephole;`scf.for`+iter_args 在线 softmax carry(`flash_fused.mlir`,rel 1.1e-7)已跑;`lower_memref_module_to_kernels`(device pipeline,无 bufferize)。
- **⭐ M7 关键 de-risk(本轮 workflow 实测)**:`gpu.subgroup_mma_{load,compute,store}` 在 MLIR 20.1.8 可解析,经现有 `--gpu-lower-to-nvvm-pipeline=cubin-chip=sm_120a` → 真实 `wmma.load/mma.sync.aligned.row.m16n16k16.f32.f32/store` PTX → `ptxas -arch=sm_120a -O3` 出 6232-byte cubin。**Blackwell tensor-core 这条路是通的**(M7→M8→M9-A 整条 compute-bound 链的承重柱)。M7 第一动作必须重新确认此探针未失效。
- **测量纪律**:kernel-only 中位数(两侧预分配、~1s 暖机、interleaved `do_bench`、**计时区内无 host sync**);被撤回的 0.77× 教会的是测量纪律,不是"beat 不可能"。

## Scope:29 个 kernel / 27 个交付物(8 个语义族)

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

### 交付计数口径(已统一,sum=27)

**29 算子 → 27 lowering 交付物**:唯一的合并是 flash 反向栈(3 入口 = 同一算法的 host wrapper / grid entry / inner body)→ **1 交付物**;29 − 2 = 27。一个"交付物"= 一个有独立 PK baseline 的库 kernel(其 backward 与 forward 共享 pass 时并入同一交付物)。**共享 C++ 路径不减少计数**(M2 两个 kernel 共享一条 DAG+pass,仍是 2 个交付物 / 2 个 PK)。

| M | 交付物数(PK baselines) | 备注 |
|---|---|---|
| M1 | 2 | rms_norm, softmax(已交付) |
| M2 | 2 | liger layer_norm + flash 1pass,共享 1 条 DAG/路径 |
| M3 | 8 | dyt, geglu, swiglu, rope, qwen2vl_mrope, llama4_rope, tvd, flash `_bwd_preprocess_do_o_dot` |
| M4 | 3 | kl_div, jsd, sparsemax(fwd+bwd 同交付) |
| M5 | 5 | group_norm, poly_norm, fused_add_rms_norm, xformers rms_norm, rms_norm_add |
| M6 | 1 | cross_entropy |
| M7 | 1(+1 bonus) | linear.kernel_fwd;`kernel_bwd` 是同 pass 落出的额外交付,**不计入 27** |
| M8 | 3 | flash `_fwd_kernel`, xformers `_fwd_kernel_splitK`, `_splitK_reduce_varargs`(非 varargs 版并入) |
| M9 | 2 | flash bwd 栈(3→1), xformers `_splitK_reduce_varargs_backward` |
| **合计** | **27** | M7 bonus `kernel_bwd` 另计 |

## 成熟编译器的硬标准("非玩具"的定义)

1. lowering = C++ `OpRewritePattern`,用 `OpBuilder` 构造真实 IR;**禁止** Python 字符串拼接 MLIR、正则解析 module、per-kernel recognizer。
2. **通用算子驱动**:lower `intent.reduce`/`map_*`/`contraction`/`loop`/`scan`/`select` 一次,各 kernel 作为组合自然下降。判据:**没有任何 per-kernel 的"认出这是 X"分支**(M9-A 的 5-contraction+recompute 结构匹配是 shape/DAG 驱动,非 kernel-name 匹配——见 M9 maturity 注)。
3. 在 `Passes.td` 注册、`intent-opt` 驱动、可被 lit 测试;复用 MLIR 经验证的转换(gpu-mapping / all_reduce / mma)。

## 性能标准(≥ Triton 的精确语义 —— 已按"Forge 都快过"修正)

**目标:每个 kernel 实测 ≥ 它自己那个库 kernel,且尽力 beat;parity 只在基线已最小流量时作为诚实地板。**

- **"≥" 的对象**:库**自己**那个 kernel,**同算子、同 input contract**,head-to-head。**不**借跨算子融合的便宜(已退役的 off-target 方向)。
- **beat 的杠杆 = 调度自由**:① **少搬 HBM 字节**(更激进融合 / read-once / 少落中间量 / 少 pass);② occupancy / 向量化 / launch 几何。**每个"beat"必须点名机制**;没有机制就没有 headline 数字。Forge 实测 1.58–1.75×(rms_norm local_ratio 1.753)→ beat 这些可达。
- **统一标准,memory-bound 与 compute-bound 同一把尺(无分级门、无放水)**——研究编译器、不进真实 LLM 生产,compute-bound(M7–M9)也按实测 ≥ / 尽力 beat,不给"在 `tl.dot` 的 X% 内"软门。
- **parity 作为诚实地板,仅在基线被证明已贴硬件天花板时成立**:memory-bound = 基线已最小流量(read-once);compute-bound = tensor-core 已打满(ncu `sm__pipe_tensor_op_hmma` 贴顶)。**两种地板都必须实测论证,不能假设**;基线留 headroom 一律必须 beat。
- **"未达标"不是更软的验收门,是"还没做完"**:fused-flash 当前慢 ~570–980×(register-spill cliff)→ 这是 M8 **未完成**,不是合格线。
- **诚实门**:kernel-only 中位数、~1s 暖机、interleaved、计时区内无 host sync;bit-faithful(rel < 1e-3,通常 ~1e-7,vs fp32 ref + 真实库 kernel)**优先于**性能。
- **fp32-only IR 的诚实标注**:M1/M2 IR 为 fp32-only;库常跑 fp16/bf16(HBM 字节减半)。为保持 head-to-head 诚实,**两侧都跑 fp32**;"库-真实精度 beat" 需混精度 follow-up(`match()` 扩 fp16/bf16 行输入 + fp32 累加 + cast),明确标注。

## 里程碑总览 M1 → M9(按"逼出的新 lowering 能力"排序)

| M | 新 lowering 能力 | kernels | perf-bar(诚实) |
|---|---|---|---|
| **M1** | 行归约→block-cooperative + broadcast-affine(**✅ `b74b0ab`**) | rms_norm, softmax | **1.00×/0.97×**,1507 GB/s,rel 4-8e-8 — parity 地板 |
| **M2** | 多输出 epilogue(Y + per-row stat)+ 双相关归约 + thread-0 守卫标量写(**✅ 已交付**:扩 M1 pass,`ResultRole RowOut/PerRowOut`,`scf.if(tx==0)` 写 `memref<Mxf32>`,无 per-kernel 分支) | layer_norm(liger + flash 1pass) | **实测 0.98×/0.99×** vs 真实 liger(1535-1578 GB/s),Y/Mean/RSTD rel 3e-8~1.7e-7 — read-once 几何匹配 parity 是诚实地板 |
| **M3** | read-once 纯 elementwise pass:多输出 / in-place 别名 / `intent.select`+`intent.cast` / 索引重映射(transpose/half-split)/ constexpr-DCE-by-construction | dyt, geglu, swiglu, rope×3, tvd, flash bwd_preprocess | qwen2vl 三平面免临时量 / tvd 融合 loss-sum / dyt read-once → beat;swiglu/geglu/rope read-once → parity(≥1500 GB/s 为证) |
| **M4** | 流式 tile 融合行核 + 块内 cumsum scan(`intent.scan`)+ -inf tail-mask + atomicAdd-to-scalar 跨行折叠 | kl_div, jsd, sparsemax | jsd 免 [BT,V] loss 物化 → beat;sparsemax 是 cumsum 门 |
| **M5** | 多 pass TILED 归约(scf.for+iter_args + smem 跨 pass 缓存)+ aux 标量输出 + 2D grid + in-place residual + BroadcastSegment Col | group_norm, poly_norm, fused_add_rms_norm, xformers rms_norm×2 | xf rms_norm 实测两趟 → smem 缓存省一趟 read → beat;poly/fused 单 pass → parity |
| **M6** | 在线 softmax loop(`scf.for`+iter_args,**无 matmul**)+ 两阶段 in-place 梯度 + `gpu.barrier` 真 sync + RowInt 角色 + rank-1 loss | cross_entropy | 省一趟 X-read(融合 pass2 重算)→ beat,寄存器不 spill 为前提 |
| **M7** | **首个 tiled tensor-core matmul**:`intent.contraction`→`gpu.subgroup_mma_*` 片段累加 + scf.for over BK + 融合 epilogue(**perf-bar 转折**) | linear.kernel_fwd | 诚实地板 parity(无 cp.async 多级流水 → 重调 tl.dot 已贴顶);beat = 已命名的 cp.async follow-up |
| **M8** | **融合在线 softmax + 双 contraction**(QKᵀ→PV)+ MMA 累加片段 carry + per-row α 片段重缩放(**消 spill cliff**)+ split-K + LSE reduce | flash _fwd_kernel, xformers splitK fwd + reduce | 先消 spill cliff 达实测 ≥;TMP 免物化 + mask-DCE 为命名 beat 机制 |
| **M9** | 多 matmul 反向(5 交错 contraction + recompute-P-from-LSE)+ DQ 双模(serial/atomic 等价)+ varargs split | flash bwd 栈, xformers splitK bwd | A:库留 5-13× headroom(去 barrier 节流)但需 MMA 不 spill;B:HBM 已贴顶 → parity(Forge remote 1.06) |

---

## ⭐ M2–M9 详细执行规格

> 模板:**交付物 → 输入 IntentIR → 新 lowering 能力(C++ 机制) → 新原语 → 依赖 → perf 故事 → 验收 → 风险**。每条均经 agent 读真实源码 grounding + critic 审。引用行号针对当前 `ConvertIntentRowReductionToGPU.cpp`(M1)。

### M2 — 多输出行归约(Y + per-row Mean/RSTD)

- **交付物(2,共享 1 条 DAG/路径)**:liger `_layer_norm_forward_kernel`(`ops/layer_norm.py:26-86`);flash `_layer_norm_fwd_1pass_kernel`(`flash_attn/ops/triton/layer_norm.py`,base flags `IS_RMS_NORM=false`、无 residual/dropout/x1/w1/rowscale)。**base-flag flash 1pass 提到与 liger 逐 op 相同的 IR DAG**,只 PK baseline 不同(flash autotune num_warps;liger 固定 4)。
- **输入 IntentIR**:两个相关 last-axis reduce + 三输出。`mean=reduce{mean}(X)` → `xc=X−mean` → `xc2=xc*xc` → `var=reduce{mean}(xc2)`(**依赖第一个 reduce 结果——正是 softmax max→sum(exp(x−max)) 的依赖,`lower-softmax-to-gpu.mlir` 已证 M1 能 lower**)→ `rstd=rsqrt(var+eps)` → `Y=(xc*rstd)*W+B`。返回 `Y[M,N]`、`mean[M,1]`、`rstd[M,1]`。**⚠ 校正**:liger(L74)/flash(L263)都算**中心化** `var=Σ(X−mean)²/N`(顺序依赖 2-reduce),**不是** `E[X²]−mean²`;按源码提中心化形。
- **新 lowering 能力(C++)**:**扩展 `RowReductionLowering`,不重构、不新建 pass**。四处外科手术(全在 `match()`/`rewrite()`):(A) 删 `getNumResults()!=1`(L168),改为遍历 `ret.getOperands()`,按 result shape 分 `RowOut`[M,N] / `PerRowOut`[M,1];(B) reduce 分类循环无需改(2-reduce + rstd uniform 链已被接受);(C) signature 建造(L327-331)每个 PerRowOut 追加一个 `memref<Mxf32>`(**rank-1,丢 unit dim**,匹配库 `(n_rows,)` 指针布局,零 PK glue);(D) epilogue(L401-405)RowOut 保持 register-tile 写;PerRowOut 是 per-row uniform 标量,发 `%isLead=cmpi eq tx,c0` + `scf.if %isLead { store uniform → statMem[bx] }`(单 writer / block)。
- **新原语**:**多输出 epilogue + thread-0 守卫的 per-row 标量写**(`scf.if(tx==0) store`)。这是唯一新构造(第二个 reduce、fp32 var/+eps/rsqrt 都被 softmax 测试 + uniform-hoist + peephole 覆盖)。
- **依赖**:M1(同 pass 严格超集)。单输出 rms_norm/softmax 必须不变通过 = 回归门。
- **perf 故事**:memory-bound。两 baseline 都已单 pass read-once → HBM 流量 **parity**;beat 杠杆 = launch 几何。liger `calculate_settings` 对 N≤2048 硬编码 num_warps=4(K=N/128 大寄存器 tile、低 occupancy),我方 `blockThreads` 选最小 K 抬 occupancy 填这个缺口(Forge local_ratio 1.561),目标 N≤2048 上 1.3-1.5×;flash 已 autotune num_warps(Forge 1.007)→ **parity 诚实地板,不冒充 beat**。
- **验收**:三输出全 rel<1e-3(~1e-7)vs fp32 ref(N∈{1024,4096,768},T 整除 N);lit `lower-layer-norm-to-gpu.mlir`(3 返回 + thread-0 守卫 CHECK);M1 单输出 lit 不变;PK vs liger ≥1.0×(几何杠杆目标 1.3-1.5×,机制命名否则数字作废);PK vs flash ≥1.0×(parity)。
- **风险**:PerRow rank 归一([M,1]→[M] 索引);tx/bx 不能写反(否则 Mean 静默坏);fp32-only baseline(混精度是 follow-up);几何 beat 须实测(若 num_warps=4 已近优则 parity 是诚实结果)。

### M3 — read-once 纯 elementwise pass(`convert-intent-elementwise-to-gpu`)

- **交付物(8)**:dyt、geglu、swiglu、rope、qwen2vl_mrope、llama4_rope、tvd(liger `ops/*.py`);flash `_bwd_preprocess_do_o_dot`。后者是 `delta=Σ(o*do,axis1)` 的**行归约 [M] 输出**——**非 M3 新机制**,而是 **M1 reduce pass 放松到 rank-1 [M]/[M,1] 标量-per-row 输出**(~30 行,不新建 pass)。
- **输入 IntentIR**:全是 `{map_unary, map_binary, select, cast, transpose, const}` 直线 DAG,无 loop、(除 tvd/preprocess)无 reduce。dyt=`tanh(α*x)*γ(+β)`;geglu/swiglu=`gate(a).cast(b.dtype)*b`;rope=transpose+half-split,`Y1=q1*cos−q2*sin, Y2=q2*cos+q1*sin`,q/k in-place;qwen2vl=rope + `cos_row=Σ select(plane)`;llama4=even/odd 复数乘,`imag_sign` 是 Scalar(选 fwd/bwd 无分支);tvd=`grad=select(p>q,+s,−s)`(elementwise)+ `loss=Σ(0.5*|p−q|)`(行 sum)。
- **新 lowering 能力(C++)**:**新建 `Pass<ModuleOp> convert-intent-elementwise-to-gpu`**(`ConvertIntentElementwiseToGPU.cpp`),pipeline 中跑在 M1 之前(elementwise 在此 match,带 reduce 的 fall through 到 M1)。把 M1 原语重构进共享头 `Intent/Conversion/MapEmit.h`(`emitScalarUnary/Binary`、Role 分类、register-tile read-once load、hoistUniforms、`col[k]=tx+k*T`)。新增:(a) **多输出**(每 result 一个 destination-passing memref + 一条 register-tile 写循环);(b) **in-place 别名**(func attr `intent.inplace`,输出 memref 复用输入 arg;read-once 保证所有 load 在任何 store 前);(c) **`emitScalarSelect`**(`arith.select` over `arith.cmpf`)+ **真 cast `emitScalarCast`**(`trunc/extf`,dtype-faithful);(d) **transpose/half-split = 索引重映射**(emitAt 加 per-Row index-transform 闭包,如第二半 load `col[k]+hd/2`);(e) tvd 的行 reduce 在同 launch 内复用 M1 `gpu.all_reduce`。
- **新原语**:多输出 destination-passing、in-place 别名、`intent.select`(+ `intent.cast`)新方言面、constexpr-flag **DCE-by-construction**(未取分支在 lift 时即不在 DAG 中,寄存器永不背死分支)。
- **依赖**:M1(复用 register-tile 机制 + 下游 device leg)。**`intent.select`/`intent.cast` 由 M3 独家拥有**(加 `IntentOps.td` 验证器 + `emit_mlir` printer);M4 只加 `intent.scan`,M6 复用——见 dialect-op 归属。
- **perf 故事**:全 memory-bound。命名 beat:qwen2vl(源 load 6 个 masked tile 再求和;我方 select+add 融合 3 平面免 per-plane 临时量,cos/sin read-once,目标 ≥1.3×)、tvd(库另起 host `.sum()` pass;融合行 sum 省一 launch,≥1.2×)、dyt(库 per-row reload α/γ;read-once + hoist α,≥1.2×)。诚实 parity:swiglu/geglu/rope/llama4/preprocess 已 read-once/write-once。
- **验收(perf bar 已收紧,删"parity slack 1.5x")**:qwen2vl ≥1.3× / tvd ≥1.2× / dyt ≥1.2×(各命名机制);swiglu/geglu/rope/llama4/preprocess **实测 ≥1.00× 且实测吞吐 ≥ ~1500 GB/s peak HBM 的既定百分比**(饱和证明,非假设 tie)。correctness rel<1e-3 优先;zero residual intent.*;rope CHECK 写目标 memref == 输入 arg(in-place);PK clone in-place q/k 在计时区外。
- **风险**:`intent.select`/cast 新方言面(verifier);tvd 需 `cmp/gt` 布尔 fn(map_binary 当前无 → 加 cmp→i1 feed select);llama4 strided(d*2/d*2+1)load(建模成两个 logical Row 输入避免 emitAt 复杂化);rope 非 pow2 head 需 tail-mask(M1 follow-up,共享);多输出 host launch-plan 解析(`row_reduction.py` 当前假设单 Y)。

### M4 — 流式 tile 融合行核 + 块内 cumsum scan

- **交付物(3)**:kl_div(batchmean/sum/mean/log_target 变体折叠)、jsd(beta=0/1/generic + HAS_LABEL 折叠)、sparsemax(fwd **是门** + bwd)。**注**:kl_div reduction=none 与 backward 是纯 elementwise → 归 M3 族,非 M4。
- **输入 IntentIR**:[M=BT,N=V] 行,新增 `select`、`scan` 节点 + 外层流式 tile。kl_div=`mul(y_true, log(max(y_true,eps))−y)` → `Σ` → 跨行 `Σ`→标量;jsd=max-shift exp 稳定 + 公式分派;sparsemax fwd=`select(mask,z_sorted,0)`→`scan{cumsum}`→`t=(cssv−1)/r`→`support=cmp(z>t)&mask`→`k=Σcast(support)`、`s=Σselect(support,z,0)`→`tau=(s−1)/max(k,1)`→`y=max(x−tau,0)`;sparsemax bwd=两 pass reduce+select(无 scan)。
- **新 lowering 能力(C++)**:扩展 M1 pass(更名 `convert-intent-row-fused-to-gpu`),四块真 OpBuilder 基建:(A) 新 IR op **`intent.scan`**(`fn,axis,init`;`intent.loop` 只 yield 末值 → 不能表达 cumsum 逐元素输出 → scan 是独立 op);emitAt 加 Select 臂 + scan-result 臂;(B) match 识别 map/select/reduce/scan 为 DAG 节点;(C) **流式 tile 外循环(工程核心)**:因库 `for i in range(0,n_cols,BLOCK)`(V 可达 128k),把单一全展开 K=N/T 换成 `scf.for over ceil(N/BLOCK)`,iter_args 携 per-reduce 标量累加器 + scan running-prefix,块内只 register-tile `K_tile=BLOCK/T` 列,循环后 `gpu.all_reduce` 一次 —— **退掉 M8 式 spill cliff**;(D) **-inf tail-mask**(非 pow2 V:per-(tile,k) i1 谓词 `col<N`,combine 前 `arith.select` 注入 reduce identity,epilogue `scf.if` 谓词写);(E) **atomicAdd-to-scalar**(`memref.atomic_rmw addf` 进 1-元素 memref,消库的 torch `.sum()` launch)。
- **新原语**:**块内 cumsum scan(`intent.scan`)→ sm_120a 两级 Kogge-Stone**(per-thread 顺序前缀 → warp 内 `gpu.shuffle` inclusive scan(5 步)→ warp 间 shared-mem alloca + `gpu.barrier` + 一 warp scan warp 总和 → 各 thread 加 warp 前缀)——**未验证,GATE 0 门控**。配套:-inf masked-lane、`memref.atomic_rmw addf`。
- **依赖**:M1(扩展同 pass)。`intent.scan` 是 M4 唯一新 dialect op;`select`/`cast` 复用 M3。
- **perf 故事**:jsd headroom 最大(库物化全 [BT,V] loss 再 torch `Σ` 重读;融合块内 reduce + atomic 到标量,**永不写/重读 [BT,V] loss** → loss 路 ~2× 带宽,净 ~1.5-1.8×,实测);kl_div(atomicAdd 消两 launch+sync → ~1.2-1.7× 来自 launch 融合);sparsemax fwd(免物化 support bool、k-count 与 s-sum 融一 pass、cssv 留寄存器;~1.3-2×,**若 shfl scan 慢于 `tl.cumsum` 则 parity 诚实地板**);sparsemax bwd 两 pass read-once → parity。**不引 Forge 端到端 ratio(含 torch sum + autograd)**。
- **验收**:**GATE 0**(先于任何 sparsemax IR):独立 microbench 块-cumsum(两级 shfl)经 device pipeline 出 cubin,bit-exact vs 参考前缀和 + `gpu.shuffle` 无 ptxas 错;失败 → sparsemax fwd 降级 vs-fp32-ref+roofline 并标注(kl_div/jsd 不受影响)。correctness 各 rel<1e-3 vs 真实 liger;-inf tail 正确(V=4097/50257);大 V(≥32768)PTX 无 local spill(证外循环替代 unroll-K);zero residual;M1 lit 经更名 pass 不变。
- **风险**:cumsum on sm_120a 是项目级未知(shfl subgroup scan + 两级 shared-mem alloca 的 bare-ptr Blackwell lowering 未测);流式外循环是 M1 全展开体的结构性重写;`intent.scan` 新 op 扩 blast radius;atomicAdd float-nondeterministic(大 BT*V 可能超 1e-3 → 退确定性两阶段);sparsemax `k=Σcast(bool)` 需 bool→f32 reduce 路。

### M5 — 多 pass TILED 行归一(`convert-intent-row-norm-to-gpu`)

- **交付物(5)**:xformers `_rms_norm_kernel`、`_rms_norm_add_kernel`(`ops/_triton/rmsnorm_kernels.py`);liger group_norm、poly_norm、fused_add_rms_norm。后三个里 **poly_norm/fused_add 走 num_tiles==1 = M1 快路径**,只需多输出/aux ABI。
- **输入 IntentIR**:全是 [M,N] map/reduce DAG(同 M1 族,加 tiling)。xf rms_norm=`msq=mean(X*X)`→`rstd=rsqrt(msq+eps)`→`Y=X*rstd*W`;xf rms_norm_add=先 `A=X+resid` 再 RMS,**两输出(A in-place 到 X,h1)**;group_norm=`s=ΣX`、`sq=Σ(X*X)`→`var=sq/N−(s/N)²`→Y,affine W/B 是 BroadcastSegment Col(`idx=col//HPS+group*cpg`),aux Mean/RSTD;poly_norm=一 register tile 出 x²/x⁴/x⁶ 三 reduce,aux RSTD[M,3];fused_add=`S=X+R`→RMS,两输出(S,Y)。**均无 `loop` op**(多 pass 是 schedule tiling,非 loop-carried)。
- **新 lowering 能力(C++)**:**扩展 M1(更名 `RowNormLowering`,M1/M5 同一 recognizer+scheduler,不新建 pass)**。核心:把单一 `(T,K=N/T)` 换成 `(T,K_per_tile,num_tiles)`,`N=T*K_per_tile*num_tiles`,K_per_tile 选 ≤~64 f32(寄存器预算)。M1 是 num_tiles==1 特例(行为零变,M1 lit 不变)。五处:(A) **tiled 归约**(num_tiles>1 时每 reduce = `scf.for k_tile iter_args(acc)`,`gpu.all_reduce` 留**循环外**)+ **smem 跨 pass 缓存**(workgroup `memref.alloca[N]` 持行,normalize 不重读 HBM;N*4≤smem(~N≤25K)则缓存,否则退 HBM 重读并标注 parity);(B) 多输出;(C) **aux 标量输出**(`Role::Aux`,`scf.if(tx==0) store` 到 [M]/[M,k]);(D) **2D grid**([B,G,H] → gridX=B,gridY=G,只 reduce 末轴);(E) **BroadcastSegment Col**(attr `intent.col_index_map = affine_map`,emitAt 发 floordiv+add 索引——结构化 data-independent gather,无新 gather 原语)。
- **新原语**:**tiled cooperative 归约 + smem 跨 pass 缓存**(scf.for+iter_args 块条带喂一个循环后 all_reduce,workgroup alloca 跨 stats/normalize 持行);**aux 标量谓词写**。配套 ABI:多输出 + in-place 别名。
- **依赖**:M1。与 M4(online-softmax loop)**不相交**(M5 是 associative-tiled 非 loop-carried)→ 可任意先后。
- **perf 故事**:beat:xf rms_norm/add **源是显式两 pass loop**(pass1 load X 算平方和、pass2 重读 X 写归一)→ 8K<N≤~25K(smem-fit)我方 smem 缓存把 2 次 X read 变 1(Forge 1.21/1.18),rms_norm_add 缓存 post-add A 省第二读;group_norm(Forge 1.354)= smem 缓存 X + hoist 组 W/B(只 cpg 个值)。parity 地板(实测):poly_norm(Forge 1.723)/fused_add(1.559)已单 pass read-once,**Forge 那些 >1.5 来自被禁的跨算子融合,我们守 head-to-head 不冒认**。
- **验收**:5 个 rel<1e-3 vs fp32+真实库(≥2 尺寸,一单块一多 pass);M1 lit 不变(num_tiles==1 无 scf.for/无 smem);新 lit(tiled scf.for+workgroup alloca+循环外 all_reduce;多输出;aux scf.if;2D grid gridY>1;in-place;BroadcastSegment floordiv);PERF:xf rms_norm/add ≥1.10×(N∈8192..24576,省 HBM pass),group_norm ≥1.05×(典型 hidden,W/B-hoist)/≥1.10×(hidden>65536,X-recache),poly/fused ≥1.00×(单 pass parity 明示);smem-fit 在真机失败 → 退 roofline 标注。
- **风险**:group_norm W/B 的 affine gather 提取(15-kernel corpus 无 group_norm,此 lift 未证 → 不行则 M5.1 出 4/5);aux 输出 ABI 在 M1 未测;in-place 副作用须在 lift 保留(否则 head-to-head 漏写假过);真机 smem 预算未证;n_cols>65536 时库 `calculate_settings` RAISE(无 baseline → 退 fp32-ref)。

### M6 — 在线 softmax loop(无 matmul)+ 两阶段 in-place 梯度

- **交付物(1)**:liger `cross_entropy`(`ops/cross_entropy.py`,fwd loss + in-place grad)。
- **输入 IntentIR**:一 func,args `X[M,N]`(Row,双用:logits + in-place grad dest)、`Y[M]`(**RowInt 整数目标,新角色**)、scalars。两输出:`loss[M]`(rank-1)+ `grad[M,N]`(别名 X)。DAG:`ori_x_y=X[row,Y[row]]`(表为 `Σ select(col==Y,X,0)`,无新 gather)→ **loop**{ scan(X,axis=1),init(m=NEG_INF,d=ZERO),body:`m_new=max(m,blockmax)`,`d=d*exp(m−m_new)+Σexp(Xtile−m_new)`,yield } → `lse=m+log(d)` → `loss=lse−ori_x_y(/n)` → grad epilogue `p=exp(X−m)/d`、`g=p+z−eps`、`select(col==Y,g−(1−ls),g)/n`,写回 X。
- **新 lowering 能力(C++)**:**新建 sibling pass `convert-intent-loop-reduction-to-gpu`**(`LoopReductionLowering`,仿 M1)。match:body 含一 `intent.loop`(scan 末轴 + 标量 f32 carry),外围 map/reduce/select 产 rank-1 loss + in-place [M,N] grad;Role 加 **RowInt**(Y[M] 整数)。rewrite:一 `gpu.launch`、一 block/行(grid=M)、T 线程分 N —— **几何同 M1**,但块内归约不再是单一全展开 all_reduce,而是 launch body 内 **`scf.for %i=0..K iter_args(%m=-inf,%d=0)`**:每 trip load `col=tx+i*T`、`gpu.all_reduce maxnumf` 得块 max、`m_new`、第二个 `gpu.all_reduce add` of `exp(x−m_new)`、`d_new=d*exp(m−m_new)+sum`、yield。循环后 `lse=m+log(d)`(uniform,hoist)。grad epilogue 复用 M1 emitAt 重算 `exp(X−m)/d` + `select(col==Y)`(`emitScalarSelect`→`cmpi eq` of col-idx vs `Y[bx]`),写 X memref(in-place 别名)。grad 写与 loss 写之间发 **`gpu.barrier()`**(= debug_barrier 真 sync)。
- **新原语**:**在线 softmax scan 作 `scf.for`+iter_args 携标量(m,d)跨列 tile,`gpu.all_reduce` 嵌每 trip —— loop-carried 递推但 body 无 contraction(M8 前驱)**。`flash_fused.mlir` 已证 scf.for+iter_args+ex2.approx 在 5090D(rel 1.1e-7)。配套:**RowInt 角色 + rank-1 loss 输出**(`select`/`cast` 复用 M3);`gpu.barrier`。
- **依赖**:M1(复用 emitAt/hoistUniforms/all_reduce/peephole/ABI)+ `intent.loop` 已存在已验证。**唯一新机制 = scf.for+iter_args carry + 循环内嵌 all_reduce(未测接缝)+ RowInt + rank-1 loss + barrier**。
- **perf 故事**:带宽受限。库读 X **两次**(pass1 stats、pass2 grad 重算)+ 写 X 一次。命名 beat:**read-once register-tile 把两次 X read 折成一次**(pass1 tile 留 SSA 寄存器,pass2 从寄存器读 `exp(X−m)/d` 非重 load HBM)→ 省一整 [M,N] read ≈ 1/3 HBM(3 pass→2)→ 目标 ≥1.3×。**诚实 caveat/地板**:liger BLOCK_SIZE 至 16384、num_warps=32;大 V 时 K=V/T 寄存器 tile 有 M8 式 spill cliff,触界则 pass2 退 HBM 重读 beat 降 parity —— 寄存器预算交叉点须实测。
- **验收**:loss+in-place grad bit-faithful vs fp32 ref + 真实 liger(reduction=mean,ls∈{0,0.1},lse_square_scale∈{0,1e-4},ignore_index 路精确);perf 实测 ≥ liger(>1× 须归因省 X-read,spill 交叉 V 实测报告);zero residual(gpu.launch+scf.for(iter_args)+all_reduce+barrier,无 intent.*);M1 lit + `lower-loop.mlir` 不变;PK `test_cross_entropy_pk.py` + lit `lower-cross-entropy-to-gpu.mlir`。
- **风险**:`select`/`cast` 须已由 M3 加(M6 不重复加);RowInt 整数 load + `cmpi eq` 穿过 f32-only emitAt;两输出混 rank + in-place 别名约定;大 V spill cliff;**`gpu.all_reduce` 嵌 scf.for 是未测接缝**(M1 只证顶层 all_reduce,flash_fused 证 scf.for 但内无 all_reduce)→ G2 门;7 constexpr flag 仅 scoped 配置入验收门(weighted/softcap 是 follow-up)。

### M7 — 首个 tiled tensor-core matmul(`convert-intent-contraction-to-gpu`,perf-bar 转折)

- **交付物(1 + 1 bonus)**:flash `linear.kernel_fwd`(`flash_attn/ops/triton/linear.py`,`activation(A@Wᵀ+bias)`);**bonus** `kernel_bwd`(同 pass DAG-shape 落出:contraction + `mul` against act_grad,**非新 pass**,不计入 27)。
- **输入 IntentIR**:直线 DAG,[M,N] fp32 累加。`const ZERO` → `contraction{"mk,nk->mn"}(A[M,K],W[N,K])` → 可选 `add bias[N]`(右对齐广播)→ 可选激活(squared_relu=`select(x≥0,x,0)`+`mul`;gelu_approx=tanh 形,全在 vocab)→ Y。**无 `loop`**(tiled-K 是 Layer-C schedule,contraction 是原子 op)。gelu-exact 需 erf(不在 vocab)→ 出范围。
- **新 lowering 能力(C++)**:**新建 `Pass<ModuleOp> convert-intent-contraction-to-gpu`**,跑在 convert-intent-to-linalg 与 row-reduction 之前,结构仿 `RowReductionLowering`。match:body 恰一 `intent.contraction`(rank-2 静态 f32/f16 → [M,N])+ 可选 pointwise epilogue 链;args 按 shape 分 `Lhs[M,K]/Rhs[N,K]/Bias[N]/RowAux[M,N]/Scalar`。rewrite:bare-ptr signature(destination-passing Y[+ 可选 pre-act Y2);grid=`(cdiv(M,BM)*cdiv(N,BN),1,1)` flatten 解码 (pid_m,pid_n);**初始钉 BM=64,BN=64,BK=16 = 4×4 个 m16n16k16 WMMA tile,4 warp**(後升 Options)。每 block:fp32 累加器 = 寄存器常驻 `gpu.mma_matrix<16x16xf32,"COp">`(16 tile,`gpu.subgroup_mma_constant_matrix 0`);`scf.for k=0..K step BK iter_args(16 COp)`,body `gpu.subgroup_mma_load_matrix`(从 workgroup `memref.alloca` staging,协作 coalesced copy from global)+ `gpu.subgroup_mma_compute`;epilogue(严格序,COp 片段)bias-add + 激活(`gpu.subgroup_mma_elementwise` 或 extract→arith→repack,复用 `emitScalar*` + 新 select/relu)+ `gpu.subgroup_mma_store_matrix` → Y。输出经 `lower_memref_module_to_kernels`(同 row-reduce device leg,无 bufferize)。**显式弃用** `ConvertIntentToLinalg` 的 `linalg.generic` 标量-MAC(留作 CPU/oracle leg)。
- **新原语**:**tiled-K tensor-core matmul**(`intent.contraction` → 寄存器常驻 `gpu.mma_matrix` COp 片段 + scf.for over BK + `gpu.subgroup_mma_*`)。**⭐ 本环境实测**:MLIR 20.1.8 可解析,经 device pipeline → `wmma.load/mma.sync.aligned.row.m16n16k16.f32.f32/store` PTX → ptxas → 6232-byte sm_120a cubin。wmma f16→f32 == tl.dot fp16→fp32。配套:协作 shared-mem staging copy。
- **依赖**:M1 的 **pass 架构**(Pass<ModuleOp> + shape-match + 单 launch rewrite + bare-ptr ABI + erase-old-func)+ device leg(`lower_memref_module_to_kernels` + math peephole)。**不**依赖 `ContractionLowering`(linalg leg 留作 CPU/oracle)。
- **perf 故事**:compute-bound。**诚实姿态 = parity 是地板,非 beat**,且 parity 仍须实测 baseline。tl.dot baseline 重调(autotune BLOCK + num_stages=3-6 cp.async 软流水 + GROUP_M=8 L2 swizzle);一版 plain scf.for WMMA **无 cp.async 多级流水、无 L2 swizzle**,K-bound 形上很可能输 → **不冒认 beat**。验收要求 **ncu 测 tensor-core 利用率**:baseline 贴顶 → parity 是证明地板;有 headroom(Forge 1.084,低置信,**不作 headline**)→ gap 可达但**仅靠已命名 follow-up**(cp.async 多级流水化 K 循环),M7 之後另立。**M7 headline = "tensor core 到达 + bit-faithful + 测得 baseline 利用率下的 parity"**。
- **验收**:rel<1e-3 vs fp32 ref + 真实 kernel(id/gelu_approx/squared_relu × bias on/off);perf kernel-only ≥ tl.dot(≥3 形:512³ / 4096³ / K-bound 1024²×8192,fp16+bf16);**baseline 饱和证明**(ncu `sm__pipe_tensor_op_hmma`,parity 仅在贴顶形被接受,有 headroom 的形归因 deferred cp.async,不静默放过);zero residual;lit `lower-linear-to-gpu.mlir`(gpu.launch+`subgroup_mma_compute`+store,CHECK-NOT 标量 scf.for/linalg.generic),`lower-contraction.mlir`(linalg leg)不变。
- **风险**:**无 cp.async → K-bound 输 tl.dot**(parity 是诚实地板的原因,流水是命名 follow-up);tail-mask(M%BM/N%BN/K%BK==0,非整除退 linalg leg,共享 M1 follow-up);gelu-exact 需 erf 出范围;SAVE_ACT_INPUT 两输出;L2 swizzle 未做;WMMA 片段 epilogue 机制(`subgroup_mma_elementwise` fn 覆盖待 impl 验,否则 extract+repack 多一次 shared 往返)。

### M8 — 融合在线 softmax + 双 contraction(`convert-intent-loop-contraction-to-gpu`,消 spill cliff)

- **交付物(3)**:flash `_fwd_kernel`(非 causal/no-bias EVEN_* 先)、xformers `_fwd_kernel_splitK`(FP16/BF16,PACKED_PER_VAL==1;INT4/FP8 出范围)、`_splitK_reduce_varargs`(`_splitK_reduce` 非 varargs 经 packed-tensor 归一并入)。替换 `flash_emit.py` 的 one-thread-per-query 标量发射。
- **输入 IntentIR**:融合 flash fwd = `intent.loop` scan(K,V),init(m=NEG_INF,l=ZERO,Oacc=zeros),body:`S=contraction{qd,kd->qk}`、`m_new=max(m,reduce max S)`、`P=exp(S−m_new)`、`alpha=exp(m−m_new)`、`l_new=l*alpha+ΣP`、`PV=contraction{qk,kd->qd}(P,V)`、`O_new=Oacc*alpha+PV`、yield;epilogue `O=Oacc/l`(split-K 另写 `LSE=m+log(l)`)。LSE-merge reduce = `intent.loop` scan over split-k(body 无 contraction)。只用 loop/contraction/reduce{max,sum}/map。
- **新 lowering 能力(C++)**:**新建 `Pass<ModuleOp> convert-intent-loop-contraction-to-gpu`**(`FusedAttnLoopLowering`,跑在 convert-intent-to-linalg 前),单 OpRewritePattern,**通用 over loop body DAG,无 looksLikeFlash 分支**。match:body 一 `intent.loop` + return epilogue;carry 按 shape 分 {行标量 [Sq,1]=m,l} + {累加 tile [Sq,D]=Oacc};body 直线图恰两 contraction。rewrite:一 `gpu.launch`,**消 cliff 的协作 schedule**:grid=`(cdiv(Sq,BLOCK_M),B*H[,split_k])`,block 大小让 BLOCK_M×D 累加器入寄存器/片段预算。OpBuilder IR:(1) Q[BLOCK_M,D] **一次** load 进 smem → ldmatrix 进 MMA 片段,跨循环常驻;(2) `scf.for kt=0..Sk step BLOCK_N iter_args = Oacc 的 MMA 累加片段 + warp-uniform m,l`(iter_args 携片段/vector SSA,**非 memref.alloca**——M1 寄存器教训用于片段);(3) 每 tile:cp.async load Kj,Vj 进 smem,QKᵀ via M7 MMA → S 片段,行 reduce-max(`gpu.all_reduce`/shfl over 片段 lane 布局)→ m_new、alpha、P 片段、reduce-sum、l 更新,**Oacc 片段先 *=alpha 再 += PV(M7 MMA)**;(4) epilogue `Oacc/l` 存(split-K 另写 LSE)。reduce 复用 M1 `combineScalar`+`AllReduceOp`,map 复用 `emitScalar*`。
- **新原语**:**协作 MMA 累加片段 carry + per-row α 片段重缩放**——scf.for iter_args 的 carry 是 MMA 累加 FRAGMENT(acc[BLOCK_M,D] 分布在 warp lane),每 tile `acc *= alpha[m]`(per-row 标量按 MMA lane 布局广播)再 += PV 片段。这是 M8 真正新 codegen(区别 M7 单次累加;M8 跨 trip 携累加器并 in-flight 按归约导出标量重缩放)。lowers:QKᵀ/PV 用 `nvgpu.mma.sync`/`nvvm.wmma` 片段(承自 M7);per-row max/sum via `gpu.all_reduce` over 行-lane 组 → shfl.bfly;α 广播 = 片段-elementwise 乘;exp→ex2.approx peephole。split-K = 第三 grid 维;LSE-merge 复用 M1 all_reduce 无 MMA。
- **依赖**:**硬依赖 M7**(body 内调 M7 contraction→MMA 两次:QKᵀ、PV)。本环境已证 nvgpu→nvvm 路 + `--gpu-lower-to-nvvm-pipeline` 可下 mma;**今天 lib/ 无任何 MMA codegen**(只 linalg 标量-MAC)→ M8 不能在 M7 前 ship。+ M1(all_reduce/emitScalar)+ M6(scf.for+iter_args 在线 softmax,rel 1.1e-7)。**若 M7 只出 memory-bound contraction(无 MMA 片段),M8 整个 schedule 变(无片段可重缩放)= gating 问题**。
- **perf 故事**:mixed → compute-bound。当前**不是地板是未完成**:`flash_emit.py` 一 block/query、一 thread/query(BLOCK_M=1),单线程持全 D 累加器 + 内层 QK/PV 标量积为 iter_args → spill cliff(~570-980×)。命名机制 #1(cliff 修复,非库 beat):标量单线程 carry 换协作 MMA 片段 carry,BLOCK_M×D 累加器经 tensor core 分布 warp,per-thread 寄存器降到 spill 阈下 → 达实测 ≥。命名 HBM 机制 #2(诚实 beat,Forge `_fwd_kernel` 1.713):flash 源有 TMP scratchpad 往返("BUG: have to store and immediately load",每 K-tile 2 次 rescale 向量 HBM 往返)→ 我方 alpha/carry 留 iter_args/片段**永不物化 TMP**;+ 静态形 DCE 掉 EVEN_* mask 材料。split-K(Forge 5.442)/reduce(6.729)signal 更大 headroom(split 中间量写/重读 HBM)→ 命名 beat = TMP 消除 + mask-DCE,**非无据倍数**。诚实地板:M7 MMA 落在 cuBLAS 顶下时 M8 继承该 compute 天花板。
- **验收**:rel<1e-3 vs fp32 ref + 真实 Triton(保 1.1e-7 fp32-ref 地板);perf kernel-only median ≥ baseline,head-to-head;**spill cliff 消除**(ptxas -v/ncu 测寄存器/occupancy 不 spill,~570-980× 变 ≥);zero residual;lit `lower-flash-fused-to-gpu.mlir`(gpu.launch+scf.for iter_args+mma op,CHECK-NOT intent.loop/contraction),`flash_fused.mlir` 不变;PK 3 交付 head-to-head(FP8/INT4 排除标注);tile-invariance(变 BLOCK_N 同 Layer-A 输出)。
- **风险**:**MMA 片段重缩放是最难未知**(per-row alpha 跨 `nvgpu.mma.sync` 片段非连续 lane 布局广播 → 可能需 `vector.shuffle`/`warp_execute_on_lane_0`,lane-to-row 映射未完全钉);sm_120a mma.sync shape/ptxas codegen 须 M7 先证;m/l 存储(shfl-uniform vs smem)寄存器预算须实测;causal mask 在 BLOCK_M tiling(先 spec 非 causal EVEN_*,causal+bias 是 constexpr-DCE follow-up);varargs(packed-[split_k,...] 归一,**前端须 stamp,见 M9-B 前端前置**);Sk%BLOCK_N tail(M1 follow-up);LSE 用 exp2/log2 + log2e 因子,自然底 IR 须对齐 bit-faithful。

### M9 — 多 matmul 反向 + varargs split reduce 反向

- **交付物(2)**:flash bwd 栈(`_flash_attn_backward`+`_bwd_kernel`+`_bwd_kernel_one_col_block`+`_bwd_preprocess_do_o_dot`,4 源 fn → 1 交付;delta-preprocess 是 M1 行归约 2nd entry);xformers `_splitK_reduce_varargs_backward`。
- **输入 IntentIR**:A(flash bwd)= 一 `intent.loop` 扫 query 轴,carry dk,dv(init zeros);body **recompute-P-from-LSE 后 5 contraction**:`S=qk`、`Sm=S*scale−LSE_row`、`P=exp(Sm)`、`dv+=Pᵀ@DO`、`dp=DO@Vᵀ`、`Pds=P*(dp−Delta_row)`、`ds=Pds*scale`、`dk+=dsᵀ@Q`、`dq=ds@K`(DQ:serial 循环累加 / atomic_add)。B(splitK reduce bwd)= 无 loop-carried(两 pass scan over split 轴):split-max + per-split K-reduce 的 LSE 重分配。
- **新 lowering 能力(C++)**:两扩展共享 M1 机制。**(B) varargs-split = 可先 ship 的一半**:扩 row-reduction pass,新 `Role::RowList`(尾部 S 个同 shape Row args = unroll 後的 VAR_ARGS,match 检尾部同 shape run,S 从前端 `split` attr);一 gpu.launch + 两 `scf.for over S` 全展开(pass1 per-thread 顺序 maxnumf 得 lse_max;pass2 w_s + K-lane 写 + `corr` 经 `gpu.all_reduce ADD`)。**(A) flash bwd = 新 pass `convert-intent-attention-backward-to-gpu`,叠在 M7+M8 上**:结构匹配(contraction subscript qd/kd/qk + recompute-P 子树 + dq/dk/dv,**非 name 匹配**);一 gpu.launch,loop→scf.for over start_m,dk/dv 寄存器/shared 累加(iter_args);每 contraction 经 M7 MMA;M8 融合 elementwise epilogue 使 P/ds/dp 寄存器常驻(永不 HBM);per-(B,H) `memref.subview` 指针重定基;DQ 双模 attr 选。
- **新原语**:(1) **varargs RowList**(S 路同 shape Row 经一协作 launch,split 轴顺序 reduce vs head_dim across-thread all_reduce;= unroll_varargs 的 IntentIR 实现,bare-ptr 需静态 arg 数 = unroll 已给);(2) **双模 DQ 累加 + 可证等价**:serial(dq_mode=serial,grid(1,B*H),DQacc 为 scf.for iter_arg,确定序)vs atomic(grid(cdiv(Sk,BN),B*H),`gpu.atomic addf`→`red.global.add.f32`,跨 program,预清零)。两者算同一实值 `Σ_n ds_n@k_n`,仅 fp 加序不同(故 DQ 的 reduce 携 `assoc=float_nondeterministic`)。**硬门:serial==atomic==fp32-ref 到 rel<1e-3**。
- **依赖**:M7(contraction→MMA)+ M8(epilogue 寄存器常驻,否则 spill cliff 重现)。**B 仅依赖 M1 + varargs 扩展 = de-risk 一半,可先 ship+PK**。intent.loop→scf.for(`LoopLowering` 已建)是 A 的 start_m 循环底座;carry-privatization + query-outer interchange 是 A 需的同一变换。
- **perf 故事**:A(flash bwd)compute-bound 但**库两杠杆都留**(Forge 13.21/11.82):(i) HBM——双 launch(preprocess delta 单独 kernel)+ `dq_accum` fp32 scratch + 末 `dq.copy_(dq_accum)` 一整 DQ-sized 往返可融掉;(ii) occupancy——源遍布 `debug_barrier` 节流 + EVEN_* 重 load,clean M7 MMA + M8 寄存器常驻去掉 → 5-13× headroom 在此。**诚实 caveat:仅当 M7 MMA 不 spill 才可达;A 的 MMA 路在 5090D 实测前 parity 是地板,5-13× 是上限目标非 headline**。B(splitK reduce bwd)memory-bound 读 S split + Out/LSE/DOut/DLSE 一次写 S 梯度一次 = 最小流量 read-once,**Forge remote 1.06 → 库已贴 HBM 顶 → parity 是诚实地板**(local 4.02 是坏计时器膨胀,remote 1.06 是真值)。
- **验收**:A correctness rel<1e-3 vs fp32-ref(torch autograd of softmax(QKᵀ)V)+ 可跑 5090D baseline(dq/dk/dv,headdim∈{64,128},EVEN/非 EVEN tail,causal/非);**DQ 双模等价硬门**(serial==atomic==fp32-ref,≥3 seed,失败=未完成);B correctness vs fp32-ref + 解 unroll baseline(S∈{2,4,8});perf B ≥ baseline(parity at HBM peak Forge 1.06,cross-check torch.copy_/baseline 落已知 GB/s 否则harness artifact);perf A ≥ 可跑 baseline(命名 beat = 融 dq_accum→dq copy + 去 barrier 节流,实测;baseline 不跑 → 退 fp32-ref+roofline 标注,parity 不可认);**A 无 MMA spill cliff**(PTX 无 ld/st.local);zero residual;lit `lower-splitk-reduce-bwd-to-gpu.mlir` + `lower-flash-bwd-to-gpu.mlir`;**GENERALITY 注**:无 per-kernel 字符串分支,5-contraction+recompute 是结构匹配(见 maturity 注)。
- **风险**:**M7/M8 是不存在的前置**(今天 contraction 只 linalg.generic)→ A 在 M7 MMA + M8 寄存器常驻 land 前**不可建**;若 slip/spill → A 退 correctness-only(缓解:先 ship B);DQ atomic fp32 重结合等价须实证(源自标 headdim=128/seqlen=(108,256) ATOMIC_ADD race bug — 我们须 out-correct 不复现);baseline 可跑性真未知(debug_barrier 脆 + autotune pre_hook launcher bug → 剥 autotune 用裸单配置);causal/seqlen -inf-lane = M1 tail-mask follow-up;**B 的 RowList 需前端 stamp split=S 并发 X0..X_{S-1}**(见前端前置);B 的 DLSE corr 是 head_dim reduce per split → S 个 all_reduce/launch,大 S 串行化。

---

## 横切关注点(cross-cutting)

### 新原语账本(primitive ledger —— 谁引入、谁复用,无 used-before-introduced)

| 原语 | 引入 | 复用 |
|---|---|---|
| 多输出 epilogue + thread-0 守卫 per-row 标量写 | M2 | M5(aux Mean/RSTD)、M6(loss) |
| read-once 多输出 destination-passing / in-place 别名 | M3 | M5、M6、M9 |
| `intent.select`(arith.select/cmpf)+ `intent.cast`(trunc/extf) | **M3 独家** | M4、M6、M7、M9(**不重复加**) |
| 索引重映射 emitAt 闭包(transpose/half-split/even-odd 条带 load) | M3 | M5(BroadcastSegment)、M8/M9(片段布局) |
| M1 reduce → rank-1 [M]/[M,1] 标量-per-row 输出 | M3(bwd_preprocess) | M2 PerRowOut |
| `intent.scan` + 块内 cumsum(两级 shfl Kogge-Stone) | M4 | — |
| -inf/+inf/identity tail-mask(masked-lane 替换) | M4 | M3(rope 非 pow2)、M6、M7、M8、M9(causal) |
| `memref.atomic_rmw addf` 跨行折叠 | M4 | M9-A(`gpu.atomic` 跨 program addf) |
| tiled cooperative 归约(scf.for+iter_args)+ smem 跨 pass 缓存 | M5 | M8(tile 内重用) |
| BroadcastSegment Col(affine floordiv 索引) / 2D grid | M5 | — |
| 在线 softmax scf.for+iter_args carry(无 contraction)+ 循环内 all_reduce + `gpu.barrier` + RowInt + rank-1 loss | M6 | M8(加 contraction) |
| tiled-K tensor-core matmul(`gpu.subgroup_mma_*` 片段 + scf.for over BK)+ smem staging | **M7** | M8(×2 in-loop)、M9-A |
| MMA 累加片段跨 trip carry + per-row α 片段重缩放 + split-K 第三 grid 维 | M8 | M9-A |
| varargs RowList(S 路 unrolled)/ 双模 DQ(serial vs atomic,可证等价)/ per-(B,H) subview 重定基 | M9 | — |

### sm_120a 能力门序列(probe gates —— 各由首需者前置 de-risk 依赖者)

- **G0 — `gpu.subgroup_mma_compute` → sm_120a cubin**(M7 首动作,承重 M7/M8/M9-A)。本轮 workflow **已实测通过**(6232-byte cubin);M7 开工须重新确认未失效。
- **G1 — workgroup `memref.alloca` + `gpu.barrier` on bare-ptr Blackwell**(M4 cumsum 与 M5 smem 缓存共需;**M5 当前隐含假设 → 提为显式共享门**)。
- **G2 — `gpu.all_reduce` 嵌 `scf.for`(在 `gpu.launch` 内)**(M6 "未测接缝",M8 同需;建任一循环前先门)。
- **G3 — 块内 cumsum shfl Kogge-Stone**(M4 GATE 0,已在 M4 验收,保留)。

### dialect op 归属(避免重复添加导致 build break)

- **`intent.select` + `intent.cast`:M3 独家添加**(`IntentOps.td` 验证器 + `emit_mlir` printer)。M4 **只**加 `intent.scan`;M6 **复用** select/cast(M6-new 仅 RowInt 角色 + rank-1 loss);M7/M9 复用。interpreter 已有 `np.where`/`astype` 语义参考。
- 现状(已核):`IntentOps.td` 仅 map_binary/map_unary/reduce/contraction/yield/loop;select/cast/scan 都需新增。

---

## Acceptance Criteria

- [ ] **27/27 交付物(29 算子,flash-bwd 3→1)**通过 intent → C++ lowering → sm_120a cubin,**零残留 `intent.*` 算子**。
- [ ] **29/29 bit-faithful**(rel < 1e-3 vs fp32 ref + 真实库 kernel),含非 pow2 tail-mask(M4 交付,M1/M3/M6-9 共享)。
- [ ] **全 29 算子 kernel-only 实测 ≥**(同一把尺,memory/compute 不分级):beat 须附机制;parity 仅在基线被证明贴硬件天花板时成立并标注(memory=最小流量 / compute=tensor-core 打满 ncu 证);M8 spill cliff 必须先消除。
- [ ] **零 per-kernel Python recognizer / 零字符串拼接 MLIR**;lowering 全在 C++ `OpRewritePattern`;lit + pytest 全绿。
- [ ] 每族有 lit 回归 + 一个 PK 测试(vs 真实库 kernel,5090D 实测基线);baseline 跑不了 → PK 退 vs-fp32-ref + roofline,**显式标注**。
- [ ] **四个 sm_120a 能力门 G0–G3 全过**(或失败者降级并显式标注)。

## Risks / Open Questions

- **⚠ 最大 finishability 风险 = M7 tensor-core 原语**:`gpu.subgroup_mma_*`→sm_120a 是 M7/M8/M9-A 承重柱,今天 lib/ 不存在;**无 cp.async 多级流水时,重调的 tl.dot/_fwd/_bwd 基线意味着整条 compute-bound 主干 parity(非 beat)是诚实天花板**;若 MMA 探针失效或 spill → 三个里程碑退 correctness-only。**缓解(规格已含):M9-B(纯 M1 底座)完全 de-risk,可最先 ship。**
- **前端前置(非纯风险,是硬前置)**:M9-B/M8 的 varargs 需 `emit_mlir.py` stamp `split=S` 并发 `X0..X_{S-1}` 连续同 shape arg run(镜像 `unroll_varargs`)。今天前端不做 → **纳入 M8 范围(packed-split 归一首现处)**,否则 RowList pass 无真实输入。
- **baseline 可跑性**:xformers splitK **FP8** sm_120a codegen 不确定 → 限 FP16/BF16,M8 不以 FP8 为门;flash bwd **`debug_barrier` race 绕过**跨版本脆弱 → 基线必须 5090D 实测;**FlagGems scaled/persistent + Triton tutorial softmax** 命中 Triton-3.6 launcher arg-count bug → 用裸 `@triton.jit`。
- **三个最难 lowering**:① online-softmax + 双 matmul 寄存器不 spill(M8,核心赌注,MMA 片段 α 重缩放是最难未知);② 多 matmul 反向 + 双模归约(M9-A);③ constexpr-flag DCE-by-construction at scale(M3/M6,未取分支若物化破坏 parity)。
- **缺的新原语**:tiled-K MMA(M7,已 de-risk)、块内 cumsum(M4,G3)、`atomic_add`+device-sync(M4/M6/M9)、varargs split(M8/M9)。

## Out of Scope

- 非 GPU 后端(CPU 已有参照)、多 GPU。
- 训练循环 / autograd 编排(只 lower 单 kernel)。
- 走 Forge 的 OptGuide 搜索(kernel 只作参考/输入)。
- 新的 LLM 提取(提取侧已 real;本任务聚焦 lowering 侧)。
- cp.async 多级流水化 K 循环(M7 命名 follow-up,M7 之後)、gelu-exact(需 erf vocab 扩)、FP8/INT4 dequant 路、causal+bias 完整 mask(先非 causal EVEN_*)。

## Definition of Done

27 交付物(29 算子)全部 lower + bit-faithful + **全部实测 ≥**(同一把尺,beat 附机制,parity 仅在贴硬件天花板时并标注);C++ lowering 成熟(无玩具机制);lit+pytest 全绿;每族有回归与 PK;G0–G3 全过;本 PRD acceptance 全勾。

## Technical Notes

- 验证过的探针 / recipe 见 memory(`tianchenir-backend-recipe`, `tianchenir-workspace-layout`, `tianchenir-extraction-corpus`)。
- 29 契约:`refs/TianchenForge/artifacts/provider29_optguide/`;源:`Dataset/{Liger-Kernel,flash-attention,xformers}`(liger 真 Triton kernel 在 `src/liger_kernel/ops/*.py`,`backends/_ascend/` 是 NPU 变体勿用)。
- 既有 C++ lowering 范例:M1 `lib/Intent/Conversion/ConvertIntentRowReductionToGPU.cpp`(440 行,M2/M4/M5/M6 的扩展或仿写蓝本);`ConvertIntentToLinalg.cpp`(contraction linalg leg,M7 弃用作 GPU 但留 CPU/oracle)。

## Research References

* [`research/29-kernel-characterization.md`](research/29-kernel-characterization.md) — 30-agent workflow:每内核原子分解 + 难度/regime + baseline 可跑性 + M1→M9 难度排序 + 风险 + 缺失原语。
* M2–M9 详细规格 buildout(本轮):9-agent workflow(8 milestone-design 读真实源 + 1 completeness-critic),critic verdict = "M2–M9 可驱动到 finishable DONE";已应用 punch-list(select/cast 归 M3、27-计数口径统一、M3 perf-bar 收紧、M9-B 前端前置入范围、G0–G3 门集中)。
