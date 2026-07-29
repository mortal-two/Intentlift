# Task 17：Fresh-14 三柱能力分离与完整 Lowering-State Search

> 状态：准备执行
>
> 类型：production backend 能力扩展、三臂因果实验修正、CUDA/H100/RVV 单线程搜索
>
> 唯一主图内部口径：`C-Fixed / Schedule-Only / IntentIR`

## 1. 第一性原理裁决

当前系统已经证明了两件事：Fresh-14 可以被真实 J→Intent MLIR→target pipeline 消费；5090D 上最终
IntentIR 代码相对原始 Triton Source 已经有竞争力。现在真正未闭合的是内部归因：

```text
C-Fixed       一个由 C 机械投影得到的 target schedule
Schedule-Only 固定同一结构，充分搜索 target schedule
IntentIR      搜索全部已准入的结构/数值 MLIR states，并分别搜索 schedule
```

三个柱子的目标不是人为制造慢、较快、最快，而是形成严格嵌套的能力空间：

```text
Candidates(C-Fixed)
  ⊂ Candidates(Schedule-Only)
  ⊂ Candidates(IntentIR)
```

在同一批候选的稳定测量上应满足：

```text
latency(IntentIR) <= latency(Schedule-Only) <= latency(C-Fixed)
```

相等是合法的真实负结果；不能为了让每个 kernel 都严格递增而弱化 baseline、修改 workload 或改变
correctness。但如果全体几乎相等，说明后端没有提供足够不同且有效的 schedule/lowering states，不能把
它包装成论文结果。

## 2. 当前 5090D 三柱基线

当前 L2-controlled cold-data、同次 finalist 测量结果如下，单位为微秒：

| Kernel | C-Fixed | Schedule-Only | IntentIR | G_schedule | G_lowering | 当前主要变化 |
|---|---:|---:|---:|---:|---:|---|
| DYT | 111.264 | 92.864 | 92.864 | 1.198 | 1.000 | schedule |
| Flash bwd preprocess | 307.744 | 304.896 | 190.400 | 1.009 | 1.601 | padded subgroup-feature topology |
| Flash one-pass LayerNorm | 143.360 | 141.280 | 140.768 | 1.015 | 1.004 | FMA，小收益 |
| Fused-add RMSNorm | 94.848 | 94.848 | 93.120 | 1.000 | 1.019 | reciprocal/materialization state |
| GeGLU | 2924.096 | 2752.480 | 2752.480 | 1.062 | 1.000 | schedule |
| KLDiv | 97.696 | 97.696 | 97.696 | 1.000 | 1.000 | 无有效新增状态 |
| LayerNorm | 92.896 | 92.896 | 91.776 | 1.000 | 1.012 | FMA，小收益 |
| Liger RMSNorm | 48.352 | 47.424 | 47.424 | 1.020 | 1.000 | schedule |
| PolyNorm | 47.936 | 47.040 | 46.880 | 1.019 | 1.003 | FMA，小收益 |
| Softmax | 47.936 | 46.272 | 45.760 | 1.036 | 1.011 | reciprocal，小收益 |
| Sparsemax | 70.272 | 70.272 | 70.048 | 1.000 | 1.003 | reciprocal；缺真正 scan state |
| SwiGLU | 1454.400 | 1381.920 | 1381.664 | 1.052 | 1.000 | schedule，local state近似无效 |
| xFormers RMSNorm-add | 534.976 | 534.976 | 533.184 | 1.000 | 1.003 | materialization state近似无效 |
| SplitK reduce | 16.576 | 16.416 | 16.416 | 1.010 | 1.000 | rows-per-block schedule |

Fresh-14 几何平均：

```text
G_schedule = 1.029x
G_lowering = 1.038x
C-Fixed -> IntentIR = 1.068x
```

问题不是测量波动：当前只有 Flash preprocess 的额外 state 真正消除了另一类并行度瓶颈，得到
`1.60x`；其余大多数 expanded winner 与 Schedule-Only 相同，或只增加了 FMA、倒数近似、局部
materialization 等 0--2% 变化。

## 3. 为什么三柱没有分开

### 3.1 C-Fixed 实际上过于“聪明”

当前 C-Fixed 不是单纯消费 C。CUDA 排序会把 C 的 tile role 通过固定 elements-per-thread 重新解释，
elementwise 路径甚至用当前 J shape 和目标 occupancy hint 替换原 C value，再选择一个 target-aware
候选。这样得到的是“小型 target cost model 的答案”，自然经常已经接近 Schedule-Only winner。

正确边界应当拆开：

```text
Project_C(C,target):
  只解释 C role/anchor，做必要的 target legality/clamp
  不读取 workload-derived occupancy heuristic
  不试多个候选

Order_C(search space):
  C 作为有限预算搜索先验
  可以与 target cost model共同排序
```

C-Fixed 使用前者；Schedule-Only/IntentIR 的搜索顺序使用后者。不能再让 C-Fixed 偷带一次隐式调优。

### 3.2 Schedule-Only 不是完整的 target schedule search

当前 CUDA fixed-path schedule space 主要只有：

```text
elementwise: block_threads、少量 flat vec2/vec4
row reduction: block_threads、tile_cols=32/64、auto/scalar vector width
padded row: 少量 serial-row block/vector 组合
SplitK: rows_per_block={1,2,4,8}
```

缺少或尚未真实作用于代码的关键轴包括：

- row tile/strip：`16/32/64/128/256` 与 per-thread elements；
- vector packet、load/store width、grid-stride unroll；
- rows-per-block、warps-per-row、CTA row grouping；
- shared/register placement、pipeline/stage、partial layout；
- RVV dynamic VL、SEW/LMUL、strip mining 与 tail；
- target 上实际 launch/grid/occupancy 可改变、且在 post-pass code 中可验证的 schedule assignment。

因此当前 `Schedule-Only` 只是有限 block-size 搜索，不能代表“C 解耦后 target 能充分重调优”。

### 3.3 当前 IntentIR 也不是名义上的“全部开放”

当前正式 bundle 使用 budget 32，且没有运输/测量完整的 admitted candidate universe。E3 先放完整
Schedule-Only spine，再在剩余预算访问 expanded states；候选空间扩大后，未访问的 state 根本不会进入
target 测量。因此当前 `IntentIR` 是 bounded E3，不是完整性能上限。

更重要的是，现有 expanded space 本身偏弱：

- 多数 normalization 的 canonical path 已经是单 launch、read-once 的高度专用 lowering；
- expanded state 仍复用同一 direct pass，只切换 FMA、reciprocal、leader finalize 等局部选项；
- 当前 multi-stage reduction 会重读 producer/input，通常比单阶段更慢；
- Sparsemax 没有进入真正 vector/block prefix-scan 或 sorted-prefix state；
- SplitK 只有一个 unrolled weighted-merge 结构，四个候选只是 rows-per-block schedule；
- elementwise 的主要可获收益本来属于 E2 vector/grid schedule，E3 目前只剩近似数学候选；
- 当前 fresh-14 没有 contraction，所以 TF32/MMA 不是这批 E3 的性能来源。

Flash preprocess 之所以例外，是因为它的 expanded subgroup-feature state 真正把 serial feature traversal
变成并行 feature reduction；它改变并解决了主瓶颈，而不是只切换一个数值开关。

### 3.4 主性能上限与搜索预算被混成了一个问题

“E2 与 E3 总候选预算相同”适合回答 bounded-search efficiency，却不适合定义三根能力柱。若 E2 用完
全部预算搜索一个 state，E3 不可能在相同总预算中既完整保留 E2 搜索，又给多个新 state 相同的
schedule coverage。

因此必须拆开：

- **三柱主图**：比较每种能力开放后的 converged/exhaustive performance envelope；
- **C-prior 小表/anytime 曲线**：固定相同总预算，只比较搜索顺序和达到最好性能的速度。

这不是给 IntentIR 更多不公平预算。三柱回答“开放什么能力能得到什么代码”；C 小表回答“相同编译预算
下先验能否更快找到它”。两个因果问题不能共用一个预算约束。

### 3.5 并非所有 kernel 都应有大 lowering gain

SwiGLU、GeGLU、DYT 等大 elementwise workload 的主要自由度是 vector/grid schedule，本来更应体现在
`G_schedule`。已经单趟读写、接近带宽限制的 RMSNorm 也不可能仅凭换 reduction spelling 得到巨大
`G_lowering`。论文必须报告 Fresh-14 全体，同时报告 transformation-applicable subset；不能要求每项
都严格拉开，也不能只挑正例。

## 4. 修正后的三柱定义

### 4.1 C-Fixed

```text
同一个 J
+ 一个 production-grade canonical structural path
+ 对真实 C 的机械 target projection
+ exactly one candidate
+ no tuning/cost-model search
```

未由 C 给出的 target 参数使用统一固定默认值。默认值不能读取 kernel 名、历史性能或 workload-derived
occupancy model。C-Fixed 可以很好，但不能隐式替用户完成 Schedule-Only。

### 4.2 Schedule-Only

```text
同一个 J
+ 与 C-Fixed 完全相同的 canonical structural path
+ 完整 target schedule domain
+ converged/exhaustive best schedule
```

canonical path 必须是合理的 production MLIR path，允许标准 fusion/canonicalization，禁止使用“一 op
一个 launch”的故意低效 materialized pipeline制造几十倍差距。擦除 schedule 参数后，CF 与 E2 的
compiler graph、fusion partition、materialization、reduction topology 和 stage decomposition必须相同。

### 4.3 IntentIR

```text
同一个不可变 J
+ 包含完整 Schedule-Only state
+ 全部 hard-admitted structural/numeric MLIR states
+ 每个 state 独立获得同等级 schedule search
+ 同一 Q/D filter
+ converged/exhaustive global best
```

三柱使用同一 target-side latency map和 fresh finalist remeasurement。主图不使用 equal-total-budget 截断
“Full”；equal-budget 只保留在 C-prior/anytime 实验中。

## 5. 执行阶段

### Phase A：先修实验语义，不先调数字

1. 将 C-Fixed 的机械 C 投影与 C-guided search ordering 拆成两个函数；删除 C-Fixed 中的
   workload/occupancy 替代值。
2. 增加完整 candidate-space 测量模式；三柱直接从同一 latency map求 CF、fixed-path schedule best 和
   global state best。
3. 主三柱不再使用 equal-total-budget 截断；保留独立 equal-budget C/no-C 与 anytime 曲线。
4. 报告验证 E2 candidate set 包含 CF、E3 candidate set 包含 E2；这只是运行时集合关系，不建设
   hash、digest、freeze、receipt 或新账本。
5. 对现有 Fresh-14 在 5090D/H100 先完整测量当前 universe，确认“尚未测到”与“测到但不赢”各占多少。

### Phase B：把 Schedule-Only 做成真实 retile/search

横向扩展真实影响 target code 的 schedule axes：

1. elementwise：scalar/vec2/vec4、flat/grid-stride、block size、每线程 packet/unroll；
2. row reduction：feature tile、per-thread strip、threads/warps、rows-per-block、shared/register placement；
3. padded row：serial feature strip 的 tile/vector/threads，不把 subgroup topology混进 E2；
4. SplitK：同一 weighted-merge 结构内的 row grouping、D packet、warp occupancy；
5. RVV 单线程：dynamic VL、SEW/LMUL、strip mining、tail 与 reduction tile；
6. 所有 requested schedule 必须在真实 post-pass MLIR/launch/object 中生效；不能只增加 Python 枚举值。

### Phase C：加入真正解决瓶颈的 IntentIR states

#### C1. Producer--reduction--epilogue

面向 RMSNorm、fused-add RMSNorm、LayerNorm、Softmax、KLDiv、PolyNorm、Flash one-pass LN 与 xFormers
RMSNorm-add：

- production-grade standard fused canonical state；
- warp/subgroup-per-row、CTA/workgroup-per-row、multi-row CTA；
- 多 reduction 共享一次 input traversal 与 shared/register frontier；
- producer、statistics 与 epilogue 单趟消费，不写回再重读完整中间 tensor；
- 真正 multi-stage partial/finalize 只传 partial summary，不重读 producer/input；
- Welford、sum/sumsq、hierarchical fold 等结构作为不同真实 MLIR states，各自调 schedule。

优先复用 sibling TianchenIR 已有的通用 `ConvertIntentRowReductionToGPU` 独立 reduction grouping、
shared-memory frontier 和 read-once epilogue实现，不重新写 Python kernel。

#### C2. Sparsemax scan

- CUDA warp/block prefix scan；
- RVV vector prefix/segmented scan；
- scan、threshold、reduction 与 epilogue融合；
- sorted-prefix early stop 作为 observation-changing execution-domain candidate，经固定 Q/D筛选；
- 不把普通 reduction 冒充 scan，不按 kernel 名选择路径。

优先复用 sibling TianchenIR 的 warp-shuffle、compact shared scratch 与 sorted-prefix MLIR lowering。

#### C3. SplitK

- 当前 warp-owned unrolled weighted merge 保留为一个 state；
- 新增 CTA-cooperative D reduction/merge state；
- LSE max、sumexp、weights 与 numerator只计算一次并广播；
- f32 vec2/vec4 partial loads、D=256 strip、multi-row block分别调优；
- 只在真正需要时引入 partial/finalize，不增加无意义 dispatch。

优先复用已有 stable online LSE merge实现；扩展的是通用图 matcher与目标映射，不新增 per-kernel emitter。

#### C4. Padded/elementwise/numeric

- 将 Flash preprocess 已获胜的 subgroup-feature机制推广到所有匹配 padded reduction 图；
- elementwise 的 register tile/vector/grid主要计入 E2，只有真正改变 fusion/materialization 的 state计入 E3；
- reduction tree、FMA、native approximation与 accumulator state继续走统一 Q/D，但不把 1% 数值开关
  当作结构搜索的主要故事；
- 未来 fresh contraction 到位后再加入 IEEE/TF32/MMA，当前 14 项不伪造 TF32 搜索。

### Phase D：三个 target 独立搜索

- RTX 5090D 与 H100 共用高层 matcher/transformation，各自完整搜索 schedule；
- H100 不复用 5090D winner；
- RVV 本任务只做单线程，使用真实 MLIR→LLVM→RVV object；
- RVV C-Fixed 直接投影 GPU/Triton C，Schedule-Only开放 dynamic-VL/LMUL retile，IntentIR再开放
  fusion/reduction/scan states；
- GPU 继续以原始 Triton Source 为外部 baseline，RVV 使用独立 Native RVV baseline。

### Phase E：正式三柱与 C-prior 输出

主图：

```text
External GPU: Source / IntentIR
Internal:     C-Fixed / Schedule-Only / IntentIR
```

另表：

```text
C-guided vs Erase-C
  - equal budget best
  - candidates to 95% of converged best
```

Fresh-14 全体、per-kernel 与 transformation-applicable subset同时报告；subset不能替代全体。

## 6. 性能目标与验收原则

以下是后端工程目标，不是删除负结果或修改实验的理由：

- 5090D/H100 Fresh-14 all-corpus 目标：`G_schedule >= 1.08x`、`G_lowering >= 1.10x`；
- schedule-sensitive subset 目标：至少 5 项 `G_schedule >= 1.05x`；
- structural-applicable subset 目标：至少 4 项由擦除 schedule 后仍不同的 state获胜，subset
  `G_lowering >= 1.15x`；
- RVV 单线程应重点显示 GPU/Triton C 与 RVV target retile 的差异，不预先伪造数值门槛；
- 若充分实现、profile 和完整搜索后仍未达目标，保留全量负结果并重新评估论文 claim，而不是改变 Q、
  workload、Source 或 baseline。

完成标准：

1. Fresh-14 全部在 5090D、H100 与 RVV 单线程进入同一三臂 controller；
2. C-Fixed 不含隐式 cost-model tuning；
3. Schedule-Only 的 schedule assignments 全部真实改变 target code/launch；
4. IntentIR 完整包含 E2，并对每个 structural state提供同等级 schedule coverage；
5. 三柱从同一 converged latency map计算，C/no-C 独立使用 equal budget；
6. 新收益来自真实 MLIR structure/dataflow/topology变化，并通过相同 Q/D；
7. GPU Source/IntentIR 的竞争力不因内部消融修改而退化；
8. 不实现 Python target emission、per-kernel recipe/family、hash/digest/freeze/receipt/性能缓存；
9. 全量回归通过，工作区干净。

## 7. 本任务明确不做

- 不故意把 canonical path 退化成一 op一 launch来制造几十倍；
- 不要求每个 kernel 的三柱都严格分离；
- 不把 equal-total-budget E3 冒充完整 IntentIR 性能上限；
- 不把 FMA/reciprocal 的微小收益冒充全部 structural lowering story；
- 不修改 accepted J、Source workload、Q/tolerance 或 timed boundary追求正数；
- 不恢复旧的 implementation-family 术语或抽象；
- 不做 RVV 16-thread；
- 不把任务重心转成 artifact 管理或实验元数据整理。
