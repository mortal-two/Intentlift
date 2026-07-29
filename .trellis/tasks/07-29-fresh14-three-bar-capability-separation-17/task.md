# Task 17：Fresh-14 三柱能力分离与完整 Lowering-State Search

> 状态：执行中
>
> 类型：production backend 能力扩展、三臂因果实验修正、CUDA/H100/RVV 单线程搜索
>
> 唯一主图内部口径：`C-Fixed / Schedule-Only / Full (IntentIR)`

## 1. 第一性原理裁决

当前系统已经证明了两件事：Fresh-14 可以被真实 J→Intent MLIR→target pipeline 消费；5090D 上最终
IntentIR 代码相对原始 Triton Source 已经有竞争力。现在真正未闭合的是内部归因：

```text
C-Fixed       一个由 C 机械投影得到的 target schedule
Schedule-Only 固定同一结构，充分搜索 target schedule
Full          搜索全部已准入的结构/数值 MLIR states，并分别搜索 schedule
```

三个柱子的目标不是人为制造慢、较快、最快，而是形成严格嵌套的能力空间：

```text
Candidates(C-Fixed)
  ⊂ Candidates(Schedule-Only)
  ⊂ Candidates(Full)
```

在同一批候选的稳定测量上应满足：

```text
latency(Full) <= latency(Schedule-Only) <= latency(C-Fixed)
```

相等是合法的真实负结果；不能为了让每个 kernel 都严格递增而弱化 baseline、修改 workload 或改变
correctness。但如果全体几乎相等，说明后端没有提供足够不同且有效的 schedule/lowering states，不能把
它包装成论文结果。

## 2. 修改前的 5090D 诊断基线（不得作为最终结果）

以下数据产生于机械 C-Fixed、完整空间主三柱和扩展 row tile 全部落地之前，只用于定位“为什么弱”，
不得进入最终论文表。单位为微秒：

| Kernel | C-Fixed | Schedule-Only | Full | G_schedule | G_lowering | 当前主要变化 |
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
C-Fixed -> Full = 1.068x
```

问题不是测量波动：当前只有 Flash preprocess 的额外 state 真正消除了另一类并行度瓶颈，得到
`1.60x`；其余大多数 expanded winner 与 Schedule-Only 相同，或只增加了 FMA、倒数近似、局部
materialization 等 0--2% 变化。

## 3. 为什么三柱没有分开

### 3.1 修改前的 C-Fixed 实际上过于“聪明”

修改前的 C-Fixed 不是单纯消费 C。CUDA 排序会把 C 的 tile role 通过固定 elements-per-thread 重新解释，
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

C-Fixed 使用前者；Schedule-Only/Full 的搜索顺序使用后者。不能再让 C-Fixed 偷带一次隐式调优。

最新执行裁决进一步简化为：**C-Fixed 与 Schedule-Only 之间只允许 target schedule 自由度不同。**
二者必须使用同一 J、同一 canonical lowering structure、同一 target code generator 与 Q/D；C-Fixed
固定 C 给出的 tile/schedule及统一缺省值，Schedule-Only开放 tile、threads/warps、vector width、
unroll、pipeline和同结构 memory placement搜索。任何 fusion、materialization、reduction topology或
stage decomposition变化只能进入 Full。

这里的“tile”是用户侧的简写，不应被误读成只允许一个整数变化：它指同一 MLIR 结构上的完整 target
schedule assignment。C-Fixed 可以保持弱而机械——只运行 C 投出的那一个 assignment，C 未表达的坐标
使用全 kernel 统一默认值；Schedule-Only 则解冻这些坐标。弱化 C-Fixed 不能通过退化计算图、关闭正常
canonicalization或改输入实现，只能来自“不做搜索”本身。

### 3.2 修改前的 Schedule-Only 不是完整的 target schedule search

修改前的 CUDA fixed-path schedule space 主要只有：

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

### 3.3 修改前的 Full 也不是名义上的“全部开放”

修改前的 bundle 使用 budget 32，且没有运输/测量完整的 admitted candidate universe。Full 先放完整
Schedule-Only spine，再在剩余预算访问 expanded states；候选空间扩大后，未访问的 state 根本不会进入
target 测量。因此当时的 `Full` 只是 bounded structural search，不是完整性能上限。

更重要的是，现有 expanded space 本身偏弱：

- 多数 normalization 的 canonical path 已经是单 launch、read-once 的高度专用 lowering；
- expanded state 仍复用同一 direct pass，只切换 FMA、reciprocal、leader finalize 等局部选项；
- 当前 multi-stage reduction 会重读 producer/input，通常比单阶段更慢；
- Sparsemax 没有进入真正 vector/block prefix-scan 或 sorted-prefix state；
- SplitK 只有一个 unrolled weighted-merge 结构，四个候选只是 rows-per-block schedule；
- elementwise 的主要可获收益本来属于 Schedule-Only vector/grid schedule，Full 目前只剩近似数学候选；
- 当前 fresh-14 没有 contraction，所以 TF32/MMA 不是这批 Full 的性能来源。

Flash preprocess 之所以例外，是因为它的 expanded subgroup-feature state 真正把 serial feature traversal
变成并行 feature reduction；它改变并解决了主瓶颈，而不是只切换一个数值开关。

### 3.4 主性能上限与搜索预算被混成了一个问题

“Schedule-Only 与 Full 总候选预算相同”适合回答 bounded-search efficiency，却不适合定义三根能力柱。若
Schedule-Only 用完全部预算搜索一个 state，Full 不可能在相同总预算中既完整保留 Schedule-Only 搜索，又给多个新 state相同的
schedule coverage。

因此必须拆开：

- **三柱主图**：比较每种能力开放后的 converged/exhaustive performance envelope；
- **C-prior 小表/anytime 曲线**：固定相同总预算，只比较搜索顺序和达到最好性能的速度。

这不是给 Full 更多不公平预算。三柱回答“开放什么能力能得到什么代码”；C 小表回答“相同编译预算
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
一个 launch”的故意低效 materialized pipeline制造几十倍差距。擦除 schedule 参数后，C-Fixed 与 Schedule-Only 的
compiler graph、fusion partition、materialization、reduction topology 和 stage decomposition必须相同。

### 4.3 Full（IntentIR）

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

Source 不属于内部三柱，但仍是每个 GPU headline cell 的强制第四次执行：原始 Triton Source与最终
Full winner使用相同 shape、dtype、输入、cache policy、计时器和交替 paired batches。内部消融变强
或变弱都不能省略 Source，也不能用 RefJ/Replay代替它的性能。

## 5. 执行阶段

### 当前已落地（2026-07-29）

- CUDA C-Fixed 已从 target-aware C ordering 中拆出：只按 C role/value 投到同一路径合法 schedule，
  其余坐标使用固定 compiler default；不读取 shape、occupancy、A 中算子密度、cost model或性能；
- CUDA row C-Fixed 未由 C 指定的 block 默认固定为 64 threads（两个 warp）；这是统一、弱而可生产的
  compiler default。C 给出的 tile仍机械投影；Schedule-Only解冻 threads/tile/vector后自行重调优；
- C-Fixed 编译失败时直接失败，不再静默换成另一个候选；
- CUDA row Schedule-Only 的真实 C++ tile 轴已经扩为 `16/32/64/128/256`，并与
  `64/128/256/512/1024` threads交叉；
- CUDA row C-Fixed/Schedule-Only现在明确落到标准 `gpu.all_reduce` collective；Full才枚举
  紧凑 warp-shuffle/shared、multi-row、multi-stage 等实际不同的 reduction topology。标准 collective
  不再被误标为 observation-changing；只有 post-pass 中真实出现的结构/数值变化才进入 Full 的 Q/D 门控；
- 完整 expanded bundle 可一次运输全部可执行候选；主三柱从这同一个完整候选初测图做排名，并对
  C-Fixed、Schedule-Only provisional best和全局 provisional best做 fresh finalist重测；
- bounded C-guided/Erase-C order仍保留，但只生成 anytime/C-prior 指标，不再定义主三柱；
- GPU 最终 Full winner仍与原始 Triton Source做同 shape/dtype/cache/timer 的交替配对确认。
- elementwise 的 scalar/vec2/vec4、logical tile 与 grid-stride unroll 已成为真实 C++/MLIR schedule；
- row reduction 的 tile、threads、multi-row subgroup、leader finalize、raw-moment variance已经产生可由
  post-pass IR 区分的状态；Sparsemax sorted-prefix 与 Flash padded subgroup-feature是当前真正改变主瓶颈
  的结构路径；
- 当前 Fresh-14 的 Source baseline 14/14 都会真实执行并通过 Source--J 输出比较，不以 Replay/RefJ
  的性能代替 Source；
- 已确认当前主要缺口不是更多 Python 候选名，而是更多可复用、能改变实际 MLIR dataflow/topology且
  解决主瓶颈的 C++ lowering，以及 RVV 上真实 scalable-vector/retile能力。
- RVV 已有真实 scalable-vector 编译状态（LLVM scalable vector 与 `vsetvli` 可见）及单线程搜索；下一步
  仍需把 dynamic-VL/retile 接入可消费的 row/elementwise lowering，而不是停留在候选枚举。
- `require-sorted-prefix` 已在 C++ matcher 中严格 fail-close；非 Sparsemax 图不能再借该候选身份落到普通
  row reduction。LayerNorm/PolyNorm 的旧 provisional 归因已作废，正常候选不受影响；
- SplitK 新增 `warps-per-row={2,4,8}` 的真实 multi-warp D partition，Schedule-Only固定为 1；Fresh
  natural shape 上该状态正确但只带来约 1% 的边际收益，说明 15 us 左右已受启动/小流量下限主导；
- 首轮完整 5090D cold-data 诊断（修复伪路径和最终 C-Fixed前）为
  `G_schedule=1.048x, G_lowering=1.055x, Full/Source=1.246x`；其中 Flash preprocess
  `G_lowering=1.663x`、Sparsemax `1.155x`，其余多数仍受带宽上限约束。该轮仅用于指导继续优化，
  不作为最终表。
- 修复后的 5090D Fresh-14 natural-headline cold-data v5 已完成：
  `G_schedule=1.178x, G_lowering=1.055x, Full/Source=1.251x`，8/14 的
  `G_schedule>=1.05x`。Flash preprocess `G_lowering=1.663x`、Sparsemax `1.160x`；
  xFormers RMSNorm-add 的 `C-Fixed=2341.95 us -> Schedule-Only=534.43 us`，直接显示冻结
  GPU/Triton C 后在 feature=12288 上形成的长 strip/寄存器溢出可由 target retile 消除。Full 全体仍
  未达到 1.10 工程目标，必须原样保留并继续在 H100/RVV 与结构适用子集上验证。

以上只闭合了 Phase A 和 row tile 的一部分，不表示任务完成。正式 Fresh-14 natural-headline cold-data
数据必须全部重跑；第 2 节旧数据不得沿用。

### Phase A：先修实验语义，不先调数字

1. 将 C-Fixed 的机械 C 投影与 C-guided search ordering 拆成两个函数；删除 C-Fixed 中的
   workload/occupancy 替代值。
2. 增加完整 candidate-space 两阶段测量模式：所有可执行候选先使用相同重复次数形成完整排名图，
   再对每个主臂 provisional best与全局 top finalists交替 fresh重测；三柱由这一次搜索求 CF、
   fixed-path schedule best和global state best。
3. 主三柱不再使用 equal-total-budget 截断；保留独立 equal-budget C/no-C 与 anytime 曲线。
4. 报告只需给出三柱的实际候选数量、获胜路径和耗时；不增加与性能无关的 artifact 管理流程。
5. 对现有 Fresh-14 在 5090D/H100 先完整测量当前 universe，确认“尚未测到”与“测到但不赢”各占多少。

### Phase B：把 Schedule-Only 做成真实 retile/search

横向扩展真实影响 target code 的 schedule axes：

1. elementwise：scalar/vec2/vec4、flat/grid-stride、block size、每线程 packet/unroll；
2. row reduction：feature tile、per-thread strip、threads/warps、rows-per-block、shared/register placement；
3. padded row：serial feature strip 的 tile/vector/threads，subgroup topology只在 Full；
4. SplitK：同一 weighted-merge 结构内的 row grouping、D packet、warp occupancy；
5. RVV 单线程：dynamic VL、SEW/LMUL、strip mining、tail 与 reduction tile；
6. 所有 requested schedule 必须在真实 post-pass MLIR/launch/object 中生效；不能只增加 Python 枚举值。

### Phase C：加入真正解决瓶颈的 Full states

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
- elementwise 的 register tile/vector/grid主要计入 Schedule-Only，只有真正改变 fusion/materialization 的 state计入 Full；
- reduction tree、FMA、native approximation与 accumulator state继续走统一 Q/D，但不把 1% 数值开关
  当作结构搜索的主要故事；
- 未来 fresh contraction 到位后再加入 IEEE/TF32/MMA，当前 14 项不伪造 TF32 搜索。

### Phase D：三个 target 独立搜索

- RTX 5090D 与 H100 共用高层 matcher/transformation，各自完整搜索 schedule；
- H100 不复用 5090D winner；
- RVV 本任务只做单线程，使用真实 MLIR→LLVM→RVV object；
- RVV C-Fixed 直接投影 GPU/Triton C，Schedule-Only开放 dynamic-VL/LMUL retile，Full再开放
  fusion/reduction/scan states；
- GPU 继续以原始 Triton Source 为外部 baseline，RVV 使用独立 Native RVV baseline。

### Phase E：正式三柱与 C-prior 输出

主图：

```text
External GPU: Source / Full
Internal:     C-Fixed / Schedule-Only / Full
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
4. Full 完整包含 Schedule-Only，并对每个 structural state提供同等级 schedule coverage；
5. 三柱从同一 converged latency map计算，C/no-C 独立使用 equal budget；
6. 新收益来自真实 MLIR structure/dataflow/topology变化，并通过相同 Q/D；
7. GPU Source/Full 的竞争力不因内部消融修改而退化；
8. 不实现 Python target emission、per-kernel recipe 或与性能无关的 artifact 管理机制；
9. 全量回归通过，工作区干净。

## 7. 本任务明确不做

- 不故意把 canonical path 退化成一 op一 launch来制造几十倍；
- 不要求每个 kernel 的三柱都严格分离；
- 不把 equal-total-budget Full 冒充完整 Full 性能上限；
- 不把 FMA/reciprocal 的微小收益冒充全部 structural lowering story；
- 不修改 accepted J、Source workload、Q/tolerance 或 timed boundary追求正数；
- 不恢复旧的 implementation-family 术语或抽象；
- 不做 RVV 16-thread；
- 不把任务重心转成 artifact 管理或实验元数据整理。
