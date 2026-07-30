# Task 22：阶段二新增 8 个 J 的 Production Backend 闭合

> 状态：已建立，待执行
>
> 类型：阶段二后端能力建设、真实 C++/MLIR lowering、5090D/H100/RVV 性能解耦
>
> 唯一 intake：`doc/J来源.md` 中已通过的 22 个 Kernel
>
> 阶段边界：前 14 项属于阶段一；本任务只处理与阶段一的差集 8 项

## 1. 阶段边界

阶段一已经完成并记录以下 14 个 kernel：

```text
SwiGLU
RMSNorm
Fused Add RMSNorm
GeGLU
LayerNorm
Softmax
DYT
Sparsemax
KLDiv
PolyNorm
Flash backward preprocess
Flash one-pass LayerNorm
xFormers RMSNorm-add
SplitK reduce
```

阶段一的结果、能力与缺口统一见：

```text
doc/Fresh14_三目标后端完整进展汇报_2026-07-30.md
```

阶段一的 14 项不再计入阶段二的完成数量，也不因本任务重新定义 baseline 或重复生产整套结果。只有共享后端
修改需要做 regression 时才复测相关代表项。

阶段二新增、且恰好为 22 项总表与阶段一差集的 8 个 J 是：

| # | Kernel | 正式证据类型 | 阶段二结构类别 |
|---:|---|---|---|
| 1 | JSD | direct closure | multi-input masked stable row reduction、多输出 epilogue |
| 2 | TVD | direct closure | elementwise abs/sign/select、多输出 |
| 3 | xFormers RMSNorm | direct closure | row reduction + affine epilogue |
| 4 | CrossEntropy | Gate2 recheck | indexed gather + conditional effect/in-place output |
| 5 | GroupNorm | Gate2 recheck | grouped multi-axis statistics + affine gather、多统计输出 |
| 6 | Liger RoPE | direct closure | rank-4 f16 pairwise rotation、in-place outputs |
| 7 | Qwen2VL MRoPE | Gate2 recheck | multidimensional position gather + concat + rotation |
| 8 | Flash Linear | direct closure | f16 contraction + bias + cast、多输出 |

Flash Linear 必须使用 `doc/J来源.md` 中的正式 direct winner：

```text
/tmp/intentir-task15-v12-owned-realization-linear-splitkfwd-sol-high-b3k5-20260730-01/
  flash_attn_ops_triton_linear_kernel_fwd/fanout-001/candidate-02/report.json
```

禁止误用另一份 `gate2_failed` 且 `changes_parent_verdict=false` 的旧 Linear recheck。

## 2. 第一性目标

本任务不是为了多填 8 行实验表，而是补齐阶段一没有覆盖的 production backend 能力：

1. 同一 accepted J 能通过真实 C++/MLIR pass 在 5090D、H100、RVV 上生成可执行目标程序；
2. 当前已有 pass 能消费的图直接复用，不能因新 kernel 名称再写一套 lowering；
3. 当前只能 generic lowering 的图，要明确 generic 状态的真实 topology，再增加 typed transformation；
4. 当前无法完整 lower 的图，补的是 gather/effect/N-D alias/contraction/grouped-statistics 等通用能力；
5. Full 必须改变 traversal、materialization、reduction/synchronization、index graph、stage graph 或 contraction
   structure；只改变 tile、threads、warps、vector width、pipeline 仍属于 Schedule-Only；
6. Python 只编排、绑定、Q/D 与计时，不发射 CUDA/RVV target kernel；
7. 后端 matcher 不读取 kernel 名、Source 路径、provider、benchmark label 或历史 winner。

5090D/H100 是阶段二当前实现和调优主线。RVV 保留单线程完整入口，并补齐新增 8 项所缺的 independent Native
baseline 和目标能力，但不阻塞 GPU 能力纵切。

## 3. 固定实验定义

### 3.1 三柱

```text
C-Fixed:
  DirectLower(J) + C 的一次确定性 target schedule 投影
  一个 executable，不搜索

Schedule-Only:
  与 C-Fixed 相同的 DirectLower(J) 和相同 executable structure
  只搜索 tile、threads、warps、vector、pipeline、VL/LMUL 等 schedule

Full:
  完整包含 Schedule-Only spine
  额外构造由 typed J、B hard facts、target capability、current-state precondition
  与 pass contract 准入的结构/数值 MLIR state
  每个 state 再独立搜索 schedule
```

内部指标固定为：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_full     = latency(Schedule-Only) / latency(Full)
```

### 3.2 外部 baseline

- 5090D/H100：原始 lifted Source kernel；
- RVV：严格匹配 operator、shape、dtype、layout、observable outputs 与 timed boundary 的单线程 Native RVV；
- 有成熟、严格同边界的 ggml/llama.cpp RVV 实现时优先复用；没有时实现独立 C/RVV intrinsic baseline；
- RefJ、Replay 与 generic Python evaluator 只负责 correctness，不能作为性能 baseline。

### 3.3 测量

- 5090D/H100：natural/model-level workload、code-warm/data-cold、CUDA event；L2 eviction 在计时区间外；
- RVV：固定 CPU affinity、单线程、code-warm/data-cold；
- 同一 kernel 的三柱和 external baseline 必须使用相同 shape、dtype、layout、observable outputs 与 timed boundary；
- Source 允许用 harness 绑定合法 runtime/constexpr 参数，但不能修改算子语义或把工作移出 timed boundary；
- closure artifact 的某个具体 valuation 不自动等于论文 workload。每项 natural shape 必须来自原 Source launcher、
  provider test 或真实模型配置；找不到时标为待定，禁止凭经验猜一个对后端有利的 shape。

## 4. 八个 J 的 typed 结构与后端目标

### 4.1 JSD

正式 J：两个 `f32[x0,x1]` 输入、label/mode/scalar 输入，两个 `f32[x0,x1]` 输出。图中包含两条 row max、
sub/exp/log、mul/add/div、mask 与多输出 epilogue。

当前应优先复用 CUDA row-reduction/read-once lowering，并以 source-blind probe确认完整图是否被消费。需要补的
通用能力只能来自实际缺口，例如：

- 两个 probability path 的共同输入/summary live state；
- max、normalization statistics 与双输出 epilogue 的多阶段融合；
- mask/label 不产生中间 tensor；
- 允许时 joint/multi-accumulator reduction tree；
- RVV multi-input producer fusion + joint vector reduction。

不能把单独 approximate exp/log/FMA 的小收益作为主要 Full 机制。

### 4.2 TVD

正式 J 是 `sub -> abs -> mul -> compare -> select/sign` 的 f32 elementwise 多输出图。

它首先用于验证后端能否把多输出、mask/select 全部保存在一个 target loop/kernel 中。CUDA 现有 elementwise
matcher若能完整消费则直接复用；否则扩展必须保持 source-neutral。由于它很可能已经是单遍读写，TVD 是阶段二
重要负例：允许 Schedule-Only 提升，不强迫 Full 违反带宽下限。

### 4.3 xFormers RMSNorm

正式 J 为 `mul -> sum(axis=1) -> div/add/rsqrt -> affine`，C 含 reduction tile 128。

它与阶段一的 xFormers RMSNorm-add 不是同一个 J。CUDA 首先复用现有 row-reduction/read-once path，并纳入
已实现的 compact tree、materialization choice、per-thread multi-accumulator tree 与 schedule search。RVV 复用
explicit scalable-vector reduction 与 producer fusion。只有 post-pass 出现新的 reduction/materialization 图时才计
Full，不能把阶段一 winner 直接搬过来。

### 4.4 CrossEntropy

正式 J 的关键结构是 f32 logits、i64 label、gather、equality predicate、`effect_if` 与 in-place/额外 loss output；
不能根据 kernel 名擅自补一套 J 中不存在的 softmax 图。

仓库已经存在 CUDA gather/scatter pass 与 cross-entropy opgraph经验，阶段二必须优先复用其 C++ 能力。重点是：

- typed gather/index correctness；
- conditional effect 与 alias/in-place 边界；
- gather、predicate、loss write 的同 kernel live state；
- 若 J 中存在可融合 summary，再由实际 dataflow打开 row reduction；
- RVV 增加 label gather、conditional write 与独立 Native baseline。

### 4.5 GroupNorm

正式 J 为 rank-3 f32输入、group/channel affine参数、两条 statistics reduction、iota/gather 和三个输出
`Y/mean/rstd`。

阶段二目标是通用 grouped multi-axis normalization：

- 从 typed shape/index map恢复 group-row domain；
- 同一次 root traversal 联合累积 statistics；
- raw-moment state只在 Q/D 允许时进入 Full；
- W/B gather 与 normalization epilogue融合，不物化 broadcast/index tensor；
- CUDA复用 row-reduction grouped/ColSeg 能力；RVV扩展 joint reduction 的 grouped affine mapping。

### 4.6 Liger RoPE

正式 J 为 rank-4 f16 Q/K 与 broadcast cos/sin，两个 in-place f16输出；图本身是 pairwise mul/add/sub。

现有 CUDA elementwise specialized matcher主要覆盖 rank-1/2，不能因为 RoPE 算术简单就声称已消费。需要实现或
复用通用 N-D contiguous/broadcast pointwise lowering：

- alias-safe 读取后再原位写回 Q/K；
- rotary pair的连续 packet/vector mapping；
- cos/sin broadcast与Q/K不同head extent；
- 不物化 split/concat/rotated-half tensor；
- RVV实现相同 pairwise packet和tail-safe in-place loop。

### 4.7 Qwen2VL MRoPE

正式 J 为 rank-4 f16 Q/K、三路 position-dependent cos/sin、iota、多个 gather、concat 与 rotary arithmetic。

这项是阶段二最重要的 indexed-fusion能力之一：

- CUDA优先复用已有 gather pass，补齐 N-D position map、concat消除和in-place rotation；
- Full 对比 materialized index/gather/concat 与 fused gather-rotate state；
- index computation与cos/sin packet只计算/读取一次；
- 不能按 Qwen 名称编码 channel section recipe，必须从 typed gather/concat/index关系恢复；
- RVV需要新增 multidimensional gather/index lowering和独立 Native baseline。

### 4.8 Flash Linear

正式 J 为 `f16 A[x0,x1] * B[x2,x1] -> f32 accumulate -> bias -> f16 cast`，两个 f16输出；C 包含 output tile、
K tile、K partition与contiguity hints。

仓库已经存在 CUDA tiled contraction、MMA sync、warp-specialized MMA 与 tuned linear fast path，禁止重新写一个
Python或kernel-name linear实现。阶段二的工作是让正式 J 完整进入这些现有 pass与统一search：

- C-Fixed使用确定性 direct contraction + C projection；
- Schedule-Only固定 contraction structure，搜索M/N/K tile、warps、stages；
- Full开放SIMT tiled、MMA sync、warp-specialized、K partition/SplitK与bias/cast epilogue fusion；
- 两个observable output必须由同一J边界完整产生，不能为性能忽略其中之一；
- 5090D/H100独立选择winner，不能复用跨卡winner；
- RVV优先复用成熟matmul/linear intrinsic或库实现作为external baseline，并建设真实RVV contraction lowering。

## 5. 当前能力复用与明确缺口

### 5.1 CUDA 已有轮子

优先复用：

```text
ConvertIntentElementwiseToGPU
ConvertIntentRowReductionToGPU
ConvertIntentGatherToGPU
ConvertIntentContractionToGPU
ConvertIntentContractionToGPUMMA
ConvertIntentContractionToGPUMMAWS
generic Intent -> Linalg -> CUDA fallback
```

现有 selection 已登记 tiled contraction、MMA sync、warp-specialized contraction、linear fast path与 gather/scatter。
新增8项的第一轮必须是直接对正式 J 做 registered C++ probe，按结果分为：

```text
specialized_complete
generic_complete
missing_lowering
```

该分类来自实际 post-pass MLIR 是否完整消耗 Intent op，而不是人工按 kernel 名判定。

### 5.2 CUDA 预计缺口

- rank-4/N-D alias-safe elementwise与broadcast packet；
- multidimensional gather/index/concat fusion；
- conditional effect/in-place output lowering；
- grouped multi-axis joint statistics；
- 正式 Linear J到既有 contraction/MMA search的完整接线；
- 多输出与alias boundary在所有候选上的Q/D和runner支持。

### 5.3 RVV 已有轮子

阶段一已经实现并必须复用：

```text
fixed/scalable parallel and reduction schedule
producer-consumer fusion
reassociation-gated explicit vector reduction
joint vector reduction
raw-moment variance
fused stable Softmax
```

### 5.4 RVV 明确缺口

当前 RVV 没有新增8项的独立 Native baseline，也没有通用：

- N-D gather/index/concat；
- alias-safe in-place RoPE；
- label gather与conditional effect；
- grouped multi-axis affine normalization；
- RVV contraction/linear search；
- JSD所需的multi-input masked stable row path。

TVD与xFormers RMSNorm最接近已有elementwise/row-reduction能力，但仍必须通过正式J probe与post-pass record确认，
不能提前写成已支持。

## 6. 实施阶段

### Phase A：真实 intake、workload与当前覆盖

1. 对8个正式J逐一materialize并运行CUDA/RVV当前registered probe；
2. 记录每项当前是specialized、generic还是missing，并保留实际post-pass MLIR；
3. 从原Source launcher/provider test/模型配置确定5090D/H100/RVV共用的natural workload；
4. 为每项确认Source timed boundary、所有observable outputs与alias行为；
5. 先跑current backend的correctness与一次性能筛选，定位真实瓶颈后再选pass，不按名字预设实现。

这里不做artifact哈希、冻结账本或流程性材料；产出只服务实际编译、correctness、shape选择和性能归因。

### Phase B：5090D/H100 通用 lowering

按可复用能力并行推进四组：

1. `TVD + Liger RoPE`：多输出elementwise、N-D broadcast、in-place alias-safe packet；
2. `xFormers RMSNorm + JSD + GroupNorm`：read-once/joint/multi-axis reduction与epilogue；
3. `CrossEntropy + Qwen2VL MRoPE`：gather/index/effect/concat fusion；
4. `Flash Linear`：复用现有SIMT/MMA/warp-specialized contraction与epilogue fusion。

每组先实现完整correct lowering，再开放Schedule-Only，最后增加真正结构不同的Full状态。

### Phase C：RVV 后端与Native

1. TVD/xFormers RMSNorm先复用已有vector/row-reduction路径；
2. 扩展N-D/in-place RoPE packet和multidimensional gather/index；
3. 扩展GroupNorm/JSD的joint multi-input/grouped reduction；
4. 为CrossEntropy补label gather/effect；
5. 为Flash Linear建设RVV contraction lowering，并选择成熟库或独立intrinsic Native baseline；
6. 全部使用单线程、相同natural workload与相同observable boundary。

### Phase D：八项完整实验

对8项分别在5090D、H100、RVV单线程产出：

```text
C-Fixed latency
Schedule-Only latency
Full latency
Source/Native latency
G_schedule
G_full
external ratio
winning concrete topology
关键 traversal/materialization/stage/reduction/index/MMA 计数
```

全表必须同时保留结构不适用或接近物理下限的负例。

## 7. 每项实验必须回答的问题

| Kernel | 必须回答的因果问题 |
|---|---|
| JSD | Full是否删除双分布summary/normalization的中间遍历，而非只换exp/log指令？ |
| TVD | 单遍elementwise在schedule解冻后是否已到带宽下限，Full为何没有额外空间？ |
| xFormers RMSNorm | 同一row J在两张GPU与RVV为何选择不同规约树/VL？ |
| CrossEntropy | gather/effect融合消除了哪些index、temporary和dispatch？ |
| GroupNorm | joint statistics和affine gather是否从多遍变成一次root traversal？ |
| Liger RoPE | N-D in-place packet是否减少指令/访存，而不改变alias语义？ |
| Qwen2VL MRoPE | fused gather-index-rotate相对materialized concat路径提升多少？ |
| Flash Linear | J是否真实打开SIMT/MMA/warp-specialized/K-partition路径，C retile与结构搜索各贡献多少？ |

## 8. 验收标准

### 8.1 后端能力

- 8/8 accepted J能够在5090D/H100完整lower、编译、执行并通过所有observable output的Q/D；
- RVV 8/8至少具备correct generic或specialized executable与独立Native baseline；
- 至少新增或接通四类通用能力：N-D alias-safe elementwise、indexed/effect fusion、grouped/joint reduction、
  contraction/MMA search；
- 新pass按typed IR匹配，缺少precondition时fail closed；
- post-pass MLIR、LLVM/target assembly能看到所声称的structure，不以requested flag计命中；
- 不使用Python target-kernel emission，不写kernel-name allowlist。

### 8.2 性能与消融

- 8项在5090D/H100均有完整C-Fixed/Schedule-Only/Full/Source表；
- 8项在RVV单线程均有C-Fixed/Schedule-Only/Full/Native表；
- 5090D、H100各至少2个stage-2 cell达到`G_full >= 1.50x`；
- RVV至少2个stage-2 cell达到`G_full >= 1.50x`，或在真实完成目标结构后给出直接物理证据；
- structural-applicable subset的`G_full`几何均值目标`>=1.50x`；
- 八项全量`G_full`几何均值目标`>=1.15x`；
- Full包含Schedule-Only spine，同轮complete-space timing中不得系统性回退；
- target winner不慢于对应Source/Native，未达到时保留结果并解释，不修改baseline追数。

### 8.3 结果完整性

每个headline cell必须同时报告：

```text
C-Fixed
Schedule-Only
Full
Full-without-the-new-mechanism
Source/Native
natural shape/dtype
all observable outputs correctness
traversal/materialization/index/stage/reduction/MMA evidence
```

不能只报告挑选出的强项；八项全表与structural-applicable subset必须同时给出。

## 9. 明确禁止

- 不修改accepted J来适配后端；
- 不把closure valuation冒充natural workload；
- 不削弱C-Fixed、Schedule-Only、Source或Native制造差距；
- 不把tile/vector/warps/stages参数变化误称为结构Full；
- 不从kernel名称、provider或benchmark输入推断语义权限；
- 不把Q/D tolerance反向写成B hard fact；
- 不忽略第二输出、stats输出、in-place effect或alias边界；
- 不复用Flash Linear的failed旧artifact；
- 不重复实现仓库已有gather、contraction、MMA、row reduction和fusion能力；
- 不用哈希、冻结、receipt整理或流程文档替代真实后端能力与性能数据。

## 10. 完成条件

1. 八个新增J的CUDA/RVV lowering状态全部闭合；
2. 至少四类阶段二通用能力进入统一search并有正向、边界、fail-closed测试；
3. 5090D/H100/RVV三目标完整三柱与external baseline表完成；
4. 强收益有同轮消融和post-pass/assembly证据，负例有物理解释；
5. 阶段一14项保持已完成状态，必要regression无系统性回退；
6. 达到第8节目标，或在真实实现结构后如实记录未达到的原因；
7. 代码、Task、结果提交，工作区干净。

