# Task 20：J 驱动的 Full 瓶颈消除与 1.5× 机制闭环

> 状态：执行中
>
> 类型：production backend 能力建设、真实 C++/MLIR 结构搜索、Fresh-14 横向性能闭合
>
> 继承：Task 19 已完成 dynamic-VL 整行规约与 producer/reduction fusion 的首个 RVV 纵切；
> 本任务继续扩展 Full 能到达、而固定 lowering 加 schedule search 到不了的目标程序

## 1. 第一性目标

本任务不以“多枚举几个候选”作为进展。每个新增 Full state 必须改变目标程序的 dataflow、完整遍历数、
materialization、reduction/scan topology、同步图或 stage graph，并实际移除当前 winner 的主瓶颈。

最终要证明两件事：

1. 同一 J 的 `Schedule-Only` 只能在一个固定结构中调 tile、vector、threads、warps、pipeline 等参数；
2. `Full` 能从同一 J 经通用 compiler transformation 构造新的具体 MLIR state，并在这些 state 内分别调度，
   因而获得固定结构的参数搜索无法达到的性能。

`Full/Schedule-Only >= 1.5x` 是高杠杆机制的目标线，不是要求所有算子违反硬件下限。已经单趟读取、单次
写回且接近带宽或 launch 下限的 elementwise/read-once kernel 是合理负例；禁止通过削弱 C-Fixed、
Schedule-Only 或 Source 制造差距。

## 2. 实验臂保持不变

```text
C-Fixed:
  DirectLower(J) + C 的一次确定性 target schedule 投影
  一个 executable，不搜索

Schedule-Only:
  与 C-Fixed 相同的 DirectLower(J) 和相同结构
  只搜索 schedule 参数

Full:
  完整包含 Schedule-Only 的候选
  额外枚举由 J/typed IR、B hard facts、target capability 与 pass contract
  准入的结构或数值 MLIR state
  每个 state 独立搜索 schedule
```

内部主要指标为：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_lowering = latency(Schedule-Only) / latency(Full)
```

GPU 外部 baseline 是原始 Source；RVV 外部 baseline 是严格同语义、shape、dtype、observable outputs 与
timed boundary 的独立 Native RVV。RefJ 只负责正确性，不是性能 baseline。

## 3. 当前事实与真正缺口

### 3.1 已证明的结构空间

Task 19 的 RVV 整行 scalable-vector accumulator 与 producer/reduction fusion 已经是可观察的真实 MLIR
变化。Fused Add RMSNorm `[2048,4096]` 的工程筛选中，组合 state 相对强 Schedule-Only 达到
`5.317x`，并与独立 Native RVV 基本持平。这证明“大收益空间不存在”不是事实；缺口是该能力尚未覆盖
共同根多规约、online normalization、scan、multi-input reduction 与跨 stage 状态。

CUDA 已有两个重要信号：Flash backward preprocess 的 padded/subgroup-feature state 已出现 1.5× 级内部
收益；Sparsemax 的 prefix/scan state 已带来显著外部竞争力，但当前内部 Full/Schedule-Only 仍约为
1.2--1.4×。这说明方向正确，也说明还需要真正减少 prefix work、barrier 或尾部流量，而不是继续调几何。

### 3.2 不能继续投入的伪空间

- 对已单趟融合的 SwiGLU、GeGLU、DYT 继续堆 tile/vector 开关；
- 对已经 read-once 的普通 row reduction 再增加等价的 block/warp 参数组合；
- 将 multi-stage 用在会重读完整输入、只增加 partial buffer 和 launch 的 workload；
- 只增加 candidate label、trace 或 Python controller 分支而不改变 post-pass MLIR；
- 把普通 producer fusion、load motion 或 predicate elimination称为新的结构空间；
- 修改 accepted J literal、Q、输入规模、observable outputs 或弱化固定结构 baseline。

## 4. Fresh-14 的结构分组

| 图结构 | Fresh-14 | Full 应打开的主要能力 |
|---|---|---|
| 已融合 elementwise | SwiGLU、GeGLU、DYT | 保留负例；只需避免回退 |
| 单规约 + epilogue | RMSNorm、Fused Add RMSNorm、KLDiv、Flash backward preprocess、xFormers RMSNorm-add | multi-input producer fusion、整行 vector/subgroup tree、padded feature |
| 共同根多统计 + epilogue | LayerNorm、PolyNorm、Flash one-pass LayerNorm | 一次输入遍历携带多个 accumulator，statistics 与 epilogue 共用 live state |
| normalization recurrence | Softmax | online max/sum recurrence、删除 exp/statistics 物化、cache-or-revisit state |
| prefix/support scan | Sparsemax | warp/CTA/RVV prefix、support/sum/tau 融合、合法时 active-prefix early stop |
| staged weighted merge | SplitK reduce | J boundary 内一次性 weight/normalization merge；只有 producer 在 J 内时才做跨 stage 消物化 |

表中分组来自 J/typed IR 的 op、axis、数据依赖和 observable boundary，不是 kernel-name dispatch。实现时
matcher 必须对全部 Fresh-14 运行；不匹配项正常返回 `not_matched`。

## 5. 本任务新增的高杠杆机制

### P1. 共同根多统计单遍状态

把同一 root、同一 row domain、相同 reduction axis 上的多个独立统计从：

```text
traverse X -> statistic 1
traverse X -> statistic 2
traverse X -> statistic 3
```

改成：

```text
one traversal of X
  -> accumulator 1
  -> accumulator 2
  -> accumulator 3
  -> one finalize per statistic
  -> fused epilogue
```

首个纵切是 PolyNorm 的三个 moments，随后覆盖 LayerNorm 与 Flash one-pass LayerNorm。RVV 生成一个
dynamic-VL loop、一次 root packet load 和多个 vector accumulators；CUDA 生成共享同一 register/shared
frontier 的 subgroup/CTA state。变换必须由 C++/MLIR pass 实现，不能把三个已有 reduction executable
在 controller 中拼成一个标签。

性能来源：减少 2--3 次完整输入读取、producer 重算、中间 tensor 写回与 loop/launch overhead。

### P2. Online normalization 与无物化 softmax

当前 Softmax 的 max、exp、sum、divide 仍形成多阶段 traversal/materialization。新增两种具体 state：

1. `online-statistics + second output pass`：第一遍以稳定 recurrence 同时求 max/sum，第二遍重新读输入并
   直接产生最终输出，不物化 exp/statistics tensor；
2. `resident-row online`：当 target capacity 与静态 shape 允许时，输入或 unnormalized values 保留在
   register/shared/vector scratch，统计完成后直接归一化写回。

这不是 schedule 参数。online recurrence 与普通 two-pass graph 擦除 tile 后仍具有不同 dataflow；可能
改变浮点观察顺序的 state 标记为 observation-changing，并由统一 Q/D 筛选。

性能来源：删除完整 exp tensor 的 store/load、减少 exp 重算或输入重读、减少 barrier 与 stage。

### P3. 通用 prefix/segmented scan 与 Sparsemax state

实现 source-neutral 的 scan transformation：

- CUDA warp prefix、CTA segmented prefix 与 compact shared frontier；
- RVV vector prefix/segmented scan；
- prefix、support predicate、supported-value sum、tau 与 epilogue 共用 live state；
- 只有 J/domain 中存在可验证的 descending-order 前提时才准入 active-prefix early stop；否则执行完整
  prefix，不能从 benchmark 输入或 Q/D 反推 hard domain fact。

性能来源：用并行 prefix 替代串行依赖，消除 cumsum/support/tau 间的全量 materialization，并在合法的
sorted domain 上停止无效尾部 work。

### P4. Multi-input、mixed-dtype 与 padded reduction

把现有 RVV 一输入 f32 row-reduction matcher扩展为：

- 多个同域输入 packet；
- 显式 cast 后的 f32 accumulator；
- elementwise producer payload 与 reduction融合；
- singleton/padded/tail axis 的安全映射；
- 仍然只有 row 末一次 horizontal finalize。

CUDA 将已经在 Flash preprocess 生效的 subgroup-feature/padded path按 typed map/select/cast/reduce/pad
结构通用化。主要覆盖 KLDiv、Flash backward preprocess，以及其他结构匹配的 masked row reduction。

性能来源：避免分离 cast/mul 临时值与标量 feature traversal，让多个输入在同一 vector/subgroup packet
中被消费。

### P5. J-boundary 内的跨 stage 消物化

SplitK 的 standalone reduce 已接近一次读取 partial、一次写回结果的物理下限；仅改 warps/vector width
不能形成 1.5× Full 收益。只有当 J 的 typed graph 确实包含 upstream producer 时，才枚举：

```text
producer -> materialized partials -> weighted merge
```

到：

```text
producer + online weighted merge
```

的结构 state。若 producer 位于 J boundary 外，保留 SplitK 为真实负例，不宣称消除了不可见的工作。

### P6. 数值 state 只服务结构，不单独追小收益

reduction tree、FMA、reciprocal/native approximation、accumulator precision 可作为 observation-changing
state，经相同 Q/D 进入 Full；但它们的首要作用是启用 vector tree、online statistics、scan 或 one-pass
merge。只有数值开关带来的 0--10% 不作为本任务的主要结果。Fresh-14 没有 typed contraction，因此不
搜索 TF32/MMA；等未来 `tl.dot`/matmul J 到来后再开放相应 state。

## 6. 搜索器与实际代码要求

每个 structural transition 必须登记并实际执行：

```text
typed A/MLIR pattern
required B hard facts
current-state precondition
target capability
resulting concrete MLIR rewrite
exact-preserving or observation-changing
independent schedule domain
```

具体要求：

- `Schedule-Only` 的候选全集原样包含在 `Full` 中；
- 每个新 state 有独立的 tile/vector/threads/warps/VL/LMUL 等 schedule domain；
- winner 由同一次 complete-space timing 中的 fresh finalist 样本选出；
- mechanism label 只由 post-pass structure 得出，requested flag 本身不算命中；
- correctness 统一走 Target--RefJ search filter 与固定 winner 的 Q/D closure；
- backend pass 不读取 kernel 名、Source、Replay、provider、历史 winner 或 benchmark label；
- target 计算只能由真实 C++/MLIR/LLVM pipeline 产生，Python 只负责编排和测量。

## 7. 执行顺序

### Phase A：共同根多统计

1. 实现 joint-reduction matcher、多个 vector accumulator 与 fail-closed negative tests；
2. PolyNorm RVV 完成 MLIR、LLVM、RVV assembly、Q/D 和 cold-data性能闭环；
3. 横向覆盖 LayerNorm、Flash one-pass LayerNorm，并增加 CUDA shared-frontier state；
4. 新 state 低于 `1.2x` 时直接 profile 主瓶颈，不继续加局部开关。

### Phase B：Softmax online state

1. 实现普通 graph 到 online max/sum recurrence 的通用 transformation；
2. 分别实现 resident-row 与 second-output-pass target state；
3. 在 5090D、H100、RVV 单线程各自搜索 schedule，不复用 winner；
4. 验证 exp/statistics materialization 或完整 traversal 确实减少。

### Phase C：Scan 与 multi-input/padded reduction

1. 完成 Sparsemax CUDA/RVV prefix state与合法 early-stop admission；
2. 完成 KLDiv、Flash preprocess 的 multi-input/mixed-dtype RVV vector path；
3. 将 CUDA padded/subgroup-feature path推广到全部结构匹配 J；
4. 每个 target 留下 full-scan 与合法 early-stop 两种可区分 state。

### Phase D：跨 stage 与横向矩阵

1. 对 SplitK 判断 producer 是否在 J boundary 内；只实现真实可见的跨 stage transition；
2. 对 Fresh-14 全量生成 Source/Native、C-Fixed、Schedule-Only、Full；
3. 5090D、H100 使用 code-warm/data-cold CUDA event；RVV 使用固定 CPU、单线程、data-cold；
4. 每个 target 独立选择 winner，并报告全量与 structural-applicable subset。

## 8. 性能验收

### 8.1 Headline 机制目标

- 至少 6 个 kernel-target cells 达到 `Full/Schedule-Only >= 1.50x`；
- 覆盖至少 4 个 kernel、3 种真实 MLIR transformation 和 3 个 target；
- RTX 5090D、H100、RVV 单线程各至少 1 个 `>=1.50x` cell；
- 共同根多统计、online normalization、prefix/scan 三类机制各至少有一个跨过 `1.50x` 的代表；
- 这些 cell 的 Full 同时不得慢于对应 Source/Native baseline。

若某机制真实完成并已消除预期 traversal/materialization，但仍低于 1.5×，如实保留该 cell，并用
instruction/transaction/launch/barrier 或 roofline 证据解释剩余下限；不得调整 baseline 定义追数。

### 8.2 Fresh-14 全量目标

- 14 个 kernel 在 5090D、H100、RVV 单线程都进入同一 controller；
- Full 从同一 complete-space timing中选择，因包含 Schedule-Only spine，不能系统性回退；
- 全量报告 `G_schedule`、`G_lowering`、external ratio、几何均值、中位数与
  `#>=1/#>=1.05/#>=1.5/#<1`；
- structural-applicable subset 的几何均值目标 `G_lowering >=1.50x`；Fresh-14 全量目标
  `G_lowering >=1.15x`；
- elementwise/read-once 负例保留在全量结果中，不能被 applicable subset替代。

## 9. 完成标准

1. P1--P4 至少三类由真实 C++/MLIR pass生成并进入统一 Full search；
2. post-pass MLIR、LLVM/target assembly 能直接观察到被消除的 traversal/materialization、共同 accumulator、
   prefix topology 或 padded packet；
3. 新 state 具有正向、边界和 fail-closed MLIR tests，并通过现有 backend/controller regression；
4. Fresh-14 三 target 的三柱与外部 baseline 数据完整，使用同一 natural workload、dtype、Q/D 与测量协议；
5. 达到第8节目标，或在真实高杠杆机制完成后给出不能达到的物理证据；
6. 不修改 J、SCRC、Source baseline、Q/D、workload 语义，不使用 Python target-kernel emission；
7. 代码、任务与结果提交，工作区干净。

## 10. 当前进展（2026-07-30）

### 10.1 P1 共同根多统计首个真实闭环

已实现通用 `joint-reductions` C++/MLIR state：

- 从多个 reassociable row reduction 沿纯 elementwise、identity-mapped producer DAG 回溯共同 root；
- 要求相同 row domain、reduction axis、静态输出 carrier和可证明的零初始化；
- 在一个 dynamic-VL loop 中只对 root 执行一次 `vector.transfer_read`，同时携带多个 vector accumulator；
- 每个 statistic只在行末执行一次 horizontal finalize；
- 第二次 Linalg fusion删除统计路径不再需要的 `x2/x3` tensor materialization，最终 epilogue直接消费 root
  与三个统计结果；
- 不同 root、非零 init、ordered reduction、非 identity producer或不满足 dominance的图 fail closed；
- 搜索器把 joint state作为独立 Full transition，只枚举 scalable-vector schedule，并以实际 module attrs、
  LLVM scalable-vector IR 与 RVV `vfred*` 指令作为命中条件。

Fresh PolyNorm natural workload `[2048,4096]`、f32、RVV4 CPU0 单线程、code-warm/data-cold 的完整
122-executable工程筛选结果为：

| 实验臂/状态 | Latency |
|---|---:|
| C-Fixed | 852.908 ms |
| Schedule-Only（完整固定结构 schedule 子空间的最好值） | 589.536 ms |
| 旧 Full 最好值（fusion + 三个独立 vector reduction） | 107.429 ms |
| 新 joint Full | **20.342 ms** |
| 独立 Native RVV | 41.890 ms |

因此本次同次测量得到：

```text
G_schedule                         = 1.447x
Schedule-Only / joint Full         = 28.982x
best non-joint Full / joint Full   = 5.281x
Native RVV / joint Full            = 2.048x
```

winner 的 post-pass record 为 `1 joint group / 3 reductions / 1 root load`，目标汇编含3条RVV浮点
reduction指令（`vfred*`），并明确记录
`python_target_kernel_emission=false`。122 个实际 executable 的 candidate correctness均通过；joint winner
对两个 observable output 的最大绝对误差分别为 `1.67e-5` 与 `4.10e-5`，符合固定 Q。

这仍是 `warmup=1、cold-iterations=1、batches=1` 的 go/no-go 工程筛选，不冒充正式论文置信数据。下一步
先对同一 bundle做多批 fresh finalist重测，再将共同根统计机制推广到可改写为 raw moments 的 LayerNorm
与 Flash one-pass LayerNorm；不以 PolyNorm单点替代 Fresh-14 横向结果。
