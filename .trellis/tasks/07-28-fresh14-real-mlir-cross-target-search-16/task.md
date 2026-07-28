# Task 16：Fresh-14 J 的真实 MLIR 跨目标搜索与 Source/Native Baseline 性能闭合

> 状态：执行中（Fresh-14 已 14/14 接入 typed J→Intent MLIR；5090D 的真实 Q/D、搜索与高杠杆 lowering 正在横向推进）
>
> 类型：production backend 能力建设、真实 lowering-state search、跨目标性能与消融实验
>
> 执行分支：`codex/backend-certified-j-search-20260725`
>
> 起点提交：`db71972b`（机器可执行 CF/E2/E3、C-prior oracle、RVV real-MLIR
> producer/reduction fusion checkpoint）

## 1. 第一性原理目标

本任务不以管理 artifact、整理内部证据或制作逐 Kernel 账本为目标。唯一主线是：

```text
同一个 fresh accepted J
  -> typed Intent MLIR
  -> 通用 compiler transformations 产生不同真实 MLIR states
  -> 每个 state 独立搜索 target schedule
  -> Q/D correctness 筛选
  -> CUDA 5090D / H100 / RVV target code
  -> GPU 与原始 Source kernel、RVV 与独立 Native RVV baseline 比较性能
```

任务必须证明三件事：

1. **生成代码有竞争力**：GPU 与原始 Source kernel 比较；RVV 与独立 Native RVV baseline 比较；
2. **C 不绑定 target schedule**：Schedule-Only 相对 C-Fixed 的收益；
3. **J 打开普通调参之外的实现自由度**：IntentIR 相对 Schedule-Only 的收益。

`C` 只改变有限预算下的候选顺序，不改变合法搜索空间和大预算性能上限。J、C、Source、Replay、
Gate receipt 的内部 provenance 由实现过程保持正确，但不单独成为阶段、表格或论文贡献。

## 2. 首批正式输入：fresh Task14-14

本任务的首批正式分母只包含以下 14 个 fresh accepted J：

1. Liger SwiGLU
2. Liger RMSNorm
3. Liger fused-add RMSNorm
4. Liger GeGLU
5. Liger LayerNorm
6. Liger Softmax
7. Liger DYT
8. Liger Sparsemax
9. Liger KLDiv
10. Liger PolyNorm
11. FlashAttention backward preprocess `_bwd_preprocess_do_o_dot`
12. FlashAttention one-pass LayerNorm
13. xFormers RMSNorm-add
14. xFormers SplitK reduce varargs

边界规则只在实现中执行，不扩张成独立管理任务：

- boundary dtype、shape relation、observable output 与 selected semantic constexpr 直接读取 J；
- C 中的 tile、vector、threads、pipeline 等值可以重新投影和搜索；
- fresh J 已声明的符号轴可绑定自然 workload；literal dtype/shape 不静默改写；
- SplitK 首批保持当前 `vararg_arity=2`、`WRITE_LSE=true` 与所选 direct-symbol 路径，不把
  `K/head_dim` 误认成 split count；
- legacy `production-jc-v1`、J_dev、JSD、TVD、xFormers RMSNorm 与两个 canary 不进入 fresh-14
  performance aggregate，可继续作为独立 compiler regression。

将 14 个 J 接入普通 loader/materializer 只是开始执行所需的输入接线，不设置“正式 intake”研究阶段，
不为此制作论文结果。

## 3. Source kernel 是强制 baseline

Source 不是可选 correctness 旁证，而是正式性能 baseline。

RTX 5090D 与 H100 上，每个可执行的 `(kernel, config, dtype)` 必须同时测：

```text
Source kernel
C-Fixed
Schedule-Only
IntentIR
```

其中：

- `Source kernel` 是被 Lift 的原始 Triton/上游 kernel，不是 Replay、RefJ 或重写后的 Python 实现；
- Source 与三个后端臂使用同一 shape、dtype、layout、observable outputs、输入数据分布、cache policy、
  profiler、timed boundary 与 target；
- Source 和 IntentIR 最终 winner 使用交替顺序、等数量的 paired measurement batches；
- GPU 正式外部 baseline 只有 Source，不再加入其他库实现。

原始 Triton Source 通常不能在 RVV 上执行，因此 RVV 使用独立 native baseline：

```text
Native RVV
IntentIR
```

Native RVV baseline 按以下顺序建立：

1. 优先复用语义、shape、dtype、layout 和 timed boundary 严格匹配的 ggml/llama.cpp 等实现；
2. 没有匹配实现时，编写独立 RVV intrinsic kernel；
3. baseline 不能复用 IntentIR 的候选生成、winner 或 target object，避免自我比较；
4. Triton-CPU 只在确有可执行路径的少量 kernel 上作为补充结果，不设为 fresh-14 全量要求；
5. scalar Python、RefJ latency 或 CUDA Source 不能替代 Native RVV baseline。

## 4. 正式实验臂

### 4.1 C-Fixed

```text
J -> canonical structural path
C -> target-aware deterministic schedule projection
exactly one concrete state
no search
```

C-Fixed 不是 Source realization，也不是把 GPU BLOCK 值直接当 RVV VL；它是 target 对 C role/axis 的
确定性投影结果。

### 4.2 Schedule-Only

```text
same J
same canonical structural path as C-Fixed
search target schedule only
```

允许改变 tile、vector width、warps/threads、pipeline、unroll、memory placement、RVV VL/SEW/LMUL 和
thread count；不得产生结构不同的 MLIR path。

### 4.3 IntentIR

```text
same J
search multiple admitted structural/numeric MLIR states
search schedule independently inside every state
select global fastest Q/D-correct state
```

IntentIR 的候选集合必须包含完整 Schedule-Only spine，再在相同总预算规则下访问其他结构状态。

主因果指标：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_lowering = latency(Schedule-Only) / latency(IntentIR)
G_source   = latency(GPU Source) / latency(GPU IntentIR)
G_native   = latency(Native RVV) / latency(RVV IntentIR)
```

`G_schedule` 与 `G_lowering` 优先从同一 target timing receipt 计算，跨进程独立报告比值只作漂移诊断。

### 4.4 C-prior 搜索效率

使用相同 J、候选全集、hard admission、总预算和 latency map，只改变 C-guided/Erase-C 顺序，报告：

- 固定预算下 best performance；
- 达到完整候选全集最终 best 的 95% 所需候选数。

C/no-C 不进入主性能三柱；C 在某 target 上无收益也是合法结果。

## 5. 必须补齐的真实后端能力

所有能力必须由 C++/MLIR/upstream compiler pass 形成 post-pass MLIR 与 target binary/object。Python 只做
输入绑定、搜索编排、correctness、计时与报告，禁止 Python target kernel emission。

### 5.1 Elementwise、grid 与 tail

覆盖 SwiGLU、GeGLU、DYT，并服务所有含 pointwise producer/epilogue 的图：

- target-derived grid traversal；
- contiguous scalar/vec2/vec4 或 target vector packets；
- masked tail 与不整齐 shape；
- producer/epilogue fusion；
- 消除不必要的完整 tensor traversal 和中间 store/load。

### 5.2 Producer--reduction--epilogue

覆盖 RMSNorm、fused-add RMSNorm、LayerNorm、Softmax、KLDiv、PolyNorm、Flash preprocess、Flash
one-pass LayerNorm、xFormers RMSNorm-add：

- materialized 与 single-pass fused states；
- subgroup/warp/vector reduction；
- workgroup/block/tree reduction；
- hierarchical 或真正减少重读的 multi-stage decomposition；
- 多 reduction 共享输入 traversal；
- statistic/epilogue 直接消费 partial，不回写再重读完整输入；
- register、shared-memory、occupancy、barrier 与 dispatch 的独立 schedule 搜索。

不得再次实现已经证明会重复读取 X/W 的低效 multi-stage 路径后停止；若 multi-stage 落败，必须依据
profile 修改数据流、partial layout 或适用域。

### 5.3 Sparsemax scan

实现真实 MLIR scan/prefix 路径，而不是把 scan 当普通 reduction：

- vector prefix/segmented scan；
- scan 与后续 threshold/reduction/epilogue 融合；
- 保持 J 中已有的输入与排序语义边界，缺少合法性事实时 fail closed；
- CUDA warp/block scan 与 RVV vector scan 使用同一高层 matcher、不同 terminal lowering。

### 5.4 SplitK reduction

在固定 vararg=2 的 fresh J 上实现并搜索：

- 两个 partial Out/LSE 的融合读取；
- LSE max/sumexp/numerator 单趟合并；
- head-dim vectorization 与 mask；
- M、B/G/H grid parallelism；
- CUDA warp/block 与 RVV vector/multicore states；
- 消除逐 split 的不必要 materialization、dispatch 或重复 exponentiation。

### 5.5 Numeric states

在不改变 boundary ABI/J 的前提下，将以下状态放进同一个 Q/D 搜索框架：

- reduction tree/order candidate；
- FMA contraction；
- internal accumulator precision；
- target native approximation；
- 未来 typed contraction 到位后的 IEEE/TF32/MMA states。

这些不是单独主图开关。每个 candidate 都必须在冻结的 Q/D 上先 correctness 后 timing；不能从 tolerance
反向修改 J，也不能为得到正收益放宽 Q。

## 6. 三个 target 同步推进

### RTX 5090D 与 H100

- 共用同一高层 matcher、transformation identity 与 workload config；
- 分别搜索 target schedule、warp/block/shared-memory/pipeline 与 capability；
- 两台 GPU 都运行 Source、CF、Schedule-Only、IntentIR；
- H100 不是只复跑 5090 winner，必须重新搜索；
- 不因 Triton Source 自带 autotune 就省略 Source baseline。

### RVV4

- 使用真实 MLIR -> LLVM/RISC-V object，不生成 Python/C 伪 kernel；
- 实现 dynamic VL strip-mining、SEW/LMUL、vector reduction/scan 与 tail；
- 实现真实 single-thread 与 16-thread execution，并固定 affinity/thread count；
- 同样运行 CF、Schedule-Only、IntentIR 和 C/no-C；
- 为 fresh-14 建立独立 Native RVV baseline；Triton-CPU 仅作可用子集的补充。

## 7. Workload 与计时

每个 kernel 至少一个来自上游 benchmark或真实模型调用的 headline configuration；shape-sensitive 的
normalization、reduction、attention component 和 SplitK 保留自然 sweep。禁止 small/medium/large 人工档位。

第一批建议 workload 方向：

| Kernel | headline 方向 |
|---|---|
| SwiGLU | LLM token rows × intermediate hidden size |
| RMSNorm / fused-add RMSNorm / LayerNorm | model token rows × hidden size |
| GeGLU | token rows × intermediate hidden size |
| Softmax / KLDiv / Sparsemax |真实 row 数 × reduction/vocab length |
| DYT | token rows × hidden size |
| PolyNorm | model rows × hidden size，保留 coefficient axis=3 |
| Flash preprocess |真实 B/H/S/D 与 rounded sequence capacity |
| Flash one-pass LayerNorm |真实 M/N 与所选 fused residual/bias branch |
| xFormers RMSNorm-add |模型 M/N，保留 X inout |
| SplitK reduce | decode-shaped B/G/H/M/K，首批 vararg=2 |

具体数值由上游 benchmark/model config直接确定并写入可执行 workload catalog；不为统一图形强迫所有
kernel 使用相同数量的 shape。各实验臂和所有 baseline 对同一 cell 使用完全相同 configuration。

主性能计时：

- candidate 与 Source 预编译，输入/输出/workspace 预分配；
- correctness 位于 timed region 外且先执行；
- timed region 只包含 prepared launch 或完整 prepared multi-launch graph；
- CUDA 使用 event；RVV 使用 monotonic clock、固定 affinity/thread count；
- GPU Source、RVV Native、CF/E2/E3 使用相同 warm-up、cache policy、采样数和 profiler；
- 报告每个 cell 的 median、完整样本、dtype、shape、target 与 binary/object digest；
- Roofline/counters/cache sensitivity 只用于解释瓶颈，不替代主 latency。

## 8. 执行方式：横向能力而非纵向打补丁

每轮遵循：

```text
14-J horizontal bring-up
  -> profile 真正瓶颈
  -> 实现一个 source-neutral MLIR mechanism
  -> 回归全部14项的 match/admission/compile/QD
  -> 在5090/H100/RVV重新搜索
  -> GPU与Source、RVV与Native baseline比较
```

禁止：

- 按 kernel name/source path/provider 选择 pass；
- 每个 kernel 一个 emitter、recipe 或实现 registry；
- Python 发射 CUDA/RVV 计算；
- 只跑预计加速的 applicable subset 后声称 fresh-14 提升；
- 用 generic 故意低效 materialization 制造几十倍内部对比；
- 把只改变 tile/warps/vector width 的结果称为 lowering gain；
- 因当前 E3/E2 小就停止在现有候选空间；
- 为追求正数修改 correctness tolerance、J、Source workload或计时边界。

## 9. 阶段与验收

### Phase A：14-J 端到端 bring-up

- 14 个 fresh J 全部进入普通 materializer；
- 14 个都形成 typed Intent MLIR；
- 在三个 target 上 compile/run，缺失能力直接修后端；
- GPU 14 项原始 Source kernel baseline 接通；
- RVV Native baseline 开始接通，优先复用严格匹配实现，否则使用独立 intrinsic kernel。

验收：14/14 有真实 MLIR、target binary/object、Target--RefJ correctness 和明确性能 cell；失败项不得因
不适用被删除。

### Phase B：高杠杆 lowering-state expansion

- 横向实现第5节的 elementwise、reduction/fusion、scan、SplitK、numeric states；
- 每个 state 有真实 post-pass MLIR digest、binary/object digest 和 actual launch plan；
- Schedule-Only path 与 IntentIR 的 schedule spine 完全一致；
- IntentIR 至少对结构适用子集形成擦除 schedule 后仍不同的真实 MLIR states。

验收：不是“候选存在”即结束；若新 state 没有解决 profile 指出的流量、同步、dispatch、并行度或
occupancy瓶颈，继续修改通用 compiler mechanism。

### Phase C：跨目标搜索与性能推进

- 5090D、H100、RVV 分别搜索，不复用另一 target winner；
- 对全部14项运行 CF/E2/E3 与 C/no-C；
- GPU 与 Source paired timing；
- RVV single/16-thread 与独立 Native RVV baseline；
- 持续以 profile 修正 search space 和 codegen，不把首轮数字当最终结果。

### Phase D：正式实验输出

GPU Figure panels：

```text
External: Source / IntentIR
Internal: C-Fixed / Schedule-Only / IntentIR
```

RVV Figure panels：

```text
External: Native RVV / IntentIR
Internal: C-Fixed / Schedule-Only / IntentIR
```

另输出 C-prior 小表。所有 aggregate 同时报告 fresh-14 全体、结构适用子集和 per-kernel；主结论不能
只选择 winner。

### 当前执行 checkpoint（2026-07-28）

- fresh-14 已 14/14 由 typed ContractIR materializer 形成真实 Intent MLIR；production materializer
  不再以旧 functional-boundary matcher 作为 membership gate；
- SplitK fresh J 的真实图不是 legacy packed `concat + reduce`，而是两个 partial Out/LSE 的静态展开
  stable weighted merge。新增 C++ SSA matcher 直接识别该代数并要求所有函数体 op 都可从两个结果反向
  到达；不读取 kernel 名、Source 路径或 receipt；
- 该图已由 `Intent -> GPU -> NVVM -> PTX -> cubin` 形成单一真实 CUDA kernel，runtime ABI 为四个输入
  加两个输出指针；D=16 的非满 warp 尾部有显式界限保护；结果已与原始 Triton Source 对齐；
- 同一 concrete lowering path 已形成 `rows_per_block={1,2,4,8}` 四个真实编译状态，对应
  `block_threads={32,64,128,256}`，每个状态带 post-pass topology/trace receipt，并已进入普通搜索；
- 5090D 当前 workload 的 bring-up 计时为：Source 约 4.07 us，C-Fixed/default rows=4 约 4.86 us，
  Schedule-Only 当前最优 rows=8 约 3.35 us；即初步 `G_schedule≈1.45`、`G_source≈1.21`。这些数值尚未
  经过正式 paired batches，不进入最终 aggregate；
- generic 19-launch materialized 图约 50.8 us，而单 launch 约 3.35 us，只用于验证 whole-graph
  materialization/dispatch 瓶颈确被消除，不作为正式 `G_lowering`。当前这个 cell 的 IntentIR winner 仍是
  Schedule-Only spine，故其正式 `G_lowering` 预期接近 1；需要由其他真实结构状态补足该因果指标。
- Fresh PolyNorm 的 `tensor<1xf32>` typed singleton broadcast 已由通用 row-reduction evaluator 作为
  uniform scalar 消费，同时保留一般 ParamVector 只能经已验证 gather 访问的 fail-closed 规则。由此同一
  C++ pass 已完整消费三个共享输入的 reduction、row epilogue 与 compact `[M,3]` 输出，形成一个 cubin；
  结果已与原始 Source 对齐。预算 8 的首轮已实际编译 subgroup/workgroup、FMA 与 native-div 请求等 5 个
  状态；当前 closed workload 初测 Source 约 4.34 us，最佳 plain subgroup 约 3.41 us，仍待正式 paired/QD。
- Fresh Flash preprocess 已复用既有 `PaddedRowDotLowering`：同一 source-neutral matcher 现在同时接受
  runtime-bound mask/select 图和 J 已静态解析边界后的 direct-cast 图；后者以三指针 ABI 形成一个 cubin，
  在同一 launch 中完成 f16->f32、乘积、规约和 `[33]->[64]` 零填充。输出及 padding tail 已与 Source
  对齐；serial-row、subgroup-feature、vector packet 与线程数均进入普通搜索。当前 closed workload 初测
  Source 约 4.21 us，后端候选约 4.13 us，主要结论是能力/结构闭合而非该小 shape 的大幅加速。
- 横向 budget-8 编译审计现已使 fresh-14 **14/14** 至少有一个 C++/MLIR 高级状态形成 cubin：三项
  pointwise 图有真实 vectorized states；九项普通/scan/padded row 图覆盖 subgroup、workgroup 或
  subgroup-feature；PolyNorm 覆盖共享输入多规约；SplitK 覆盖 unrolled weighted merge。这里仅说明
  backend capability 与 target compilation，不替代后续 14 项 Q/D、Source paired timing 与跨 target 结果。
- CF/E2/E3 已改为在同一 search receipt 中强制重新测量 C-Fixed、Schedule-Only spine 与 expanded
  finalist；`G_schedule`/`G_lowering` 不再由三个独立进程的首轮 winner latency 拼接。自然 workload
  的 headline valuation 已覆盖 fresh-14 14/14，并保持 accepted J digest 不变。
- Flash preprocess 的 padded-row lowering 已从小 shape 的 serial 特例扩展为任意正 batch、head-dim
  `<=1024` 的 batch grid + subgroup feature strip。自然 `[4,16,8192,128]` 上，512-thread/16-warp 的
  expanded topology 初测约 160.5 us，Schedule-Only spine 约 279.6 us，同 receipt
  `G_lowering≈1.74`；paired Source 约 160.8 us。该结果已经是不同 MLIR topology 的真实收益，但仍是
  measurement-round=1 的 bring-up 数字。
- Softmax 自然 `[2048,4096]` 上，C-Fixed 到 512-thread row-reduction schedule 的同 receipt
  `G_schedule≈1.55`，后端约 12.1 us、paired Source 约 17--18 us；这验证了 C 可重投影/retile 的主线。
  RMSNorm 与 DYT 的自然大 shape 则分别表现为约 1.08x Source gain 和主要 schedule gain，expanded
  numeric state 在内存受限形状上没有被虚构成大收益。
- Sparsemax 的真实 Source ABI 接收 `x` 与上游已排序的 `sorted_x`，sort 明确位于 timed kernel 之外。
  旧 workload 将两者独立随机生成，使执行域不再代表实际调用，并让 Q/D 正确拒绝排序相关优化。
  Harness v4 现用通用 `sort_descending_last_axis` 派生关系从同一 `x` 构造 `sorted_x`，四腿共享同一值；
  该关系不修改 J、也不伪造 compiler permission。自然 `[2048,4096]` 已重新 3/3 通过 CF/E2/E3：
  Schedule-Only 约 17.3 us，expanded reciprocal state 约 16.2 us，同 receipt `G_lowering≈1.07`，paired
  Source 约 15.5 us。J 尚无 typed descending relation，因此 sorted-prefix early-stop 仍 fail closed；
  当前数字只属于通用 scan + Q/D numeric search，不冒充 contract-proven sorted specialization。
- Fresh xFormers RMSNorm-add 与 SplitK 的 Q/D intake 已接到同一 typed RefEval：前者消费 J returns
  中的 inout post-state，后者从唯一 node-less masked load 恢复 pointwise consumer，并用 J broadcast
  axes 验证坐标；两者均没有新增 J 字段或放宽 Q。closed SplitK 已实际完成 CF/E2/E3 搜索（约
  0.97/0.93/0.93 us 的 bring-up 值）。
- Fresh SplitK 的自然 `[B,G,H,M,D]` 上游布局与当前 J 的 `p34=H -> x1`、`p35=G -> x2` 轴声明不一致；
  `H!=G` 时 shared workload builder 正确拒绝，而不是用交换 constexpr 或改 shape 掩盖。该 cell 暂不
  计入自然 headline；需要前端重新提供 source-grounded 轴对应后，才可形成新的 accepted J。

## 10. 总体验收

1. fresh 14 J 全部由同一普通 backend 消费，无 Python target emission；
2. 14 项在 5090D/H100/RVV 均有 compile/correctness/performance 结果或准确的后端阻断；
3. GPU 原始 Source kernel baseline 覆盖全部可执行 headline cells；
4. RVV fresh-14 接通严格匹配、独立实现的 Native RVV baseline；
5. CF/E2/E3 使用同一 workload、target、compiler、Q/D、profiler与可比较预算；
6. `G_schedule`、`G_lowering`、`G_source` 和 `G_native` 可由机器报告直接计算；
7. C/no-C 使用同一候选全集与 latency map；
8. 新收益来自真实 MLIR dataflow/topology/numeric-state变化，不是报告开关或弱 baseline；
9. 结果包含 fresh-14 全量与 per-kernel，不只报告加速项；
10. 全量 Python/MLIR regression 通过，文档、实验配置与代码提交保持一致，工作区干净。

## 11. 非目标

- 不修改 frontend Lift/Replay/Gate verdict；
- 不为后端性能制作新的 SCRC Gate；
- 不把 artifact/digest/使用域管理写成研究阶段；
- 不创建 implementation-family object、family registry 或 per-kernel emitter；
- 不把 legacy v1/canary/J_dev 混入 fresh-14 正式 aggregate；
- 不以一次 benchmark 数字代替完整后端能力。
