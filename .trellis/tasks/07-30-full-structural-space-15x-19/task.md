# Task 19：Full 真实结构空间扩展与 1.5× 性能闭合

> 状态：执行中
>
> 类型：production backend 能力扩展、Fresh-14 横向性能闭合、CUDA 5090D/H100 与 RVV 单线程
>
> 继承：Task 18 已建立的三柱、cold-data、Source/Native baseline、Q/D 与首批 Sparsemax/RVV 实现；
> 本任务接管尚未完成的 1.5× 目标，重点不再是增加候选数量，而是增加真正改变瓶颈的 MLIR 状态

## 1. 第一性目标

Full 只有在生成了 Schedule-Only 无法到达、并且实际移除了主瓶颈的目标程序时，才可能稳定超过
Schedule-Only。继续增加 tile、warps、vector width 或布尔开关，不能解决当前约 `1.02--1.05x` 的
GPU `Full/Schedule-Only` 差距。

本任务只投入以下四类高杠杆变化：

1. 消除一次或多次完整 tensor traversal、temporary materialization 或 launch；
2. 把串行 reduction/scan dependency 改成真实的并行 tree、prefix 或分层 finalize；
3. 让多个 statistics、exp、weight 或 output 共用一次输入读取和中间状态；
4. 把原本跨 loop/stage 的 live state 保留在 register/vector/shared memory，而不是重新落入全局内存。

目标不是让每个 memory-traffic 已接近下限的 kernel 都达到 1.5×，而是：

```text
在 Fresh-14 全量不删负例的前提下，
让所有真实存在结构 headroom 的 cell 进入 Full，
并让至少一组跨 kernel、跨 target 的结构结果稳定达到 Full/Schedule-Only >= 1.5x。
```

## 2. 固定实验语义

### 2.1 外部比较

- CUDA：原始 Triton Source 与最终 Full；
- RVV：语义、shape、dtype、observable outputs 和 timed boundary 一致的独立 Native RVV 与最终 Full；
- RefJ 只作正确性裁决，不作性能 baseline。

### 2.2 内部三柱

```text
C-Fixed:
  DirectLower(J) + C 的一次确定性 target schedule 投影
  一个 executable，不搜索

Schedule-Only:
  与 C-Fixed 完全相同的 DirectLower(J)
  只搜索 tile/vector/threads/warps/unroll/pipeline/grid 等 schedule

Full:
  完整包含 Schedule-Only
  额外从同一个 J 构造真实改变 dataflow、reduction/scan topology、
  materialization 或 stage graph 的 MLIR states
  每个 state 独立、充分地搜索自己的 schedule
```

本文不使用 `implementation family`。后端对象是同一个 J 上经通用 compiler transformation 得到的
具体 MLIR state；擦除 schedule 数值后结构仍不同，才算 Full 新空间。

### 2.3 不可改变项

同一比较 cell 固定：J、operator、natural shape、dtype、layout、输入分布、observable outputs、Q/D、
target、cache policy、profiler 和 timed boundary。禁止通过削弱 C-Fixed/DirectLower、缩小 workload、修改
accepted J literal、放宽 tolerance 或减少输出制造 Full 收益。

## 3. Fresh-14 横向分母

1. SwiGLU
2. RMSNorm
3. Fused Add RMSNorm
4. GeGLU
5. LayerNorm
6. Softmax
7. DYT
8. Sparsemax
9. KLDiv
10. PolyNorm
11. Flash backward preprocess
12. Flash one-pass LayerNorm
13. xFormers RMSNorm-add
14. SplitK reduce

每项都进入 5090D、H100 和 RVV 单线程 controller。某个结构不适用时保留正常 Schedule-Only/Full
结果，不为它伪造新路径；SwiGLU、GeGLU、DYT 等已经单趟融合的 elementwise kernel 很可能是合理负例。

## 4. Full 能力地图与优先级

### M1. RVV vector-accumulator row reduction

本任务起点的实验性 pass 只是把固定宽度 reduction tile vectorize，并可能在每个小 tile 内做一次
horizontal reduction；这不是高性能 row reduction。Phase A 已将其替换为下述整行 accumulator结构。

目标结构：

```text
dynamic-VL strip-mined row loop
  -> vector partial accumulator 跨完整行存活
  -> 尾部 masked load
  -> 行末一次 horizontal finalize
  -> epilogue
```

要求：

- 在真实 C++/MLIR pass 中生成 `scf.for`、scalable `vector` load/mask/reduction；
- MLIR 结构显式存在，不能只给 LLVM 设置 scalable-vector flags；
- LLVM IR 出现 scalable vector，RVV 汇编出现对应 `vsetvli`、vector load/store 与 reduction；
- exact order 与 observation-changing reassociation 分成不同 state，后者统一经过固定 Q/D；
- 支持自然 shape 的 tail，而不是仅支持 `extent % 4 == 0`。

主要覆盖：RMSNorm、Fused Add RMSNorm、LayerNorm、Softmax、KLDiv、PolyNorm、Flash one-pass
LayerNorm、xFormers RMSNorm-add。

性能来源：消除标量 reduction dependency chain，让一个 horizontal reduction 替代每个小 tile 的反复
scalar finalize，并减少 accumulator load/store。

### M2. Producer--statistics--epilogue 单趟状态

将：

```text
producer -> temporary -> statistics -> temporary -> epilogue
```

改成一个 row program：producer value、partial statistics 和 epilogue 所需 live value 保存在
register/vector/shared state。多个 reduction 共享一次 traversal；只有最终 observable result 落地。

主要覆盖：LayerNorm、PolyNorm、Softmax、Fused Add RMSNorm、Flash one-pass LayerNorm、xFormers
RMSNorm-add、KLDiv。CUDA 与 RVV 共用 typed graph matcher，各自生成 target structure。

只有 post-pass 确实减少完整 load/store、alloc 或 kernel stage 时才计入 Full；普通 elementwise tiling/fusion
不能冒充该机制。

### M3. CUDA padded-feature / joint-statistics reduction

把 Flash backward preprocess 已有的 subgroup-feature winner推广为按 A/typed-IR 结构匹配的通用 pass：

- serial feature traversal 改为 subgroup packet；
- 多个 statistic 使用同一 warp/shared frontier；
- 对长 reduction 搜索 CTA tree 或 partial/finalize，但第二阶段只读 partial statistics，不重读完整输入；
- 5090D 与 H100 分别搜索 mapping，不复用 winner。

该机制已有 5090D `Full/Schedule-Only ≈ 1.66x` 信号，优先补齐 H100 并推广到所有结构匹配的 J。

### M4. Sparsemax fused prefix/scan

目标不是继续堆 Sparsemax 专属几何，而是完成可复用的 scan lowering：

- CUDA warp/CTA segmented prefix；
- RVV vector prefix/segmented scan；
- prefix、support predicate、supported-value sum、tau 与完整 epilogue 共用 state；
- 只在 J/QD 足以准入时使用 active-prefix early stop；否则执行完整 scan；
- pass 按显式 scan/dataflow 匹配，不读取 kernel 名。

当前 5090D/H100 已约为 `1.39x/1.24x`，只有新的 prefix work、barrier 或 tail traffic 被实际消除后才继续；
小于 `1.2x` 的新变体不横向扩散。

### M5. SplitK one-pass weighted merge

针对 J 中明确可见的两组 partial Out/LSE：

- 每个 split 的 max、sumexp 与 weight 只计算一次；
- numerator 与 normalization 在同一 pass 合并；
- head dimension 使用 target vector packet，标量 weight 在 warp/vector lanes 中广播；
- 消除 J boundary 内真实存在的重复 exp、dispatch 和 materialization；
- J boundary 之外的 upstream partial producer 不宣称被融合。

CUDA 搜索 warp/block merge，RVV 搜索 vector merge。只有 dataflow 真正变化才进入 Full；单纯改变
warps-per-row 属于 Schedule-Only。

### M6. 数值 state 只作结构助推器

reduction tree、FMA、accumulator precision、reciprocal/native approximation 可以形成
observation-changing MLIR state，并经过预先固定的 Q/D。它们不修改 J，也不能授权 alias/effect/ABI
变化。单独只有 0--10% 收益时不作为 1.5× headline；当它们是启用 vector tree、online statistics 或
one-pass merge 的必要组成时，与对应结构一起评估。

当前 Fresh-14 没有 typed contraction，因此不搜索 TF32/MMA。

## 5. Headroom-first 执行规则

每个机制先在一个自然 workload 上完成纵切，但纵切必须直接验证主瓶颈是否被消除：

1. 比较 Schedule-Only winner 与新 state 的 post-pass MLIR/LLVM/assembly；
2. 确认减少了完整 traversal、global transaction、barrier、launch、串行 dependency 或重复高代价运算；
3. 使用同一 Q/D 通过 target execution；
4. 给新 state 独立调 schedule；
5. cold-data fresh remeasurement 达到 `>=1.2x` 才推广到全部结构匹配 J；
6. 低于 `1.2x` 时先根据真实 profile判断瓶颈是否仍在；若结构已消除但物理 headroom 不存在，保留负例并转向下一机制，不继续堆局部开关。

这个门槛只决定工程资源是否继续投入，不用于从最终 Fresh-14 数据中删除负例。

## 6. 搜索与 J 的真实作用

每个 transformation 必须显式声明：

```text
A / typed-IR structural match
required B facts
current-state preconditions
target capabilities
resulting MLIR transformation
exact-preserving or observation-changing
independent schedule domain
Q/D policy
```

J 的作用不是直接指定 warp、VL 或某个后端 recipe，而是暴露 computation/dataflow/reduction/scan/update、
domain、effect、alias 和 observable boundary，使同一个通用 transformation 能安全匹配多个 kernel。

禁止：

- kernel-name/source-path/benchmark allowlist；
- Python 发射 CUDA/RVV 计算 kernel；
- 仅增加 candidate label、trace 或开关而不改变目标 MLIR；
- 把 fusion、materialization、reduction/scan topology 混入 Schedule-Only；
- 从 tolerance 反推 memory/effect/ABI 许可；
- 为制造 1.5× 而退化 DirectLower 或 C-Fixed；
- hash、冻结、receipt、artifact 身份链或与性能无关的账本。

## 7. 测量与主结果

- natural workload 与 accepted J 的 shape/dtype保持一致；
- code/executable 预编译，launch 参数在 timed region 外准备；
- 主结果为 code warm、data cold 的 L2/LLC-controlled single-shot；
- CUDA 用 event，RVV 用 monotonic clock、固定 CPU affinity、单线程；
- Source/Native 与 Full 用交替顺序、等数量 paired batches；
- 每个 target独立搜索，不跨 5090D/H100/RVV 复用 winner；
- C-prior/Erase-C 只作相同总预算下的搜索效率小表，不进入最终性能上限归因。

最终每个 kernel/config 报告：

```text
Source or Native
C-Fixed
Schedule-Only
Full
G_schedule = C-Fixed / Schedule-Only
G_lowering = Schedule-Only / Full
G_external = Source-or-Native / Full
winning concrete MLIR transformation
```

主图仍只有外部比较与内部三柱；具体 transformation 用于解释 winner，不新增主柱。

## 8. 1.5× 与竞争力验收

### 8.1 结构收益

- 至少 5 个 kernel-target cells 达到 `Full/Schedule-Only >= 1.50x`；
- 覆盖至少 3 个 kernel、3 种具体 MLIR transformation traces 和 2 个 target；
- 5090D、H100 各至少 1 个；RVV 单线程至少 2 个；
- 由 J graph facts 预先判定的 structural-applicable subset，几何均值 `>=1.50x`；
- Fresh-14 全量同时报告 geometric mean、median、`#>=1/#>=1.05/#>=1.5/#<1`，工程目标
  `G_lowering >=1.15x`，不能用 applicable subset替代全量。

### 8.2 外部竞争力

- GPU 的 `>=1.5x` headline cell 同时不得慢于对应 Triton Source；
- RVV 至少 3 个代表 kernel 的 Full 延迟达到独立 Native RVV 的 `<=1.25x`；
- 若真实高杠杆能力完成后仍未达标，如实报告，不修改实验语义追数。

## 9. 执行顺序

### Phase A：收口当前 RVV vertical slice

1. 修正现有显式 row-reduction pass 的 reduction/fusion record 与 pass 顺序；
2. 增加正向、fail-closed MLIR test和真实 Fresh J lowering smoke；
3. 把固定 `vector<4>` tile升级为 dynamic-VL vector accumulator + one finalize；
4. 在 RVV4 单线程完成 Q/D、汇编和性能闭环。

### Phase B：三个 GPU 高杠杆 vertical slices

1. Flash backward preprocess padded-feature 在 5090D/H100 闭环并通用化；
2. SplitK one-pass weighted merge；
3. multi-stat producer--statistics--epilogue state，优先 Flash one-pass LayerNorm、PolyNorm、Softmax。

Sparsemax保留当前跨 target强状态；只有确认能继续减少 prefix/barrier/tail work时再扩展。

### Phase C：横向推广

按 A/B/typed-IR facts将每个已验证机制应用到 Fresh-14全部匹配项；每个 state在三个 target分别调度，
elementwise和无headroom项保留真实负结果。

### Phase D：正式数据

运行 Fresh-14 的 Source/Native、C-Fixed、Schedule-Only、Full；输出全量、structural-applicable subset、
per-kernel与C-prior小表。

## 10. 完成标准

1. 达到第8节的真实结构收益与外部竞争力门槛，或在高杠杆机制真实完成后给出不可达的物理证据；
2. 至少3种新增结构由 C++/MLIR pass真实生成，并在 MLIR/LLVM/assembly 中可观察；
3. Fresh-14在5090D、H100、RVV单线程全部进入同一三柱controller；
4. 每个新 state独立调 schedule并通过同一Q/D；
5. 不弱化任何 baseline，不修改J/Q/workload，不使用Python target-kernel emission；
6. 相关编译、MLIR lit、Python controller与target execution测试通过；
7. 代码、task和最终数据提交，工作区干净。

## 11. 当前进展（2026-07-30）

### 11.1 M1/M2 首个真实闭环

已实现通用 `vectorize-linalg-row-reduction` C++/MLIR pass：

- 支持 fixed/scalable vector，scalable strip size由 `vector.vscale`产生；
- vector accumulator跨完整 reduction axis存活，行末只做一次 horizontal finalize；
- `vector.transfer_read`处理自然 shape tail；
- 支持任意数量的前导 parallel/unit dims；
- 支持 fused `linalg.generic` region中的任意 Elementwise scalar producer payload，而不是只认裸 `addf`；
- 只有 reduction自身带 `reassoc` 时准入；ordered reduction继续fail closed；
- 最终 object必须包含真实 RVV `vfred*` 指令，否则候选编译失败；
- 显式 dynamic-VL state只枚举scalable schedules，不复制到不合法的fixed-VLEN组合。

Fused Add RMSNorm natural workload `[2048,4096]`、f32、RVV4 CPU0单线程、code-warm/data-cold
工程筛选结果：

| State | Latency |
|---|---:|
| Schedule-Only strong scalable canonical | 168.122 ms |
| Fusion only | 37.438 ms |
| Explicit whole-row vector reduction only | 51.983 ms |
| Fusion + explicit whole-row vector reduction | **31.619 ms** |
| Independent Native RVV | 31.683 ms |

组合 state 成为 winner：`Full/Schedule-Only = 5.317x`，`Native/Full = 1.004x`。三个 observable
outputs全部通过比较；winner由真实MLIR pipeline生成，含2条RVV floating-point reduction指令，
`python_target_kernel_emission=false`。

这是一次cold调用的go/no-go工程筛选，不冒充最终论文置信区间；它已经证明M1/M2存在远大于1.5×的
真实headroom，下一步应横向推广并做fresh finalist重测。

### 11.2 Softmax边界

同样的8-state筛选在Softmax `[2048,4096]` 上得到约`1.254x` Full/Schedule；fusion-only仍是winner，
组合vector reduction没有消除max、exp、sum与divide之间的多阶段materialization。因此Softmax不继续堆
reduction vector参数，后续只在online/joint-statistics dataflow真正减少阶段或完整遍历时再投入。
