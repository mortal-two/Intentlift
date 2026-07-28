# Accepted J+C production backend：真实 MLIR 搜索与跨目标实验

> 状态（2026-07-28）：fresh Task14 的 14 个 accepted J 已进入统一 typed J → Intent MLIR
> materializer。本文只描述 production backend 当前要实现和测量的能力，不承担 artifact 管理、
> SCRC 归档或内部账本职责。

## 1. 第一性原理

production backend 的对象不是 kernel 名称、Python recipe 或 implementation family，而是同一个 J 下由
通用 compiler transformation 产生的具体 MLIR state：

```text
accepted J=A+B
  -> typed Intent MLIR M0
  -> B/current-state/target/pass-contract hard admission
  -> concrete transformed MLIR M1, M2, ...
  -> each state searches its own target schedule
  -> target executable
  -> Q/D correctness filter
  -> performance selection
```

其中：

- A 提供 typed computation、dataflow、axis、reduction/scan/update 与 observable state；
- B 固守 domain、predicate、access、alias/effect、output mapping、ABI 等 hard boundary；
- C 只给候选排序和初始 schedule，不授权 transformation；
- Q/D 只在 compiler 外层裁决可能改变数值观察的 executable；
- Python 只做绑定、搜索编排、执行、比较、计时和报告，不发射 CUDA/RVV 计算 kernel。

core lowerer 只读取 J、可选 C 与 target capability，不读取 Source/Replay/π/Gate diff、kernel 名或
历史 benchmark 来生成候选。

## 2. 已删除的非目标能力

后端不实现 artifact 身份链、冻结工具、内容寻址、证明收据或性能结果缓存。具体规则是：

- 不为 J/C、输入、执行域、MLIR、toolchain、object、binary、spec 或 report 计算内容标识；
- 不按代码内容折叠候选，不把内容标识用作候选名、缓存键、准入、正确性或验收条件；
- 不保留“默认关闭”的兼容入口、隐藏工具或备用 schema；
- accepted artifact 历史 JSON 中与后端计算无关的旧字段由 loader 忽略；
- report 直接记录 J 声明的 operator/shape/dtype、具体 valuation、target、transformation、schedule、
  launch、correctness 与 latency。

每个显式枚举候选占一个预算单位；编译失败也占一个预算单位。即使两个候选偶然生成相同代码，仍按
两次显式访问计算，不建立代码内容身份。

候选编译完成后，benchmark 在同一次运行中持有 executable 和 prepared launch closure，以保证 timed
region 只含 launch。这是一次测量的显式运行对象，不是跨搜索复用的结果缓存。

## 3. 首批正式输入

fresh-14 是首批正式横向分母：

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

后端直接消费这些 J 的 typed 内容：

- symbolic extent 可由 workload valuation 绑定；
- literal shape/dtype/selected constexpr 不由 backend 改写；
- C 中 tile/vector/threads/pipeline 等 prior 可以针对 target 重投影和搜索；
- legacy corpus 与 canary 只作独立 compiler regression，不进入 fresh-14 aggregate。

## 4. Materializer 与 ContractView

`materialize_j(J, nu)` 只做 J 的 typed compiler encoding：

- 输出真实 Intent MLIR；
- 统一降低 typed ContractExpr AST，不依赖旧字符串 matcher 作为成员资格 gate；
- 完整投影 boundary domain/access/effect/output/value-dimension facts；
- 支持 J-declared constexpr、state/inout、predicate/fill 与 full-axis logical step；
- 对影响 hard semantics、memory safety 或 ABI 且尚未实现的 relation fail closed。

`Attach_C(M0,C,target)` 在 materialization 之后附加可擦除 prior；`Erase_C` 恢复同一个 M0。C 的变化
不能改变 ContractView capability、hard admission 或候选全集，只能改变有限预算下的访问顺序。

ContractView 是 invocation-local typed analysis，不是新 artifact。每个 transformation 声明：

```text
A/IR structural match
required B facts
current-state preconditions
target capabilities
transformation result
exact-preserving or observation-changing
schedule domain
```

## 5. 真实搜索空间

### 5.1 Elementwise 与 grid/tail

通用能力包括：

- row/flat grid traversal；
- scalar/vec2/vec4 或 target vector packet；
- 非整齐 shape 的安全 tail；
- pointwise producer/epilogue fusion；
- 消除不必要的完整 tensor traversal 和中间 load/store。

主要覆盖 SwiGLU、GeGLU、DYT，并服务所有图中的 pointwise 子图。

### 5.2 Producer--reduction--epilogue

搜索不只改变 threads/tile，而要产生结构不同的 MLIR state：

```text
materialized producer -> temporary -> reduction -> temporary -> epilogue
producer/reduction/epilogue single-pass fusion
subgroup/warp/vector reduction
workgroup/block/tree reduction
hierarchical or multi-stage decomposition
```

每个 state 独立搜索 register/shared placement、barrier、occupancy、vector width、threads 和 stage 参数。
multi-stage 只有真正减少重读、提高并行度或解决寄存器/共享内存瓶颈时才有价值；仅把一个强单阶段
kernel 拆成多个重复读取 X/W 的 launch 不是目标。

### 5.3 Sparsemax scan

Sparsemax 需要真实 prefix/scan lowering，而不是把 scan 当作普通 reduction：

- CUDA subgroup/workgroup prefix；
- RVV vector prefix/segmented scan；
- scan、threshold、reduction 与 epilogue 的结构融合；
- 缺少 sorted relation 等 hard fact 时，对相应 specialization fail closed。

### 5.4 SplitK

首批 fresh J 固定当前 vararg=2，搜索：

- 两组 partial Out/LSE 的融合读取；
- max/sumexp/numerator 单趟 merge；
- head-dim vectorization 与 tail；
- CUDA warp/block 与 RVV vector/multicore mapping；
- 减少重复 exponentiation、dispatch 和中间 materialization。

### 5.5 数值 state

reduction tree、FMA、内部 accumulator precision 和 target native approximation 可以形成
`observation-changing` concrete MLIR state。它们不修改 J，也不能从 tolerance 反向得到 memory/effect/ABI
许可；统一在预先固定的 Q 与 D_search 上直接比较 Target--RefJ，通过后才计时。最终 winner 再在独立
D_cert 上直接比较 Target--Source、Target--Replay 和 Target--RefJ。

当前 fresh-14 没有 typed contraction，因此不能把 TF32 当作 row reduction 加速轴。只有后续 accepted J
真正含 typed dot/contraction 时，才搜索 IEEE/TF32/MMA 等 target state。

## 6. 正式实验臂

```text
Source / Native RVV baseline

C-Fixed:
  canonical structural path
  + target 对 C 的确定性 schedule 投影
  + one candidate, no search

Schedule-Only:
  same canonical structural path
  + target schedule search

IntentIR:
  same J
  + complete Schedule-Only spine
  + additional hard-admitted structural/numeric MLIR states
  + independent schedule search per state
```

主指标：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_lowering = latency(Schedule-Only) / latency(IntentIR)
G_source   = latency(Source) / latency(IntentIR)       # CUDA
G_native   = latency(Native RVV) / latency(IntentIR)   # RVV
```

`G_schedule` 与 `G_lowering` 从同一次 candidate timing run 的 fresh finalist 样本直接计算。独立进程的
ratio 只作漂移诊断。

C-prior/Erase-C 使用相同显式候选序列、hard admission、总预算和 latency map，只改变访问顺序，报告：

- 固定预算下的 best latency；
- 到达完整序列最终 best 的 95% 所需候选数。

## 7. Baseline 与计时

CUDA 5090D/H100 的外部 baseline 是被 Lift 的原始 Source kernel。RVV 没有 Triton Source 时，使用
语义与 timed boundary 严格匹配的独立 native implementation；优先复用 ggml/llama.cpp 等实现，否则
编写独立 RVV intrinsic kernel。RefJ latency 不能替代性能 baseline。

同一个比较 cell 必须具有相同 operator、shape、dtype、layout、observable outputs、输入分布、target、
cache policy、profiler 与 timed boundary。

计时规则：

- Source 和所有后端候选预编译；
- tensor/workspace/launch 参数在 timed region 外准备；
- correctness 在 timed region 外执行；
- timed region 只包含 prepared single launch 或完整 prepared graph；
- CUDA 使用 event，RVV 使用 monotonic clock、固定 affinity/thread count；
- Source 与最终 winner 使用交替顺序、等数量 paired batches；
- 主表使用预热后的 L2/LLC-controlled single-shot；warm-cache 与 streaming 另表；
- report 保留全部 latency samples、median、launch/code-size 和直接结构字段。

Roofline 只解释瓶颈：J-derived logical bytes 不是硬件 DRAM transaction；低算术强度也不自动等于当前
shape 已经 memory-bound。需要 target copy ceiling、足够大 working set、实际 latency 与可选 counters
共同支持 memory-wall 结论。

## 8. 三个 target

### CUDA 5090D / H100

- 共用高层 typed matcher 与 C++/MLIR transformations；
- 分别重新搜索 target schedule，不复用另一 GPU winner；
- 都执行 Source、C-Fixed、Schedule-Only、IntentIR；
- H100 portable runner 只执行本地已编译的真实 cubin/launch graph，不安装 Python target emitter。

### RVV4

真实编译链是：

```text
Intent MLIR -> Linalg -> structured tiling/fusion/vectorization
            -> bufferized loops -> LLVM dialect/IR
            -> llc riscv64 (+v,+zfh) object
```

后端需要补齐 dynamic VL、SEW/LMUL、vector reduction/scan、tail、single-thread 与 16-thread execution，
并执行独立 Native RVV baseline、C-Fixed、Schedule-Only、IntentIR 和 C/no-C。

## 9. 当前推进顺序

每次只围绕横向 compiler capability 循环：

```text
fresh-14 match/admission
  -> compile all applicable states
  -> Q/D correctness
  -> profile real bottleneck
  -> improve one generic MLIR mechanism
  -> rerun 5090D/H100/RVV search and baselines
```

当前优先级：

1. 完成 fresh-14 在 5090D 的全部 Source/CF/E2/E3 cells；
2. 提升 producer--reduction--epilogue read-once 和真正有用的 multi-stage；
3. 完成 Sparsemax scan 与 SplitK dataflow；
4. 同步推进 H100 搜索；
5. 完成 RVV dynamic vector/multicore 与独立 native baselines；
6. 输出 fresh-14 全体、适用子集和 per-kernel 三种视图。

禁止按 kernel 名选择 pass、为单个 benchmark 编写 emitter、用弱 materialized baseline 制造内部大收益、
只报告加速项、修改 J literal 或放宽 Q 来追求正数。
