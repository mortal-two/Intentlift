# Accepted J+C production backend：真实 MLIR lowering、B 合同与 C 搜索

> 状态：production catalog 已 append-only 扩展为 content-addressed 的 **18 个**闭合 J；2026-07-27。
> RTX 5090D、H100 与 RVV4 的 all-18 closed-valuation 报告已冻结；它们是横向消费/因果基线，
> 不冒充 production-shape 主表。下一里程碑是 all-18、all-target、L2-controlled 的符号 valuation
> 矩阵；历史 closed-12/16 数字只作追溯。

本文记录已冻结 18 个 SCRC Gate1/Gate2 闭合 Kernel 的 production backend 实现。当前权威结果、
证据边界、压缩原始报告与 SHA-256 见
[18-J production backend 结果](results/production-backend-closed-18/README.md)。原
[closed-12 报告](results/production-backend-closed-12/production-backend-report.md)和
[closed-16 smoke](results/production-backend-closed-16/backend-implementation-smoke-2026-07-26.md)
只保留为历史，不参与当前 aggregate。

新增六项已经通过同一个 catalog loader 导入；snapshot 只复制已接受 artifact/C/evidence，未重新 Lift、
未修改 J，也不把 `/tmp` 作为长期 truth：

| case | frozen snapshot digest | evidence kind |
|---|---|---|
| xFormers RMSNorm-add | `e8e5451fa837443024a435b34b125f5c9f2261bdaf178842f80898875cf727be` | zero-model Gate2 recheck |
| Liger Sparsemax | `ca974aed6b5954b197a063f0e96d62652d569b5728608818adcdbfbc3cb11fda` | zero-model Gate2 recheck |
| Liger KLDiv | `b39a3df6d64782be475b5b459f6b7dc01efdb3356fa85abe22a75334f296df7a` | certified run report |
| FlashAttention one-pass LayerNorm | `76ffde5e8eb4e53d0fca5afad8bb1d44ad4ba3f6c9df764dc67b21aa95402480` | zero-model Gate2 recheck |
| masked copy canary | `10d67ee0b4f061e1ce28b3e6b1ebfc371be2825692671c927e80b8e743e4ad49` | certified run report |
| masked row square-sum canary | `31208517fc91dca13e48573198090ca46c6905e2c8d814109f0149cd72d9c892` | certified run report |

当前 18 项均无 typed contraction；Flash `do_o_dot` 是 FP32 multiply + row reduction。因此当前性能
推进不把 TF32 误套到 row reduction；TF32/IEEE 精度状态留给后续 `tl.dot`/matmul/完整 Attention J。

## 1. 输入边界

core lowerer 的持久输入只有：

- accepted canonical `J=A+B`；
- 可选、可擦除的 `C`；
- target capability。

它不读取 Source、Replay、π、Gate verdict、Q、kernel 名或 provider 身份来决定
lowering。`corpus/production-jc-v1/` 保存 18 个 content-addressed
`intentir.production.jc-snapshot.v1` 对象和一个 stable-name catalog。catalog loader
会同时核验 snapshot、J、C digest；验收导出目录的 `<name>.json` 和冻结后的
`<sha256>.json` 使用同一个消费入口。

```text
accepted J+C snapshot
  -> Emit(J, concrete invocation)              M0 Intent MLIR
  -> ContractView(J, target)                   hard-boundary admission
  -> Attach_C(M0, C, target) / Erase_C         ordering only
  -> registered C++ MLIR pass candidates
  -> CUDA: GPU dialect -> NVVM -> PTX -> ptxas cubin
     RVV: Linalg -> bufferized loops -> LLVM -> riscv64 object
  -> generic Q/D candidate filter
  -> performance selection among passing states
  -> held-out final-winner backend closure
```

Python 只负责 snapshot transport、候选编排、CUDA Driver ABI 绑定、计时与报告。
所有目标计算和 local/structural/schedule transformations 都由注册的 C++/upstream MLIR pass 生成；不存在
Python/Triton 替代 kernel emission。仓库中历史 `cuda/flash.py` 只服务旧 benchmark tests，不在
production catalog/search/bundle 的可达依赖图内；架构测试禁止 production 模块导入它。

这里的 Q/D filter 是已经实现的唯一通用 `SearchAndCloseController`，不是每个 Kernel 的 Python
emitter 或 validator。core MLIR compiler 不读取 Q；controller 只通过 signature-driven opaque harness
执行 Source/Replay/RefJ/target，并使用冻结 comparator。`D_search` 对每个具体 post-pass MLIR state 做
direct target–RefJ；性能选择后的固定 winner 在输入 digest 不相交的 `D_cert` 上直接比较
target–Source、target–Replay、target–RefJ，禁止 tolerance 传递。缺 Replay 时可以留下 search evidence，
但不能产生 formal backend-closure receipt。

## 2. J、B 与 constexpr

`materialize_j()` 先对 canonical J 做 deterministic materialization。当前支持：

- functional boundary projection；
- dense predicate/fill materialization；
- exact value-dimension relations；
- full-axis logical step：包括 no-scan/null-init carry，也包括 scan 输入与 typed literal/reference
  carry init；该投影与 RefJ 的完整逻辑轴一步语义相同，不把 C tile 当语义；
- J-declared constexpr 对 typed `effect_if` 的精确分支专化；普通 scalar `value_if` 不伪装成
  constexpr，而在 M0 中保留为标准 `scf.if`；
- J.signature 明确声明的 rank-0 `constexpr` specialization。
- direct input/output binding 的确定性 identity materialization；它生成通用 `intent.map_unary identity`
  而不是 Python copy kernel，并在 C++ generic/CUDA scalar lowerer 中闭合。

constexpr specialization 不是从 Source 猜常量。只有 J.signature 的
`ParameterKind.CONSTEXPR` 可以被绑定；普通 scalar/pointer 输入会被拒绝。若 constexpr
同时受 B 的 value-dimension equality 约束，materializer 必须先验证值与 dimension
一致，再删除 runtime ABI 参数。该检查曾真实拦截 xFormers RMSNorm 的旧索引投影问题。

每次 CUDA 搜索前都会复用 `analyze_j_for_cuda()` 建立 ephemeral
`CudaContractView`，记录：

- J、target 与 boundary digest；
- 已消费 B facts；
- unsupported B facts；
- deterministic boundary projection。

存在未消费的 alias/effect/access guard/storage/update/reduction/output relation 时，搜索在
候选编译前 fail-closed。C 不能绕过该入口。编译审计会向每个已闭合 kernel 的同一 M0
注入未支持 `may_alias` 作为负对照：历史 closed-12 的 with-C 与 Erase_C 两臂都在 codegen 前以
`alias_relation` 拒绝。

Sparsemax 复用同一个 functional-boundary projection 消费 authored access-local guards；scan 上的
reduction relation 与 reduce 使用同一闭合合同。若 projection receipt 未覆盖某个 guard，ContractView
仍返回 `missing/unsupported`，不会仅因图中出现 select 而猜测它已被消费。

该负对照只证明显式 `may_alias` 不会被 C 绕过，不能单独证明 v7 的空 alias list 等价于完整
no-alias。当前 `b.readonly_nonalias_inputs.v1` 对空列表的解释属于待收紧的 B0/B1 contract gap；规范
要求后续 capability receipt 同时给出 exact J carriers、artifact/schema completeness rule 与 derivation
version。无法证明 alias relation set complete 时应返回 `missing_contract`，而不是从具体 benchmark
buffers、测试通过或空数组猜 no-alias。

## 3. C 的真实消费

`Attach_C` 把 C slot 解析为带 role、value、subject/axis attachment 的 search prior；
`BaseIR(S_C)` 始终是同一 M0，且 `Erase_C(S_C) == M0`。C 不进入 pass legality，也不
改变 candidate ID、performance-cache identity 或 candidate universe。

当前候选枚举在 `elementwise/row_reduction/padded_row_reduction` 等 registered lowering path
之间 round-robin，再用 C 的 tile/thread prior 排同一路径种类内的 schedule。这些名字只作为
`lowering_path_kind` 与 pass receipt，不是可发射的 compiler family 或 semantic object。round-robin
只避免有限预算下某种 matcher 候选被饿死；C
仍不决定哪条 path 合法。

正式消融固定 J=A+B、ContractView、target、compiler、workload、candidate universe 和
budget，只切换 `Attach_C/Erase_C`。当前 role-aware scorer 完整消费 C 的
`tile_extent/reduction_tile_extent/execution_warps/constexpr` roles；它只对已有 schedule dimension
计算距离，不生成 candidate 或改变 admission。RTX 5090D all-18、budget=8 的配对报告中，with-C
相对 Erase-C 的 best-latency 几何平均优势为 **1.0195x**；更大 budget 下两者趋于收敛。因此 C 的
贡献是有限预算收敛速度，而不是 `ReachHard`、Q/D feasibility 或 correctness。

portable exporter 现在不会只编译 with-C prefix 再把未执行的 Erase-C ID 写进报告。它分别沿两条
order 消耗相同 distinct-state/failure budget，运输两者实际访问 state 的并集，以 post-pass MLIR
digest 去重；target runner 在同一批 correctness/timing 样本中分别选择两条 order 的 best。报告中的
`c_prior_over_erase_c_speedup` 因而是真实 target-side 配对，不是 requested-order 的代理指标。

## 4. J 驱动的大粒度机制与 local-B 次级消融

### 4.1 首要机制：切断 materialization、dispatch 与规约瓶颈

第一性原理上的搜索对象是 concrete post-pass MLIR state，不是几个局部开关。materializer 从 A 的
typed producer--consumer DAG、region、axis 与 observable state 出发；`ContractView` 再从 B 的
domain/predicate/effect/update/output mapping 与 target/pass contract 判定 transition 是否 hard-safe。
后端由此可以在同一个 immutable J 下同时形成：

```text
materialized producer -> temporary -> reduction -> temporary -> epilogue
whole-graph producer/reduction/epilogue state

sequential/subgroup/workgroup/hierarchical reduction states

exact/FMA/precision-or-approximation states
```

这里的 readonly/noalias capability 只审计 A DAG 中实际流向 observable output 或 effectful state 的
active inputs；闭合 artifact 为 ABI/Replay 保留的 dead legacy input 不再无理由阻塞 fusion。该
active-input 集合由通用 DAG use 分析确定，不按 kernel 名维护 allowlist。

RTX 5090D 的正式 all-18 E3 回执中，6 个 J 同时形成 materialized multi-kernel 与 whole-graph
state。两侧各自调 schedule 后，消除中间物化与 dispatch 的几何平均收益为 **17.1258x**，最高
PolyNorm **45.0139x**；其余为 GeGLU 10.7662x、LayerNorm 19.8063x、masked row square-sum
39.7813x、Softmax 15.4004x、SwiGLU 4.2902x。这是 J 发挥作用的核心证据：收益来自编译图、全张量
内存流量和 dispatch 数量改变，不来自 Python 发射或 per-kernel recipe。

在相同 all-18 numeric ladder 中，加入 reduction topology 带来全体 **1.0409x**、12 个适用 J
**1.0622x**；FMA 仅 **1.0003x**，如实作为负结果；precision/approximation 带来全体
**1.0676x**、2 个适用 J **1.1271x**。它们属于 observation-changing states，统一通过冻结
Q/D，而不是让后端从 tolerance 或 `float_nondeterministic` 推出静态数值许可。

### 4.2 local-B 只作次级证据

B 不能从 J 中关闭，因为持久语义对象就是 `J=A+B`。L0/L1 固定完整 J、ContractView、结构/数值
state、C、target、workload、budget 与计时协议，只切换注册 local transformations 是否进入候选：

- L0 仍完整消费 B 并执行 fail-closed legality，但关闭 rank-0 capture/load motion 等 local transition；
- L1 加入 `required_b_capabilities` 被 ContractView 满足的 local transition；
- 适用性只从 C++ pass 产出的 post-pass `cuda.b.*` trace 和 actual count 判断，不相信 requested toggle。

当前正式 L0/L1 两臂各 18/18 完成 held-out backend closure；L1/L0 全体为 **1.0162x**，10 个真实
观察到 local-B state 的 J 为 **1.0295x**，Flash preprocess 最大 **1.1827x**。这些数值只证明
B-conditioned local optimization，不作为图级 lowering-state expansion 的创新主证据。C 消融则只
回答有限预算顺序，二者不能混写。

RVV whole-graph Linalg fusion同样改变 materialization/loop topology。实现把它从 schedule 拆成
独立 `RiscvTransformationState`；materialized 与 fused state 分别搜索 O2/O3、vectorize on/off，
post-pass Linalg op count 与 topology digest 来自真实 MLIR receipt，不是 pass 开关自报。

## 5. 注册的真实 MLIR lowering paths

### Elementwise row lowering path

`ConvertIntentElementwiseToGPU.cpp` 对完整 typed map/cast/select DAG 生成一个
`gpu.launch`。搜索轴包括 block threads、row/flat layout、vector width，以及由只读非别名
B capability 授权的 singleton/rank-0 host-by-value ABI。非 2 次幂 tail
使用安全 in-bounds 地址 + guarded store，不产生越界，也避免每个 input load 的发散
`scf.if`。

已消费：DYT、GeGLU、SwiGLU、TVD。TVD 的 optional-label 输入在 constexpr
`HAS_LABEL=false` 后成为 dead ABI input；MLIR host signature 仍保留该 memref，但 C++
lowering 不生成虚假设备 load。stable/native/approx math 都作为不同 actual MLIR state 留下
math-op receipt；只有通过 Q/D 的状态才参与性能选择。

### Row reduction lowering path

`ConvertIntentRowReductionToGPU.cpp` 对 normalization/loss 型完整 DAG 生成 read-once
register tile、warp/shared reduction 和 typed epilogue。搜索轴包括：

- block threads；
- register tile extent；
- scalar/vec4 memory path；
- deferred/preloaded epilogue Col；
- rank-0 host-by-value / device-uniform-load ABI。

reduction topology 与 FP32 FMA 已进入 production lowering-state search。C++ pass 在结果 MLIR 上记录
实际 shuffle/barrier/all-reduce/FMA 数量与 transformation trace；请求 FMA 但实际未产生 FMA 的状态会
按 post-pass MLIR digest 去重。它们标记为 `observation-changing`，在 `D_search` 通过 target–RefJ 后
才可计时，固定 winner 再走 `D_cert` 三条 direct edges。现有 `reduce.assoc`、value-dimension equality、
predicate/output/alias projection 继续参与 hard admission 或 codegen，而不是数值权限开关。

已消费：Liger RMSNorm、fused-add RMSNorm、LayerNorm、Softmax、JSD、PolyNorm、
xFormers RMSNorm、xFormers RMSNorm-add、KLDiv、Sparsemax、Flash one-pass LayerNorm 与
masked row square-sum。KLDiv 的
scan/carry 在 J-only full-axis materializer 展开；Sparsemax 的 scan 与 logical `and` 有 generic/target
lowering；Flash 的普通 scalar `value_if` 保留为 `scf.if`，C++ row state 再对 pure arms 做 predication。
该 state 的实际 receipt 为 2 次 value-if predication、10 次 constexpr select folding，而非 Python 分支。

### Masked short-dot + pad lowering path

`ConvertIntentPaddedRowReductionToGPU.cpp` 结构化匹配：

```text
same B-derived (p < validP && n < validN) mask
  -> masked A/B [1,M,P,N]
  -> f32 multiply
  -> reduce(axis=3)
  -> authored zero pad P -> Q
```

matcher 不读函数名。搜索 block threads、full-feature vector load、`serial_row/subgroup_feature`
reduction topology 和 rank-0 scalar ABI。
正式 E2/E3 中 Flash preprocess 的 expanded-state winner 相对 fixed-path schedule-only 为
**1.4047x**；正式 L0/L1 中 rank-0 host capture 的局部增量为 **1.1827x**。两者分别归因于
state-space expansion 与 local-B，不能相互替代。shuffle/topology、host capture count 和
`cuda.b.rank0_host_value_capture.v1` 都由 post-pass C++ receipt 记录，不由 Python toggle 自报。

## 6. 搜索与计时

`search_cuda()` 对每个 candidate 执行：

1. complete Layer-B audit；
2. 指定唯一 registered fast path 和 pass options；
3. C++ whole-graph admission；
4. GPU/NVVM/cubin compilation；
5. 在 `D_search` 上 direct target–RefJ filter；
6. 通过者的 device timing；
7. top-finalist robust tournament 与 fastest candidate selection；
8. 固定 winner 在独立 `D_cert` 上 direct target–Source/Replay/RefJ closure。

v2 实验合同把预算单位定义为不同的 canonical post-pass MLIR state；重复 state 保留在 audit log，
不消耗候选预算。finalist 使用 fresh round-robin、正反交替 traversal 并排除初次排序样本。最终性能
确认使用四组等数量 Source/target paired batches并交替执行顺序；确认前两侧按相同声明时长做
round-robin untimed warm-up。只计 prepared launch/graph，不把 allocation、cache eviction 或
correctness synchronization 放进 timed closure。

运行结果不用于放宽 memory/effect/ABI hard boundary。正确性测试与 performance selection 分离；
历史 frozen-12 winner 另外与 canonical J 比较；新 controller 已对 observation-changing state 执行通用
Q/D feasibility 与 direct-edge closure。baseline 是 SCRC 已使用的 exact declarative Triton
workload adapter/source slice，而不是新写的替代 kernel。

performance cache key 包含 M0 digest、target、candidate、toolchain 和 benchmark workload，
但不包含 C digest；同一 candidate 在 Attach_C/Erase_C 两臂可安全共享测量。

缓存/工作集与 Roofline 使用独立 machine spec
`spec/lowering/backend-performance-measurement-v1.json`；当前 canonical JSON content digest（不是文件字节 SHA-256）为
`aaf6dbbc5c48925cc58d81d66d2d68410e94953bc95350ebdeb29f998d20eae0`，每份 bundle/report 绑定当时 canonical
digest。参考 TileLang/Triton 的 kernel benchmark 口径，正式 CUDA 主数字为：先完成
code/module/page/frequency warm-up，然后每次单次调用前在 event interval 外驱逐 L2，用 CUDA
event 计时。当前 runner 将这个 **L2-controlled steady-state** 协议记为 `cold-data`。

warm、streaming 与 cold-data 仍是不同 estimand，禁止混合 aggregate：

- warm：persistent address 的 warm-cache prepared launch/graph，只作调试或 deployment-specific 附录；
- streaming：轮转互不相交的 exact-invocation replicas，实际 working set 必须严格超过 LLC，只作压力敏感性；
- cold-data：每次 single-shot 前用严格超过 LLC/L2 的 target working set 驱逐，驱逐与同步不计时；这是主表口径。

J workload-cost receipt 给出 boundary logical bytes、audited FLOP、special-function/comparison/control
独立计数和算术强度。GPU 在目标端实测 sustained D2D copy 与 TF32-disabled FP32 ceiling；H100 还能用
Nsight Compute 12.8 读取真实 DRAM/L2 counters。logical bytes 永远不冒充 hardware transaction。
如果实际 latency 小于 `0.95x` logical DRAM Roofline lower bound，说明 cache reuse 或
traffic/ceiling 模型不匹配，runner 会记 `cache_reuse_or_roof_model_mismatch` 并禁止
memory-wall claim。大于 `2x` 同样不再称为 roof-approaching；大于 `10x` 是明确的
`latency_or_unmodeled_work_dominated`。带未建模 special-function ops 或不完整 J cost 的点也
不单独支持 memory-wall claim。

closed-18 原始 case 的 boundary 只有约 1--8 KiB，主要用于闭合与小延迟性能，不是
production-shape 主表。`roofline-large-j-only-v1` 是 masked-copy Target--RefJ 压力工具，不因
尺寸大就升格为 Source 性能证据。正式放大只允许绑定 frozen J 中已声明的符号轴，
不从 kernel 名猜 shape scaling。

### 6.1 当前 symbolic/literal 执行边界

目标无关的 workload valuation catalog 已为 **18/18** 登记 `symbolic-medium-v1` 与
`symbolic-large-v1`，CUDA 与 RVV exporter 共用同一份封闭 `HarnessBindingSpec`。profile 只对
frozen J 真实声明的符号 extent 生效：

- Flash preprocess：只放大 `heads=x0`；J 仍为 `[1,x0,33,16] -> [1,x0,64]`，因此
  batch/sequence/head_dim/rounded extent 均不变；
- Flash one-pass LayerNorm、JSD、LayerNorm、Sparsemax、TVD：绑定各自 J 已声明的
  row/feature symbols；
- DYT、fused-add RMSNorm、GeGLU、KLDiv、Liger RMSNorm、masked row square-sum、PolyNorm、
  Softmax、SwiGLU 与两个 xFormers RMSNorm：只放大已声明的 feature symbol，外层 literal
  `rows=2` 保持不变；
- `masked_copy` 放大 J 的 `length=x0`，但 accepted Source 没有 `program_id`，所以 expanded
  valuation 明确为 `Target--RefJ-only`；它仍在 18-J 矩阵中，不能伪造 Source speedup。

profile receipt 同时绑定 base/active snapshot digest、不变的 J digest、具体 extent valuation、
C digest、保留的 literal constraints 与 per-kernel baseline policy。18 个 medium 与 18 个 large
valuation 已逐项通过 `build_workload_jc_suite`，并在 `sm_120a` 完成 canonical true-MLIR
post-pass/cubin 编译烟测。该烟测证明 profile 可被真实后端消费，不是 performance/correctness 主结果；profile
本身也不证明所有 shape、不修改旧 SCRC 证书。

### 6.2 all-18 横向评测矩阵

主性能与 correctness 覆盖是 **18 个 J x 每个 target x 每个已登记 legal valuation**，
不是仅对少数代表 Kernel 运行。E2/E3/E4、L0/L1、N0--N3 在 18/18 上执行；
`not_matched`、`literal_shape_only`、target 缺 Source/Replay、compile/QD failure 都保留在矩阵中。
主报告同时给出 all-corpus、applicable subset 和 per-kernel p50/speedup。只有 Nsight/CUPTI、
Roofline 机制分解等高开销解释证据可在每个结构类上选 1--2 个 valuation；这不能
替代全 18 主表或消融。

首个按此矩阵落地的高级 CUDA 扩展不是 kernel-name recipe，而是通用的多规约 tiled MLIR
状态：同一依赖层的独立规约共享一次 feature-tile traversal，依赖规约从 workgroup frontier
继续遍历；每个 partial 使用可复用 shared-memory tree，epilogue 也消费该 frontier，从而保持
global row read-once。当前 matcher 在 18-J large profile 上真实准入 7 项，其余 11 项仍以
`not_applicable` 或精确 rejection 留在矩阵中。该路径已经通过 true-MLIR 生成与 lit 检查；只有
Target--RefJ 和稳定计时完成后，才能写成性能收益，不能因“候选已生成”预先宣称加速。

## 7. H100 分支

H100 机器不安装/调用 IntentIR compiler。`export_cuda_benchmark_bundle.py` 在本地分别用
`INTENT_GPU_CHIP=sm_90a --arm E2|E3` 生成 cubin、launch plan、输入、canonical-J reference 和 exact
Source baseline；`run_cuda_benchmark_bundle.py` 在 `ssh h100` 只加载 cubin并做 CUDA Driver
launch、correctness 和 paired timing。

all-18、equal-budget=32 的 portable 结果为：E2/E3 都 18/18 执行成功，E3/E2 几何平均
**1.1196x**，E3/Source **1.0539x**，15/18 E3 达到或超过 Source，15/18 的 E3/E2 超过 1%。
runner 直接检查 Source–RefJ 与 Target–RefJ；由于 bundle 尚无 opaque Replay 和 held-out
Target–Source/Replay/RefJ 三边，它是跨 target 性能与 correctness 证据，不冒充 formal backend
closure。

## 8. RVV 分支

RVV 后端不生成 C/Python kernel。它从与 CUDA 相同的 deterministic Intent MLIR M0 出发，
执行以下真实编译链：

```text
registered ConvertIntentToLinalg + ExpandRsqrtForLibm
  -> materialized/fused upstream Linalg structural arm
  -> registered TileLinalgByIteratorKind(parallel_tile, reduction_tile)
  -> one-shot-bufferize + buffer-results-to-out-params
  -> registered ForwardBufferResultsToOutParams
  -> registered LowerDisjointMemrefCopyToLoops
  -> Linalg/SCF/MemRef/Math -> LLVM dialect -> LLVM IR
  -> LLVM O2/O3 vectorization search -> llc riscv64 (+v,+zfh) PIC object
```

`ssh rvv` 实测机器为原生 riscv64、64 核、2.6 GHz、VLEN=128。portable bundle 在本地完成
MLIR/LLVM 编译，远端只用系统 clang 链接 object，并通过 ctypes C ABI 做 canonical-J
correctness 和固定 CPU 计时。all-18 结果为 18/18 Target–RefJ correct，E3/E2 几何平均
**1.1817x**，15/18 winner 使用 B-admitted whole-graph transformation。代表性增量包括 KLDiv
1.593x、Flash preprocess 1.458x、PolyNorm 1.430x、Flash one-pass LayerNorm 1.296x。

RVV winner 不强制“必须含向量指令”：标量、vectorized、O2/O3 states 都由同一真实 MLIR pipeline
形成并按实测选择，receipt 同时保留 RVV 指令数与 topology digest。当前没有 RVV Source/Replay
`D_cert`，因此 1.1817x 只解释为同一 J、同一 target 下从 materialized E2 到 expanded/fused E3 的
内部因果增量，不能解释为超越 Source 或正式端到端 backend closure。

retile 不再是 report 中的虚拟参数：C++ pass 通过 upstream TilingInterface/SCF tiler 对 parallel 与
reduction iterator 使用独立 extent，并在 module receipt 记录实际 tiled Linalg op/loop 数。搜索使用
固定、C-independent 的 52-state 请求宇宙；C 只重排相同 concrete native-object universe。
`LowerDisjointMemrefCopyToLoops` 也已按 `memref.copy` 的非重叠有效程序合同覆盖等形 ranked ABI 参数
copy，避免合法 `masked_copy` state 残留外部 `memrefCopy` 符号；所有 11 个不同 native object 已在
RVV 实机通过 Target--RefJ，而不是只通过本地链接。

## 9. 复现

```bash
export PYTHONPATH=$PWD/python
export INTENT_OPT=$PWD/build/tools/intent-opt/intent-opt

# RTX 5090D：正式 CUDA E2/E3。arm 由 machine-consumed v2 spec 冻结；
# 缺 opaque Replay 或任一 held-out direct edge 时 fail closed。
python tools/run_production_backend_ablation.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/cuda-e2-e3.json --arms E2 E3 \
  --candidate-budget 32 \
  --replay-manifest corpus/production-replay-v1/manifest.json \
  --require-backend-closure

# 数值阶梯、local-B 和低预算 C prior 分别运行，避免混淆因果轴。
python tools/run_production_backend_ablation.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/cuda-numeric.json --arms N0 N1 N2 N3 --candidate-budget 32
python tools/run_production_backend_ablation.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/cuda-local-b.json --arms L0 L1 --candidate-budget 32 \
  --replay-manifest corpus/production-replay-v1/manifest.json \
  --require-backend-closure
python tools/run_production_backend_ablation.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/cuda-c-prior.json --arms E3 E4 --candidate-budget 8

# sm90a：本地分别冻结 exact E2/E3 cubin bundle，远端只执行。
INTENT_GPU_CHIP=sm_90a python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/sm90a-e2.json --arm E2 --search-budget 32
INTENT_GPU_CHIP=sm_90a python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/sm90a-e3.json --arm E3 --search-budget 32

# J-only 128 MiB masked-copy 压力工具：仅 Target--RefJ，不报 Source speedup。
python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 --kernels masked_copy \
  --workload-profile roofline-large-j-only-v1 --arm E3 --search-budget 6 \
  --out /tmp/masked-copy-roofline.json
python tools/run_cuda_benchmark_bundle.py \
  --bundle /tmp/masked-copy-roofline.json --out /tmp/masked-copy-roofline-report.json \
  --data-regime cold-data --timing event --cold-iterations 16 \
  --cache-flush-mib 256 --roofline-calibration

# all-18 symbolic-large CUDA 主口径：分别冻结 equal-budget E2/E3；未匹配与失败仍留在报告。
python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --workload-profile symbolic-large-v1 --arm E2 --search-budget 32 \
  --out /tmp/symbolic-large-sm120a-e2.json
python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --workload-profile symbolic-large-v1 --arm E3 --search-budget 32 \
  --out /tmp/symbolic-large-sm120a-e3.json
python tools/run_cuda_benchmark_bundle.py \
  --bundle /tmp/symbolic-large-sm120a-e2.json \
  --out /tmp/symbolic-large-sm120a-e2-report.json \
  --data-regime cold-data --timing event --cold-iterations 16 \
  --cache-flush-mib 256
python tools/run_cuda_benchmark_bundle.py \
  --bundle /tmp/symbolic-large-sm120a-e3.json \
  --out /tmp/symbolic-large-sm120a-e3-report.json \
  --data-regime cold-data --timing event --cold-iterations 16 \
  --cache-flush-mib 256
python tools/summarize_cuda_benchmark_ablation.py \
  --e2 /tmp/symbolic-large-sm120a-e2-report.json \
  --e3 /tmp/symbolic-large-sm120a-e3-report.json \
  --out /tmp/symbolic-large-sm120a-e2-e3.json

# RVV：本地 true-MLIR 编译；当前环境的可用 SSH alias 为 rvv。
python tools/export_rvv_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --workload-profile symbolic-medium-v1 \
  --out /tmp/rvv-bundle.json
scp /tmp/rvv-bundle.json tools/run_rvv_benchmark_bundle.py rvv:/tmp/
ssh rvv python3 /tmp/run_rvv_benchmark_bundle.py \
  --bundle /tmp/rvv-bundle.json --out /tmp/rvv-report.json
```

RVV exporter 已接入与 CUDA 相同的 all-18 `--workload-profile` 与 per-kernel valuation receipt；
RVV 没有 Triton Source executable，所以性能报告仍统一标记 `Target--RefJ-only`。CUDA 与 RVV
可以共享 `(j_digest, nu, workload_profile)`，但不能跨 target 复用 latency 或把 CUDA Source
speedup 转写成 RVV 结果。

## 10. 当前边界

当前冻结实现达到 **18/18 deterministic materialization、generic cubin compilation**；17/18 至少有
一个 C++ whole-graph path，masked copy 由 generic C++ MLIR 生成单 cubin。真实 B/J consumers 覆盖
inout/post-state、access-local guard、scan/reduction relation、loop scan/carry、runtime value-if、
multi-output half epilogue、direct identity binding 与 masked square-sum reduction。

当前 closed-valuation 结果已经不是小修小补：RTX 5090D 正式 E3/E2 **1.1302x**，H100 portable E3/E2
**1.1196x**，RVV4 E3/E2 **1.1817x**；6 个图的 whole-graph/materialized 内部对照为
**17.1258x**。reduction topology 与 precision/approximation 分别提供 **1.0409x** 和
**1.0676x** 的 all-corpus 增量，FMA 的 **1.0003x** 负结果保留。local-B 的 **1.0162x** 只作辅助，
不冒充核心创新。这些数字仍需在 all-18 production valuation 的 L2-controlled 主协议上
重新验证，不与新矩阵混合 aggregate。

当前 18 项没有 typed contraction；Flash `do_o_dot` 仍是 FP32 multiply + reduction，所以不存在可诚实
声称的 TF32 state。但下一步不是停下现有 18 J 等新 contraction：

| 横向工作 | CUDA 5090D/H100 | RVV4 | 覆盖/验收 |
|---|---|---|---|
| tail-safe contiguous vector state | vec2/vec4 masked load/store，消除非 4 整除 scalar fallback | dynamic VL + SEW/LMUL | 18/18 match receipt；elementwise/reduction applicable subset |
| producer + one/multiple reduction + epilogue | 单趟多 accumulator，subgroup/workgroup/multi-stage 独立 retile | vector partial/fold + reduction tile | 11 个 reduction J 主要覆盖，其余记 not_matched |
| scan/effect/gather/contraction | prefix/segmented scan、effect-safe update、PolyNorm contraction/gather tiling | vector scan、multicore state、layout-aware tiling | Sparsemax/TVD/PolyNorm，通用 matcher 不读 case name |
| portable production valuations | all-18 closed + 所有 legal symbolic profiles | 接入同一 profile/valuation receipt | 每 target 全 18 matrix，literal-only 不删除 |

新 `tl.dot`/matmul/完整 Attention J 到来后，再复用同一 transformation/QD framework 加入
MMA、IEEE/TF32 accumulator 和更大 stage decomposition；不为现有或新 kernel 增加名字驱动
recipe。H100 缺 Replay，RVV 缺 Source/Replay，二者的 formal `D_cert` 仍是明确证据边界。
