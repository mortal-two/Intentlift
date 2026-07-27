# Task 15：16-Certified + 2-Provisional J Development 与真实跨目标后端能力闭合

> 状态：执行中
>
> 类型：production backend 能力建设、lowering-state search 与论文实验落地；不是 SCRC Gate、
> 新 Agent 或 per-kernel emitter。
>
> 执行分支：`codex/backend-certified-j-search-20260725`

## 目标

以 backend 当前可逐 digest 核验的 **16/28 个 certified high-performance J** 为不可变语义来源；
RoPE 与 SplitK 在 exact Gate evidence 补齐前只作为两个 provisional `J_dev` 规划项，canary 独立分账。
在这个事实边界上允许构造有 provenance 的 `J_dev` shape/dtype variants，真实完善 CUDA、H100 与
RVV 后端，使同一套 J-grounded compiler framework 能够：

1. 消费真实 shape/dtype/layout，而不被旧 c0 小 shape 限制；
2. 生成真实 MLIR、target object/cubin 与多 launch graph，不使用 Python target kernel emission；
3. 在固定结构上搜索 schedule，并搜索不同的合法 lowering states；
4. 以统一 Q/D 过滤 observation-changing numeric states；
5. 形成 `C-Fixed / Schedule-Only / IntentIR` 与独立 C-prior 消融；
6. 在 RTX 5090D、H100 与 RVV 上完成横向 correctness、性能与搜索效率证据；
7. 以全量矩阵和通用 compiler mechanisms 推进，而不是只优化少数 winner。

权威设计为：

- `doc/IntentIR_后端能力建设与最终实验协议_2026-07-27.md`；
- `doc/IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md`；
- `doc/production-backend-jc-search.md`。

## 当前断点

### 真实 corpus

backend `production-jc-v1` 当前可核验事实是：

```text
16 high-performance accepted J
+ 2 scrc_canary
= 18 catalog entries
```

两个曾被报告为“已闭合”、但当前给定路径不能支持 production intake 的 kernel 是：

- Liger RoPE：
  `/tmp/scrc_task11_v112_latest_liger_b3_20260726_01/liger_rope/fanout-001/candidate-01/attempt-03/artifact.json`；
- xFormers SplitK reduce varargs：
  `/tmp/scrc_task11_v99_continuous_replay_fanout_20260726_01/xformers_ops_fmha_triton_splitk_kernels_splitK_reduce_varargs/fanout-001/candidate-02/attempt-03/artifact.json`。

逐 evidence 审计结果：RoPE report 为 `GATE2_FAILED`，SplitK candidate 为 `NO_CLOSURE`。因此二者
不得导入 certified catalog，也不得与 16 项合并报告 SCRC aggregate；在新的 exact closure 到来前，
它们只能使用 `certification_status=uncertified_backend_dev` 做 backend capability/Target--RefDev/性能。

`masked_copy` 与 `masked_row_square_sum` 继续作为 compiler/correctness canary，但不得进入 18/28、
论文主表或性能 geomean。

### shape/dtype

当前 active 28 adapters 没有 BF16 specialization；Liger 多数 c0 为 FP32，Flash 主 tensor 为 FP16，
xFormers norm/reduce 多为 FP32。已逐 JSON 确认 Softmax、RMSNorm、Sparsemax 三项把 rows 固定为 2；
Flash preprocess 还固定 `B=1,S=33,D=16`。其余项即使 materialization 出现 concrete extent，也不能
据此断言 J 非参数化，必须读取 accepted J 的 shape expression。上述 concrete artifact 对原 c0 有效，
但不足以单独检验自然 workload 下的真实后端能力。

普通 workload valuation 仍只能绑定 J 已声明的 symbolic axes，不能修改 literal shape 或 dtype。
Task15 已新增物理分离、不可冒充证书的 `intentir.backend.dev-j-variant.v1` 与 digest-pinned
`intentir.backend.dev-workload.v1`；这不改变 accepted snapshot 或旧 SCRC verdict。

### 已落地基线（截至 `bc4ef9b5`）

- `dcb82c1d` 已闭合真实 C++/MLIR multi-stage row reduction、direct opaque `malloc !llvm.ptr`
  transport 与 2/3-launch 执行；Python 只做候选调度和 host graph transport；
- `f70a624d` 已加入 exact-edit J_dev、canonical type re-elaboration、base replay validation、自然 Source
  workload 与 ordinary `materialize_j -> MLIR -> cubin` 导出；
- `bc4ef9b5` 已让收据区分 requested schedule 与 compiler-emitted launch plan；`vector_width=0` 的 auto
  状态现在会如实记为实际 vec4/scalar，并删除显式 vec4 与 auto 的重复/必败请求；
- backend Python tests 为 `283 passed`；自然 workload 的 Source--RefDev 与全部计时候选
  Target--RefDev 均先通过 Q/D。

首批自然 FP32 capability slices（不冒充上游 BF16 主结果）：

| case | config | 同回执 IntentIR / Schedule-Only | Source / Target | 结论 |
|---|---|---:|---:|---|
| Liger RMSNorm | `M=2048,H=1024` | `1.0531x` | `1.0491x` | subgroup shuffle + 实际 vec4 胜出 |
| Liger RMSNorm | `M=2048,H=4096` | `1.0000x` | `1.0174x` | 单阶段强；multi-stage 增加重读而失败 |
| Liger DYT | `BT=4096,H=4096` | `1.0024x` | `1.0275x` | 约 128 MiB 工作集已接近带宽，旧小输入 1.57x 不可外推 |

这些是 development/warm 机制数据，尚不是最终 cold-data 主表；它们的作用是校准真实 backend
mechanism 和 shape applicability，而不是提前冻结论文 aggregate。

## 第一性原理约束

1. core lowerer 只读取 J、可选 C 与 target capability；禁止读取 Source、Replay、pi、kernel name、
   provider、历史 winner 或 benchmark allowlist选择路径；
2. accepted J 永不原地修改；手工能力变体必须产生新 digest、显式 lineage 与
   `certification_status=uncertified_backend_dev`；
3. `J_dev` 可以用于 compiler capability、Target--RefDev/Source correctness与性能，不继承 Gate1/Gate2；
4. C 只排序，不改变 candidate universe、hard admission 或 Q/D；
5. boundary dtype/shape属于 J；FMA、TF32、内部 accumulator precision等是 backend states；
6. 所有 target code 必须来自注册 C++/MLIR/upstream compiler passes；
7. 每项新能力对 18/18 运行 match/admission/compile/correctness coverage，`not_matched` 也留证；
8. correctness 先于性能，性能失败不能放宽 hard boundary 或 Q；
9. 不预写必须获得正 speedup，但持续根据真实 profile修正机制、schedule space与codegen，直到形成
   具有竞争力且因果口径成立的结果。

## Phase A：修正 production intake

1. 等待并核验 Liger RoPE 与 SplitK reduce 的新 exact closure artifact/C/evidence；
2. 核验 case id、source/symbol、artifact/J/C/evidence digest；
3. catalog 保持 append-only/content-addressed，stable name 不与现有项冲突；
4. canary 保留独立 catalog/标记，但 high-performance view 必须严格是18项；
5. production adapters、Source/Replay sidecars、workload cost与coverage tests同步扩展；
6. intake 不重新 Lift、不手写 accepted J、不追认旧 verdict。

### Phase A 验收

- 当前里程碑 high-performance certified count=`16`；未来只有 exact evidence 到位后才可达到 `18`；
  canary count=`2`且不进入高性能 aggregate；
- 18项逐个 snapshot/J/C/evidence digest可验证；
- Liger RoPE与SplitK reduce可通过统一loader、materializer、generic target smoke或准确fail closed；
- 不再有文档/报告把16+2 canary或16+2 provisional写成18/28 certified。

## Phase B：`J_dev` 与自然 workload catalog

实现 `intentir.backend.dev-j-variant.v1`：

```text
base_j_digest
variant_j_digest
source_kernel_id
shape_config_id
dtype_config_id
edits[]
upstream_evidence[]
certification_status = uncertified_backend_dev
j_unchanged = false
```

要求：

1. 复用 canonical J parser/content digest/ProductionSnapshot/HarnessBindingSpec；
2. variant storage与accepted snapshot物理分离，禁止覆盖production corpus；
3. shape/dtype编辑同步更新signature、A op/cast、B carriers、access/output、Harness、RefEval与C valuation；
4. 建立 versioned 18-item natural configuration catalog；配置直接使用模型/参数名，不使用
   small/medium/large；
5. 首批至少覆盖 FP32、BF16、FP16/FP32-accumulation等上游真实组合；unsupported dtype保留负例；
6. 每个variant可生成Source adapter（上游支持时）、RefDev与target invocation，所有输入复用统一ABI
   validation；
7. receipt明确区分accepted、symbol valuation与manual dev variant。

### Phase B 验收

- 18/18至少一个natural/capability config；shape-sensitive项保留上游sweep；
- 18/18当前目标dtype与至少一个非c0 dtype（上游存在时）形成compile/correctness matrix；
- accepted snapshot digest完全不变；错误dtype/shape/半更新variant在compiler前稳定拒绝；
- dev variant绝不产生或冒充`SCRC_CERTIFIED`。

## Phase C：真实 CUDA lowering-state expansion

优先横向补齐以下能力，而非逐kernel配方：

1. tail-safe scalar/vec2/vec4 contiguous load/store与grid-stride traversal；
2. producer + one/multiple reduction + epilogue单趟路径；
3. subgroup/workgroup/hierarchical/multi-stage reduction；
4. materialized/fused graph与真实多launch scratch transport；
5. scan、indexed/in-place RoPE、effect-safe update；
6. contraction/Flash/SplitK partial-reduce路径；
7. FP32/BF16/FP16 boundary、FP32 accumulation、FMA/approx与typed contraction的IEEE/TF32 states；
8. 每条路径独立schedule domain与post-pass MLIR/topology/launch receipt。

当前 multi-stage 已闭合 direct `malloc !llvm.ptr` graph transport并进入 search universe；下一步补齐
逐 launch-site geometry receipt，并在 16 certified + 2 provisional J_dev 的 planned matrix 中分别记录
matched/admitted/rejected。无适用 graph 时保留既有单阶段或 generic lowering。

### Phase C 验收

- no Python target emission；
- multi-stage state在真实production J/dev variant上形成cubin、2/3 launch graph并执行正确；
- 18/18 coverage报告含matched/admitted/rejected/compile/QD/timing状态；
- 结构扩展不是requested toggle自报，而来自post-pass MLIR与binary receipt；
- 对适用kernel至少形成两个擦除schedule参数后仍拓扑不同的concrete MLIR states。

## Phase D：RVV真实retile与跨目标能力

1. 复用与CUDA相同的J/J_dev、shape/dtype config和高层transformation identity；
2. 实现dynamic-VL strip mining、SEW/LMUL、vector reduction/scan与tail；
3. 实现真实single-thread与16-thread execution，不把single-shot计时误写成single-thread；
4. materialized/fused、vector/scalar、O2/O3与multicore states均由真实MLIR/native object receipt区分；
5. 补Triton-CPU或语义匹配CPU baseline，以及可用子集上的native RVV baseline；
6. C只重排同一C-independent object universe。

### Phase D 验收

- RVV18/18 compile/Target--RefDev或准确fail closed；
- schedule tile、VL/SEW/LMUL、thread count真实改变MLIR/LLVM/native object；
- C-Fixed与retuned路径不是报告中的虚拟参数；
- single/16-thread均有固定CPU affinity、线程数与correctness receipt。

## Phase E：最终机器实验臂

新增并机器化：

```text
C-Fixed:
  canonical lowering path
  + one deterministic target projection of C
  + no search

Schedule-Only:
  same canonical path
  + schedule search

IntentIR:
  multiple admitted lowering paths
  + per-path schedule search
```

当前E2映射Schedule-Only，E3映射IntentIR，E4保留为Erase-C；E1不能冒充C-Fixed。

另做C-prior小表：同一candidate universe、相同预算、相同latency map，只切换C/no-C order，报告固定
预算best与达到最终best 95%所需候选数。

### Phase E 验收

- 三臂共享exact config、dtype、target、compiler、Q/D、profiler；
- C-Fixed exact one candidate且schedule来自可审计C projection；
- Schedule-Only与IntentIR的canonical path identity相同；IntentIR包含Schedule-Only spine；
- external panel使用Source/Best Applicable/IntentIR；RVV缺baseline时不得伪造；
- canary、synthetic profile、component/composed graph不混合aggregate。

## Phase F：性能推进与证据冻结

对 RTX 5090D、H100、RVV 依次执行：

1. 18项capture-seed regression；
2. 18项natural/capability config correctness；
3. C-Fixed/Schedule-Only/IntentIR；
4. C/no-C bounded-search；
5. Source/Best external comparison；
6. 仅对需要解释的机制做counters/Roofline/cache敏感性，不膨胀主图。

每轮性能分析从瓶颈出发：global traffic、dispatch、parallelism、reduction dependency、occupancy、register/
shared pressure、vector utilization、RVV VL/multicore。优化必须落到通用 MLIR/compiler mechanism，再回归
18项；不能按名字修 winner。

正式主表使用上游natural/model config；capture-seed与旧symbolic-large只作regression/stress。

## 总体验收

1. production high-performance catalog只包含有 exact evidence 的 certified 项；当前为16，未来目标18；
   provisional J_dev 与 canary分别分账；
2. 18项均有versioned natural shape/dtype config与可执行J/J_dev；
3. 18项在5090/H100/RVV形成横向compile/correctness矩阵；
4. CUDA/RVV均具有真实schedule与structural lowering search；
5. C-Fixed/Schedule-Only/IntentIR与C-prior消融机器可执行；
6. Source/Best与IntentIR在相同op/shape/dtype/layout/timing边界比较；
7. 报告同时给all-corpus、applicable subset与per-kernel结果，不只选加速项；
8. 所有发布winner有binary/object digest与correctness receipt；
9. docs/spec/result SHA与compiler commit冻结；
10. Python与MLIR全量回归通过，工作区无未提交改动。

## 非目标

- 不修改Task11/Task14的Lift、Replay、Gate1、Gate2 verdict；
- 不让fixed code手写accepted J或把J_dev冒充证书；
- 不建立implementation-family emitter/registry；
- 不以kernel name、provider或benchmark path决定lowering legality；
- 不因追求正speedup放宽Q、alias/effect/ABI hard boundary；
- 不把warm/cold/streaming、Roofline或局部pass开关当主贡献。

## 当前执行顺序

```text
task/docs checkpoint
  -> exact evidence audit（16 certified；2 provisional）
  -> dev-j-variant + 首批 natural config（已完成）
  -> direct !llvm.ptr multi-stage graph transport（已完成）
  -> requested/actual launch schedule receipt（已完成）
  -> real 5090 natural-shape mechanism calibration（执行中）
  -> horizontal CUDA states
  -> RVV VL/retile/multicore
  -> C-Fixed + unified arms
  -> H100/RVV/5090 planned-18 high-performance evidence（certified/provisional分账）
  -> full regression + clean commits
```
