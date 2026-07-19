# SCRC Gate1 17 Kernel 规模性能实验（2026-07-17）

## 1. 结论

本轮不是缓存重放或手写 J 探针，而是对 17 个已安装 FlagGems Triton Kernel 逐例执行：

```text
真实 source preflight
→ fresh gpt-5.6-sol/xhigh Lift
→ deterministic π finalize
→ source-blind fresh replay
→ source/replay TTIR hard Gate1
→ 必要时 source-aware relift 后完整重启
→ paired intervention + actual-π masked-J
→ π 消融
→ 独立 Gate2 source/replay/Lower2C 矩阵
```

最终结果：

- 17/17 都得到 terminal report，无 harness error；
- 17/17 真实源 Kernel preflight 编译成功；
- 15/17 生成了合法完整 LiftArtifact，Lift 成功率 88.2%；
- 只有 8/17 在第一次模型响应即得到合法 LiftArtifact，一次成型率 47.1%；
- 14 个样本进入 normal replay，19/19 份 normal replay 全部编译成功；
- 12 个样本 Gate1 `REPLAY_CLOSED`；
- 1 个 `LEAKAGE_REJECTED`，1 个 `NO_CLOSURE`，1 个在 Gate1 TTIR 解析阶段
  `LIFT_REJECTED`，2 个 `LIFT_FAILED`；
- 14 个有可执行 replay 的样本全部完成 Gate2，累计 153/153 个三方 trial PASS；
- Gate1 闭合率为 12/17=70.6%；在 Lift 成功样本中为 12/15=80.0%；在真正进入
  normal replay 的样本中为 12/14=85.7%；
- Gate2 PASS 同时出现在 Gate1 `REPLAY_CLOSED`、`LEAKAGE_REJECTED` 和
  `NO_CLOSURE` 三类样本中，再次证明 Gate1/Gate2 不是包容关系。

因此，当前 Gate1 已经不是五核玩具原型：在 fill、基础/复合 reduction、runtime
scalar、gather 等真实 family 上形成了稳定证据；但它还不能称为“数百 Kernel 正式
就绪”。主要边界已经从猜测变成四类具体证据：Lift 一次成型率、复杂 access 的 Lift
失败、TTIR region SSA 解析错误、以及低熵 Kernel 的 π 拓扑泄漏。

## 2. 本轮代码与样本扩展

### 2.1 性能观测

`test/scrc_live_e2e.py` 新增 observation-only 性能记录：

- `initial_lift`；
- `gate1_loop`；
- `specialization_family`；
- `pi_ablation`；
- `gate2`；
- `gate1_core = initial_lift + gate1_loop`；
- `total`。

同时记录可从报告中严格恢复的逻辑调用量：Lift/Relift response、normal/masked/
intervention/π writer、normal replay compile、family compile 和 Gate2 trial。底层 SDK
内部 retry 不可见，因此报告明确不把这些数字称为精确 API request 数。

这些字段：

- 不进入 Lift/Relift prompt；
- 不进入 `ReliftEvidence`；
- 不进入 J/C/π；
- 不进入 TTIR fingerprint；
- 不参与 Gate1/Gate2 verdict。

`test/scrc_heldout_batch.py` 的 resume identity 升为 schema 26，并增加 nearest-rank
p50/p95 聚合。未执行阶段保持 `null`，空样本集合不会被记成 0 耗时或满分。

### 2.2 新增 ready Kernel

原 12 个 ready case 扩为 17 个：

| 新增样本 | 真实函数 | 主要覆盖 |
|---|---|---|
| `fg_zeros` | `flag_gems.ops.zeros::zeros_kernel` | 0 常量 fill、多 program tail mask |
| `fg_amax` | `flag_gems.ops.amax::amax_kernel_2` | 独立真实 max-reduce 实现、边界长度和值域 |
| `fg_eye` | `flag_gems.ops.eye_m::eye_kernel` | 二维 iota/broadcast/select、双轴 mask |
| `fg_trace` | `flag_gems.ops.trace::trace_kernel` | 对角 stride access、跨 1024 的两次循环、reduce |
| `fg_log_softmax` | `flag_gems.ops.log_softmax::log_softmax_kernel` | online stable reduce、两段循环、M/N tail |

这五例均先用真实 TritonCompiler contract 做 source preflight，5/5 编译成功；整个
17-case ready 集合也再次逐个 preflight，17/17 成功。

`fg_eye` 和 `fg_log_softmax` 还声明了 fixed-C runtime profile family，但二者分别在
Lift 和 TTIR parse 阶段终止，因此本轮真正执行 family Gate1 的仍只有 `fg_ones`。

## 3. 正式执行命令与原始证据

```bash
PYTHONPATH=python python test/scrc_heldout_batch.py \
  --model gpt-5.6-sol \
  --effort xhigh \
  --relift-rounds 3 \
  --gate2-matrix \
  --out-dir /tmp/scrc_gate1_scale_schema26_20260717
```

本轮没有使用 `--resume` 或 `--lift-cache-dir`。原始报告位于：

```text
/tmp/scrc_gate1_scale_schema26_20260717/summary.json
/tmp/scrc_gate1_scale_schema26_20260717/fg_*.json
```

项目内相关保护性回归：

```text
324 passed, 33 warnings
```

本轮没有使用 Trellis task/workflow。

## 4. 每个 Kernel 的终态

| Kernel | Lift responses | Relift | Gate1/批次终态 | masked-J | Gate2 trials | 总墙钟 |
|---|---:|---:|---|---|---:|---:|
| `fg_dot` | 1 | 0 | `REPLAY_CLOSED` | hard mismatch | 3/3 PASS | 391.6 s |
| `fg_addr` | 1 | 0 | `LEAKAGE_REJECTED` | **hard closure** | 3/3 PASS | 557.7 s |
| `fg_ones` | 1 | 0 | `REPLAY_CLOSED` | hard mismatch | 18/18 PASS | 339.9 s |
| `fg_zeros` | 1 | 0 | `REPLAY_CLOSED` | hard mismatch | 18/18 PASS | 330.7 s |
| `fg_arange` | 1 | 2 | `NO_CLOSURE` | 未进入 | 9/9 PASS | 434.8 s |
| `fg_eye` | 3 | 0 | `LIFT_FAILED` | 未进入 | 未执行 | 594.9 s |
| `fg_trace` | 3 | 0 | `LIFT_FAILED` | 未进入 | 未执行 | 828.0 s |
| `fg_embedding` | 3 | 0 | `REPLAY_CLOSED` | hard mismatch | 6/6 PASS | 1138.0 s |
| `fg_amax` | 2 | 0 | `REPLAY_CLOSED` | hard mismatch | 27/27 PASS | 403.3 s |
| `fg_sum` | 2 | 1 | `REPLAY_CLOSED` | hard mismatch | 3/3 PASS | 740.0 s |
| `fg_mean` | 2 | 0 | `REPLAY_CLOSED` | hard mismatch | 3/3 PASS | 535.3 s |
| `fg_max` | 2 | 0 | `REPLAY_CLOSED` | hard mismatch | 3/3 PASS | 441.5 s |
| `fg_min` | 2 | 0 | `REPLAY_CLOSED` | hard mismatch | 36/36 PASS | 443.4 s |
| `fg_prod` | 2 | 0 | `REPLAY_CLOSED` | hard mismatch | 3/3 PASS | 438.4 s |
| `fg_vector_l2` | 1 | 2 | `REPLAY_CLOSED` | hard mismatch | 3/3 PASS | 618.2 s |
| `fg_vector_l1` | 1 | 0 | `REPLAY_CLOSED` | hard mismatch | 18/18 PASS | 426.8 s |
| `fg_log_softmax` | 1 | 0 | Gate1 `LIFT_REJECTED` | 未进入 | 未执行 | 292.6 s |

## 5. 性能结果

### 5.1 总体墙钟

- 17-case 顺序总墙钟：8954.8 s，约 149.25 分钟，即 2.49 小时；
- 每 case 平均：526.8 s（8.78 分钟）；
- p50：441.5 s（7.36 分钟）；
- p95/最大值：1138.0 s（18.97 分钟，`fg_embedding`）。

### 5.2 分阶段

| 阶段 | 样本数 | 平均 | p50 | p95 |
|---|---:|---:|---:|---:|
| source preflight | 17 | 3.45 s | 3.23 s | 4.47 s |
| initial Lift | 17 | 287.1 s | 242.5 s | 824.3 s |
| Gate1 loop | 15 | 209.8 s | 145.1 s | 456.2 s |
| Gate1 core | 15 | 440.9 s | 366.7 s | 924.2 s |
| π ablation | 12 | 65.6 s | 40.8 s | 208.6 s |
| Gate2 | 14 | 2.04 s | 1.14 s | 7.15 s |

结论很明确：当前吞吐瓶颈是复杂 Lift 和外层 Gate1 模型调用，不是 Triton preflight
或 Gate2 执行。以当前顺序/xhigh 配置直接扩大到数百 Kernel，在不做并发调度和失败
分层的情况下成本不可接受；这属于吞吐工程问题，不能通过放宽 Gate1 硬条件解决。

### 5.3 逻辑调用量

本轮记录：

- initial Lift response：29；
- Relift response：6；
- Lift/Relift response 合计：35；
- normal writer：19；
- paired semantic intervention writer：7；
- actual-π masked-J writer：13；
- π mechanism ablation writer：12；
- writer 合计：51；
- 高层模型交互记录合计为 86 次；底层 SDK retry 不在此数内。

normal replay compile 为 19/19 成功，说明当前扩大样本后，writer 的首要问题不是
“生成的 Triton 普遍不可编译”，而是更上游的 Lift 合法性、Gate1 结构闭合和反泄漏。

## 6. 失败与边界根因

### 6.1 `fg_eye`：Lift 修复未收敛，不是源编译失败

三次 Lift response 依次失败于：

1. `ComposeDependencyError`：同一 `n2` 产生重复 `realize_broadcast` step；
2. `write_once storage 'b0' requires exactly one semantic write event`；
3. `storage 'b0' last_use_event is unrelated to its members`。

这说明二维 select/broadcast 的 J op 本身不是唯一问题；当前 Lift 对 boundary broadcast
去重、输出 storage 写事件和 lifetime endpoint 的一次成型能力不足。三个错误都在完整
LiftArtifact 交给 writer 前被拒绝，Gate1 没有用宽松路径掩盖问题。

### 6.2 `fg_trace`：静态 shape 约束与二维对角 access 表达不稳定

三次 Lift response 依次失败于：

1. J 输入 shape 使用不存在于 compiler scalar arguments 的符号 `M`；
2. 模型生成当前 RuleRegistry 不支持的 `intent_op:triu`；
3. 再次使用不存在的 shape symbol `M`。

真实函数只提供 `num_diag/stride0/stride1`，不提供矩阵 `M/N` runtime scalar；compiler
view 中的 `1031×1025` 因而必须保持静态 extent。当前 repair prompt 没有稳定做到这一
点。第二次响应选择 `triu`，说明一般二维对角 gather/reduce 尚未形成清晰、唯一的现有
J/B 表达路径。不能据此直接新增 `triu` op；应先确认现有 iota/gather/access 是否足以
表达，再按既定 A/B/C 缺口流程论证。

### 6.3 `fg_log_softmax`：Gate1 TTIR parser 对 region SSA 的 false reject

该样本一次 Lift 即生成合法完整 artifact，deterministic π equality 也通过；Gate1 在
解析真实 source TTIR 时返回：

```text
duplicate SSA definition %n_offset
```

重新编译源 Kernel 可见 `%n_offset` 分别定义于两个 sibling `scf.for` body。这在 MLIR
region scope 下合法，但当前 `TTIRParser._parse_operations` 使用全函数唯一的
`result_to_node: dict[str, int]`，没有把 region context 纳入 SSA identity，因而把合法
的跨 region 名称复用误判为重复定义。

所以报告中的 `LIFT_REJECTED` 是现有 Gate1 状态枚举的粗粒度表现；真实根因是 TTIR
fingerprint parser scope bug，不是 Lift/J/C/π 被验证为错误。该问题应作为 P0 parser
修复，且不需要新增 verifier 或放宽任何 hard category。

### 6.4 `fg_addr`：actual-π masked-J 发现真实拓扑泄漏

`fg_addr` 的 normal replay 已 hard-close；三个 paired semantic intervention 全部精确
PASS，说明 normal writer 在 J 可见时确实响应 J 语义。

但遮蔽 J 后保留同一份实际 Lift π，masked writer 仍重建出完整二维 addr 计算：

```text
beta * input + alpha * outer(vec1, vec2)
```

并与源 TTIR 所有 hard category 完全闭合。因此 Gate1 正确返回：

```text
LEAKAGE_REJECTED
anti_leakage_basis = masked_j_full_pi_hard_closure
```

Gate2 3/3 PASS 不能覆盖这个结论。该样本证明：opaque ref 名称已经避免词法泄漏，但
π 的规则序列、ref 拓扑、B/access 关系、interface/C 与低熵计算形状组合后，仍可能
泄露足够多的算法结构。当前 π 不能被宣称为对所有 Kernel 无泄漏。

### 6.5 `fg_arange`：合理 no-closure，但 stagnation 未提前停止

三轮 artifact 的 J 始终严格保持：

```text
iota(size) → cast(f32) → mul(step) → add(start)
```

J、C.strategy、C.schedule 和 π 均未被 BLOCK/pid/lane 污染。三轮 hard diff 也完全
不变：

- `op_multiset`：5 条；
- `dataflow`：23 条。

预算耗尽后 Gate1 返回 `NO_CLOSURE`，Gate2 9/9 PASS，这是正确四象限结果。

但当前 `Gate1Runner` 只检查 relift 总预算，没有实现文档已经要求的“连续确定性
diff signature 不再变化时提前停止”。因此本例在明确 stagnation 后仍消耗了两次
Relift 和两次 fresh writer。这是性能与协议一致性缺口，不应靠 score/模糊阈值解决。

## 7. Relift 实际效果

本轮只有三个 Kernel 触发 Relift：

- `fg_arange`：2 次，最终 `NO_CLOSURE`；
- `fg_sum`：1 次，第二份 fresh replay 闭合；
- `fg_vector_l2`：2 次，第三份 fresh replay 闭合。

五次 Relift 中，J、C.strategy、C.schedule 和 canonical π 均与初始 artifact 相同；
只有 `fg_arange` 第二次 Relift 改变了 location_map。也就是说，本轮没有一次 Relift
实际修正 J/C/π 语义核心。

`fg_sum` 和 `fg_vector_l2` 的最终闭合来自相同 artifact 上新的 source-blind writer
样本，而不是 Lift artifact 的可观察改进。这不等于 replay repair：writer 从未看到旧
replay code；但它说明当前 Relift 通道在本批次更多充当“完整重启并再次采样 writer”，
尚未证明能稳定依据 semantic diff 改善 J/C。

下一步应同时报告：

- canonical artifact digest 是否变化；
- deterministic hard-diff signature 是否变化；
- 若 artifact 与 diff 均连续不变，按既定协议提前 stagnation stop；
- 不把“换一个 writer 样本后偶然闭合”误称为 Relift 修复成功。

## 8. π 与消融结果

12 个 closed control 执行 π 消融：

- `plan_step_dropout`：12/12 在 writer 前 `canonicality_rejected`，证明 direct π tamper
  仍被 deterministic compose equality 拒绝；
- `plan_mechanism_substitution`：6 个 `semantics_preserved`，6 个
  `off_target_drift`。

本轮 mechanism substitution 都是在 `materialize_index_tile` 与
`materialize_pointer_tile` 间切换：

- final scalar reduction family（sum/mean/max/min/prod/amax）为
  `semantics_preserved`；
- dot/embedding/ones/zeros/vector_l1/vector_l2 为 `off_target_drift`。

因此这个消融目前不能被解释成统一的“π 必要性准确率”：

- `semantics_preserved` 可能表示两个合法实现机制在该具体 access 上观察等价，不是
  π 泄漏；
- `off_target_drift` 表示本来面向 C.strategy 的干预改变了 J hard component，说明
  该干预也不够纯；
- 真正直接的泄漏证据仍是 actual-π masked-J，当前为 12 个 hard mismatch、1 个
  hard closure（`fg_addr`）。

后续 π 消融应只选择能预先证明具有可辨识因果效果的 target/family；观察等价的
mechanism pair 应标记为 non-discriminating，而不是强行计成成功或失败。

## 9. 当前能力上限

按本轮真实证据，可以合理声称：

- fill、final reduction、复合 map-reduce、runtime scalar、axis-0 gather 已具备可执行
  Gate1/Gate2 主线；
- writer 生成的 normal Triton 编译稳定性较高，本轮 19/19；
- actual-π masked-J 能发现真实低熵拓扑泄漏；
- Gate1/Gate2 四象限和 rejected-sample Gate2 diagnostic 工作正常；
- budget exhaustion 后继续 Gate2 的路径真实可用。

仍不能声称：

- 已覆盖全部 Kernel 特性；
- 二维 select/broadcast、对角 access、online structured reduction 已通过完整主线；
- fixed-C multi-specialization 已扩大到多个真实 family；本轮仍只有 `fg_ones` 6/6
  真正执行；
- Relift 已证明能稳定修正 J/C；本轮五次都未改变语义核心；
- π 已完全消除 topology leakage；`fg_addr` 明确反证；
- 当前顺序/xhigh 吞吐可直接扩到数百 Kernel。

## 10. 后续主线优先级

### P0：修复 scope-aware TTIR SSA 解析

以 region context 作为 SSA identity 的一部分，允许 sibling region 重用局部名字；增加
真实 `fg_log_softmax` TTIR 回归。修复后必须重新 fresh 执行该 case，不能用本轮 artifact
直接宣称 Gate1 闭合。

### P0：实现既定的确定性 stagnation stop

使用 canonical artifact digest + hard-diff signature，不使用 fuzzy score。连续轮次二者
都不变时提前退出并保留 `NO_CLOSURE`/最佳证据。

### P0：增强首次 Gate1 硬差分对下一轮 Relift 的语义指导

本阶段不以提高初始 Lift 响应“一次成型率”为目标。只有 source/replay 已完成确定性
hard compare 后，才从 exact difference 生成结构化 Relift worklist：失败/已确认硬类别、
去内部 hash 的 grouped dataflow delta、应复查的 J semantic surface、允许考虑的既有
C.strategy family，以及当前 J 摘要。Relift 必须把每个失败类别判成 J semantic gap、
既有 strategy gap 或当前词汇不可表达；没有充分语义依据时返回 J 不变，由 stagnation
机制停止，不能为获得另一次 writer 随机样本而改 J。

### 暂缓：π topology leakage 结构优化

当前不修改 π 数据模型。继续保留 actual-π masked-J 终检和批次计数；只有后续跨 family
规模测试出现成批 hard closure，而不是孤立样本时，才发出警告并提交独立必要性报告。

### P1：收紧 π mechanism ablation 的可辨识性

预先证明被替换机制在该 target 上应产生确定 hard delta；观察等价时记录
`non_discriminating`，避免把等价实现错误解释成 π 泄漏或 π 必要性成功。

### P1：继续扩大真实 family，而不是只增加同类 reduce

在上述 P0 修复后优先重跑：

1. `fg_eye`；
2. `fg_trace`；
3. `fg_log_softmax`；
4. 再进入 fused in-place/multi-output、scan、atomic、sort/top-k。

这些 family 必须继续走真实 LLM Lift；不得用手写 MLIR/J 代替正式覆盖。

## 11. 实验后修复状态（schema-27/28，尚未替代本报告结果）

本报告记录的是不可改写的 schema-26 实验事实。其后代码已完成两项 P0 修复：

- TTIR fingerprint schema 升为 `triton-ttir-3.6-gate1-v2`，SSA 解析按 lexical region
  scope 绑定；真实 `fg_log_softmax` source TTIR 已能提取 102-op、2-loop 指纹，同一
  region duplicate SSA 仍 fail-closed；
- Gate1 外循环加入 `exact_artifact_and_hard_diff_v1`：连续两次完整 fresh replay 的
  canonical Lift artifact digest 与确定性 hard-diff digest 同时不变时，返回
  `NO_CLOSURE` 并停止，不再继续 Relift/writer 采样。compile/parser 等不可比较失败
  不参与该判断，score 与 replay source 不进入 digest。

schema-27 fresh `fg_log_softmax` 已确认 source fingerprint 不再受 SSA scope 阻塞，但
provider model-cooldown 使 writer 未形成候选，不能算新的语义结果。随后 Relift guidance
协议升为 v9，held-out resume identity 随后升到 schema-29：只有完成 hard compare 后才
生成结构化 semantic worklist，纯 provider failure 不再调用 Relift。上述实现和单元/
集成回归不能替代完整 fresh `gpt-5.6-sol/xhigh` 复验；`fg_log_softmax` 与 `fg_arange`
的新正式结果应另行记录，不得回填或改写本报告的 schema-26 统计。
