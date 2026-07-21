# SCRC v8 后续实施与规模验证流程

> 状态：当前活动流程文档
> 日期：2026-07-19
> 当前协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`
> 适用范围：Triton SCRC Gate1、Gate2、真实 LLM Lift、确定性 π、held-out 与
> fixed-C multi-specialization。
> 本文不使用 Trellis task/workflow；只沿用项目 spec，不恢复旧 verifier 或 JSON
> interpreter 主线。

## 1. 文档目的

本文只回答三个问题：

1. 当前还有哪些部分没有完成；
2. 哪些部分已经实现，但需要继续优化或扩大真实证据；
3. 后续每一轮应按什么顺序执行、失败后允许修改什么、怎样判断阶段完成。

当前唯一主线保持不变：

```text
真实源码
  → 完整 Lift {
      source-aware LLM 生成 J / interface / C.schedule / C.strategy / location_map
      Lift 内 deterministic finalizer 生成唯一 π 与 provenance
      返回完整 J + C + π artifact
    }
  → source-blind writer 根据 J + interface + C + π + RuleCards 生成 replay
  → Gate1 比较 source/replay 的硬 TTIR 语义指纹
  → Gate1 闭合后进入独立 Gate2 三方执行
```

其中：

- J 保存算法语义和可移植依赖；
- C.schedule 保存冻结的 specialization/调度值；
- C.strategy 保存闭合枚举内的实现机制选择；
- RuleRegistry 保存跨 Kernel 的 frontend 施工知识；
- π 只是上述权威输入的确定性组装结果，不保存独立智能决策；
- Gate1 与 Gate2 是两份独立证据，不是包容或覆盖关系。

## 2. 当前已完成基线：后续不重复建设

截至 2026-07-17，以下基础已经完成：

- Lift 内部的 LLM 子响应 `LiftCore` 已禁止 provider 直接书写 π 或 compose
  provenance；完整 Lift 返回前必须生成两者；
- J 已加入 closed loop carry、portable scope、semantic storage/lifetime、effect/access
  binding 等 schema；
- C 已拆分为冻结的 `C.schedule` 和可 relift 的 `C.strategy`；
- RuleCard 已有 composer 所需的 executable metadata；
- `deterministic_composer_v3` 已实现 canonical compose、100% leaf provenance、
  exact compose equality 和 opaque-id/order 稳定性；v3 进一步将同一 node_ref 的多条
  BroadcastSpec 合成一个节点级 `realize_broadcast`，同时完整绑定成员语义、依赖和
  provenance；
- relift 通道常开，normal generation/compile/interface/hard-diff 失败只回流 Lift，
  不修 replay code；
- direct π drop/add/ref/order 在 writer 前拒绝；合法 mechanism 变化只通过
  C.strategy→recompose；
- fixed-C multi-specialization production runner 已实现，采用一份 artifact/一份 replay
  和逐 profile strict all-pass；
- Gate2 已实现 source/replay/production Lower2C 三方执行与独立归因；
- add、row-sum、softmax、RMSNorm、matmul 五个 fresh
  `gpt-5.6-sol/xhigh` 样例已有 normal Gate1 与 Gate2 PASS 历史证据；在恢复
  `masked J + 实际 Lift π` 终检并升级 batch identity 后，五例必须重新执行，旧结果
  不能自动计作当前正式 Gate1 `REPLAY_CLOSED`；
- 当前默认 RuleRegistry 有 22 张卡；第 22 张 `realize_iota` 来自真实
  `fg_arange` fresh Lift 的重复 `ZeroRuleMatchError(intent_op:iota)`，不是预先枚举；
- actual-π masked-J、replay fresh-only、归档边界与 Lower2C
  iota/integer read-back 接入后的 Gate1/SCRC/Gate2 相关回归为
  `324 passed, 33 warnings`；其中已覆盖 source-free `ReliftEvidence` 边界、活动
  report schema 中 repair-era 字段清除、symbolic-iota 合同和 fixed-C family
  same-replay/random-profile 路径；
- 2026-07-17 已将 reviewed ready case 从 12 个扩为 17 个，真实 Triton source
  preflight 17/17 成功，并完成一次无缓存的 `gpt-5.6-sol/xhigh` 正式批次：12 个
  `REPLAY_CLOSED + Gate2 PASS`、1 个 `LEAKAGE_REJECTED`、1 个 `NO_CLOSURE`、
  2 个 `LIFT_FAILED`、1 个 TTIR parser 导致的 `LIFT_REJECTED`；153/153 个已执行
  Gate2 trial PASS。完整证据见
  `doc/SCRC_Gate1_17Kernel规模性能实验_2026-07-17.md`。

后续工作不得重新建设下列旧路线：

- JSON interpreter 作为 Gate2 第三腿；
- source-aware LLM 子响应自由书写 π；完整 Lift 内的确定性 π finalization 必须保留；
- replay-local repair；
- 用 `mask J -> recompose` 诊断替代 `masked J + 实际 Lift π` 终检；
- fuzzy ratio 或空集合相似度决定 Gate1；
- 新建 verifier、第三道 Gate 或 op “verify-ready”资格体系。

## 3. 后续唯一执行流程

```text
[0] 真实 Kernel 候选登记
      │ 记录真实 source、frontend、函数、compiler/execution contract
      ▼
[1] Source preflight
      │ 同前端、同 C 编译源 Kernel
      ├─ 失败：记 setup/source_compile failure，不调用 Lift
      ▼
[2] Fresh complete Lift
      │ [2a] source-aware LiftCore 生成 J/interface/C/location
      │ [2b] 校验 J/interface/C
      │ [2c] Lift 内 deterministic finalizer 生成 canonical π/provenance
      │ [2d] 返回完整 LiftArtifact(J,C,π,interface,location)
      ├─ 任一子阶段失败：归因 LiftCore/J/C/schema/RuleCard，不调用 writer
      ▼
[3] Source-blind replay
      │ writer 只看 replay packet 和选中的 RuleCards
      ▼
[4] Replay compile + Gate1 hard fingerprint
      ├─ 全部硬项闭合：进入 paired intervention / π audit
      │                  → masked J + 同一实际 Lift π 终检
      │                  ├─ compile rejection / hard mismatch：允许 Gate1 闭合
      │                  ├─ hard closure：LEAKAGE_REJECTED
      │                  └─ 不可评估：NO_CLOSURE
      └─ 不闭合：生成确定性 semantic diff
                    │
                    └─ source + 当前完整 LiftArtifact + diff 回流 Lift
                         → 修改 J/C.strategy 并在 Lift 内重新 finalize π
                         → 从 preflight/静态检查/fresh replay 全流程重跑
      ▼
[5] Gate1 终止
      ├─ REPLAY_CLOSED：进入正式 Gate2
      └─ LEAKAGE_REJECTED / NO_CLOSURE / budget exhausted / stagnated：
         保留最佳失败证据；
         可运行 rejected-sample Gate2 diagnostic，但不得改写 Gate1 verdict
      ▼
[6] Gate2 三方矩阵
      │ O_src / O_rep / O_low
      │ shape × value × seed，逐 case strict all-pass
      ▼
[7] 失败归因与改进
      ├─ J/A 缺口
      ├─ J/B 缺口
      ├─ C.strategy 缺口
      ├─ RuleCard 施工不唯一或缺失
      ├─ Gate1 normalization/fingerprint 缺口
      ├─ production Lower2C 缺口
      └─ source/compiler/provider/infrastructure 问题
      ▼
[8] 修改后重跑受影响样本 + 全部相关回归
```

Gate1 循环的停止判据只能是：

- 所有硬类别闭合；
- relift 总预算耗尽；
- 连续轮次的确定性差分签名不再变化；
- provider/编译基础设施导致本轮无法继续。

score 或向量距离只能用于失败排序和分析，不能把硬差异改判为通过。

## 4. 当前未完成部分

### 4.1 17-Kernel 正式 pilot 已完成，仍不是数百 Kernel 覆盖

schema-26 正式 pilot 时 `test/scrc_heldout_batch.py` 有 25 个候选，其中 17 个具备
reviewed case contract；source preflight 17/17 成功，17/17 获得 terminal report，
12 个 Gate1 闭合样本和两个 Gate1 非闭合/拒绝但可执行样本累计完成 Gate2 153/153
trial PASS。当前 schema-33 manifest 已包含真实 `fg_diag_extract` 候选，并把既有
`fg_skip_layer_norm` 提升为 ready，并新增 `fg_index_select` two-axis indirect indexed
gather，清单为 27 candidates / 20 ready。三个新 contract 均已用原始 FlagGems source
完成 compile + scope-aware TTIR fingerprint：`index_select` 为 58 ops、2 masked loads/
1 masked store，`diag_extract` 为 28 ops、masked read/write，`skip_layer_norm` 为 48 ops、
4 masked reads/1 masked write。三者现均已有 schema-31 fresh terminal report：
`diag_extract` normal/family 闭合但 Lower2C gather contract incomplete；`index_select` 在
两轮相同 exact family diff 后 stagnation，且安装 source 的 tensor `and` 暴露动态输出
异常；`skip_layer_norm` normal 在一次 Relift 后闭合，但 paired intervention 因
off-target drift 拒绝。

schema-31 fresh `fg_log_softmax` 已完成：2 份 normal replay 均编译，但四个 fixed-C
profiles 全部在 `op_multiset + dataflow + predicate_effect` 不闭合。Gate2 对两个
single/exact-tile profiles 6/6 PASS，对两个 multi-N profiles 6/6 `replay_diverged`。
该结果把覆盖从同类 reduce 扩到 runtime loop/traversal 结构，并直接驱动了 scoped
loop witness 与 family diff 压缩。scope-v3 后续证据显示这些 source loop 没有 carry
pair，不能仅凭 loop 数量差异认定 J 缺 semantic recurrence。

此外，复用 schema-26 正式批次保存的 14 份真实 Lift/replay observation 重新执行当前
Gate1 compile/fingerprint/comparison 与独立 Gate2 三方诊断：Gate1 normal hard-close
13/14，Gate2 14/14 PASS。`fg_arange` 明确落在 `Gate1 no-closure + Gate2 pass` 象限，
证明报告实现没有把 Gate2 当作 Gate1 的覆盖关系。该结果是迁移诊断，不替代 fresh SCRC。

本轮同时暴露 `fg_eye/fg_trace` 的 Lift 失败、`fg_log_softmax` 的 region-scoped SSA
parser false reject、`fg_addr` 的 actual-π masked-J hard closure，以及 Relift 没有改变
J/C/π 语义核心的问题。完整数据和根因见
`doc/SCRC_Gate1_17Kernel规模性能实验_2026-07-17.md`。仍未完成：

- 将曾用于规则增长的 development/growth case 与真正未见的 evaluation case 分开
  报告；这只是数据归属，不新增 Gate 或评分；
- 为剩余 queued Kernel 建立真实 compiler/execution contract；
- atomic、scan、sort/top-k、indexed elementwise 等家族尚未形成当前 v11 的系统证据；
  structured loop 已能完成 scope-aware fingerprint，但尚无 replay closure；
- fused norm、strided gather 与 indirect indexed gather 已有 fresh terminal evidence，
  但分别暴露 paired-writer off-target drift、Lower2C gather contract incomplete 和
  trusted-source 动态异常，尚不能称为完整 family 支持；
- 形成当前 27-case 完整批次后，再扩大到 50～100 个类型分层样本；
- 在规则库冻结的评价批次上形成正式 closure/三方执行结果后，才能进入数百 Kernel。

当前 manifest 将 `fg_arange` 列入 ready compiler case。2026-07-17 已补齐通用
Lower2C 的 iota lowering 和 i1/i8/i16/i32/i64 输出读回，并完成 schema-24 fresh
`gpt-5.6-sol/xhigh` 完整运行：初始 J 和两次 Relift J 均严格保持
`iota("size")->cast->mul(step)->add(start)`，Gate1 因稳定的 affine association
结构差异返回 `NO_CLOSURE`，Gate2 三 profile × 三 seed 共 9/9 三方零误差 PASS。
该四象限结果是正式证据，不能被改写成 Gate1 PASS；若要补足现有 C.strategy 的
表达能力，必须先审查独立的计划外候选报告。

2026-07-19 更新：`fg_skip_layer_norm` 已完成 fresh `gpt-5.6-sol/xhigh` 全流程。
initial Lift 一次成功；第一份 replay 在 `op_multiset + dataflow` 不闭合后回流一次
source-aware Relift，第二份 normal replay 全硬项闭合。最终 3 个 paired interventions 中
两个 masked-fill exact delta 通过，`final add→sub` mutant 同时错误删除两个 `/N` 和一个
整数转浮点，因 off-target drift 返回 `LEAKAGE_REJECTED`。独立 Gate2 原 contract 3/3
三方 PASS。该结果不是 π topology hard closure，masked-J 因 paired 阶段已拒绝而未运行。

### 4.2 fixed-C normal/paired/masked 均已接入 Gate1 full-family

schema-25 已完成第一阶段 live/held-out 接入：

- `LiveKernelCase` 可声明有序 fixed-C profile family 和 primary profile；
- held-out resume identity 保存完整 family/profile runtime contract；
- family runner 可直接消费 primary Gate1 的同一个不可变 `ReplayGeneration`，不再调用
  第二个 writer；
- report 保存逐 profile source/replay compile 与 hard comparison；
- `fg_ones` fresh `gpt-5.6-sol/xhigh` 已用固定 `BLOCK_SIZE=1024` 验证
  `1/1023/1024/1025/2051/8209` 六个 runtime profile，同一 replay 6/6 hard-close；
- 同轮 Gate2 六 profile × 三 seed 共 18/18 三方零误差 PASS；primary Gate1 为
  `REPLAY_CLOSED`，actual-π masked-J 为 hard mismatch。

schema-30 已将 `fixed_c_family_full_sweep_relift_v1` 接入每个 artifact round：一份 replay
完成全部 profile 后才作 all-pass 判断；任一失败由
`exact_ordered_fixed_c_family_hard_diff_v1` 聚合为一次 source-aware Relift，已通过
profile 是不可回归约束。连续两轮相同 artifact + 完整 family diff 才触发 stagnation；
compile/setup failure 没有 family digest。保存的真实 `fg_ones` artifact/replay 已用当前
stack 复验 6/6 normal hard-close，Gate2 18/18 三方零误差 PASS。

schema-31 已完成的正式接入项：

- 每个 paired J/B intervention 只生成一份 mutant replay，在完整 fixed-C family
  上逐 profile 检查注册 exact delta；
- actual-π masked-J 只生成一份 masked replay，所有 profiles 都必须形成
  可比 fingerprint 并 hard mismatch；任一 hard closure 拒绝，任一不完整则
  `NO_CLOSURE`；
- 保存的真实 `fg_ones` 证据迁移为 normal 6/6、paired 6/6、masked
  6/6 hard mismatch、Gate2 18/18 PASS。

仍未完成的是 schema-31 fresh LLM 的其余跨 family 正式批次，以及 add、
row-sum、`fg_addr`、matmul 等更多真实边界 profile。

`fg_skip_layer_norm` 已增加固定 `BLOCK_SIZE=512` 下的 `N={1,257,511,512}` 四 profile；
保存真实 normal replay 在当前 stack 上 4/4 hard-close。这是 fused norm 的第一份 fixed-C
normal family 证据，paired/masked full-family 尚未完成，因为 fresh control 已在 primary
paired intervention 阶段被拒绝。

第一阶段仍只允许 fixed-C。跨 BLOCK/warps/stages 的多 C family 需要单独论证和批准，
不混入当前实施。

### 4.3 J 的复杂语义只有部分 schema 证据

已经补齐 schema 不等于已经有真实 Kernel 证据。

未完成项：

- loop carry：真实 structured-loop Kernel，更多 carry dtype 和对应 MLIR lowering；
- buffer lifetime：多阶段临时 buffer、alias/复用和跨 scope lifetime 的真实 Lift；
- frontend scope：真实嵌套 helper/loop/program-instance 的 captures、definitions、effects；
- branch-region：当前没有一等 IntentIR branch op，继续 fail-closed；只有真实 Kernel
  证明现有 J 无法表达并经用户批准后，才进入 schema/op 扩展；
- effect/access：atomic、scatter、scan 等复杂副作用的真实闭合证据；
- 动态 rank、动态输出、稀疏/间接访问等现有规划能力的真实 extraction 证据。

每个 J 扩展必须由真实失败报告驱动，报告至少回答：

1. 哪个真实 Kernel 和源码结构无法表达；
2. 为什么现有 J.A/J.B 组合不够；
3. 缺口属于 A、B 还是 C；
4. 对 composer、RuleCard、Gate1 指纹和 Gate2 Lower2C 的影响；
5. 为什么不能只修 prompt、C.strategy 或 RuleCard。

未经批准，不因“以后可能需要”预先增加 op、字段或 strategy family。

### 4.4 C.strategy 与 RuleRegistry 仍需真实失败驱动扩展

当前四个 override family 只覆盖：

- program coordinate width；
- address association；
- mask materialization；
- reduction call shape。

未完成/待验证：

- `materialize_pointer_tile` v3 要在未参与该规则修正的真实 Kernel 上验证，避免只拟合
  matmul；
- 每张 RuleCard 的施工语言是否足够确定，不能给 writer 留下发散的表达自由度；
- atomic、scan、sort/top-k、复杂 gather/scatter 等真实 Kernel 是否需要现有卡的新
  coverage，或确实需要新卡；
- RuleCard 增长样本和冻结后的 held-out 样本必须分开；
- 新卡或新 strategy family 只能从稳定、可复现的 source/replay diff 中增加；
- 修改规则后必须重跑该卡覆盖的全部 development case 和未见 evaluation case。

禁止通过以下方式提高闭合率：

- 把 operator、predicate、index expression 或源码拓扑复制进 π；
- 用自由文本 strategy 代替闭合 enum；
- 给单个 Kernel 创建专用规则卡；
- 把 writer 的随机选择包装成新的 π 语义。

### 4.5 Gate1 规模路径仍需优化

Gate1 核心裁决已实现，但仍需：

- fixed-C family runner 已接入 live/batch，但 schema-26 正式批次实际仍只有
  `fg_ones` 进入 family execution；`fg_eye/fg_log_softmax` 在更早阶段终止；
- 在 held-out 中验证当前 normalization 只消除固定、可枚举等价差异；
- 对多 profile relift 使用确定性差分签名判断 stagnation，不使用 score 变化率；
- 保证每一轮 relift 后 source、schema、compose、静态扫描和 fresh replay 全部重跑；
- 保留 Gate1/Gate2 四象限，允许合理出现 `Gate1 no-closure + Gate2 pass`；
- 继续验证 direct π tamper 在 writer 前拒绝，C.strategy change 则走普通完整 π；
- 旧 fuzzy-ratio API、`Gate1Result(repair_attempts/leakage_ratio)` 和 fallback policy
  alias 已移入 `python/intentir/verify/back/legacy_v0/`；masked writer 仍是正式
  terminal attack。活动 Gate1 不再携带这些旧配置或报告字段。
- 内部 `CandidateRecord` 只用于审计；外层 Relift 仅接收 source-free
  `ReliftEvidence`，其中不含 `ReplayGeneration`、replay code、`CompileResult` 或原始
  编译输出。schema 24 已删除
  `replay_code_repair=disabled`/repair-count 一类容易被误认为可开启子系统的否定字段；
  schema 26 绑定有序 fixed-C family contract 与 `scrc_case_performance_v1`；schema 27
  又绑定确定性 stagnation policy 和 TTIR fingerprint schema；schema 28 进一步绑定
  Gate1 hard-diff semantic guidance 与“无 hard diff 的 provider failure 不 Relift”；
  schema 29 绑定结构化 NO_CLOSURE termination；schema 30 进一步绑定完整
  fixed-C family sweep/Relift/stagnation 与 Gate1/Gate2 四象限 v2；schema 31
  绑定 full-family paired/masked 和组合 closure policy；schema 32 显式绑定
  `deterministic_composer_v3`；schema 33 绑定 Lower2C boundary-broadcast v4；当前
  schema 34 再绑定 compressed along-axis gather v5，
  防止旧 compose/lowering 语义、旧提示、旧停止归因、post-hoc family diagnostic 或
  错误恢复逻辑混入正式聚合。

schema-26 真实批次证明了当时尚未实现文档要求的 stagnation 提前停止：
`fg_arange` 三轮 hard-diff signature 完全相同，却仍消耗完整两次 Relift。当前
schema-27 已实现 `exact_artifact_and_hard_diff_v1`：只在连续两次**完整 fresh normal
replay** 后，canonical J/interface/C.schedule/C.strategy/π/registry digest 与全部硬分量
的结构化差分签名同时完全相等时，返回 `NO_CLOSURE` 并停止。它不使用 score、错误
文本或 replay code；generation/compile/parser 不可比较失败会打断连续链。仅 artifact
相同不会在 writer 前提前停止，从而保留“同一 artifact 的下一次 fresh replay 可能
闭合”的真实情形。该协议变化与 scope-aware TTIR parser v2 一同进入 schema-27
resume identity；`fg_arange` fresh 复验仍待执行。

schema-28 新增 `deterministic_gate1_hard_diff_guidance_v1`。只有完成 hard compare 的
失败才生成 semantic worklist；其中包含 exact diff digest、失败/已确认类别、按语义
edge role 分组的 dataflow delta、J 检查面、允许考虑的既有 strategy family 和 J 摘要。
纯 provider timeout/quota/model-cooldown 没有 replay candidate/hard diff，不再浪费
Relift。schema-31 fresh `fg_log_softmax` 已形成真实 hard diff，并暴露旧 family
worklist 重复展开约 68 KB、未突出 source 2 loops/replay 0 loop/J 0 carry 的问题。
当前已增加 deterministic loop/reduction/effect 结构 guidance、family consensus exact
diff 和 bounded prompt views，并将活动 policy 升为
`deterministic_gate1_hard_diff_guidance_v2` /
`deterministic_fixed_c_family_hard_diff_guidance_v2`；完整 audit comparison 保留，
prompt 不再重复全部 edge。

schema-29 新增 `structured_no_closure_termination_v1`。所有活动 Runner 产生的
`NO_CLOSURE` 必须区分 deterministic stagnation、replay provider unavailable、Relift
budget exhausted、Relifter unavailable、Relift failed、intervention evidence incomplete 和
masked evidence incomplete；
live/batch 不再通过 `stagnation != null` 或诊断文本猜测原因。该字段只改变审计与聚合，
不改变 Gate1 verdict；Gate2 diagnostic 仍可在有不可变 replay observation 时独立执行。

schema-30 新增 `fixed_c_family_full_sweep_relift_v1`。完整 family sweep 位于 writer 后、
Relift 前；secondary profile failure 现在是 normal Gate1 verdict 条件，并由
`deterministic_fixed_c_family_hard_diff_guidance_v1` 一次性回流 Lift。family stagnation
使用完整 profile 集的 canonical hard-diff digest。batch 同时升级为
`scrc_gate1_gate2_conditional_v2`，直接报告四象限与 incomplete coverage。

schema-31 进一步将 paired/masked 接入完整 family，分别使用
`fixed_c_family_exact_delta_all_profiles_v1` 和
`actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1`。活动报告现在可以明确
`anti_leakage_scope=full_fixed_c_family`；不完整证据仍 fail-closed。

### 4.6 Gate2 production Lower2C 尚未覆盖全部 J

本轮已经完成：

- `intent.iota` 的 backend-neutral linalg lowering；
- integer output 的通用 read-back、Boundary broadcast rank insertion 与 compressed
  along-axis gather legalization，当前
  production 协议为
  `intent_linalg_cpu_symbolic_iota_integer_readback_boundary_broadcast_compressed_gather_v5`；其中
  `attrs.n` 可由 Gate2 `dims` 绑定 runtime symbolic extent，已有 BroadcastSpec axes
  通过 `tensor.expand_shape` 接入通用 linalg affine broadcast maps；compressed gather
  通过 expand-index、等 rank gather、collapse-result 接入既有严格 op；
- Gate2 对 i1/整数输出采用精确比较，对浮点输出继续采用逐输出容差，协议为
  `exact_integer_tolerant_float_v2`；大整数不会先转换成浮点再用 allclose；
- i8/i16/i32/i64 iota、i1 布尔读回、signed-i64 边界，以及 runtime-symbolic
  `start + step * iota` 浮点组合的 MLIR/CPU execution regression。

另用 2026-07-16 保存的真实 LLM `fg_arange` J 与 replay 做了诊断性三方矩阵：
`forward_fractional / descending_fractional / forward_unit × 3 seeds` 共 9 个 case，
source↔replay、source↔Lower2C、replay↔Lower2C 全部零误差 PASS。该结果证明当前
production Lower2C 接缝对真实 J 有效；随后 schema-24 fresh Gate1/Gate2 已单独完成，
结果见 4.1，不再由该历史 artifact 代替。

schema34 fresh `fg_diag_extract` 尚未形成真实闭环：本轮 Lift 生成了整数
`intent.contraction` 来人为构造 `[N,1]` index，而 production contraction lowering
明确只支持浮点，因此 9 个 Gate2 cases 均为 `execution_failed`，Gate1 也未闭合。
该结果不应归因于 compressed-gather legalization；后续需要把首轮 Gate1 差异后的
Relift 指导收紧到 J 已有 compressed gather 或其他受支持的整数索引构造，不能为了
迁就一次错误 Lift 擅自增加整数 contraction。

当前仍未完成：

- `fg_trace` 已有真实 compiler contract，但三次 Lift 分别因非法动态 shape symbol 和
  未注册 `triu` op 失败，尚未进入 Gate2；
- `fg_eye` composer-v3 fresh 已 Gate1 closed；Lower2C v4 使用同一保存真实
  artifact/replay 完成 12/12 Gate2 PASS，四个 fixed-C profiles 与 masked-J family
  结论保持不变；
- `fg_log_softmax` parser 缺口已修复，schema-31 fresh 已完成 2 replay/4-profile
  Gate1 与 12-case Gate2。当前未完成的是用改进后的 Relift guidance 再次 fresh 验证
  是否能把 multi-tile recurrence 表达到 J loop/carry，或明确判为当前 schema 无可表达；
- structured loop/scan、atomic/scatter、sort/top-k、动态输出等复杂 artifact 的 production
  lowering 与真实三方执行；
- 更广 dtype/rank/shape 组合的输入绑定和输出读回。

`fg_skip_layer_norm` 的 4 shape × 3 value × 3 seed Gate2 matrix 新增了
normal、large-dynamic 和 near-constant。normal/large-dynamic 27/27 PASS；near-constant
在 `N={257,511,512}` 上 9/9 `lowering_diverged`，source↔replay 精确通过，Lower2C 最大
绝对偏差约 `9.7e-5～4.3e-4`。float64 reference 显示 source/replay 更接近高精度结果。
当前保留该 Lower2C 数值稳定性缺口，不放宽统一 tolerance，也不反向修改正确 J。

Gate2 的处理原则：

- lower 不支持时返回 `LOWERING_UNSUPPORTED`；
- 不回退归档解释器；
- 不把 unsupported 计成已经完成的语义失败；
- Gate2 PASS 不覆盖 Gate1 no-closure；
- Gate1 closed 但 Gate2 replay diverged 是高优先级缺口，必须检查 Gate1 指纹是否漏检；
- source≈replay、lower 偏离时优先修 Lower2C，不反向篡改 J。

## 5. 已实现但需要优化的部分

| 优先级 | 组件 | 当前状态 | 待优化目标 | 验收方式 |
|---|---|---|---|---|
| P0 | held-out batch | schema-26 的 17-ready 正式批次已完成；当前 schema-33 为 27 candidates / 20 ready，新增三个 non-reduce source contract 已通过 compile/fingerprint，`fg_eye` 已在 composer-v3 下 fresh closure 且 Lower2C-v4 三腿通过 | fresh 复验 parser/stagnation/family-diff-guided Relift 与 full-family anti-leakage，继续处理真实 family 缺口，区分 growth/evaluation | 每例有结构化 terminal report，无静默 cache 升级；任何 replay 都无 prior-code 输入；composer/Lower2C/family/performance/stagnation/TTIR/Relift guidance/termination 均进入 identity |
| P0 | RuleCard v3 | 五核 matmul 已闭合 | 在未参与设计的 address Kernel 上复验 | hard diff 为空或失败得到固定可枚举归因 |
| P0 | Gate1/Gate2 报告 | `scrc_gate1_gate2_conditional_v2` 已直接输出四象限和 coverage | 用 fresh batch 扩大真实象限覆盖 | 不生成 overall pass 覆盖两道门；unsupported/incomplete 不计 non-pass |
| P1 | multi-specialization | schema-31 normal/paired/masked 均已接入完整 fixed-C family；保存真实 `fg_ones` 为 6/6 + 6/6 + 6/6 | 扩大 schema-31 fresh family 覆盖 | 每个实验臂只有一份 replay；normal 全闭合、paired 全 exact-delta、masked 全 hard-mismatch |
| P1 | relift | 单 profile 与 fixed-C family 均常开 | 用 fresh secondary failure 验证 family guidance 真正改变 J/C.strategy 或明确无可表达变化 | 不修 replay，终止原因完整保存 |
| P1 | Gate2 matrix | 已有 shape/value/seed suite | ready case 按合法输入域启用矩阵 | 每个子 case 必须 PASS 才算 suite PASS |
| P1 | Lower2C | iota/integer read-back 与 Boundary broadcast rank insertion 已补；`fg_arange`、`fg_eye` 保存真实 artifact 三腿通过 | 继续按真实 unsupported 补 gather/loop/scan/atomic 等复杂组合 | 真实 artifact 三方执行 + MLIR regression |
| P2 | J complex features | schema/原型部分存在 | loop/lifetime/scope/effect 真实覆盖 | fresh Lift + Gate1 + Gate2 证据 |
| P2 | 兼容表面 | 旧 interpreter 已归档 | 继续清理旧 masked/score 工具与历史报告引用 | production 路径零反向依赖 |
| P3 | 大规模 | 27 候选、20 ready；17-case 首轮耗时约 2.49 小时 | 先完成新增非 reduce family 的 fresh 证据，再到 50～100 分层样本和数百 Kernel | 分家族报告，不用单一总体数字掩盖缺口 |

## 6. 分阶段实施计划

### 阶段 A：v11 / schema-31 正式复验

输入：当前 20 个 reviewed ready case；首轮 17-case schema-26 pilot 已完成，后续以 schema-31
协议先重跑受 parser/stagnation 影响的样本，再扩大正式批次。
执行：

1. source preflight；
2. fresh `gpt-5.6-sol/xhigh` 完整 Lift，在 Lift 内生成 J/C/π，不使用
   strict-v5-v7 artifact；
3. Gate1 relift-only，使用当前默认预算；
4. closed control 后执行 paired intervention 和 π audit；
5. Gate1 closed 进入正式 Gate2；rejected case 可运行独立 Gate2 diagnostic；
6. 对有 matrix contract 的 case 开启 shape/value/seed suite。

阶段输出：每例 JSON、batch summary、Gate1/Gate2 paired observation、failure taxonomy。
退出条件不是“全部通过”，而是：

- 17/17 都得到可复核 terminal 状态；
- 无 harness error 被伪装成语义 non-pass；
- 每个失败都能归入 Lift/schema/compose/writer/compile/hard-diff/Lower2C/infrastructure；
- growth case 与 evaluation case 分开报告。

### 阶段 B：fixed-C multi-specialization 接线

首批 family：

1. `fg_ones`：`N={1,B-1,B,B+1,non-divisible,multi-tile}`；
2. add：固定 BLOCK 的多 N/grid；
3. row-sum：M/N tail 与多 row；
4. `fg_addr`：已审查的 stride/profile；
5. matmul：M/N/K tail 与多 K tile。

退出条件：

- 一份 artifact、一份 replay、固定 C；
- 每个 profile source/replay 都编译并做硬比较；
- family verdict strict all-pass；
- 任一 profile 失败都统一回流 Lift；
- 不出现 profile-local replay repair；
- report/resume identity 包含完整 profile family。

### 阶段 C：复杂 J 特性真实样本

按真实样本可获得性推进：

1. structured loop / loop carry；
2. buffer lifetime / alias；
3. frontend nested scope；
4. gather/scatter/atomic effect；
5. scan/sort/top-k；
6. 动态/间接访问和动态输出。

每个特性先运行现有 J；只有得到确定失败并完成缺口报告后，才决定修改 prompt、J、
C.strategy、RuleCard 或 Lower2C。手写 MLIR/固定 J 只作单元回归，不计作本阶段完成。

### 阶段 D：完成当前 27-case 队列

为剩余 queued case 建立 reviewed compiler/execution contract，逐个进入阶段 A/B/C
流程。规则增长必须留下：

- 触发该变化的真实 case；
- source/replay 硬差分；
- 修改落点与为什么属于 J/C/规则库/Lower2C；
- 受影响回归集合；
- 新规则冻结后未见样本的结果。

退出条件：27 个候选全部有明确状态，不能只报告通过子集。

### 阶段 E：50～100 Kernel 分层验证

在 27-case 流程稳定后扩展，至少按以下家族分层：

- elementwise/broadcast；
- reduce/norm；
- contraction；
- indexed/gather/scatter；
- loop/scan；
- sort/top-k；
- atomic/effect；
- fused/attention 类。

正式评价批次运行前冻结 J schema、RuleRegistry 和 comparator 版本；批次中出现失败时
先完整记录，不边跑边修改后继续混算同一批成功率。需要修复时结束该批，形成新版本
后重新运行受影响批次。

### 阶段 F：数百 Kernel

只有阶段 E 稳定后进入。沿用相同 Gate1/Gate2 流程，不增加新的综合评分或验证模式。
规模化重点是批处理、失败分诊和覆盖分层，不是放宽 Gate1。

## 7. 失败后的允许修改路径

| 失败阶段 | 首要判断 | 允许修改 | 禁止修改 |
|---|---|---|---|
| Lift schema | 模型是否误填现有字段 | prompt、schema 错误提示 | replay code |
| Lift 内 deterministic finalize | J/C 是否缺引用，RuleCard metadata 是否不唯一 | LiftCore 的 J/C、composer、RuleCard metadata | Gate1 补 π、手写 π |
| writer generation | packet 是否可执行、规则说明是否确定 | Lift 的 J/C、共享 RuleCard | 对失败 replay 打补丁 |
| replay compile | J/C/规则施工是否形成非法 Triton | relift、RuleCard | source-specific adapter |
| Gate1 hard diff | J 漏错、C 选择错、规则施工歧义、固定 normalization gap | 对应权威层 | fuzzy score 放行 |
| Gate2 replay diverged | Gate1 是否漏检或运行 specialization 不一致 | Gate1 fingerprint/contract | 用 Gate2 覆盖 Gate1 |
| Gate2 lowering diverged | production Lower2C 是否错误 | Lower2C/已有 J 映射 | 为迁就 lower 篡改正确 J |
| Gate2 lift diverged | J 与 replay 是否自洽地错 | J schema/prompt/relift | 只修 lower 或 replay |
| lowering unsupported | 现有 J 是否已表达、lower 是否缺实现 | Lower2C；必要时提交 J 缺口报告 | 回退解释器伪报 PASS |

如果拟新增 IntentIR op、J 字段、C.strategy family、跨 C family 或改变 Gate1/Gate2
verdict，必须先提交必要性报告并获得用户批准。

## 8. 每轮报告要求

不新增综合分数，只保存项目已有的确定性原始证据：

- source hash、真实来源与 compiler view；
- Lift/relift response id、attempts 和失败原因；
- J/interface/C/π canonical digest 与 composer provenance；
- source/replay compile 状态；
- Gate1 各硬类别及精确 difference；
- relift 轮次、终止原因和 best failed candidate；
- paired intervention 与 π/C diagnostic；
- Gate2 每个 shape/value/seed case 的三腿状态和 pairwise agreement；
- Gate1/Gate2 四象限与 Gate2 completion coverage；
- unsupported、setup failure、provider failure 与语义 non-pass 分开计数。

任何成功率必须同时给出：总样本数、实际进入该阶段的分母和未完成原因。不能把
unsupported 或未运行样本静默排除后只报告通过率。

## 9. 当前最近执行顺序

建议下一步严格按以下顺序推进：

1. `fg_arange → fg_diag_extract → fg_skip_layer_norm → fg_index_select` 的 schema-31
   fresh 顺序已完成；保留各自 terminal 状态和 Gate1/Gate2 正交证据，不合并成 overall
   pass；
2. 暂不优化 π topology，只保留 actual-π masked-J 监测；跨 family 批次出现成批泄漏
   时再报警并提交计划外必要性报告；
3. `fg_eye` deterministic broadcast compose 重复已由 composer-v3 修复，schema-32
   fresh Gate1 已闭合，schema-33 Lower2C-v4 保存实物复验 Gate2 12/12 PASS；`fg_trace`
   仍停在 Gate1 前 Lift 合规失败，保留真实诊断，不以
   generic initial Lift 一次成型为优化目标；
4. `fg_embedding` 暴露 grid-derived tensor extent 尚无一等 J/interface 绑定；非法
   1/17/33 profile 已撤回。该 schema 扩展属于计划外候选，先审查
   `doc/SCRC_composerV3跨Family实测与GridExtent缺口_2026-07-19.md`，未获批准前不实现；
5. 扩展 add、row-sum、`fg_addr`、matmul 的 fixed-C family，并从 queued case 中优先推进
   scan/atomic/sort 等非同质 family；
6. 完成 27-case 队列后，再建立冻结规则库的 50～100 Kernel evaluation batch；
7. evaluation batch 稳定后进入数百 Kernel。

## 10. 阶段性完成定义

### 可以称为“正式 pilot 完成”

- 当前 ready case 全部得到 terminal report；
- 每例均为真实 source + fresh LLM Lift；
- Gate1/Gate2 独立保存；
- 失败完成机械归因；
- 没有旧 interpreter、手写 J 或旧 cache 混入正式证据。

### 可以称为“fixed-C Kernel family 验证完成”

- 一份 J/C/π/replay 覆盖 manifest 声明的全部 runtime profiles；
- Gate1 strict all-pass；
- Gate2 matrix strict all-pass；
- 任一失败不会被平均 score、多数票或主 profile 覆盖。

### 可以称为“规模验证基础完成”

- 27-case 队列全部有明确状态；
- 复杂家族不再只有手写 MLIR/固定 J 证据；
- development 与 evaluation 样本分离；
- RuleRegistry 冻结后有未见样本结果；
- Gate2 completion coverage 与 unsupported 情况完整报告。

### 仍不能宣称

- Gate1 是一般 Triton 语义等价判定器；
- 五核或 12-case 结果代表数百 Kernel；
- Gate2 PASS 说明 Gate1 no-closure 不合理；
- 所有 IntentIR op、dtype、scope、effect 和 lowering 已完整覆盖；
- 通过补规则卡可以绕过真实 J 语义缺口。
