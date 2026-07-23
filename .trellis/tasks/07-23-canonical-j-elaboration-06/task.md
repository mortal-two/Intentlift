# Task 06：CanonicalJ 标准化与统一 lowering 消费边界

> 状态：已完成
>
> 类型：SCRC 系统重构；不是实验内反馈，也不是让 Lift 学习具体 MLIR consumer 拼写。

## 直接目标

建立唯一、确定性的标准化前端：

```text
LLM Lift -> (J_wire,C_wire,pi_wire)
              |
              v
       Decode + Canonicalize/Elaborate
              |
              v
       typed (CanonicalJ,C,pi)
              |
              +-> TypeCheck -> Replay
              +-> RefEval -> Intent MLIR -> CPU reference
              `-> Production lowering(J,C,target)
```

开放世界的理解、语义提取与候选选择仍归 Lift LLM；封闭 IR 内可唯一决定的语法归一、类型推导、
执行和裁决归固定代码。RefEval 与 production lowering 必须共享同一个 `CanonicalJ` 和同一段
`CanonicalJ -> Intent MLIR` 前端，不得各自重新解释 J wire 字符串。

## 为什么现在做

Task05 已用真实 DYT 得到 `SCRC_CERTIFIED`，同时证明当前所谓 canonical J 仍残留 consumer-facing
自由拼写：二维 direct index 的 `k1,k0` 与 `[k1,k0]` 被不同层分别解释，producer type 与 identity
output mapping 也被迫交给 Lift 手工满足。这会把固定 IR elaboration 的责任错误推给 LLM，并让
RefEval/production consumer 漂移。

## 语义边界

允许固定代码完成且必须 many-to-one 的事项：

- bounded direct-index surface syntax 唯一解析为结构化 axes，例如 `k1,k0` 与 `[k1, k0]` 均成为
  `DirectAxes([k1,k0])`；
- predicate/domain 进入现有受限 ContractExpr AST，而不是由每个 consumer 临时解析；
- dtype、rank、shape、dimension handle 与能从 J dataflow 唯一决定的 result type 完成 typed
  elaboration；
- 完全相同的 direct same-handle output binding 进入唯一 canonical 表示；
- canonical serialization 稳定且 `Canon(Canon(x)) = Canon(x)`。

不得由标准化猜测或修复：

- 缺 predicate 的 fill、缺失 operator/effect/alias/authorization；
- 多义 affine/free text、未知前端 API、Source 意图；
- 需要 Source、Harness concrete value、Gate diff/output 或 LLM 才能决定的关系。

这些情况必须在 Decode/Elaboration/TypeCheck 的准确阶段 fail closed。

## 实施范围

### Phase 0：移除下一次 canary 的 fanout barrier

现行 `lowest_branch_after_gate1_search_barrier.v1` 在 Task05 中让 winner 只需 14 次 Lift call，而全局
仍执行 103 次 Lift call、墙钟约 40 分钟。先实现 cooperative first-closure stop：

- 首个正式 Gate1 closure exactly-once 成为 winner并立即进入 Gate2；
- 其他 branch 已开始的 provider call自然返回，在阶段边界停止，不再启动下一轮；
- loser 记录 `superseded`，不得冒充 `NO_CLOSURE` 或 infrastructure failure；
- Gate2 不等待 fanout drain，结果不回流任何 branch；
- 不粗暴 kill provider call，不改变 SCRC 接受谓词。

### Phase 1：定义 wire 与 canonical 类型

- 清点 `BoundaryAccess.index_expression`、predicate/domain、output mapping、result type 在 parser、
  TypeCheck、Replay packet、RefEval 和 production lowering 的所有重复解释点；
- 建立版本化 `CanonicalJ`/typed access 表示与单一 canonical serializer；
- raw LLM response 只作为 proof-irrelevant sidecar 保存，证书和消费者绑定 canonical artifact。

### Phase 2：确定性 elaboration

- 实现 bounded direct-index parser 和 ContractExpr/type elaboration；
- 只做唯一、可证明语义不变的归一；模糊或缺语义信息时 fail closed；
- 将现有 orphan-fill 等 invariant 保留在 canonicalization/TypeCheck 权威边界，不静默修复。

### Phase 3：统一消费者

- Replay packet读取 canonical serialization；
- RefEval 不再解析 wire index string；
- `CanonicalJ -> Intent MLIR` 成为 RefEval 与 production lowering 的共享入口；
- 两条下游路径只在 reference execution 与 target/C lowering 处分叉。

## 验收

1. `k1,k0`、`[k1,k0]` 和等价空白形式得到完全相同的 structured axes/canonical serialization；
2. ambiguous/general affine text fail closed，不由 consumer 猜测；
3. canonicalization 幂等，raw wire 不进入 Replay verdict、RefEval 或 production lowering；
4. RefEval 与 production lowering 不再各自调用 direct-index/predicate wire parser；
5. 唯一可推导的 type/output binding 被机械 elaboration，缺失语义仍拒绝；
6. 聚焦正反例覆盖 canonical equivalence、idempotence、consumer sharing 与 fail-closed；
7. fanout 单测证明 Gate2 在 loser provider call 返回前启动，停止后无额外 Lift/Replay；
8. 使用 fresh real canary 验证新 pipeline；只运行与改动相称的聚焦测试，不再对每个 prompt/spec
   checkpoint 重复执行全量回归。

## 非目标

- 不新增 Agent、Gate、hash/commitment、per-kernel RuleCard 或前端 formal semantics；
- 不把 CanonicalJ 变成第二份语义 artifact；它是 J wire 的唯一确定性内部形式；
- 不让 canonicalizer读取 Source、Gate1/Gate2 结果、历史 candidate 或 sibling branch；
- 不把 production optimization 放进 RefEval，也不让 C/π进入 `RefEval(J,logical_inputs)`。

## 2026-07-23 Phase 0 checkpoint

- fanout protocol/selection rule 升级为 `candidate-fanout.v2` /
  `first_gate1_closure_cooperative_stop.v2`；winner 不再由 search barrier 后的最低 branch id 决定；
- `FirstClosureControl` exactly-once 选择首个正式 Gate1 closure，并在 drain loser 之前执行 Gate2
  callback；
- 单 branch search 在 `before_lift`、`after_lift`、`after_typecheck/before_replay` 与
  `after_gate1/before_relift` 边界协作停止，真实已开始调用保留预算，停止后不再新增调用；
- branch 状态新增 `SUPERSEDED` 与 `GATE1_CLOSED_NOT_SELECTED`，停止原因、winner id 与观察阶段进入
  evidence，不复用 `NO_CLOSURE`/infrastructure failure；
- 聚焦回归仅运行 fanout、SCRC runner 与 CLI 三个相关文件：`27 passed`。未运行全量回归；
- 下一步进入 Phase 1，清点并定义 J wire 与 typed CanonicalJ 的唯一边界。

## 2026-07-23 Phase 1--3 completion checkpoint

- 新增 `intentir.ir.canonical_j`：ContractExpr v1 使用受限 AST parser，不执行文本；
  `CanonicalIndex.components` 保存 ordered structured expression，`k1,k0`、`[k1,k0]` 与空白变体
  canonicalize 为同一 `[k1, k0]` wire dump，malformed/call/attribute/wrong-sort fail closed；
- `ElaborateJ` 同时物化 closed op signature 与 J graph 唯一确定的 result dtype/shape，并消去经
  producer/store 机械确认的 direct same-handle identity output mapping；显式冲突在 consumer 前
  拒绝，不从 Source 猜测；
- raw provider response/trajectory 继续作为 proof-irrelevant sidecar 保存；Runner 将 Lift result
  替换为 canonical artifact 后才执行 binding、TypeCheck、Replay，TypeCheck 升级为 v2 并包含
  `j_elaboration_failed`；Replay packet 只序列化 canonical J；
- `ScalarLoweringSpec` 与 production `LoweringRequest` 持有同一个缓存 CanonicalJ view；新增唯一
  `emit_canonical_j_module`，scalar CPU RefEval 与 CUDA production lower 均通过它进入现有 Intent
  MLIR emitter；RefEval 的 dense/mask parser 已改为消费 structured indices/predicate AST，旧 helper
  只保留测试兼容，不再是 consumer 解释权威；
- Lift schema/prompt 不再要求模型猜 `k1,k0` 是否加括号、重复填写唯一 result type或手工删除恒等
  mapping；orphan fill、缺语义、类型冲突与 general affine-as-dense 仍 fail closed；
- Task04 真实 DYT `run-003/attempt-03` 的原始 artifact 未 relift、未手改：五个 pointwise type 被唯一
  推导，`k1,k0 -> [k1,k0]`，`t5 -> t5` mapping 被消去；原先失败的 complete-boundary 与实际
  scalar CPU RefEval 均通过，输出 shape `[2,8]`、profile `logical_domain_mask_erasure_v1`。这不
  retroactively 改写历史 run-003 verdict，只作为新系统能力的真实回归输入；
- 聚焦回归分两组执行：CanonicalJ/Gate2/backend/TypeCheck/runner `76 passed`；workload/canary/
  MiniSWE ingress `36 passed`。另有真实 DYT RefEval smoke 通过；按推进要求未运行全量回归，也未
  再消耗 provider 调用重做已经闭合的 Lift/Replay 搜索。
