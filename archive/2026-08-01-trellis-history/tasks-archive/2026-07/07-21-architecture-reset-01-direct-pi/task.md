# Architecture Reset 01：清除旧控制面并切换到 LLM-direct π

> 状态：已完成（2026-07-21）
> 优先级：P0，IntentOp 健壮性研究之前的唯一前置任务
> 流程：普通 Markdown 工作项；不运行 Trellis workflow

## 目标

彻底清除已经错误进入活动系统的 deterministic construction/control plane，建立与
[LLM-first spec](../../spec/index.md) 一致的最小可运行主链：

```text
Lift LLM(source) -> anonymous J + pure C + π
Replay LLM(J,C,π,frontend public context/tools) -> complete source
Gate1 normal closure + at most one masked-J
Gate2 Source↔Replay and Source↔E_J
```

本任务完成前，不开展 IntentOp/atomic/fusion/pipeline 扩展，也不优化 backend 性能。

## 为什么这是第一步

当前错误不是一个孤立模块，而是一条依赖链：

```text
C.strategy
  -> composer work items
  -> RuleRegistry selection
  -> ReplayPlan + selected cards
  -> normal/masked prompt
```

另一条重复语义支线是：

```text
J fragments -> program_adapter -> IntentProgram + LogicalABI -> lowering
```

如果先继续补 IntentOp，新 op 会再次被迫增加 RuleCard、strategy、adapter 和 wrapper，
扩大错误控制面。因此必须先断代、迁移最小通路，再一次性删净残留。

## 范围 A：新 artifact 断代

1. 定义全新 schema epoch，只接受：

   ```text
   Intent: {J, C}
   Gate1 witness: {π, evidence}
   Harness: invocation metadata
   ```

2. 旧 artifact/cache 明确报 unsupported version；不写兼容转换器、不保留双轨。
3. 删除 registry digest、strategy digest、compose provenance 和 selected-card identity。

## 范围 B：Lift LLM 直接输出 π

1. Lift response 直接包含匿名 `J`、纯 `C`、`π`。
2. π 最小 schema 只允许 instance/region/route/slot/order/share/placement、必要的
   source-neutral realization tag 与 opaque semantic reference；tag 不得携带逻辑
   operator/combiner/predicate/value。
3. 确定性代码只检查 schema、ID 唯一和引用存在；不得推导、排序、补步骤或展开规则。
4. 源参数在 Lift 边界按声明顺序匿名化；真实名字只在 harness sidecar。

## 范围 C：Replay 直接消费 J+C+π

1. Replay packet 只包含 J、π、C slots，以及公共 frontend 文档/工具访问能力。允许 LLM
   动态检索，但检索内容不得来自私有源码或由核心代码按当前 op/pattern 选择。
2. 删除 selected RuleCard、per-op API mapping、kernel-specific instruction 注入。
3. frontend API 安全白名单若仍需要，必须成为固定全局 policy，不能按 J/Kernel 选择。
4. Replay 失败回到 source-aware Lift，禁止使用旧 candidate 或 source/replay diff 修补。

## 范围 D：删除 Gate1 错误四件套

完整删除活动实现中的：

- `python/intentir/verify/gate1/composer.py`；
- `python/intentir/verify/gate1/rules.py`；
- `RuleCard`、`RuleRegistry`、rule release 和 selected-card payload；
- `C.strategy`、`StrategySpec/Family/Choice` 及所有 source-realization family；
- `compose_replay_plan`、`materialize_replay_plan`、compose provenance；
- reduction mask placement、preheader/fusion/address-association 等 strategy patch。

不要保留 fallback composer、default registry 或“临时兼容”路径。

## 范围 E：在线协议收敛

1. core Gate1 只运行 normal Replay；normal hard-close 后最多运行一次 masked-J。
2. 重写 masked-J，使其保留匿名 π、纯 C 和相同的公共 frontend 工具能力，但不再暴露
   rule ID、selected API 或 strategy；工具/检索 trace 纳入审计 receipt。
3. semantic intervention、π swap/dropout、C swap、fixed-C family sweep 全部从在线
   orchestration/config/report 移除；需要时迁入独立离线实验目录。
4. 成功 artifact 的 LLM 调用上限：一次 Lift、一次 normal Replay、最多一次 masked-J。

## 范围 F：删除 Program/ABI 重复语义支线

1. 删除持久化 `IntentProgram`、`LogicalABI/LogicalInterface` 和 `program_adapter`。
2. 保留真正属于 J 的 IntentOp 与 B 内容；不要整文件误删语义数据模型。
3. scalar executor、MLIR emitter 和 production lowering 直接接收完整 `J=A+B`。
4. C 作为独立 lowering hint 传入；不得重新附着到 J。
5. source/replay invocation 使用 harness positional binding，不建立第二份 ABI 真值。
6. 删除 wrapper-only digest、obligation wrapper、adapter cache 和对应兼容 shim；若 B
   backend consumption 尚未实现，必须 fail closed，而不是退回 A-only。

## 范围 G：测试和残留清理

删除或重写旧架构测试，不建立兼容断言。活动代码、tests、CLI、spec 中以下符号应零引用
（archive 与历史 artifact 除外）：

```text
RuleCard
RuleRegistry
compose_replay_plan
materialize_replay_plan
C.strategy
StrategySpec
selected_rule_cards
compose_provenance
IntentProgram
LogicalABI
LogicalInterface
program_adapter
run_semantic_interventions
max_semantic_interventions
reduction_mask_placement
```

新增最小测试：

- Lift direct-π strict schema/round trip；
- π 非法引用和语义原子拒绝；
- Replay packet source isolation；
- normal Gate1 exact closure；
- masked-J packet 不含旧 side channel；
- Gate2 两条 source-anchored 边仍执行；
- 新 schema 拒绝旧 cache/artifact；
- repository ban-list regression。

## 验收标准

1. 活动 production path 对禁用符号零引用。
2. Lift LLM 直接产生 J、纯 C 和 π；没有中间 construction engine。
3. π 只含 opaque realization topology/ref，不含 semantic atom、源码自由文本或前端 API。
4. Replay LLM 直接生成完整源码，不依赖 per-op/per-kernel规则。
5. C 只有 schedule/hint，不存在 strategy 兜底。
6. 至少一个简单 Kernel 和一个包含 mask/reduction 的 Kernel 完成 normal strict Gate1。
7. 每个成功 artifact 至多额外执行一次 masked-J；其他消融不在在线代码中。
8. Gate2 的 `Source↔Replay` 与 `Source↔E_J` 两条边完整保留。
9. scalar/production lower直接消费完整 J，不存在 A-only 静默退化。
10. 旧 artifact/cache 不兼容且明确失败；没有 hidden fallback。

## 完成记录

活动主链已经断代为 `intentir.gate1.llm-direct.v1`：Lift 一次直接返回匿名 J、纯 C 与 π；
Replay 一次直接消费 `J+C+π+public frontend context`；normal hard-close 后在线最多再运行
一次 masked-J。旧 composer/rule/strategy/intervention/fixed-family 控制面、Program/LogicalABI
第二 semantic wire、旧 live/heldout 驱动和历史 artifact 已移出 importable tree并归档。

关键闭合点：

- π 使用 closed region/instance/route/order/sharing schema，只接受匿名引用；
- Source 在 Lift 前 alpha-anonymize，实名仅存在 harness sidecar；
- Gate1 按声明位置归一 Source/Replay 参数 spelling，同时 hard-check arity、kind、rank、
  dtype、constexpr 位置和 C 值；TTIR 数据流硬比较没有放宽；
- scalar `E_J` 与 CUDA production boundary 直接接收同一 canonical J；未实现的 B relation
  在执行/codegen 前 fail closed；
- runtime manifest 只保留无 dtype/mode/rank 语义的 invocation slots；C 始终独立；
- 旧 schema、旧字段和活动旧符号均由 regression test 硬拒绝。

提交前证据：

```text
605 passed, 4 skipped  (完整 python/tests，含两个真实 CUDA Gate1)
227 active JSON files validated
compileall passed
git diff --check passed
active legacy-symbol scan: zero matches
```

真实 CUDA Gate1 覆盖一个 masked copy 与一个 `masked load -> square -> reduce` kernel；两者
在匿名参数名下均通过 interface/op/dataflow/reduction/predicate-effect 五类 Level-2 hard
closure。

## 明确非目标

- 不冻结新的 IntentOp primitive set；
- 不新增 Atomic/Fusion/Pipeline named op 或合同；
- 不提高 replay 成功率或添加修复循环；
- 不优化 CUDA/RVV 性能；
- 不重跑论文完整消融；
- 不恢复 Trellis workflow。

## 完成后的下一项工作

独立开展 IntentOp robustness audit：用 copy/reverse、fusion、logical update/atomic、
async pipeline、dynamic indexing 和 semantic recurrence 的真实代表样本，验证小型可组合
J 核心是否足够，再决定最终 op schema。
