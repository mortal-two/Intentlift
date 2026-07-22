# Lift Envelope / Boundary 10：闭合 Root Ownership 与 Access 完整性

> 状态：进行中
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

## 为什么这是独立任务

Protocol-v6 的模型响应正常 completed、未截断，也正确生成了三组
`address_base -> address_lane -> transfer_direct`。但它把本应位于 root 的
`interface/schedule/pi/location_map` 放进 `j`，导致 Lift parser 在 Replay 前 fail closed。

诊断性内存移动（未落盘、未 Replay）还发现三个 accesses 的
`index_expression/predicate/fill/domain` 被省略。因此这不是 address π capacity 问题，也不能靠
自动补一个右括号解决；root ownership 和 J boundary 都需要模型一次完整生成。

## 最终目标

让 source-aware Lift LLM 在一次响应中稳定产生：

```text
root = schema + j + interface + schedule + pi + location_map
j    = kernel_id + intent + node_ids + boundary + scopes + storage
```

并保证每个 load/store/update access 的 logical index relation、源码逻辑 mask/fill/domain 不会因 frozen invocation
恰好全覆盖而被删除。固定代码仍只验证并 fail closed，不 hoist、不补字段、不从 source/v5 artifact
复制内容。

## 架构不变量

- Intent 仍是 `J(A+B)+pure C`，π 在 Intent 外；本任务不改变 address mechanism 或增加新层。
- Lift LLM 直接生成完整 artifact；不引入 composer、RuleCard/Registry、IntentProgram、LogicalABI、
  section assembler、source-to-J/π extractor 或 deterministic response repair。
- root grammar 只表达对象 ownership/required keys，不携带 kernel 语义或 source spelling。
- J boundary grammar 只投影 parser 已有的全局 access contract，不按 Swiglu/LayerNorm/opcode 分支。
- 每个 load/store/update access 的 index expression、logical domain/predicate/fill 属于 J；π 不得承载或修复它们。
- Replay 仍 source-blind；Gate1 strict topology 不弱化；normal 未闭合不运行 masked-J。
- provider `strict=false` 事实显式保留，本地 parser 是最终权威；同一 epoch 不自动 retry。

## 阶段与提交节奏

### 0. 冻结 v6 反例并审计 parser 真相（0 次 LLM）

- 固定 v6 root/j key sets、raw completion/usage/receipt SHA；
- 枚举 Gate1Artifact root required/allowed fields与 SemanticArchive J fields，形成唯一 ownership 表；
- 审计 BoundaryAccess/Layer-B parser：rank、index_expression、predicate/fill/domain/effect 的真实条件；
- 明确哪些条件应投影 prompt/schema description，不能扩大 J 或用 frozen v5 作为模板 repair。

完成后提交。

### 1. 共享 Envelope / Boundary Grammar（0 次 LLM）

- root/J required field declarations只维护一份，由 parser、Responses schema与 Lift prompt投影；
- prompt 给出短小通用 skeleton，明确 `j` 在 `storage` 后关闭，其他 section 是 root siblings；
- boundary grammar 同时进入 schema description 与 Lift prompt：每个 load/store/update access 必须有 ordered logical
  index expression；源码存在 logical mask/fill/domain 时必须保留，即使 frozen extent 全覆盖；
- 不增加 parser 宽容路径，不接受 misnested alias，不做 hoist/repair。

完成后提交。

### 2. 双向 Conformance 与全量回归（0 次 LLM）

- canonical v5 artifact root 与 boundary 在 schema/parser 双侧通过；v6 raw 在 root 精确拒绝；
- 将 root section 移入 j、遗漏任一 root/J field、access 缺失/null/空 index 等结构 mutation 均在
  schema/parser 双侧 fail closed；删除源码 logical predicate/fill/domain 不能由单个 artifact 的 parser
  证明，须由共享 Lift ownership grammar 与真实 Gate1 strict topology 暴露；
- parse/to_json 幂等，source-private location map不进入 Replay；address π 原样存活且 masked-J 不泄漏；
- v1--v6 evidence SHA 不变，全量回归通过。

完成后提交。

### 3. Protocol-v7 单一 Canary

- v7 28-case manifest/source invocation 与 v6 相同，只改变 Lift wire；
- v7 recovery 精确绑定 v6 receipt SHA、一次 failed Lift、0 Replay/Gate1、completed raw response、
  misnested root failure 与 v6→v5 lineage；
- 0-LLM preflight 28/28 ready 后，fresh Swiglu 最多 1 Lift + 1 normal Replay；normal hard-close 后
  最多 1 masked-J；任何失败立即停止；
- 若 normal 闭合且 masked-J 不闭合，进入剩余 27 case；否则冻结实际下一边界。

完成后提交、归档 Task10，并创建下一任务。

## 验收标准

1. root/J ownership 与 required fields 只有一份权威声明；
2. boundary parser 既有全局规则对 schema/prompt 可见，没有 kernel-specific 规则；
3. v6 misnest/boundary 反例精确拒绝，且不存在 deterministic hoist/repair；
4. address base/lane contract 保持不变，J/π 所有权不回退；
5. full regression 通过，历史 evidence 不修改；
6. v7 0-LLM preflight 28/28 ready，唯一 canary provenance与预算完整；
7. 只有实际 normal Gate1 hard-close 后才执行一次 masked-J。

## 当前第一步

阶段 0--1 已完成并提交：v6 反例、parser 真值、共享 root/J/boundary grammar、双向 mutation 与
v7 wire/lineage 已冻结；最终完整回归为 `867 passed, 4 skipped, 60 warnings`，未调用 LLM，也没有
repair 路径。下一步生成并复验 28/28、0-LLM preflight；只有 preflight evidence 提交且 worktree
clean 后，才允许唯一 Swiglu canary。
