# Pi Graph Integrity 08：闭合 realization 外键合同并恢复单一 Canary

> 状态：进行中
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

## 为什么这是独立任务

Protocol-v4 Swiglu Lift 已正确恢复二维 J、non-C `n_cols`、pure-C schedule，并试图表达 runtime
stride realization，但它把 raw Interface/C handles 写进 `pi.routes[*].source`。冻结 parser 要求
route/order/sharing 只连接已声明 `i*` instances，因此 artifact 在 Lift 边界 fail closed，Replay
和 Gate1 都没有运行。

这个反例尚未证明 π 缺少 address/cast 等信息容量；它证明的是现有合同虽然机器可验证，却没有
向 LLM 足够清楚地表达两层引用：

```text
J/C/interface handle --anchor--> realization instance --graph edge--> instance
```

Task08 只解决这一个通用 wire 闭合问题。它不能借机增加 kernel 规则、确定性修复器或新的语义层。

## 最终目标

建立一个简洁、单义、双向一致的 π graph contract，使 source-aware Lift 直接生成合法 graph，
source-blind Replay 原样消费：

- J/C/interface refs 只作为 realization event 的匿名 anchor；
- graph nodes 只有一套 declaration-order instance IDs；
- route/order/sharing 的每个 foreign key 都闭合到已声明 graph node；
- 本地 validator 只拒绝并精确报告错误，不猜映射、不补节点、不触发自动 retry；
- 冻结新 protocol 后，以一次 fresh Swiglu canary 重新到达 Replay/Gate1 边界。

## 架构不变量

- Canonical Intent 仍只有 `J(A+B) + pure C`；π 在 Intent 外，interface 只提供匿名 callable
  declarations，harness 只持有真实调用值。
- Lift 是唯一 source-aware proposal；Replay 看不到源码、location map、privileged binding、source
  TTIR/fingerprint、diff、历史 candidate 或失败诊断。
- 不引入 composer、RuleCard/Registry、IntentProgram、LogicalABI、per-kernel prompt、AST/regex
  source-to-π extraction 或 response repair。
- π 不允许 operator、predicate、fill、constant、formula、dtype、shape、frontend API、source spelling
  或自然语言策略；本任务不得增加 `address_base/address_lane` 等 capacity。
- Gate1 继续比较 strict normalized compiler topology，不弱化 dataflow/address/predicate/effect。
- normal Replay 不闭合就停止；只有 normal strict hard-close 后才允许一次 masked-J。
- v4 request/response/receipt 是只读历史；任何 prompt/schema/code 变化都必须使用新 epoch。

## 阶段与提交节奏

### 0. 冻结反例并确定唯一 graph grammar（0 次 LLM）

- 以 v4 response 的 17 instances、30 routes、9 个 invalid route sources 为固定反例；
- 审计 schema description、Lift instructions、Replay instructions、研究文档和 parser，删除
  `p* -> access` 这类混淆 raw anchor 与 graph edge 的简写；
- 在“不改 capacity”的前提下，决定保留现有字段并强化类型，或做一次纯命名性 wire rename；
  选择必须使 graph node namespace 只有一个真值，不能同时接受两种写法；
- 写出短小的 source-neutral 正例和 namespace-confusion 负例，不包含 Swiglu/LayerNorm 特例。

完成后提交。

### 1. Schema / prompt / parser 双向闭合（0 次 LLM）

- schema 与 parser 对 instance declaration order、dangling endpoint、duplicate ID、cycle、slot、
  repeated anchor 的接受域一致；
- error 指明关系路径、非法 endpoint、所需 `i*` namespace；不得列出或建议一个猜测修复结果；
- Lift prompt 明确“anchor 不等于 graph node，edge 不隐式声明 node”；Replay prompt使用同一语法；
- provider `strict=false` 的事实必须显式记录，本地 parser 始终是权威边界；不得用自动 retry
  掩盖可生成性问题。

完成后提交。

### 2. Conformance 与 runner 状态机（0 次 LLM）

- 增加 raw `p*/c*/n*` endpoint、dangling `i*`、duplicate ID、order/sharing foreign key、cycle 的
  negative tests，以及 repeated anchor/multi-stage/share/round-trip positive tests；
- 模拟当前 v4 invalid Lift，证明 receipt 只有一次 failed Lift，Replay/Gate1/masked-J 均为 0；
- 新 epoch lineage 只接受冻结 v4 receipt SHA、manifest identity、一次 failed Lift、0 Replay/Gate1；
- 全量回归通过，v1--v4 evidence 一个字节不改。

完成后提交。

### 3. 新 epoch 单一 Canary

- 冻结完整自包含 protocol-v5 manifest/wire/preflight，先证明 28/28 ready 且在线调用为 0；
- fresh Swiglu 最多 1 Lift；合法 Lift 后最多 1 normal Replay；normal strict hard-close 后最多
  1 masked-J；任一失败立即停止；
- 若 Lift 仍违反 graph contract，冻结证据且不重试；若合法 Lift 后 Gate1 `no_closure`，才审计
  是否达到 π capacity 升级门槛；若 normal 闭合且 masked-J 不闭合，Task08 成功。

完成后提交、归档 Task08，并创建后续任务。

## 验收标准

1. π anchor 与 graph-node foreign key 只有一种合法、可解释写法；
2. schema、prompt、parser、fixtures、Replay packet 对该写法一致；
3. validator 精确 fail closed，但不存在 deterministic repair/composer；
4. v4 反例被离线拒绝在同一路径，正例可 round-trip 且保持 chain/share；
5. runner 对 invalid Lift 保证一次调用、0 下游动作，新 epoch lineage 绑定完整；
6. 历史 evidence 不修改，新 epoch 调用前 28/28 preflight ready；
7. 唯一 fresh canary provenance 完整，并根据实际边界决定后续任务，不能预写成功。

## 当前第一步

只做阶段 0，不改实现、不调用 LLM。先从 v4 frozen response 精确枚举所有 endpoint namespace，
再把 schema、prompt、parser 和文档压缩成一份唯一 graph grammar；在这份语法通过正反例审计前，
不得冻结 v5，也不得讨论新增 address mechanism。
