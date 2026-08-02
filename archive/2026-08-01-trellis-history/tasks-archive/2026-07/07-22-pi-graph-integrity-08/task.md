# Pi Graph Integrity 08：闭合 realization 外键合同并恢复单一 Canary

> 状态：已结束（π graph 外键合同已闭合；v5 合法 Replay 暴露 address base/lane 表达缺口）
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow
> 归档原因：Task08 已让 fresh Lift artifact 通过 parser 并进入 Replay/Gate1；剩余失败满足预先
> 冻结的最小 capacity escalation gate，转入 Task09，禁止在 v5 内重试或追加 prompt patch

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

阶段 0 已完成，详见 `research/stage0-graph-namespace-audit.md`。审计确认保留现有 JSON shape：
raw J/C/interface handle 只作为 `semantic_ref` anchor，所有 graph endpoint 只接受 declared `i*`。
v4 除 9 个 invalid route sources 外还有 1 个 invalid sharing source；这仍是可生成合同歧义，
不是 π capacity 证据。

阶段 1 也已完成：

- `render_realization_witness_wire_grammar()` 成为 schema description、Lift instructions 与 Replay
  instructions 的唯一 graph-reference 文本来源；
- instance ID、anchor、route/order/sharing endpoint 均有 role-specific schema description；
- parser 现在区分 raw anchor namespace 与合法但 dangling 的 `i*`，错误精确到数组下标和字段；
- unknown `semantic_ref` 同样精确指向对应 instance，不 repair、不猜 anchor-to-instance mapping；
- raw `p*/c*/n*`、dangling `i99`、sharing consumer 与共享 grammar 投影测试全部通过；定向回归
  `120 passed`，在线 LLM 调用新增为 0。

下一步进入阶段 2：补真实 v4 frozen response、runner invalid-Lift 状态机和完整关系 mutation
conformance，然后运行全量回归；在这些证据闭合前不得冻结 v5。

## 阶段 2 与 v5 调用合同记录（2026-07-22，0 次 LLM）

- conformance 现在覆盖 raw anchor/graph-ID 双向混淆、dangling `i*`、duplicate declaration、
  route/order/sharing 所有 endpoint、order cycle、repeated anchor、round-trip 与输入不变性；真实
  v4 response 离线精确拒绝在 `pi.routes[0].source='c1'`，并核对 9 个 invalid route sources 与
  1 个 invalid sharing source。
- formal runner 离线模拟 invalid Lift：ledger 只有一次 failed Lift，artifact/Gate1 不存在，
  normal Replay 与 masked-J 均为 0；冻结 v4 receipt 的 call/hash/token/lineage 也有 committed test。
- 第一轮全量回归得到 `769 passed, 4 skipped, 2 failed`；两个失败都只因旧测试仍把已冻结 v4
  wire/preflight 当 current。没有功能测试失败，因此没有刷新 v4，而是切换新 epoch。
- protocol-v5 manifest 完整复制 v4 的 28 cases、repositories、call budget 与除 Lift 外的所有
  protocol；语义/调用 diff 为 0。仅 schema/batch ID/Lift wire identity 改变：
  - manifest digest：`bf6716ba75c0087a506031101a5628f30c06c29d08dfdaf60ef5a0bfcea5ceeb`
  - manifest file SHA：`98056f9035d839e5d51f8b209accf0806606b7fbab2bca283a302c50265716cf`
  - Lift wire：`intentir.lift-wire.v5`
- v5 Lift wire 首次把共享 grammar 所在的 `verify/gate1/pi.py` 纳入 request-path hash；v4 identity
  保留为历史常量，current verifier 明确拒绝 v4 冒充 v5。
- v5 recovery 绑定 exact v4 receipt SHA，同时验证 v4 manifest/preflight、一次 failed Lift、
  invalid response SHA、0 artifact/Gate1、v4→v3 lineage、source/provider identity；不只按路径或
  `status` 软匹配。定向回归 `161 passed`，在线 LLM 调用新增为 0。

下一步运行切换 v5 后的完整全量回归。只有全绿并提交后，才生成 0-LLM v5 preflight；任何
preflight identity 覆盖的代码若再改变，必须废弃并重新生成，不能直接开始 canary。

v5 epoch 切换后的全量回归已通过：`782 passed, 4 skipped`，60 个既有 warning，耗时
354.15 秒。此前两个“v4 仍为 current”的失败已由显式 historical-v4/current-v5 断言关闭，
没有刷新或重写任何 v4 evidence。

下一步只生成并复验 v5 formal preflight；该命令必须保持 `online_llm_calls=0`，并证明 26/26
Triton、2/2 CuTe、28/28 source observer/replay arm ready。preflight 提交后冻结受保护代码，
才允许唯一一次 Swiglu v5 canary。

v5 formal preflight 已生成并复验通过：26/26 Triton、2/2 CuTe、28/28 source pins、symbol pins、
source observer 与 Replay arm 全部 ready，`online_llm_calls=0`，overall ready。receipt SHA 为
`f0c93e6230914a9f750173ef6853315b9a1df9c212c4592e0a258a6f48222b6e`，详细身份见
`research/v5-preflight.md`。

下一步提交 preflight evidence，确认 worktree clean 与 v5 output absent，然后只运行一次绑定
v4 `lift_failed` receipt 的 Swiglu `protocol_v5_recovery` canary。

## 阶段 3 唯一 Canary 结果（2026-07-22）

- v5 Lift 一次调用后被完整 schema/parser 接受；artifact 保留二维 J、non-C `p4`、pure C，且
  π graph 所有 foreign keys 闭合。
- source-blind normal Replay 一次生成并真实编译；它使用了 p3、logical extent 与 shared offset，
  证明 v4 的 graph namespace blocker 已关闭。
- strict Gate1 为 `no_closure`，score `0.6407238924050632`；interface、reduction、predicate/effect、
  schedule-only 通过，op multiset/dataflow 不通过。normal 未闭合，masked-J 为 0。
- 合法 π 已包含 p3/program instance 到三路 access realization 的 fanout、sharing 与
  realization→transfer chain；Replay 仍把 program-uniform pointer base 与 lane-varying address
  合并成 vector offset。
- 当前通用 `realize` tag 无法区分 address base/lane/flat 三种 source topology，故 Task07 冻结的
  capacity escalation gate 已满足。精确 TTIR、hash、usage 与差异见
  `research/v5-gate1-address-audit.md`。

Task08 到此完成其目标：π graph 外键可生成合同已闭合，并把系统推进到一个合法 artifact 的
strict Gate1 address-realization 反例。后续不能在 v5 重试；应归档 Task08，并创建只增加
`address_base/address_lane` 两个 source-neutral mechanism tags 的 Task09。
