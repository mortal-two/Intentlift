# IntentIR：J 主导的最小约束 π 与 Replay 内部 Composer 重构设计

> 日期：2026-07-31  
> 状态：Task 16 的实现依据；是现有四份 SCRC 权威文档对 Replay 作者化接口的窄化修订，不增加证明对象、认证 Gate 或 production backend 输入。

## 1. 要解决的真实矛盾

当前复杂 Kernel 的主要瓶颈不是 Codex 不能理解 Source，而是 Replay agent 被迫从一份巨大、扁平、交叉引用密集的 π JSON 直接手写完整目标源码。这样同时产生三类浪费：

1. π 重复携带 J 已经声明的语义，既扩大上下文，又形成语义泄漏风险；
2. Replay agent 要手工维护 statement、block、loop、carrier、route 和目标语法，机械错误掩盖语义能力；
3. host 侧继续增加 ledger、RuleCard 或固定 recipe，会逐步退化成隐藏的 deterministic composer，违反 LLM-first。

重构目标不是让固定代码替模型 lower，而是让一个 fresh、source-blind Replay coding agent 在同一 episode 内：

```text
读取 J + π_min + MaskC(C) + U_f
  -> 自己建立 proof-irrelevant composer scratch
  -> 分步生成、编译、检查并修正 Replay source
  -> 只提交最终 Replay source
```

## 2. 不变的数学边界

外部接口和认证谓词保持不变：

```text
Lift_H(Source) = (J,C,π)
Replay = W_f(J,π,MaskC(C);U_f)

SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1(Source(C), Replay(C))
  and Gate2(Source(C), Replay(C), RefEval(J))
```

唯一语义权威是 J。C 只承载可重调/实例绑定信息；π 的类型仍是：

```text
Γ_(J,C) ⊢ π : RealizationWitness(J,C)
SemAtoms(π) = ∅
```

composer scratch 记作 `K`，只用于说明 Replay agent 的内部行为：

```text
K := AgentCompose_f(J,π,MaskC(C);U_f)
Replay := Render_f(K)
```

`K` 不是 artifact、证书字段、第四个 IR 或 production 输入；episode 结束即销毁。`Render_f` 只做目标语法序列化，不根据 J/π 选择 operator、predicate、formula、effect 或算法。

## 3. π_min 到底保留什么

π 不是目标程序，也不是 lowering plan。它只保留无法由 J、C 和全局固定表示约定唯一恢复、但重建 Source realization 又必需的关系：

- 一个 J 语义实例在 Source realization 中的 occurrence/multiplicity；
- logical instance、physical region、iteration/carrier、route 之间的拓扑关系；
- Source 的物理循环、并行轴、terminal control、materialization、sharing/order 等 realization 事实；
- representation-only 参数或 C slot 的 opaque attachment；
- packing/layout/cast 等实现见证，但只能通过 typed opaque refs 与 relation kind 表达。

loop carrier 的 `init/state/next/result` 是四个 physical occurrence，而不是四个可退化为同一 handle 的
标签。它们即使引用同一 J semantic value，也必须以 ordinary routes 显式形成
`init -> state -> next`；carrier relation 自身再给出 next-to-state 与 loop-exit 时间语义。这样 π
只增加 occurrence/time topology，不复制 update formula，Replay 也不能自行更换 recurrence。

同一 physical program axis 可以 realization 多个 J logical consumer domain。每个 d* 在
`program_axes` 中仍恰有一个 binding，但不同 d* 可以共享 axis；对应 consumer occurrences 放在
独立 branch region，并由 program-region d* instance 的 empty-slot route 控制。J domain 是唯一条件，
π 只保存共享 axis 与 branch topology，不保存比较式或 extent。

若 Source 恰在某个 authored J program domain 的补集上终止当前 invocation，则继续使用同一条
domain-to-branch empty route，并以 closed
`terminal_controls{domain_ref,continuation_region}` 声明“域内进入该 continuation、域外终止”。它不
复制 domain predicate、polarity、return 或 Source/API，也不要求 Lift 发明假的 `effect_if`。若终止
由普通 scalar predicate 控制，则仍使用 J `effect_if` 两臂 instance 的 terminal/continuation 形式。
两种 row 都只增加 control topology；同一个 domain-controlled region 在 Replay 最低结构检查中只能
计作 terminal fork，不能再重复计作普通 branch。

当 Source 的一个 program ordinal 还要经过 mixed-radix 解码时，仅靠 `program_axes` 与 route 不能
唯一重建 realization。π_min 因而允许 closed
`program_linearizations{domain_ref,dimension_refs}`：按 outer-to-inner 的 J `x*` 顺序，唯一表示
row-major quotient/remainder projection。它只引用 J 已作者化的 opaque dimensions，不携带公式
文本、具体 extent、grid、predicate、Source/API 或 target spelling；删除 J 后不能恢复计算语义。
`program_id` 及其物理解码树不进入 J，J 只描述解码后的逻辑坐标。

相反，任何 block interval、comparison、mask 或 select 若会改变 logical value，就必须留在 J 的
typed graph/domain 中。π 不得另写 partition relation；否则 Mask-J 后仍可能从 π 恢复这段语义。

π 禁止携带：

- J operator、计算公式、常量填充值、predicate 表达式或 reduction 语义；
- Source 片段、符号名、frontend API、target API 或目标源码；
- C 的具体值、Harness valuation、Gate diff、旧 Replay 或 kernel 名称；
- `emit_flash_attention`、`lower_contraction` 一类可独立恢复语义的 recipe/template 标签。

下列冗余允许从 wire 中省略，但只能由固定 Decode 机械恢复，不能由 composer 猜测：

- 已声明 graph 中唯一的 1:1 identity route；
- 唯一拓扑顺序；
- 与 typed owner 唯一一致的默认 region/instance binding；
- 可由显式端点类型唯一决定的 handle kind。

凡存在 multiplicity、分支、多个候选 endpoint 或非 identity realization，就必须显式写入 π，fail closed。压缩 π 是 lossless canonical encoding，不是删除困难语义。

## 4. Agent 内部 Composer 的允许边界

Replay agent 可以在自己的工作区建立和修改一个小型结构化构造文件，例如：

```text
work/replay.source
work/replay.compose.json      # 可选
work/composer/*.source       # 可选
```

小型 packet 可以像普通 Codex 一样直接维护一份完整源码。交叉引用密集的 packet 则使用 grouped
route composer：按 lexical region/block 保持少量 fragment，而不是“每个 π occurrence 一个文件”。
两条路径都在同一 episode 中局部 patch、编译和修正；选择编辑粒度不改变 Replay 输入或认证谓词。

host 只提供通用、syntax-only 的能力：

- 创建/移动/删除 block、statement 和 expression；
- 生成稳定的局部 ID，维护树结构与引用；
- `if/for/while/call/assign/return/raw` 等目标语法节点；
- render、format、compile、运行正式 Replay check、查看自身 TTIR/编译诊断；
- checkpoint、局部修改和原子 submit。

公开 `U_f` 只保留短 ABI 约束，不再携带按 relation/op 罗列的 target spelling RuleCard。relation 的
含义来自版本化 schema/J/π 自身，目标写法由 agent 选择；编译和 construction diagnostics 只拒绝。

对交叉引用密集的 Replay，通用 editor 允许 agent 在任意 region/block fragment 中作者化
`{{out:iK}}`、`{{slot:iK:N}}` 与 `{{dep:iK:N}}` 占位符；一个 fragment 可以含多个 occurrence。
`slot` 对应目标 iK 的显式 operand slot，`dep` 只按该 target 的 empty-slot route 声明顺序编号。固定
renderer 只能沿这些显式、唯一的 route 代入稳定 local ID，并核对 sharing/order。fragment 内调用
什么 API、写什么表达式、建立何种 control 仍完全由 agent 根据 J 选择。缺 route use、重复 slot、
producer 晚于 consumer、未定义 instance 或 order 冲突均 fail closed；renderer 不重排，也不从
semantic_ref 生成任何 target statement。

host 不得提供：

- `J op -> target API` 映射表；
- `π relation -> code fragment` 规则；
- per-kernel helper、模板库或历史 winner；
- 从 Source、Gate1/Gate2、Harness 或 kernel identity 获得的提示；
- 自动补写 semantic node、predicate、effect、route 或 control。

为了避免 agent 在长 JSON 中漏掉 J 节点，工具可以提供 declaration-order 的 exact requirements view：
每行只是当前 J 节点的 `id/op/operands/attrs/result`，以及当前 π 的 time/control/route relation。它是
同一 packet 的有界重索引，不是 lowering plan；不得给出 target statement、API、代码片段或推荐。
fixed host 也不依据该 view 宣告节点已经被正确实现，真正的语义/realization 闭合仍由 fresh Gate1
裁决。

Replay agent 可以在本次 episode 内自己抽象局部 helper；这等价于 Codex 自己写程序，不是共享模板。helper 不得跨 formal Replay round 保留。

## 5. 为什么它仍然由 J 主导

Replay agent 同时看见 J 与 π，但二者权力不同：

- J 回答“算什么”；
- π 回答“这些 J 实例在 Source 中怎样发生和连接”；
- composer 回答“本轮 agent 怎样组织自己的目标源码编辑过程”。

如果 J 改为 `J'`，旧 `(J,C,π)` 的 dependent TypeCheck 不再自动成立；必须得到匹配的新 `(J',C',π')`，并启动 fresh Replay episode。没有任何 composer checkpoint、旧源码或历史 helper 可以继承。

J 权威由三条 Gate2 直接比较与独立消融观测：

```text
Source -- Close_Q -- Replay
Source -- Close_Q -- RefEval(J)
Replay -- Close_Q -- RefEval(J)
```

Mask-J 只回答 π 是否泄漏到足以独立恢复语义，不替代正式证书。推荐的补充检查是：保持 π/C/RealSig 不变，改变 J 的一个闭合语义事实后，fresh Replay 必须跟随新 J；若 Replay 仍复现旧行为，则拒绝该 π/composer 边界。

## 6. Replay episode 生命周期

每个 formal Gate1 round 恰有一个 fresh Replay episode：

1. runner 创建空工作区，只挂载 `J、π、MaskC(C)、U_f` 与短工具说明；
2. agent 按需读取文件，不把整个 artifact/手册重复嵌入 prompt；
3. agent 自己建立 composer scratch，逐块 render Replay source；
4. agent可多次 compile/check/inspect-own-TTIR，并根据自身诊断局部修改；
5. agent只提交一个最终完整 Replay source；
6. runner在 agent 外重新编译，执行正式 Gate1；失败 feedback 只回到同一个 Lift owner 做 relift，不续用 Replay episode；
7. workspace 和 composer scratch 随 episode 销毁。

不以工具调用次数硬切断正常 coding loop。资源边界使用 wall-time、上下文窗口和 formal Gate1 round 上限；统计内部 tool/compile 次数，但不把它们误记为 relift 或 Gate1 round。

Replay 启动只给文件路径和字节数；agent 按需读取 packet。若某个前端桥接器额外提供 attention
projection，它只能是已认证 packet 的有界、exact 行式视图，并必须有精确续查路径；不能变成
target recipe，也不能把整份 artifact/RuleCard 重复塞入 prompt。

wall-time 截止也不应丢弃已完成的 LLM 输出：只有当截止点的 exact Replay bytes 已被固定 ingress 与 synthetic compile 双检查接受时，host 才可将其作为本 episode 的最终输出送往 Gate1；这只消除未来得及调用 submit 的协议失败，不修复任何字节或源码。

## 7. 实现与迁移

### 7.1 先删除旧负担

- 删除强制 `AUTHORING_INDEX.json` 大 ledger；
- 删除第二个可写 reviewer/semantic self-review 阶段；
- prompt 只保留权威边界、输入路径、工具入口和提交条件；
- 不再把 π 全量翻译成另一份自然语言教程。

### 7.2 再建立内部 composer

- 默认让 Codex 直接维护完整 source，提供 exact-text patch、checkpoint/rollback、compile/check 与
  own-TTIR inspection；只有真实交叉引用需要时才使用 route-bound fragment；
- composer state 保存在 episode workspace，runner不持有语义状态；
- compile/check/own-TTIR 读取当前 render 结果；
- raw escape 保留开放世界能力，避免固定 AST 覆盖不了新 frontend 语法。

### 7.3 最后压缩 π wire

- accepted artifact 中的 dependent π 不改对象、不改证书身份；runner 只在 Replay packet 中把同一
  relation list 编成带显式 columns 的 `π_min` 行式 wire；
- 编码前后必须精确 round-trip 相等，schema 增加或字段不匹配立即 fail closed；
- TypeCheck、CandidateCheck 与正式 artifact 继续消费 canonical π，Replay construction check 消费
  round-trip 后的同一 canonical relation；
- 因此既有 accepted `(J,C,π)` 无需 relift。只有真实 Kernel 证明 π 缺少不可恢复的 realization 事实
  时才升级 artifact schema，不能为了压缩 token 修改证明对象。

## 8. 真实验收，不写 mock tests

实现只通过真实 SCRC closure 验收：

1. 先用剩余复杂 Kernel 覆盖 recurrence、program-id/address chain、terminal CFG、多输出和 conditional post-state；
2. 每项都要求 current-version TypeCheck、fresh Replay、Gate1 和 Gate2 三边；
3. 记录 Replay packet bytes、time-to-first-compilable、agent tool/compile 次数、Gate1 frontier 与总 wall-time；
4. 至少执行一次 Mask-J/J-authority 消融，确认 π/composer 没有独立语义通道；
5. 剩余项若明显闭合，再让原先 22 项全部经过新 Replay composer，删除只对个别复杂 Kernel 有效的接口假设。

最终原则：

> **J 决定语义，π 只约束 realization，Codex 在一次 source-blind episode 内自己写 composer 和 Replay；固定代码只提供通用语法编辑、编译、检查与隔离，不替模型 lower。**
