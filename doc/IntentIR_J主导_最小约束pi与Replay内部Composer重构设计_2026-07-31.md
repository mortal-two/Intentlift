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
work/replay.compose.json
work/replay_source.py
```

host 只提供通用、syntax-only 的能力：

- 创建/移动/删除 block、statement 和 expression；
- 生成稳定的局部 ID，维护树结构与引用；
- `if/for/while/call/assign/return/raw` 等目标语法节点；
- render、format、compile、运行正式 Replay check、查看自身 TTIR/编译诊断；
- checkpoint、局部修改和原子 submit。

对交叉引用密集的 Replay，通用 editor 允许 agent 在每个 occurrence fragment 中作者化
`{{out}}`、`{{slot:N}}` 与 `{{dep:N}}` 占位符；`slot` 对应 π 显式 operand slot，`dep` 只按同一
target 的 empty-slot route 声明顺序编号。固定 renderer 只能沿这些已显式、唯一的 route 代入稳定
local ID，并核对 sharing/order。fragment 内调用什么 API、写什么表达式、建立何种 control 仍完全由
agent 根据 J 选择。缺 fragment、同一 slot 多 source、未定义 source、region/order 冲突均 fail
closed；renderer 不从 semantic_ref 生成任何 target statement。

host 不得提供：

- `J op -> target API` 映射表；
- `π relation -> code fragment` 规则；
- per-kernel helper、模板库或历史 winner；
- 从 Source、Gate1/Gate2、Harness 或 kernel identity 获得的提示；
- 自动补写 semantic node、predicate、effect、route 或 control。

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

## 7. 实现与迁移

### 7.1 先删除旧负担

- 删除强制 `AUTHORING_INDEX.json` 大 ledger；
- 删除第二个可写 reviewer/semantic self-review 阶段；
- prompt 只保留权威边界、输入路径、工具入口和提交条件；
- 不再把 π 全量翻译成另一份自然语言教程。

### 7.2 再建立内部 composer

- 在现有 Codex Replay bridge 上增加通用 syntax-tree/editor 操作；
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
