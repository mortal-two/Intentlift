# IntentIR：J 语义物化与 π 约束的 Replay Compiler Agent

> 日期：2026-07-31
> 状态：Task 18 的活动方法设计；取代 Task 16/17 的 fragment、hole、binding 与 fixed structural
> renderer 路线。本文不增加 artifact、认证 Gate 或 production backend 输入。

## 1. 裁决

Replay 不再要求 LLM 从复杂 J JSON 中重新恢复一份 semantic program，也不再让宿主搭建目标程序
骨架后要求 LLM 填洞。采用两层职责：

```text
exact J
  -> deterministic SemanticMaterializer
  -> read-only parameterized semantic program S_J

S_J + exact π + MaskC(C) + U_f
  -> one fresh source-blind Replay Compiler Agent
  -> one complete target source R_f
```

`S_J` 是编译器输入 IR，不是待补全的 Triton/CUDA/TileLang 模板。Replay agent 是完整编译器，
可以全局组织目标程序；它不是 per-node binding 作者，也不接受 fixed route assembler 的施工结果。

## 2. 数学接口不变

外部仍只有：

```text
Lift_H(Source) = (J,C,π)
Replay = W_f(J,π,MaskC(C);U_f)

SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1(Source(C), Replay(C))
  and Gate2(Source(C), Replay(C), RefEval(J))
```

内部确定性派生：

```text
S_J = M(J)
R_f = A_f(J,S_J,π,MaskC(C);U_f)
```

因为 `S_J` 是 J 的确定性函数，它不是第四输入、artifact、证书字段、sidecar 或 production policy。
exact J 仍挂载为权威回查；Agent 的主要编译视图是 `S_J`。

## 3. SemanticMaterializer 的边界

`M` 直接消费 canonical J，确定性完成：

- J canonicalization 与 dependent type/shape elaboration；
- operator、operand、result 与 declaration order；
- structured control、reduction、recurrence 与 logical carrier；
- domain、predicate、fill、access、effect 与 output binding；
- closed result-relation class；
- 每项事实到 `n*/t*/x*/d*/a*/e*` 等 J handle 的可追溯锚点。

`S_J` 必须：

- 保留 symbolic extent，不读取 Harness concrete valuation；
- 保留 J-owned control/node 边界，不因 production specialization 删除 π 所引用的锚点；
- 不读取 C、π、Source、Replay、Gate、kernel identity、target capability 或历史结果；
- 不包含 Triton/CUDA/TileLang API、target recipe 或性能选择；
- 作为只读文件挂载，Replay agent 不能修改。

文件形态分为同一值的两种 view：`semantic-program.json` 保留 exact structured form；
`semantic-program.intent` 把它确定性排成紧凑 typed SSA，直接显示
`result = op(operands)`、lexical nesting，以及 loop 的 trip/scan/carry(init/state/next/result)/yield。
这不是 target lowering；目的只是阻止 Agent 看懂算法名称后用熟悉实现覆盖 J 的真实 operand edge。

第一版直接迁入后端分支已经成熟的 canonical-J、typed-boundary、contract/predicate materialization
逻辑，放在 Replay 内部作为自包含 core；暂不建设跨仓库共享 API。不得原样复制需要
`concrete_extents`、constexpr specialization、`Attach_C` 或 backend search state 的 production 入口。

## 4. Replay Compiler Agent 的职责

每个 formal Gate1 round 启动一个 fresh、source-blind、长生命周期 coding agent。它直接阅读完整
`S_J` 与 π，并负责：

- instruction selection 与目标 API；
- program/thread/block 映射；
- address/pointer 表达；
- layout、memory space、temporary materialization；
- loop/carrier/control/synchronization 的前端实现；
- fusion、分块、代码重排和为目标编译器限制进行的整体重构；
- 编写完整 Triton、CUDA、TileLang 或其他前端源码；
- 编译自己的源码、查看自己的 compiler IR、局部修改、checkpoint 与提交。

Agent 输出的是 `S_J` 的编译结果，不是对 `S_J` 文件的修改。最终目标源码不要求逐行保留 semantic
IR，也不要求每个 J node 对应一条 target statement。

## 5. π 的角色与消费方式

π 仍是 typed anonymous realization witness，不是 target plan。Agent 直接消费它并满足：

- occurrence 与 multiplicity；
- physical region/control；
- program axis 与 linearization；
- physical iteration 与 loop carrier；
- route、order 与 sharing；
- materialization、terminal control 与 representation refinement。

宿主可以提供 exact、bounded、可回指原字段的查询：

- 按 region/instance/route 查询；
- 将 `semantic_ref` 定位到只读 `S_J`；
- 查看局部前驱、后继、order、sharing 与 carrier 关系；
- 对 flat/nested π wire 做严格无损编解码。

`pi.intent` 采用无损 compiler view：先按 region 列出 occurrence，在每条 exact relation 旁解析其
opaque instance/region handle，并把 occurrence 已引用的 exact local `S_J` anchor 就地并置。loop anchor
保留 trip/scan/carry/yield，access anchor 保留 tensor type/index order；这只是把原本允许的
`semantic_ref -> S_J` 查询文件化，不增加事实，也不把 relation 翻译成 target statement。

还必须区分 π 与 Harness：`program_axes` 约束 device 中读取哪一个 program coordinate；launch-grid
cardinality 始终是隐藏 Harness state，不进入 π。`route_extent_transforms` 则按权威定义只表示某条
existing route slot 上的 target-local physical extent partition，Replay 必须消费；不得把 host grid
关系误作者化为该字段。若 Source device body 没有对应 floor/ceil-div，而 Lift candidate 声明了它，
TypeCheck 只能证明依赖类型合法，Gate1 必须拒绝并反馈 persistent Lift owner fresh relift，不能放宽
ReplayCheck 来追认错误 π。

宿主不得：

- 从 π 生成 target skeleton、loop、route source 或 API；
- 建立 `π relation -> target statement` 映射；
- 自动拼接 Agent fragments；
- 根据 π 补造 J 缺失的 predicate、formula、effect 或 semantic constant。

## 6. Replay 的自由不是 production 搜索自由

Production backend 寻找任意 J-preserving 高性能实现，并且不读取 π。Replay 的目标不同：它必须重建
π 所见证的 Source realization。因此：

```text
R_f ∈ Realize_f(S_J, π; U_f)
```

而不是任意：

```text
R_f ∈ AnyImplementation_f(J)
```

Agent 可以自由选择满足 π 的目标拼写和工程结构，但不能忽略 π 的 occurrence、time/control、route、
sharing 或 materialization。若采用 online recurrence 等算法 realization，其算术必须从 J/S_J 推导；
π 只能约束 recurrence topology，不能成为隐藏公式来源。

## 7. “Agent 不得改变 J 语义”的真实保证

只读 `S_J` 保证语义权威不被 Agent 原地重写，但不能静态证明任意 Agent 源码正确实现了 J；做到后者
等价于再建设完整 verified compiler。当前 SCRC 的边界仍是生成后认证：

- ReplayCheck 只检查接口、构造和可编译性，不把某种 target AST 拼写当作 π membership；
- Gate1 检查 Source 与 Replay 的 realization observation；
- Gate2 独立比较 Source–Replay、Source–J、Replay–J。

因此宿主不通过 binding/hole 假装提供静态语义证明，Agent 也不能把自身 compile PASS 写成闭合。

## 8. MaskJ、C 与无泄漏

正常轮：

```text
S_J = M(J)
```

MaskJ 消融必须从同一 masked view fresh 派生：

```text
S_mask = M(MaskJ(J))
```

若 masked J 已不足以 materialize 完整 semantic program，则不提供完整 `S_J`。禁止先由完整 J 生成
`S_J` 再只遮盖 JSON J，否则 MaskJ 会被 deterministic view 泄漏。

SemanticMaterializer 不读取 C。Replay 只获得 `MaskC(C)` 的匿名 slot/调用边界，真实 C 按原协议在
Replay 后确定性绑定。production backend 继续只消费 J+C，不读取 π、`S_J` 文件或 Replay 状态。

## 9. Agent 工作区

只读输入建议保持为少量文件：

```text
input/j.json                 exact authoritative J
input/semantic-program.*     deterministic read-only S_J（exact JSON + typed SSA text）
input/pi.json                exact π 或严格无损 wire
input/pi.intent              同一 π 的逐关系、声明序、无损只读文本视图
input/mask-c.json
input/frontend.md            U_f
```

可写区只需要：

```text
work/                        Agent 自己的笔记/helper/checkpoint
output/replay.source         唯一正式输出
```

Agent 首次完整读取 `semantic-program.intent` 与 `pi.intent`；后者把 region/instance/route/order/
sharing/carrier 等原字段交叉索引，并置其 exact local S_J anchor，但不建立 relation-to-target 映射。允许工具：普通文件编辑、
exact J/S_J/π 查询、compile、own-IR/own-TTIR、checkpoint/rollback 和原子 submit。construction
PASS 后应查看自己的 compiler IR，删除没有 J/π 授权的 address/predicate/control/arithmetic，并把
inactive constexpr 路径的计算留在其分支内。资源以 episode wall-time 与 formal Gate1 round 管理，
不用极小工具调用次数打断 coding loop。

禁止恢复：

- `replay.binding.json`；
- per-node hole 或 per-occurrence target binding；
- fixed region/block scaffold；
- route-bound fragment renderer；
- target RuleCard、Kernel helper 或历史 winner；
- Gate1 diff、Source、Harness value 或旧 Replay 回流。

## 10. 失败归因与生命周期

每次 Replay episode 内，Agent 可以反复编译和修正自己的目标源码。提交后才进入正式 Gate1：

- `S_J` 缺少 J 已声明的语义：修 SemanticMaterializer；
- J 本身缺少 Source 语义：反馈给同一 persistent Lift owner，fresh relift；
- π 缺少必要 realization witness：反馈 Lift owner，fresh `(J,C,π)`；
- Replay 未正确消费完整 `S_J+π`：启动新的 fresh Replay，不修改正确 J；
- Gate1 已过而 Gate2 失败：先判断 J 错误还是 RefEval/Gate2 消费缺口，不无条件 relift。

Replay agent 不看 Gate1 diff，也不跨 formal round继承源码或 helper。Lift owner 可以消费 Gate1
source-aware 反馈并修改 artifact；随后必须启动 fresh source-blind Replay。

## 11. 多前端与后端关系

同一个 `S_J` 可以被不同 Replay Compiler Agent 编译为 Triton、CUDA、TileLang 等前端源码；差异由
π、`U_f` 与 Agent 处理，不为每个前端建设完整 deterministic J→source lowerer。

Production 路径仍是：

```text
S_J -- instantiate with declared valuation ν --> M0
M0 + C -- Attach_C --> backend search state
```

Replay 路径不读取 ν、C value、target search 或 production π consumer。两条路径共享 J 的语义解释，
但职责和允许输入保持分离。

## 12. 迁移与验收

Task 16/17 的 fragment、hole、binding、compact block 与 fixed structural renderer 全部退出默认路径。
只保留以下通用能力：

- exact packet/S_J/π 查询；
- strict source-blind sandbox；
- persistent single-owner coding session；
- own compile/own IR；
- checkpoint/rollback 与 exact submit；
- nested π wire 的严格无损编解码。

第一项真实验收使用 FlashAttention forward：它已有可编译但 dataflow 错误明显的旧 Replay frontier，
能直接检验 `S_J` 是否减少语义恢复负担、Agent 是否真正消费 π。随后闭合 Llama4 RoPE、SplitK
forward、FlashAttention forward/full backward，再对原已闭合项 fresh Replay。

最终原则：

> **固定代码只把 closed、typed J 物化为只读参数化语义程序；fresh Replay Compiler Agent 在 π
> 约束下承担完整目标前端编译；宿主不搭目标骨架、不填洞、不拼 Agent 源码。**
