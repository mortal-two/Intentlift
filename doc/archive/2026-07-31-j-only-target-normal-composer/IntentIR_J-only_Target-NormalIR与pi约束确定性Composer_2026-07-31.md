# IntentIR：J-only Target-Normal IR 与 π 约束的确定性 Composer

> 日期：2026-07-31  
> 状态：历史方案；Task 19 已归档并由 Task 20 的 J-only Replay / typed π-relative Gate1 observer 路线取代。本文只用于追溯，不再约束当前实现。

## 1. 裁决

Replay 内部改为三段，而外部认证对象仍只有 `(J,C,π)`：

```text
S_J = M(J)
F_f = A_f(J, S_J; U_f)
R_f = Compose_f(F_f, π, MaskC(C); U_f)
```

- `M` 是已有 J-only、参数化、只读的 SemanticMaterializer；
- `A_f` 是 fresh source-blind Replay Semantic Agent，只看 J/`S_J` 与公开 target capability `U_f`；
- `F_f` 是一个完整、闭合、target-friendly 的 Target-Normal IR 实例，不是模板、洞、源码片段或
  第四个认证 artifact；
- `Compose_f` 是固定、版本化、target-specific 的确定性 Composer。它读取 `F_f + π + MaskC(C)`，
  只做物理 realization、约束匹配与目标语法物化；
- `R_f` 才是进入外部 Replay ingress、Gate1 与 Gate2 的完整目标源码。

## 2. 为什么必须切开

Task 18 已让 Agent 读懂 J，但同一个上下文仍需同时完成：

- 维护 J 的 exact operand/value graph；
- 交叉引用 π 的 region/instance/route/carrier；
- 选择目标算法、地址、program mapping 与 target API；
- 手写并编译完整 Triton。

Flash-forward 的 `S_J` 已令 interface、op、reduction、effect 和 loop profile 闭合，最终仍因 carrier
operand 被 Agent 的熟悉算法先验覆盖而留下 112 个 dataflow 差异。这不是再加 π 文本或延长 Agent
时间能根治的问题；应把“J 的语义编译”和“π 的物理实现约束”分到不同权限边界。

## 3. `F_f` 的定义

`F_f` 是一份完整 Target-Normal IR 实例，必须包含：

- exact callable signature 的 J handle/provenance；
- 完整 semantic dataflow、structured control、recurrence 与 observable outputs；
- target-friendly、closed primitive vocabulary 下的 instruction selection；
- 每个语义 op/value 到 J node/value/access/domain 的 origin refs；
- Agent 选择的、仅由 J 语义允许的等价算法结构。

`F_f` 不得包含：

- π instance/region/route handle；
- C value 或匿名 C slot 的具体值；
- program-id 轴、pointer arithmetic、physical placement、cache/layout hint；
- Source、Harness、Gate、历史 Replay、Kernel identity 或 target source bytes；
- 等待 Composer 补语义的 hole、per-node source fragment 或多个候选模板。

因此 `F_f` 是一个实例，不是 recipe library：新 J 产生新的 fresh `F_f`；MaskJ 必须从 masked J
重新产生 `F_f^mask`，禁止复用 normal `F_f`。

## 4. π 的权限

π 继续是 target-independent、typed、anonymous realization constraint/witness，而不是可执行计划。
它只约束：

- occurrence multiplicity、region/control 与 placement；
- route、order、sharing 与 materialization；
- program axis/linearization；
- structured/physical iteration 与 carrier；
- terminal control 与 representation refinement。

π 不得选择 Target-Normal semantic op、提供公式/predicate/fill、携带 template ID，或告诉 Lift
某个 Composer 如何工作。Lift 仍只从 Source/Harness/compiler view 作者化 `(J,C,π)`，完全不看
`F_f` schema、target template library 或 Composer 实现。

现有完整 π schema 在第一纵切中可继续作为 exact constraint 输入；把它收缩为 compact π 是后续
真实 schema/数学修订，不能伪装成 view。只有经证明可由 closed typed endpoints 唯一恢复的重复
结构才可在 vNext wire 中省略；否则仍显式作者化。

## 5. Composer 的允许复杂度

Composer 可以比旧“最小 renderer”更完整，因为它是一个 target compiler component。允许：

- 解析/校验 `F_f` 的 closed Target-Normal vocabulary；
- 以 π 约束选择通用 physical template family；
- 生成 ABI、program/thread coordinate、地址、load/store、loop/carrier、control 与 target API；
- 使用 generic map/reduce/fold/scan/contraction/update/access templates；
- 根据 MaskC 的匿名 slot 做已声明 schedule/representation binding；
- 生成 source map 和完整 consumption receipt；
- 对不支持、歧义或不满足约束的组合 fail closed。

Composer 禁止：

- 按 Kernel 名、Source 路径或历史 winner 分支；
- 从 π 发明或替换 operator、operand、predicate、fill、constant 或 output semantics；
- 接受 per-kernel algorithm template；
- 让 Gate1 diff 成为内部 template membership oracle；
- 修改 `F_f`、J 或 π 来使源码通过。

删除 J/`F_f` 后，Composer 最多得到无法产生计算结果的 physical shell；删除 π 后，Composer 可以
得到逻辑 target-normal program，但不得声称重建了 Source realization。

## 6. 形式边界

外部接受谓词保持：

```text
SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1(Source(C), R_f(C))
  and Gate2(Source(C), R_f(C), RefEval(J))
```

但 Replay writer 的内部定义明确为：

```text
W_f(J,π,MaskC(C);U_f)
  := Compose_f(A_f(M(J);U_f), π, MaskC(C);U_f)
```

`F_f` 是 proof-irrelevant episode evidence，必须保存 exact bytes、Agent trajectory、parser/Composer
receipt，便于归因；它不是 TypeCheck 输入，也不进入 production backend。

Gate 的职责因此更清楚：

- Source–J 失败：J/Lift 或 RefEval 消费有问题；
- Replay–J 失败：`F_f` 语义不忠实或 Composer 不保语义；
- Gate1 失败而 Replay–J 通过：π/Composer 未重建 Source realization；
- Source–Replay 失败：最终目标执行不闭合，不能用其他两边替代。

## 7. Source-blind 隔离

Semantic Agent 的只读输入仅为：

```text
input/j.json
input/semantic-program.json
input/semantic-program.intent
input/frontend.json
input/TARGET_NORMAL_IR.md
```

它绝不能看到 π、MaskC、Source、Harness valuation、Gate、旧 Replay 或 packet 全景。输出只有：

```text
output/target-normal.json
```

Composer 在 Agent episode 结束后由宿主调用，读取 exact `F_f + π + MaskC(C)`。Composer 不是 Agent
工具，Agent 不能通过 diagnostics 反向查询 π。Target-Normal parser/checker 只返回 J/IR 自洽错误。

## 8. 开放世界与工作量

Target-Normal vocabulary 应比某一 frontend API 稳定：逻辑 map/reduce/fold/scan/contraction、structured
control、typed access/update 与 origin refs。新增 target 只需要一个 Composer/emitter；新增可由现有
primitive 表达的 J op，不需要改 Composer。只有真正无法分解的新 primitive 才扩展版本化 vocabulary。

这不是“零 target 代码”：严格隔离下，每个 target 至少需要一个确定性 emitter。减少的工作是把
每个 Source frontend × 每个 target 的完整手写编译器，因子化为共享 Target-Normal IR 加每 target
一个通用 Composer；LLM 承担开放世界的 semantic instruction selection，而非 pointer/π bookkeeping。

## 9. 实施顺序

1. 归档 Task 18，冻结本文与 Task 19；
2. 定义 v1 Target-Normal schema/parser、J-origin completeness 与 MaskJ 隔离；
3. 将 Codex Replay 拆成 J-only Semantic Agent episode 与 host-side Composer；
4. 先实现 Triton generic pointwise/reduction/control/access vertical slice；
5. 用一个已闭合 Kernel 验证新 Replay 路径不退化；
6. 用 Flash forward 验证 loop/carrier 与 π constraint consumption；
7. 按真实 failure class 扩展 generic primitive/Composer，而非 Kernel patch；
8. 稳定后再提出 compact π vNext，并 fresh Lift 受影响 artifact。

最终原则：

> **Agent 只从 J 生成完整语义实例；π 只约束物理实现；Composer 只能物化二者的交集。任何一方
> 缺失时，系统都不能静默猜出另一方。**
