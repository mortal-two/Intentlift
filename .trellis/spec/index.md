# IntentIR Trellis Specs

本 spec 树定义 **IntentIR**(代号 tianchenir)重写项目的长期目标与约束。它是项目的"宪法 + 设计规范",所有 agent 在改动相关代码前必须先读对应 spec。

## 这个项目是什么

IntentIR 是一套**带证书的语义提升(certified semantic lifting)**系统,也是论文 *IntentIR: Certified Semantic Lifting for Cross-Hardware Kernel Portability* 的实现。它把已经优化死的 Triton / TileLang / CUDA kernel **反向恢复**成一个可执行、可移植、带契约的"意图表示"(IntentIR),用于验证与跨硬件重定向。

本项目的定位升级为:**让 IntentIR 成为"给 LLM 用的 kernel 理解 / 提取规范",并作为一个真正的 MLIR 生态组件,而不是玩具。**

## 重写背景(为什么从零写)

`/home/kingdom/phdworks/TianchenIR/` 下的 `intentir`、`IntentIR`、`TianchenForge` 三个文件夹**只作参考,一律不动、不基于其修改**。它们大量草率、错误、互相缠绕(屎山)。本项目是全新重写,只锚定论文的科研构思,不继承旧实现的任何包袱。

`TianchenForge` / `org` / `OptGuide` / `provider29` / `four-ratio` 是**另一条科研思路**,与本项目的 IntentIR 思想无关,不纳入。我们只从它那里借用它整理过的**前端输入 kernel**(见 [extraction/frontend-inputs.md](extraction/frontend-inputs.md))。

## 命名纪律

- **IntentIR** = 本项目恢复出的三层意图表示(也是 paper 的系统名)。
- **Intent MLIR** = IntentIR 的 MLIR dialect 形态(我们自己的 dialect)。
- **Layer A / B / C** = IntentIR 的三层(算法意图 / 可移植正确性结构 / 非绑定调度提示)。
- 旧仓库里的 `intent_ir` 包名、`org`、`OptGuide` 等仅是历史标识,本项目 prose 一律不沿用其语义。

## 核心链路(不可偏离)

```
kernel 源码 (Triton/TileLang/CUDA)
  → [LLM 提取]  IntentIR JSON(人可读,三层 A/B/C)
  → [1:1 转换]  Intent MLIR dialect(我们自己的 dialect)
  → [验证]      schema + obligation + 动态源码执行一致性 → 契约 FULL/PARTIAL/OOS
  ───────────── 以上是当前阶段焦点 ─────────────
  → [后续阶段]  lower 到 linalg / vector / scf / gpu / llvm → RVV / GPU 后端
```

科研核心在 **LLM 提取** 与 **验证**;MLIR 生态价值在 **Intent dialect + 后续 lowering**。lower 到标准 MLIR 是 IR/前端稳定后的下游阶段,当前不展开。

## Spec 导航

| Spec 层 | 索引 | 作用 | 何时读 |
|---|---|---|---|
| project | [project/index.md](project/index.md) | 科研定位、不可偏离论文的约束、能力契约理念、anti-drift 规则。 | 任何架构决策、review、对外定位之前。 |
| ir | [ir/index.md](ir/index.md) | Intent MLIR dialect 的具体形态:ABC 三层、张量级原子算子、结构化循环、JSON↔dialect。 | 改动 IR 定义、算子集、dialect、JSON schema 之前。 |
| extraction | [extraction/index.md](extraction/index.md) | LLM 前端提取:直接提取到原子 IR、evidence 角色、前端输入清单。 | 改动提取管线、prompt、evidence、前端输入之前。 |
| capability | [capability/index.md](capability/index.md) | 能力契约:能力矩阵、三种"不能"、诚实边界。 | 决定支持/不支持某类 kernel 模式之前。 |
| verification | [verification/index.md](verification/index.md) | SCRC Gate1 源码隔离重放 + Gate2 source/replay 双腿实跑。 | 改动 Gate1/Gate2、规则库、指纹或动态输入之前。 |
| lowering | [lowering/index.md](lowering/index.md) | (后续阶段)Intent dialect 下降到标准 MLIR 与 RVV/GPU 后端。 | 进入 lowering 阶段时再读。 |
| guides | [guides/index.md](guides/index.md) | Trellis 自带的跨层思维方法论(工程机制,非项目内容)。 | 编码前按需。 |

本 spec 树只记录稳定的项目约束与设计规范,不记录任务状态、进度或计划(那些由 Trellis 的 task/workspace 管理)。
