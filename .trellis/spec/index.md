# IntentIR Trellis Specs

本 spec 树定义 **IntentIR**(代号 tianchenir)重写项目的长期目标与约束。它是项目的"宪法 + 设计规范",所有 agent 在改动相关代码前必须先读对应 spec。

## 这个项目是什么

IntentIR 是一套**带证书的语义提升(certified semantic lifting)**系统,也是论文 *IntentIR: Certified Semantic Lifting for Cross-Hardware Kernel Portability* 的实现。它把已经优化死的 Triton / TileLang / CUDA kernel **反向恢复**成一个可执行、可移植、带契约的"意图表示"(IntentIR),用于验证与跨硬件重定向。

本项目的定位升级为:**让 IntentIR 成为"给 LLM 用的 kernel 理解 / 提取规范",并作为一个真正的 MLIR 生态组件,而不是玩具。**

## 历史与参考边界

当前仓库已经是重写后的项目主体。历史实现与 refs 只能作为只读证据或技术参考；
任何借鉴都必须重新满足本 spec 的 A/B/C、Gate 与 backend 合同，不能把旧命名、旧
耦合或旧 verdict 直接带回主线。

`TianchenForge` / `org` / `OptGuide` / `provider29` / `four-ratio` 属于另一条科研
思路，不拥有本项目的语义真值或验证裁决权。可复用的前端 Kernel 来源见
[extraction/frontend-inputs.md](extraction/frontend-inputs.md)。

## 命名纪律

- **IntentIR** = 本项目恢复出的三层意图表示(也是 paper 的系统名)。
- **Intent MLIR** = IntentIR 的 MLIR dialect 形态(我们自己的 dialect)。
- **Layer A / B / C** = 逻辑值语义 / 可观察执行与合法性合同 / 可替换实现策略。
- **J** = 完整绑定语义对象 `A+B`；A 单独不是完整 Kernel 合同。
- 旧仓库里的 `intent_ir` 包名、`org`、`OptGuide` 等仅是历史标识,本项目 prose 一律不沿用其语义。

## 核心链路(不可偏离)

```text
Kernel 源码（Triton / TileLang / CUDA）
  → [LLM Lift / Relift]  IntentIR：J=(A,B) + optional C
  → [确定性转写]        Intent MLIR dialect
  ├→ [Gate1]            同前端、同冻结 C 的 source/replay 结构闭合
  ├→ [Gate2]            source/replay 声明输入域内的动态比较
  └→ [Neutral Lowering] IntentProgram(A+B+ABI, optional C)
                         → standard/target MLIR → executable
                         → 独立 backend conformance + performance evidence
```

Lift、Gate1、Gate2、backend conformance 与 performance 是不同命题。Lowering 已是
当前工程主线的一等组成，但不得读取或参与 Gate verdict。

## Spec 导航

| Spec 层 | 索引 | 作用 | 何时读 |
|---|---|---|---|
| project | [project/index.md](project/index.md) | 科研定位、不可偏离论文的约束、能力契约理念、anti-drift 规则。 | 任何架构决策、review、对外定位之前。 |
| ir | [ir/index.md](ir/index.md) | Intent MLIR dialect 的具体形态:ABC 三层、张量级原子算子、结构化循环、JSON↔dialect。 | 改动 IR 定义、算子集、dialect、JSON schema 之前。 |
| extraction | [extraction/index.md](extraction/index.md) | LLM 前端提取:直接提取到原子 IR、evidence 角色、前端输入清单。 | 改动提取管线、prompt、evidence、前端输入之前。 |
| capability | [capability/index.md](capability/index.md) | 能力契约：represent/extract/verify/lower 矩阵、四种“不能”、诚实边界。 | 决定支持/不支持某类 Kernel 模式之前。 |
| verification | [verification/index.md](verification/index.md) | SCRC Gate1 源码隔离重放 + Gate2 source/replay 双腿实跑。 | 改动 Gate1/Gate2、规则库、指纹或动态输入之前。 |
| lowering | [lowering/index.md](lowering/index.md) | verification-neutral 的标准／目标 MLIR lowering、selection、runtime 与能力证据。 | 改动 backend、pass、selector、runtime 或性能路径之前。 |
| guides | [guides/index.md](guides/index.md) | Trellis 自带的跨层思维方法论(工程机制,非项目内容)。 | 编码前按需。 |

本 spec 树只记录稳定的项目约束与设计规范,不记录任务状态、进度或计划(那些由 Trellis 的 task/workspace 管理)。
