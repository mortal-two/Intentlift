# IR 层:Intent MLIR Dialect 形态

定义 IntentIR 的具体表示:三层 A/B/C、张量级原子算子集、结构化循环、以及 JSON ↔ Intent MLIR dialect 的关系。这是项目的技术核心。

## 何时读

- 改动 IR 数据结构、算子集合、dialect 定义、JSON schema 之前。
- 决定"某个 kernel 模式在 IR 里怎么表示"之前。

## 当前职责

本层定义 portable semantic contract，而不只是 dialect 语法：

- `J=A+B` 的逻辑值与可观察执行合同；
- optional C 的可替换实现策略；
- JSON、Python model 与 Intent MLIR 的确定性对应；
- frontend、Gate 和 target backend 共同消费时不可丢失的 ABI／boundary 信息。

Lowering 已经是当前工程的一等消费者（见 [../lowering/index.md](../lowering/index.md)）。
IR 设计不得被某个 target 的临时缺口绑架，但也不能设计成 backend 无法完整消费 B 的
纸面 schema。

## 文件

| 文件 | 内容 |
|---|---|
| [layers-abc.md](layers-abc.md) | A/B/C 三层定义、分层不变量、softmax/FlashAttention 分层示例。 |
| [atomic-ops.md](atomic-ops.md) | 张量级原子算子集、粒度原则、数据移动 / 间接寻址 / 归约结合性。 |
| [op-signatures.md](op-signatures.md) | **具体形态**:JSON 顶层结构、每个 op 的 operands/results/attrs/层归属、`loop` 形态。 |
| [structured-loop.md](structured-loop.md) | 带状态结构化循环、FlashAttention 因式分解、数据依赖控制流逃生舱。 |
| [flashattention-example.md](flashattention-example.md) | **Worked example**：FlashAttention 的 A/B/C 语义所有权与递推×tiling 因式分解。 |
| [json-and-dialect.md](json-and-dialect.md) | JSON(人可读,LLM 产出)↔ Intent MLIR dialect 的 1:1 关系。 |
| [representation-decisions.md](representation-decisions.md) | 地基表示决策：控制流、逻辑结果／destination 分层、named op 边界。 |

## Pre-Development Checklist

- [ ] 新构造是张量级粒度吗?(不是标量、不是库级大算子)
- [ ] 它是否清晰落在 A / B / C 某一层,或被正确因式分解?
- [ ] 改变逻辑输出的 view/index 是否位于 A，而非混入 access 地址或 C?
- [ ] correctness-relevant B 字段是否有闭合类型和明确 backend 消费规则?
- [ ] 它在 JSON 与 dialect 里是 1:1 对应吗?
- [ ] 它的语义能被解释器执行、从而做动态验证吗?

## Quality Check

- 任何"高层算子内部隐藏了 mask/bound/归约"的设计都不合格(B 层约束必须显式可见)。
- 任何依赖"下游 pass 来决定语义分解"的设计都不合格(分解是 LLM 的活)。
- 任何只向 `IntentFunction` 发射 A、却把会影响正确性的 function-level B 留在验证层
  且不交给 backend 的设计都不合格。
