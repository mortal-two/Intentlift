# IntentIR 项目宪法

## Purpose

让 IntentIR 成为一套**带证书的语义提升系统**,把已优化的 Triton / TileLang / CUDA kernel 用 LLM 提取成可执行、可移植、带契约的三层意图表示,既能验证、又能跨硬件重定向;并把这套表示沉淀为**"给 LLM 用的 kernel 理解 / 提取规范"**,以一个真正的 MLIR dialect 形态进入 MLIR 生态。

## 核心科研问题

> 能否把"算什么"和"怎么调度"故意纠缠在一起的优化 kernel,通过"LLM 受约束提案 + 验证裁决"反向恢复成一个三层(算法意图 / 可移植正确性结构 / 非绑定调度提示)的意图表示,使其既能被验证,又能在新硬件上重新调度?

这是论文 *IntentIR: Certified Semantic Lifting for Cross-Hardware Kernel Portability* 的科研构思。**论文是锚:本项目可以用全新思路重写实现,但不得偏离论文的科研构思。**

## 一句话脊柱

提取 = 把 Kernel 从 **SPMD 单程序视角**（每个 program/thread 在干什么）升回
**whole-tensor 整张量视角**（这个程序对逻辑 tensor 算什么），同时保留实现该值
语义所必需的可观察边界合同。由此分出三层：

- **Layer A**：逻辑值语义——整张量计算、逻辑轴、逻辑 view/index、归约与递推。
- **Layer B**：可观察执行与合法性合同——访问域、mask/fill、destination、alias、
  effect、冲突、数值与 scope/storage 义务。
- **Layer C**：可替换实现策略——tiling、线程映射、局部 layout、unroll、pipeline 等。

完整绑定语义对象是：

```text
J = A + B
```

A 单独不是完整 Kernel 含义；C 不改变 J，但可作为 fixed-C replay 控制条件和
目标 backend 的调优先验。

## Required(必须坚持)

- **IntentIR 是 LLM 提取规范**:它的设计标准是对 LLM 友好(好生成、好自检)、对验证友好(每个算子语义明确)、对后端友好(能 lower)。这是项目护城河。
- **三层 A/B/C 不变量**：A/B 是绑定的完整 J；C 是可替换的实现策略。一个源结构若同时承载多层信息，必须因式分解，不能二选一归类。
- **逻辑 index、合法性与实现地址必须分开**：改变 logical tensor denotation 的 transpose/reverse/gather/scatter 属于 A；bounds、predicate、fill、store guard 属于 B；`program_id/tile/lane` 与物理 staging 属于 C 或 validation-only witness。
- **LLM 只提案，验证裁决**：LLM 是受约束的提案者；正确性证据来自 strict schema、
  Gate1 source/replay 结构闭合与 Gate2 声明输入域动态比较，不由 LLM 自证。
- **LLM 直接提取到原子 IR**:原子化分解由 LLM 在提取阶段完成(这是科研核心价值),产出张量级原子的 IntentIR JSON。re-roll / 识别展开循环也属于 LLM 提取的一部分。
- **张量级原子粒度**:原子算子是 `map`/`reduce`/`contraction`/`scan` 等张量级结构化原语(linalg 风格),不是标量级 `add`/`mul`,也不是库级 `softmax`/`attention` 大算子。
- **结构化循环是一等公民**:IR 必须能表达"带状态携带的结构化循环",以支持 FlashAttention 这类递推算子;同一构造放宽即覆盖数据依赖控制流。
- **atomic / 并发归约是一等公民**:提取成归约意图(Layer A 的 reduce/scatter)+ 结合性标注(Layer B),浮点非结合用 tolerance 验证,不 scope-out。
- **能力契约理念**：成熟工具卖的是清晰、可审计的 represent/extract/verify/lower
  边界，不是“号称万能”（见 [capability](../capability/index.md)）。
- **Gate 与 lowering 解耦**：Gate1/Gate2 分别裁决 Lift/replay 的结构与动态证据；
  backend conformance 检查目标实现是否正确消费 J；performance 单独评价。任何一方都
  不能代替另一方。
- **中立 lowering 合同**：backend 必须消费完整 `A+B+ABI`，可选消费或重调 C；不得读取 Gate verdict、provider response、source path 或数据集身份来授权 lowering。
- **B obligation 必须 fail closed**：backend 可以保守忽略 optimization permission，但不能静默忽略 value-driving 字段或 legality obligation。
- **MLIR 流程固定**：`JSON → Intent MLIR dialect → 标准 MLIR／目标代码`。JSON 到 dialect 不做语义分解，lowering 只做受合同约束的硬件下降与合法优化。
- **成熟工程,非原型**:目标是真实、可用、可维护的工程。Intent dialect 走真 C++ MLIR + TableGen(MLIR 20.1.8),不走 Python binding / 纯 Python 捷径;每步要能 build、能过测试(见 [code-layout.md](code-layout.md))。

## Forbidden(禁止)

- **不得把原子化分解交给 compiler pass**:那会把 LLM 从"语义提升者"降级成"打标签的",偏离论文卖点。分解是 LLM 的活。
- **不得号称万能 / 隐藏边界**:覆盖不了的模式必须诚实标注,不假装能做。
- **不得直接复制历史耦合**：旧 `intentir`/`IntentIR`/`TianchenForge` 只作证据或
  技术参考；借鉴必须重新满足当前 A/B/C、Gate 与 backend 合同。
- **不得纳入 org / OptGuide / provider29 / four-ratio 思路**:那是另一条科研线,与 IntentIR 思想无关。
- **不得把 certificate 描述成全域形式证明**：它是 scoped source-conditioned replay
  equivalence 与声明输入域动态证据，不是无条件 denotational proof。
- **默认不引入 named op（库级命名算子）**：未来若确需扩展，必须给出固定原子展开
  等价定义，且不能让 compiler pass 猜测语义分解。
- **不得让 lowering 反向篡改语义**：不能因为某个 backend 暂不支持某字段，就把 correctness-relevant 信息移出 A/B、偷偷烘焙 specialization，或回退到解释器伪装成功。
- **不得让 Gate verdict 参与 backend 选择**：selector 只能依据中立 IR/ABI、目标能力、显式 schedule 与结构 probe。

## Check Questions

- 这个改动是否守住了 `J=A+B` 与 C 可替换的分界？
- 改变逻辑输出的 index/view 是否明确落在 A，而不是埋在地址字符串或 C 中？
- destination、alias、partial/conditional writeback 是否与逻辑返回值分开？
- 原子化分解是不是仍由 LLM 完成,而不是被偷偷塞进了某个 pass?
- 算子粒度是不是张量级(不是标量、也不是库级大算子)?
- 对覆盖不了的模式,我是诚实标注了边界,还是假装能做?
- 我说的"验证通过"是测试域执行一致,还是被夸大成了形式证明?
- backend 是否消费了完整 A+B+ABI，并对未知 obligation fail closed？
- Gate、backend conformance 与性能证据是否仍是三个独立命题？
- 我是不是不小心又沿用了旧仓库 / org / OptGuide 的实现或思路?
