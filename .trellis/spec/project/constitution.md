# IntentIR 项目宪法

## Purpose

让 IntentIR 成为一套**带证书的语义提升系统**,把已优化的 Triton / TileLang / CUDA kernel 用 LLM 提取成可执行、可移植、带契约的三层意图表示,既能验证、又能跨硬件重定向;并把这套表示沉淀为**"给 LLM 用的 kernel 理解 / 提取规范"**,以一个真正的 MLIR dialect 形态进入 MLIR 生态。

## 核心科研问题

> 能否把"算什么"和"怎么调度"故意纠缠在一起的优化 kernel,通过"LLM 受约束提案 + 验证裁决"反向恢复成一个三层(算法意图 / 可移植正确性结构 / 非绑定调度提示)的意图表示,使其既能被验证,又能在新硬件上重新调度?

这是论文 *IntentIR: Certified Semantic Lifting for Cross-Hardware Kernel Portability* 的科研构思。**论文是锚:本项目可以用全新思路重写实现,但不得偏离论文的科研构思。**

## 一句话脊柱

提取 = 把 kernel 从 **SPMD 单程序视角**(每个 program/thread 在干什么)升回 **whole-tensor 整张量视角**(这个算子对整个 tensor 算什么)。由此自动分出三层:

- **Layer A**:整张量算子算什么(算法意图,被验证的语义对象)。
- **Layer B**:使"切分到 grid 上"这件事合法的条件(mask / 边界 / 同步 / 归约结合性)。
- **Layer C**:整张量算子如何切分到 grid 上(tiling / 线程映射 / unroll / 向量宽度等 schedule,可重新推导)。

## Required(必须坚持)

- **IntentIR 是 LLM 提取规范**:它的设计标准是对 LLM 友好(好生成、好自检)、对验证友好(每个算子语义明确)、对后端友好(能 lower)。这是项目护城河。
- **三层 A/B/C 不变量**:A/B 是绑定的(验证它们);C 是非绑定的(后端可重调)。一个源结构若同时含 A 和 C(如 FlashAttention 的 K 循环),必须**因式分解**成 A + C,而不是二选一归类。
- **LLM 只提案,验证裁决**:LLM 是受约束的提案者,正确性由 schema + obligation + 动态源码执行一致性决定,不由 LLM 自证。
- **LLM 直接提取到原子 IR**:原子化分解由 LLM 在提取阶段完成(这是科研核心价值),产出张量级原子的 IntentIR JSON。re-roll / 识别展开循环也属于 LLM 提取的一部分。
- **张量级原子粒度**:原子算子是 `map`/`reduce`/`contraction`/`scan` 等张量级结构化原语(linalg 风格),不是标量级 `add`/`mul`,也不是库级 `softmax`/`attention` 大算子。
- **结构化循环是一等公民**:IR 必须能表达"带状态携带的结构化循环",以支持 FlashAttention 这类递推算子;同一构造放宽即覆盖数据依赖控制流。
- **atomic / 并发归约是一等公民**:提取成归约意图(Layer A 的 reduce/scatter)+ 结合性标注(Layer B),浮点非结合用 tolerance 验证,不 scope-out。
- **能力契约理念**:成熟工具卖的是"清晰的能力边界",不是"号称万能"。能力边界是**设计出来的**,在启动阶段就写进 [capability](../capability/index.md)。
- **动态执行一致性反哺契约**:静态证不出但动态执行一致的 kernel 应被接受,而不是烂在 PARTIAL 或被拒。
- **MLIR 流程固定**:`JSON → Intent MLIR dialect → (后续) lower 到标准 MLIR`。lower 是 IR/前端稳定后的下游阶段。
- **成熟工程,非原型**:目标是真实、可用、可维护的工程。Intent dialect 走真 C++ MLIR + TableGen(MLIR 20.1.8),不走 Python binding / 纯 Python 捷径;每步要能 build、能过测试(见 [code-layout.md](code-layout.md))。

## Forbidden(禁止)

- **不得把原子化分解交给 compiler pass**:那会把 LLM 从"语义提升者"降级成"打标签的",偏离论文卖点。分解是 LLM 的活。
- **不得号称万能 / 隐藏边界**:覆盖不了的模式必须诚实标注,不假装能做。
- **不得基于三个旧文件夹改、不得继承其实现**:`intentir`/`IntentIR`/`TianchenForge` 只作参考。
- **不得纳入 org / OptGuide / provider29 / four-ratio 思路**:那是另一条科研线,与 IntentIR 思想无关。
- **不得把 certificate 描述成全域形式证明**:它是"测试输入域执行一致 + 静态义务"级别的保证。坦白这个天花板。
- **不得在起步阶段引入 named op(库级命名算子)**:张量级原子已消除长图问题;named op 留作未来标准库扩展,且它绝不等于 pass 分解。
- **不得把 lower 到标准 dialect 当成当前阶段重点**:当前焦点是 JSON 规范 + Intent dialect 本身。

## Check Questions

- 这个改动是否守住了 A/B(绑定)与 C(非绑定)的分界?
- 原子化分解是不是仍由 LLM 完成,而不是被偷偷塞进了某个 pass?
- 算子粒度是不是张量级(不是标量、也不是库级大算子)?
- 对覆盖不了的模式,我是诚实标注了边界,还是假装能做?
- 我说的"验证通过"是测试域执行一致,还是被夸大成了形式证明?
- 我是不是把 lower 到标准 MLIR 提前当成了当前重点?
- 我是不是不小心又沿用了旧仓库 / org / OptGuide 的实现或思路?
