# LLM 提取

## 核心立场:LLM 直接提取到原子 IR

LLM 的工作是把任意 kernel **直接提取成张量级原子的 IntentIR JSON**(三层 A/B/C)。这个"理解 + 原子化分解"的动作**就是论文的科研核心价值**。

- LLM **是** 受约束的语义提案者 + 原子化分解者。
- LLM **不是** 正确性裁判(由验证裁决),也不是"直接生成目标 kernel 代码"的生成器。
- **禁止**把原子化分解外包给下游 compiler pass——那会把 LLM 降级成"打标签的",偏离论文卖点(见 [../project/constitution.md](../project/constitution.md))。

## 提取流程

1. 取 kernel 源码(+ 可选 evidence hint,见 [evidence.md](evidence.md))。
2. LLM 在 IntentIR 规范约束下,提出张量级原子的 IntentIR JSON 候选,并附上它声称的 witness/claim。
3. SCRC Gate1 用源码隔离重放和双 TTIR 结构闭合裁决 J/π 是否装下源结构。
4. Gate1 闭合后，Gate2 对源、重放、production lowering 三方实跑。
5. 失败 → 结构化诊断回到重写员小修或提升官重提；超过上限则记录未闭合，不强行接受。

## SCRC 下必须区分两种 LLM 角色

- **提升官/lifter**：可看源 Kernel，负责提出 J、受限 π、接口、C 和定位表；
- **重写员/rewriter**：绝对不可看源码，只能接收 J、π、接口、C 槽和可信
  规则卡，生成用于 [../verification/scrc-gate1.md](../verification/scrc-gate1.md)
  compile-only/TTIR 闭合的重放 Kernel。

两者不能复用同一个包含源码的 conversation/context。LocationMap、源 TTIR、
C 绑定值与源侧差异明细只能留在提升侧；重写员的小修反馈只允许固定失败
类别和重放自身的清洗后编译错误。重写员生成代码是验证探针，不改变
“LLM 直接提取到原子 IntentIR”这一科研主线，也不能成为生产语义真值。

## re-roll 是提取的一部分

- 手写高性能 kernel 大量手工 unroll / strip-mine。LLM 要把展开的重复体**卷回(re-roll)**成紧凑的张量算子 / 循环。
- 识别重复模式是 LLM 强项(优于脆弱的规则匹配),这也强化了"LLM 做提取"的主线。
- re-roll 在**提取阶段**完成,不是下游 pass(见 [../ir/structured-loop.md](../ir/structured-loop.md))。

## 为什么用 LLM 而不是纯规则

优化 kernel 把同一个张量计算用不同前端抽象、不同 schedule 编码表达,纯规则提取器很脆。LLM 作为有泛化能力的识别器,能处理没见过的 kernel——前提是**不被 evidence 完备性卡死**(见 [evidence.md](evidence.md))。
