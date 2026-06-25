# LLM 提取

## 核心立场:LLM 直接提取到原子 IR

LLM 的工作是把任意 kernel **直接提取成张量级原子的 IntentIR JSON**(三层 A/B/C)。这个"理解 + 原子化分解"的动作**就是论文的科研核心价值**。

- LLM **是** 受约束的语义提案者 + 原子化分解者。
- LLM **不是** 正确性裁判(由验证裁决),也不是"直接生成目标 kernel 代码"的生成器。
- **禁止**把原子化分解外包给下游 compiler pass——那会把 LLM 降级成"打标签的",偏离论文卖点(见 [../project/constitution.md](../project/constitution.md))。

## 提取流程

1. 取 kernel 源码(+ 可选 evidence hint,见 [evidence.md](evidence.md))。
2. LLM 在 IntentIR 规范约束下,提出张量级原子的 IntentIR JSON 候选,并附上它声称的 witness/claim。
3. 验证裁决(schema + obligation + 动态源码执行一致性,见 [../verification/index.md](../verification/index.md))。
4. 失败 → 结构化诊断反馈给 LLM,做**有界**修复;超过修复上限则记弱化 / 不支持结果,不强行接受。

## re-roll 是提取的一部分

- 手写高性能 kernel 大量手工 unroll / strip-mine。LLM 要把展开的重复体**卷回(re-roll)**成紧凑的张量算子 / 循环。
- 识别重复模式是 LLM 强项(优于脆弱的规则匹配),这也强化了"LLM 做提取"的主线。
- re-roll 在**提取阶段**完成,不是下游 pass(见 [../ir/structured-loop.md](../ir/structured-loop.md))。

## 为什么用 LLM 而不是纯规则

优化 kernel 把同一个张量计算用不同前端抽象、不同 schedule 编码表达,纯规则提取器很脆。LLM 作为有泛化能力的识别器,能处理没见过的 kernel——前提是**不被 evidence 完备性卡死**(见 [evidence.md](evidence.md))。
