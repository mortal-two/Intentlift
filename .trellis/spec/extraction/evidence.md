# Evidence 的角色

## 矛盾(必须化解)

旧设计让 evidence 同时背两个互斥的 KPI:

- **当"约束生成"**:要它强。但越强越把 LLM 的泛化锁死——规则提不出 evidence 的新 kernel 直接抓瞎,违背"LLM 能识别没见过的 kernel"。
- **当"做验证"**:要它完备。但它本质是字符串 / 规则提取,天生不完备。

## 解法:三个独立角色,谁都不当硬门槛

1. **对生成 = 可选 hint**:有 evidence 就喂给 LLM 增强,**没有也能靠 LLM 直接读源码生成**。evidence 不是必需前置条件,所以**新 kernel 不会因为提不出 evidence 就废**。泛化能力保住。
2. **对验证 = 事后标尺**:规则能提多少提多少;提不到的让 obligation 判 UNKNOWN,**靠动态源码执行一致性兜底**,契约降级而非失败(见 [../verification/dynamic-feedback.md](../verification/dynamic-feedback.md))。
3. **LLM 自报 claim**:LLM 在 JSON 里附上它声称的 witness/claim(access pattern / mask / bound),验证去**核对这些 claim 真不真**。这样 evidence 不再是"前端规则的专属产物",新 kernel 靠"LLM 自报 + 执行一致性"也能验。

## 一句话

> **evidence 进生成,但作 hint 不作门槛;验证靠"规则 evidence + LLM 自报 claim + 动态执行一致性"三条腿。**

## 设计约束

- 禁止把"规则能否提出 evidence"作为"能否提取该 kernel"的门槛。
- evidence 提取器可以逐前端做(Triton TTIR / TileLang TIR / CUDA PTX),但它们的缺失只导致 obligation UNKNOWN,不导致提取中止。

## 与论文机制的关系(有意识的 refinement)

论文 §Evidence-Constrained Inference(main.tex:546)写:"The LLM receives this evidence package **rather than raw source code**."。本项目**有意识地 refine 了这个机制细节**:

- **保留的科研构思**(不偏离):LLM 受约束提案 + 验证裁决。
- **改变的机制细节**:evidence 从"生成的唯一输入 / 硬约束"改为"**可选 hint + 验证标尺 + LLM 自报 claim**";LLM 读源码 + 可选 evidence,而非只读 evidence package。
- **理由**:原措辞把 LLM 泛化锁死在 evidence 提取覆盖范围内(见上文矛盾),与"LLM 能识别没见过的 kernel"自相矛盾。
- **行动项**:论文该句措辞需相应软化(由"rather than raw source code"改为"evidence 作为约束/提示,与源码一起进入提案")。这是 spec 与论文之间已知的、需对齐的一处。
