# Evidence 的角色

## 矛盾(必须化解)

旧设计让 evidence 同时背两个互斥的 KPI:

- **当"约束生成"**:要它强。但越强越把 LLM 的泛化锁死——规则提不出 evidence 的新 kernel 直接抓瞎,违背"LLM 能识别没见过的 kernel"。
- **当"做验证"**:要它完备。但它本质是字符串 / 规则提取,天生不完备。

## 解法:三个独立角色,谁都不当硬门槛

1. **对生成 = 可选 hint**:有 evidence 就喂给 LLM 增强,**没有也能靠 LLM 直接读源码生成**。evidence 不是必需前置条件,所以**新 kernel 不会因为提不出 evidence 就废**。泛化能力保住。
2. **对 Gate1 = 诊断辅助**:规则能提多少提多少，用于帮助提升官形成 J/B 与定位表；缺失不阻断 LLM 读源码。Gate1 的裁决仍来自源码隔离重放后的双中层指纹。
3. **对 Gate2 = 输入设计提示**:evidence 可提示需要覆盖的边界 shape/mask 条件，但最终只以源与重放的实际输出比较为准；production lowering 不参与裁决。

## 一句话

> **evidence 进提升和诊断，但作 hint 不作门槛；正确性裁决统一由 SCRC Gate1 + Gate2 完成。**

## 设计约束

- 禁止把"规则能否提出 evidence"作为"能否提取该 kernel"的门槛。
- evidence 提取器可以逐前端做(Triton TTIR / TileLang TIR / CUDA PTX)，但缺失不导致提取中止；后续由 Gate1/Gate2 给出是否闭合的实际结果。

## 与论文机制的关系(有意识的 refinement)

论文 §Evidence-Constrained Inference(main.tex:546)写:"The LLM receives this evidence package **rather than raw source code**."。本项目**有意识地 refine 了这个机制细节**:

- **保留的科研构思**(不偏离):LLM 受约束提案 + 验证裁决。
- **改变的机制细节**:evidence 从"生成的唯一输入 / 硬约束"改为"**可选 hint + 验证标尺 + LLM 自报 claim**";LLM 读源码 + 可选 evidence,而非只读 evidence package。
- **理由**:原措辞把 LLM 泛化锁死在 evidence 提取覆盖范围内(见上文矛盾),与"LLM 能识别没见过的 kernel"自相矛盾。
- **行动项**:论文该句措辞需相应软化(由"rather than raw source code"改为"evidence 作为约束/提示,与源码一起进入提案")。这是 spec 与论文之间已知的、需对齐的一处。
