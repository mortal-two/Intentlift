# Extraction 层：LLM 前端提取

本层定义如何把真实 Kernel 提升为 IntentIR：可信输入、source-aware Lift、严格 schema、
source-blind replay 所需的前端适配，以及 evidence／corpus 的角色。

## 长期合同

- LLM 负责理解源码并直接产出张量级原子 IntentIR；compiler pass 不替它做语义分解；
- 提取目标是 `J=A+B` 与 optional C，不是只产出一个可 lower 的 A op graph；
- source/launch/compiler-view ABI 在进入 Lift 前必须精确绑定，发现不一致立即 fail closed；
- 不同前端分别建立 `P_i ~ J_i`，不要求 Triton/TileLang/CUDA 产生字节相同的 J；
- 前端专属 API、compiler fingerprint、RuleRegistry 与 execution adapter 留在前端验证层，
  不进入 portable A/B；
- provider 型号、请求并发、某次 corpus 数量和 campaign 结果属于任务／报告，不属于
  稳定 spec。

## 何时读

- 改动 Lift/Relift、prompt、provider、strict schema 或 evidence；
- 新增 Triton/TileLang/CUDA 等 source frontend；
- 改动 source capture、launch binding、compiler view 或 replay adapter；
- 扩充 corpus、held-out family 或 provenance 记录。

## 文件

| 文件 | 内容 |
|---|---|
| [llm-extraction.md](llm-extraction.md) | LLM 直接提取到张量级原子 IR、Lift/Relift 与 source-blind writer 边界。 |
| [evidence.md](evidence.md) | evidence 作为生成 hint、审计标尺与输入设计提示，不拥有 verdict。 |
| [frontend-inputs.md](frontend-inputs.md) | 前端来源、capture/provenance 合同与 corpus 证据边界。 |

## 新增 Frontend 的完成定义

1. source、launch、compiler-view ABI 参数顺序／kind／dtype／rank 可机械核对；
2. 产出共享 IntentIR schema，不注入 target backend 特例；
3. frontend-specific compiler representation 可用于 Gate1 fingerprint；
4. source/replay 可在隔离 module 与同一输入合同下实际执行；
5. unsupported source feature 在 Lift 前或 strict validation 中明确归类；
6. provenance 绑定 source、capture、compiler-view 与 artifact digest；
7. 不把其他前端的 J equality 当作本前端证书前置条件。

## Pre-Development Checklist

- [ ] 原子化分解仍由 LLM 完成吗？
- [ ] ABI/capture 输入是否可信且 fail closed？
- [ ] logical index、boundary 与 source realization 是否正确分层？
- [ ] evidence 是否只作 hint／标尺，而不是隐藏答案或 verdict？
- [ ] 新前端是否能独立完成 compile fingerprint 与 source/replay execution？

## Quality Check

- 提取不依赖 Kernel 名、固定答案或当前 backend 支持表来生成 J；
- 产物是完整 `A+B`，不是库级标签、标量长图或只有 A 的 lowering 输入；
- capture、Lift、replay writer、Gate comparator 与 backend 的错误责任可分别归类；
- corpus 数量、模型版本和单次通过率只在报告中陈述，并链接可审计 artifact。
