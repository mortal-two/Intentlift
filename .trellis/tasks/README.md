# 工作项目录

本目录只借用 Trellis 的文件布局。工作项是普通 Markdown，不使用 `task.json` 状态机、
Trellis CLI、自动 session 或 workflow phase。

## 当前新设计序列

- [Task 07：Canonical Boundary Projection 与 Production Lowering 消费闭合](07-23-canonical-boundary-lowering-07/task.md)（执行中）
- [Task 06：CanonicalJ 标准化与统一 lowering 消费边界](07-23-canonical-j-elaboration-06/task.md)（已完成）
- [Task 05：Gate2 dense pointwise RefEval 合同闭合](07-23-gate2-dense-refeval-05/task.md)（已完成）
- [Task 04：mini-swe-agent 有界 Lift Worker 接入](07-23-miniswe-lift-worker-04/task.md)（已完成）
- [Task 03：28 项真实 Kernel Manifest、Batch 与代表性纵切](07-22-scrc-real-kernel-batch-03/task.md)（执行中；先完成 Task 04 的 Lift 生成能力替换）
- [Task 02：真实 SCRC Exact / Masked / Reduction Canary](07-22-scrc-real-canary-02/task.md)（已完成）
- [Task 01：SCRC Gate Core 整体门禁能力重构](07-22-scrc-system-rebuild-01/task.md)（已完成）

编号从 01 重新开始，因为 2026-07-22 的 C、π、最终执行流程和 SCRC 数学总述已经替换此前
formal/hash/epoch 实验口径；新任务不得继承旧序列的协议状态。Task 01 只负责门禁系统能力的
代码重构；当前 Task 02 从三个顺序 canary 开始使用真实 LLM，不直接跳到 28-kernel 或性能结论。

## 历史任务序列

此前 01–10 只用于追溯，不约束当前实现。其中旧 Task 10 与 formal batch v1–v7 已迁入：

- [旧 Lift Envelope / Boundary 10 与 formal/hash 实验归档](../archive/experiments/2026-07-22-formal-hash-protocol/task/task.md)

最近完成：

- [Address Realization 09：新增 base/lane tags，v6 暴露 Lift envelope/boundary 缺口](archive/2026-07/07-22-address-realization-09/task.md)
- [Pi Graph Integrity 08：闭合 graph 外键，v5 暴露 address base/lane 缺口](archive/2026-07/07-22-pi-graph-integrity-08/task.md)
- [Semantic/Realization Residual 07：完成 J/C/π 所有权回收，v4 暴露 graph 外键歧义](archive/2026-07/07-22-pi-realization-topology-07/task.md)
- [Lift Wire / Parser Isomorphism 06：wire/parser 闭合，v3 暴露 pi topology 缺口](archive/2026-07/07-22-lift-wire-parser-isomorphism-06/task.md)
- [Formal Batch V2 05：protocol-v2 recovery canary 失败并完整留证](archive/2026-07/07-22-formal-batch-v2-05/task.md)
- [28 Real Kernels 04：protocol-v1 首次真实 Lift 失败并完整留证](archive/2026-07/07-22-28-kernel-llm-batch-04/task.md)
- [Typed Relations 03：闭合 logical bounds、update 与 allowed-result relation](archive/2026-07/07-22-typed-relations-03/task.md)
- [IntentOp Robustness 02：真实机制审计与错误能力回收](archive/2026-07/07-21-intentop-robustness-02/task.md)
- [Architecture Reset 01：清除旧控制面并切换到 LLM-direct π](archive/2026-07/07-21-architecture-reset-01-direct-pi/task.md)

此前所有活动任务已原样归档到
[07-21 reset 快照](archive/2026-07/07-21-pre-llm-first-reset/README.md)。
