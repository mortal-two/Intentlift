# 工作项目录

本目录只借用 Trellis 的文件布局。工作项是普通 Markdown，不使用 `task.json` 状态机、
Trellis CLI、自动 session 或 workflow phase。

## 当前任务

2026-07-28 有两条对象所有权互不重叠的活动主线。Task 11/14 负责 fresh SCRC corpus；Task 16
以当前 fresh 14 J 为首批正式输入，建设真实 MLIR 跨目标搜索并完成 Source baseline、统一三臂和
C-prior 实验。Task 16 不调用 provider、不修改 Lift/Replay/Gate verdict。

- [Task 11：SCRC v4 的 28-Kernel 有界并行全闭合](07-24-scrc-v4-full-closure-11/task.md)（执行中；SCRC 主线）
- [Task 16：Fresh-14 J 的真实 MLIR 跨目标搜索与 Source Baseline 性能闭合](07-28-fresh14-real-mlir-cross-target-search-16/task.md)（已建立；backend主线）

Task 15 已完成阶段性 checkpoint 并冻结为历史执行记录；其中 legacy 16-high-performance + 2-canary
与 J_dev 结果继续用于 compiler regression，但不再决定 fresh 正式分母：

- [Task 15：16-Certified + 2-Provisional J Development 与真实跨目标后端能力闭合](07-28-backend-18j-capability-search-15/task.md)（阶段性冻结）

## 本轮已归档任务

Task 01–10 均已结束并迁入 `archive/2026-07/`。其中任务正文与 evidence 是历史记录，不再作为
当前执行状态或后继调度的权威来源。

- [Task 10：J Semantic Readiness 与 SCRC vNext 横向闭合](archive/2026-07/07-24-j-readiness-scrc-vnext-10/task.md)
- [Task 09：退役 RefEval 重复 Layer-B 解释器](archive/2026-07/07-23-retire-duplicate-boundary-interpreter-09/task.md)
- [Task 08：唯一 J 的可执行 CUDA ABI 与 Backend Conformance 闭环](archive/2026-07/07-23-executable-j-cuda-backend-08/task.md)
- [Task 07：Canonical Boundary Projection 与 Production Lowering 消费闭合](archive/2026-07/07-23-canonical-boundary-lowering-07/task.md)
- [Task 06：J 构造标准化与统一 lowering 消费边界](archive/2026-07/07-23-canonical-j-elaboration-06/task.md)
- [Task 05：Gate2 dense pointwise RefEval 合同闭合](archive/2026-07/07-23-gate2-dense-refeval-05/task.md)
- [Task 04：mini-swe-agent 有界 Lift Worker 接入](archive/2026-07/07-23-miniswe-lift-worker-04/task.md)
- [Task 03：28 项真实 Kernel Manifest、Batch 与代表性纵切](archive/2026-07/07-22-scrc-real-kernel-batch-03/task.md)
- [Task 02：真实 SCRC Exact / Masked / Reduction Canary](archive/2026-07/07-22-scrc-real-canary-02/task.md)
- [Task 01：SCRC Gate Core 整体门禁能力重构](archive/2026-07/07-22-scrc-system-rebuild-01/task.md)

编号从 01 重新开始，因为 2026-07-22 的 C、π、最终执行流程和 SCRC 数学总述已经替换此前
formal/hash/epoch 实验口径；新任务不得继承旧序列的协议状态。当前序列的 Task 01–10 已成为
Task 11 的历史输入，不得因正文中的旧“执行中”或“后续”措辞被重新激活。

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
