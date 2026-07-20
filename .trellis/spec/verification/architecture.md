# 验证架构

当前项目的验证架构只有两个过程：

1. [SCRC Gate1](scrc-gate1.md)：源码隔离重放与中层结构闭合；
2. [SCRC Gate2](scrc-gate2.md)：Gate1 最终 J 的 replay 与 scalar MLIR lowering
   两腿动态实跑和逐输出比较。

两者在正式流水线中按 Gate1→Gate2 顺序执行。Gate1 hard closure 是活动 Gate2 的
资格前提；Gate1 拒绝样本的任何离线动态实验都只是诊断，不是 Gate2。Gate2 不是
Gate1 的真值标签，也不得覆盖 Gate1 的结构或反作弊拒绝。

Gate2 使用真实的 backend-neutral scalar MLIR reference lower，但 IntentIR production
target lowering、归档解释器和 backend benchmark 均不拥有 Gate2 裁决权。旧三方
Gate2 已归档到
[back/gate2_production_lowering_v1/scrc-gate2-three-way.md](back/gate2_production_lowering_v1/scrc-gate2-three-way.md)。
历史 `source_replay_dynamic_equivalence_v1` 报告也不得进入当前 v2 统计。

旧的 L0–L9/schema-obligation/FULL-PARTIAL 架构已归档到
[back/legacy_v0/architecture.md](back/legacy_v0/architecture.md)，不再指导新实现。
