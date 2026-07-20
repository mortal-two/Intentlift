# 验证架构

当前项目的验证架构只有两个过程：

1. [SCRC Gate1](scrc-gate1.md)：源码隔离重放与中层结构闭合；
2. [SCRC Gate2](scrc-gate2.md)：源与重放两腿动态实跑和逐输出比较。

两者在正式流水线中按 Gate1→Gate2 顺序执行，但验证命题不是包含关系。
Gate1 拒绝而离线 Gate2 通过可以合理；Gate2 不是 Gate1 的真值标签，
也不得覆盖 Gate1 的结构或反作弊拒绝。

IntentIR production lowering、归档解释器和 backend benchmark 均不拥有 Gate2
裁决权。旧三方 Gate2 已归档到
[back/gate2_production_lowering_v1/scrc-gate2-three-way.md](back/gate2_production_lowering_v1/scrc-gate2-three-way.md)。

旧的 L0–L9/schema-obligation/FULL-PARTIAL 架构已归档到
[back/legacy_v0/architecture.md](back/legacy_v0/architecture.md)，不再指导新实现。
