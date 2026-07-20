# 验证裁决

当前 `verify` 的裁决由 [Gate1](scrc-gate1.md) 与 [Gate2](scrc-gate2.md) 完成，
不再维护独立的 schema/obligation verifier 或 FULL/PARTIAL/OOS 契约流程。

Gate2 verdict 只消费 source/replay 动态比较，不消费 production lowering、解释器
或性能 benchmark 的输出。

旧文档仅为历史参考：
[back/legacy_v0/validation.md](back/legacy_v0/validation.md)。
