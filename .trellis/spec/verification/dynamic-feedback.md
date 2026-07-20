# 动态执行

动态正确性检查统一属于 [SCRC Gate2](scrc-gate2.md)。边界 shape、对抗数值、
随机 seed 与 tolerance 均在那里定义，并同步施加到相互隔离、且绑定同一最终 J 的
replay/scalar 两腿。Gate2 结果不反哺或修改已经接受的 J；失败必须形成新 Lift/Gate1
任务或 scalar/reference 能力修复，不能在 Gate2 内 repair。

旧“动态反哺契约”文档已归档到
[back/legacy_v0/dynamic-feedback.md](back/legacy_v0/dynamic-feedback.md)。
