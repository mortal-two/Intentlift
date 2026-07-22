# 动态执行

动态正确性检查统一属于 [SCRC Gate2](scrc-gate2.md)。边界 shape、对抗数值、
随机 seed 与 tolerance 均在那里定义，并同步施加到相互隔离的 source、final-J replay
与 scalar lowering 三腿。Gate2 结果不反哺或修改已经接受的 J；失败可以形成后续
能力建设任务，但不能在 Gate2 内 repair、反馈或增加准入门槛。

旧“动态反哺契约”文档已归档到
[back/legacy_v0/dynamic-feedback.md](back/legacy_v0/dynamic-feedback.md)。
