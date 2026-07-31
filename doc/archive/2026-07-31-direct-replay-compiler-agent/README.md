# 2026-07-31 直接 Replay Compiler Agent 归档

本目录保存 Task 18 的直接编译路线：固定代码从 J 生成只读 `S_J`，一个 fresh source-blind Agent
同时读取 `S_J + π + MaskC(C)` 并手写完整目标源码。

该路线证明了 `S_J`、持久 coding episode、own-compile/own-TTIR 与事务提交有效；但 FlashAttention
forward 的正式纵切也证明，要求 Agent 同时维护 value graph、60+ realization occurrence、地址、
program mapping 和 carrier 会继续覆盖 J 的真实 operand edge。最佳正式结果仍有 112 项 dataflow
差异，三分支九个 formal round 均未闭合。

活动路线由 Task 19 接管：保留 `S_J=M(J)`；fresh Agent 只从 J/`S_J` 产生完整 Target-Normal IR
实例；确定性 target Composer 再以 π 为约束物化目标源码。归档材料只用于追溯，不是活动规范。
