# Gate2 production-lowering 三方路径归档

这里保存 2026-07-20 前 Gate2 的历史三方实现：source、replay 与 IntentIR
production lowering（CPU Lower2C 或 CUDA）共同参与一次 Gate2 裁决。

该实现仅用于历史报告复核与实验复现，不属于活动验证路径。当前
`intentir.verify.gate2` 只执行 source/replay 两腿；活动 Gate2、live/held-out
入口和正式 campaign 均不得导入本目录，也不得把本目录的 lowering 结果写入
Gate2 verdict。

归档内容包括旧结果模型、三方 orchestration、Gate1/Gate2 统计口径、CPU
Lower2C adapter 与 CUDA adapter。通用后端代码仍位于 `intentir.backend`，但它是
独立编译/性能能力，不是当前 Gate2 的正确性裁决腿。

该目录是 compatibility archive：历史 adapter 继续共享活动 Gate2 的通用输入生成、
Triton 执行与输出比较工具，并非完全冻结的源码快照。需要逐提交复现实验时应使用
对应历史 commit；活动主线仍严格保持 `gate2 -> back` 零依赖。
