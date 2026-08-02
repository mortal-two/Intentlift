# run-001 控制面切换审计

本 run 使用旧的单 branch 生命周期启动 xhigh Lift。Source projection、adapter、request 与 `c0` 已
保存，但在 provider 返回任何 Lift response 以前，用户明确要求把并发粒度改为同一 Kernel 的多个
J/π candidate。该进程随后人工中断，以免旧单链继续占用 provider。

因为没有产生 candidate、TypeCheck、Replay、Gate1 或 Gate2 evidence，本 run 不是语义失败，也
不能说明 swiglu 不闭合。正式 swiglu 搜索必须在新的 candidate-fanout 协议下重新开始。
