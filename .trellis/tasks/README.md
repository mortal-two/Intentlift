# 工作项目录

本目录只借用 Trellis 的文件布局。工作项是普通 Markdown，不使用 `task.json` 状态机、
Trellis CLI、自动 session 或 workflow phase。

## 当前任务

- [最终 Kernel 逐核 SCRC 闭合](08-01-final-three-kernel-closure/task.md)（执行中；唯一活动执行记录）

按 [`AGENTS.md`](../../AGENTS.md)，同时只维护一个活动 final task：它只保留当前合同、未解决
阻塞、下一步动作与权威 closure 证据，不累积历史叙述。

## 已归档任务

Task 01–20 与其后的子任务已全部结束，2026-08-01 起迁入仓库根的
[`archive/2026-08-01-trellis-history/tasks-archive/`](../../archive/2026-08-01-trellis-history/tasks-archive/)，
按 `2026-06/`、`2026-07/`、`2026-08/` 分批。其中：

- `2026-08/07-24-scrc-v4-full-closure-11/` — Task 11，28-Kernel 闭合主线
- `2026-08/07-31-j-only-replay-jc-structural-gate1-20/` — Task 20，把 Replay 收窄为只消费
  `J + MaskC(C)`，π 降为 Gate1 内部的 certification-only normalization witness
- `2026-08/08-01-rope-pi-consumer-lift-closure/` — RoPE 手工纵切 + 自动 Lift 双闭合
- `2026-07/` — Task 01–19，含 Task 16–19 已停止的 fragment/hole/binding/Composer 路线

更早的 formal/hash 协议实验在
[`archive/2026-08-01-trellis-history/trellis-archive/experiments/`](../../archive/2026-08-01-trellis-history/trellis-archive/)。

归档任务正文与其中的 evidence 都是历史记录，**不再作为当前执行状态或后继调度的权威来源**；
正文里的"执行中""后续"等措辞不得据以重新激活。每份归档任务只按其落盘时的协议版本解释，
后续 schema/TypeCheck/observer 升级不追认、不重判。

production backend 仍只在 SCRC 之后消费 accepted `J+C`，不读取 π；其实现与实验在
`TianchenIR-backend-certified-j-search` 分支。
