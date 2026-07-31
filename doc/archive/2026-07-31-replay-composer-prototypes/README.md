# 2026-07-31 Replay Composer 原型归档

本目录保存 Task 16/17 的两条已停止路线：

- flat/grouped source-fragment composer；
- typed skeleton、per-block target binding 与 fixed structural renderer。

它们证明了 route 查询、持久化 coding episode、own-compile/own-TTIR、checkpoint 等工具有价值，
但没有显著降低 Replay agent 的编译负担。最新 FlashAttention-forward file-native pilot 虽形成可编译
Replay，正式 Gate1 仍有 232 项 dataflow 差异，且 Agent 仍需手写大部分目标语义源码。

该目录描述的后继 Task 18 路线也已在真实 Flash-forward 上证明仍然过重，并另行归档到
`../2026-07-31-direct-replay-compiler-agent/`。当前活动路线见 Task 19。

历史实现由 Git 提交 `81b873aa`、`a0fbddea`、`e3e9ff7b`、`7590f753` 以及 Task 17 交接提交保留，
不复制成活动代码的第二份真值。归档材料不得被当前实现或 spec 当作规范输入。
