# 2026-08-01 归档：Trellis 历史任务与退役流程

本目录保存 `.trellis/` 下已完成的任务记录与退役流程材料。仓库只借用 Trellis 的目录布局，
不使用其 CLI、状态机或 session 流程（见 `.trellis/README.md`）。

**没有任何代码读取这些文件**——归档是纯文件移动，不影响运行时。

## 内容

| 路径 | 数量 | 说明 |
|---|---:|---|
| `tasks-archive/` | 793 | 原 `.trellis/tasks/archive/`，含 2026-06/07/08 三批已结束任务 |
| `trellis-archive/` | 208 | 原 `.trellis/archive/`：退役 workflow、formal-hash 实验、pre-LLM-first spec |
| `workspace/` | 3 | 原 `.trellis/workspace/`，自述 "2026-07-21 起不再使用"，journal 为空 |

另删除 4 个空目录残留：`.trellis/tasks/07-31-{16,17,18,19}-*`（Task 16–19 归档后的空壳，
未被 git 跟踪）。

## 仍留在 `.trellis/` 的活动材料

- `spec/` — **当前派生规范**，12 个文件中 5 个在 2026-08-01 当天仍在更新；
  它把五份权威设计投影为 schema/API/runner 要求
- `tasks/08-01-final-three-kernel-closure/` — 当前唯一活动任务
- `tasks/README.md`、`README.md` — 索引

## 历史任务的解释规则

每份归档任务只按其**落盘时的协议版本**解释。后续 wire/TypeCheck/observer 升级不追认、
不改写、不重判旧 verdict——这条纪律在协议演进期间保护了证据的可信度，归档后依然适用。
