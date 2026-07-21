# `.trellis` 目录说明

本项目只借用 `.trellis/spec`、`.trellis/tasks`、`.trellis/workspace` 的文件布局，不使用
Trellis workflow、task CLI、developer session 或自动状态机。

- `spec/`：当前规范性设计；
- `tasks/`：普通 Markdown 工作项；
- `workspace/`：可选的人类工作记录；
- `archive/`：历史快照，非规范。

旧 `workflow.md`、`scripts/` 与 `config.yaml` 已移入
[archive/2026-07-21-retired-workflow](archive/2026-07-21-retired-workflow/README.md)。任何任务
不得以未运行 Trellis 脚本为由阻塞实现。
