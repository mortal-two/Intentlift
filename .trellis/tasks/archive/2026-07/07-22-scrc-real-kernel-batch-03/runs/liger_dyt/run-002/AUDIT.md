# run-002 证据审计

本文件只补充历史运行记录，不修改原始 failure 或 terminal verdict。

- Gate1 已经闭合，但旧异常路径只保存了准备阶段与 `failure.json`，没有保存 accepted artifact、
  TypeCheck、Gate1 或 provider call sidecar。
- Gate2 adapter 随后以 `Harness input bindings differ from accepted J inputs` 失败。原因是当时 runner
  尚未把 semantic constexpr 所锚定的 rank-0 J input 绑定给 RefEval。
- 因为 accepted candidate 证据不完整，本 run 只能保持 `INFRASTRUCTURE_FAILED`，不能在事后补成
  `SCRC_CERTIFIED`。相同问题在证据路径修复后的 run-003 被完整复现。
