# run-001 证据审计

本文件只补充历史运行记录，不修改 `report.json`、candidate、Gate 裁决或 terminal verdict。

- 正式结果仍为 `NO_CLOSURE`。
- 该版本 runner 在 normal Replay response 被 validator 拒绝时没有保存 response body，并把三次
  实际 normal Replay provider request 低记为一次 `replay_physical_calls`。目录中只有一次成功
  进入 `ReplayGeneration` 的源码，不能据此重建另外两次被拒绝的完整回复。
- 后续 runner 已在 validator 外侧记录每次 response/receipt，并按 `ReplayAttempt.attempts` 统计
  physical calls；历史 JSON 保持原样，以免伪造当时实际产生的证据。

已知候选路径是：第一次 Lift schema 拒绝；三个正式 Gate1 round 中，一次 Replay 把 tanh realization
写成 sigmoid identity 而结构不闭合，另外两次 Replay 使用了当前固定 validator 不允许的直接
`libdevice` function import。这里不把任何一次失败重新解释为认证通过。
