# masked_row_square_sum run-001

- provider: `gpt-5.6-sol` / `gpt-5.6-sol`
- main_status: `GATE2_FAILED`
- accepted_attempt: `3`
- j_necessity_status: `not_proved`
- call_budget: `{"formal_gate1_rounds": 3, "gate2_cases": 1, "lift_calls": 4, "lift_physical_calls": 4, "refeval_executions": 1, "replay_executions": 1, "replay_masked_calls": 1, "replay_normal_calls": 3, "replay_physical_calls": 4, "source_executions": 1, "typecheck_calls": 3}`
- gate2_status_counts: `{"reference_unsupported": 1}`

本目录中的 receipt/digest 仅用于普通追溯，不参与候选合法性或认证。

## 轮次解释

- Lift 物理调用 1：C slot role 违反纯 schedule schema，TypeCheck 前拒绝；
- 正式 Gate1 round 1（目录 `attempt-02`）：Replay compile failed，`c0` 被局部 alias 后不再被 Triton
  识别为 `tl.constexpr`；
- 正式 Gate1 round 2（目录 `attempt-03`）：Replay 错把两行向量化并在错误轴 reduction，hard diff；
- 正式 Gate1 round 3（目录 `attempt-04`）：TypeCheck 与 Gate1 closure 通过；
- Gate2：Source–Replay 通过，RefEval 因旧实现不支持 reduction padding 而
  `reference_unsupported`，因此历史 terminal 保持 `GATE2_FAILED`。

## 结束运行后的离线复验

新增保守 `logical_domain_reduction_padding_v1` 后，用 `attempt-04` 的原始 artifact/Replay 和同一 canary
输入离线复验；这不是历史 verdict 重判：

- RefEval 成功，profile 为 `logical_domain_reduction_padding_v1`；
- Source–Replay：通过，最大绝对误差 `0`；
- Source–J：通过，最大绝对误差 `0.0010986328125`；
- Replay–J：通过，最大绝对误差 `0.0010986328125`；
- 三条边均由既有 external Q 裁决。

正式结论仍需新的 `run-002` 从 Lift 开始完整重跑。
