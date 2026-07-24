# masked_row_square_sum run-003

- provider: `gpt-5.6-sol` / `gpt-5.6-sol`
- main_status: `SCRC_CERTIFIED`
- accepted_attempt: `2`
- j_necessity_status: `not_proved`
- call_budget: `{"formal_gate1_rounds": 2, "gate2_cases": 1, "lift_calls": 3, "lift_physical_calls": 3, "refeval_executions": 1, "replay_executions": 1, "replay_masked_calls": 1, "replay_normal_calls": 2, "replay_physical_calls": 3, "source_executions": 1, "typecheck_calls": 2}`
- gate2_status_counts: `{"gate2_pass": 1}`

本目录中的 receipt/digest 仅用于普通追溯，不参与候选合法性或认证。

## 轮次与认证细节

- Lift call 1：π refinement `auth_ref` 不是 J authorization handle `u*`，TypeCheck 前拒绝；
- 正式 Gate1 round 1（目录 `attempt-02`）：已正确使用 `tl.program_id(0)` 与直接 `c0`，但二维 lane
  加 `sum(axis=1)` 产生 block value并写入 scalar pointer，Replay 编译失败；
- 正式 Gate1 round 2（目录 `attempt-03`）：一维 lane、`sum(axis=0)`、scalar store，全部 Gate1
  structural categories hard-close；
- Gate2 RefEval profile：`logical_domain_reduction_padding_v1`；
- Source–Replay：通过，最大绝对误差 `0`；
- Source–J：通过，最大绝对误差 `0.0010986328125`；
- Replay–J：通过，最大绝对误差 `0.0010986328125`；
- masked-J 也 hard-close，故 `j_necessity_status=not_proved`；该独立诊断不改变
  `SCRC_CERTIFIED`。
