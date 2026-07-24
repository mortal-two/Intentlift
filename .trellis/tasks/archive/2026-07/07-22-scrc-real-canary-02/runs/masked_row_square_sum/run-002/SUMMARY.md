# masked_row_square_sum run-002

- provider: `gpt-5.6-sol` / `gpt-5.6-sol`
- main_status: `NO_CLOSURE`
- accepted_attempt: `None`
- j_necessity_status: `NOT_RUN`
- call_budget: `{"formal_gate1_rounds": 3, "gate2_cases": 0, "lift_calls": 3, "lift_physical_calls": 3, "refeval_executions": 0, "replay_executions": 0, "replay_masked_calls": 0, "replay_normal_calls": 3, "replay_physical_calls": 3, "source_executions": 0, "typecheck_calls": 3}`

本目录中的 receipt/digest 仅用于普通追溯，不参与候选合法性或认证。

## 三轮诊断

- round 1：TypeCheck 通过、Replay 编译成功，但把 row program coordinate 实现成长度 2 的 lane
  range；因此缺少 `tt.get_program_id`，多出 broadcast/expand/range，reduction axis 从 Source 的
  0 变为 Replay 的 1，Gate1 hard diff；
- round 2：TypeCheck 通过；Replay 执行 `i3 = c0; tl.arange(0, i3)`，`i3` 不再是 Triton 可识别的
  `tl.constexpr`，编译失败；
- round 3：TypeCheck 通过；新的 Replay 再次执行 `i4 = c0; tl.arange(0, i4)`，同类编译失败；
- 三轮耗尽后为 `NO_CLOSURE`，未进入 Gate2，不能据此判断新 reduction RefEval。

结束本次运行后，项目在共享 π grammar 与 Triton `U_f` 中明确 program-base/lane-coordinate 的
机械投影，并明确 compile-time-only API 直接消费 `c*`。历史 verdict 保持不变；修正后需全新
run-003。单个 `c*` realize 由参数 binding/direct use 满足，不需要恒等 alias；J logical reduce axis
与 B physical `tile_axis` 也在同一公共合同中明确区分。
