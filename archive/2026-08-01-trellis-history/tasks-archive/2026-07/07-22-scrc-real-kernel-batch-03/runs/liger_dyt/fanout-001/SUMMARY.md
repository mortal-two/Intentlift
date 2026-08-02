# high_performance:liger_dyt fanout-001

- candidate_count: `5`
- selection_rule: `lowest_branch_after_gate1_search_barrier.v1`
- winner_branch_id: `None`
- pre_gate2_status_counts: `{"infrastructure_failed": 1, "no_closure": 4}`

所有 branch 使用同一 Source/Harness/c0，但拥有独立 Lift、Replay、TypeCheck、Gate1 与 relift history。Gate2 只对预先选定的 winner 执行。
