# high_performance:liger_dyt fanout-001

- candidate_count: `5`
- selection_rule: `lowest_branch_after_gate1_search_barrier.v1`
- winner_branch_id: `1`
- pre_gate2_status_counts: `{"gate1_closed": 4, "no_closure": 1}`
- winner_main_status: `SCRC_CERTIFIED`

所有 branch 使用同一 Source/Harness/c0，但拥有独立 Lift、Replay、TypeCheck、Gate1 与 relift history。Gate2 只对预先选定的 winner 执行。
