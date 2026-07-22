# Task09 Protocol-v6 Formal Preflight

## 结果

protocol-v6 的完整离线 preflight 已构建并由 `verify_formal_preflight()` 复验：

```text
online_llm_calls       = 0
source_pins_verified   = 28
symbol_pins_verified   = 28
triton_compiled        = 26 / 26
cute_compiled          = 2 / 2
source_observer_ready  = 28 / 28
replay_arm_ready       = 28 / 28
overall_ready          = true
```

命令：

```text
PYTHONPATH=python python -m intentir.evaluation.formal_preflight \
  test/real_llm_batch_28_v6.json \
  --output artifacts/real_llm_batch_28_v6/preflight.json \
  --cache-dir artifacts/real_llm_batch_28_v6/preflight_cache \
  --timeout 300 \
  --observed-date 2026-07-22
```

## 冻结身份

- manifest digest：`c3144d17dc7a691dbe7127e8fea2aeb6be526cfb8788239ecf9466b27142212d`
- manifest file SHA：`162c86ef069434ae74c993d6003eb127de0b75382910481861491b06e4e84e95`
- preflight schema：`intentir.real-llm-batch-preflight.v6`
- preflight file SHA：`ffda048dfdf00738359e4fe335d9b6d188bcd7590fdcc9297f7cf9e2052fec52`
- Lift wire：`intentir.lift-wire.v6`
- Lift schema SHA：`10834a38fb7f7bc00228599ec123d63fff2f25d017fd01247c0fb5a56b499393`
- Lift prompt SHA：`51176bb9b508b4d22482f597ea8cd27debb6da908cdcfe58c4d28a7f9fe7f6d2`

Swiglu source TTIR/TTGIR SHA 与 v4/v5 相同：

- TTIR：`92e14ee61cbc4adce23bc6bf6813853d3983ff167732d40b1e2a934ce514f5fb`
- TTGIR：`75bc6af49a8d8d5b9e219d0036e60a29304029227f2d867b74aa156a516f4a16`

所以唯一实验变量是 Lift/Replay 可见的 address mechanism contract；source、specialization 和
observer 未变化。

## Canary 前置约束

- preflight 提交后不再修改受 identity 覆盖的代码；
- worktree clean，v6 Swiglu output 不存在；
- predecessor 是 SHA
  `5a286725e22f1fadc83771eb460a26e823e0c9675606bde311412563f2921b71`
  的 v5 `gate1_failed` receipt；
- fresh v6 最多 1 Lift + 1 normal Replay；normal hard-close 后才允许 1 masked-J；
- 任一失败立即停止，不在 v6 重试。
