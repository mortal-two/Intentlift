# Task08 Protocol-v5 Formal Preflight

## 结果

在 protocol-v5 manifest、Lift wire、runner lineage 与全量回归提交后，执行一次完全离线的
formal preflight：

```text
PYTHONPATH=python python -m intentir.evaluation.formal_preflight \
  test/real_llm_batch_28_v5.json \
  --output artifacts/real_llm_batch_28_v5/preflight.json \
  --cache-dir artifacts/real_llm_batch_28_v5/preflight_cache \
  --timeout 300 \
  --observed-date 2026-07-22
```

该构建函数没有 provider call path，receipt 再次记录：

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

随后用 `verify_formal_preflight()` 对同一 manifest 复验通过；它重新核对 manifest、Lift wire、
Gate1 code identity、environment 和 ordered case receipts，并没有调用 LLM。

## 冻结身份

- manifest schema：`intentir.real-llm-batch-28.v5`
- manifest digest：`bf6716ba75c0087a506031101a5628f30c06c29d08dfdaf60ef5a0bfcea5ceeb`
- manifest file SHA：`98056f9035d839e5d51f8b209accf0806606b7fbab2bca283a302c50265716cf`
- preflight schema：`intentir.real-llm-batch-preflight.v5`
- preflight file SHA：`f0c93e6230914a9f750173ef6853315b9a1df9c212c4592e0a258a6f48222b6e`
- Lift wire：`intentir.lift-wire.v5`
- Lift schema SHA：`1cff7d55922ee0e91c212ac1964af066984a92513ab24d54d7900dbedfe66aba`
- Lift prompt SHA：`10a7a58b8a4482f33a33dcf732d43e995ce2da6cc3f676e42c331bc459cae4fb`

Swiglu source observer 仍与 v4 完全相同：

- TTIR SHA：`92e14ee61cbc4adce23bc6bf6813853d3983ff167732d40b1e2a934ce514f5fb`
- TTGIR SHA：`75bc6af49a8d8d5b9e219d0036e60a29304029227f2d867b74aa156a516f4a16`

这证明 v5 没有改变 source invocation 或 compiler observation；新 epoch 的唯一实验变量是 π
graph-reference contract/wire。

## Canary 前置约束

- 提交 preflight 后不得修改其 `gate1_code` 或 `lift_wire.client_files` 覆盖的文件；
- worktree 必须干净；
- output root 中不得已有 v5 Swiglu receipt；
- predecessor 必须是 SHA
  `98a3efbc9b9027e2480e68347e053ea62ce7014b6f34db95b707fd39364573d2`
  的 v4 `lift_failed` receipt；
- fresh v5 最多一次 Lift；合法后最多一次 normal Replay；normal strict hard-close 后才允许
  一次 masked-J；任何失败都停止且不得在同一 epoch 重试。
