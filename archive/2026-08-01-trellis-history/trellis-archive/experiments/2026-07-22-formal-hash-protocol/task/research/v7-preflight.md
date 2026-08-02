# Task10 Protocol-v7 Formal Preflight

## 结果

protocol-v7 的完整离线 preflight 已生成，并由 `verify_formal_preflight()` 独立复验：

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

执行命令：

```text
PYTHONPATH=python python -m intentir.evaluation.formal_preflight \
  test/real_llm_batch_28_v7.json \
  --output artifacts/real_llm_batch_28_v7/preflight.json \
  --cache-dir artifacts/real_llm_batch_28_v7/preflight_cache \
  --timeout 300 \
  --observed-date 2026-07-22
```

编译 cache 仅用于本次离线探测，复验后删除，不作为 formal evidence 提交。

## 冻结身份

- manifest digest：`66bf2db6af8b7c26d6ea04d045bc3e50eb1f64dcf667890e801c938621312a34`
- manifest file SHA：`fc9e8902b07da4687b67f1b5c11884e8dd73d3f3785a11b2f1e1e820a2df2c50`
- preflight schema：`intentir.real-llm-batch-preflight.v7`
- preflight file SHA：`0e7a6df61b2d11cfab4fb6e8092fed453c9a14897a474fe3777ec1654e988e2c`
- Lift wire：`intentir.lift-wire.v7`
- Lift schema SHA：`f7612a64c587cdcdfd07b754b9f8f881443520a4dc2dafb716aaadd1360444b7`
- Lift prompt SHA：`8f97ce5928725e837e69bf3a22cfea35a2ef8d42439e6344db4c52536115085a`

Swiglu source observer identity 与 v4--v6 不变：

- TTIR：`92e14ee61cbc4adce23bc6bf6813853d3983ff167732d40b1e2a934ce514f5fb`
- TTGIR：`75bc6af49a8d8d5b9e219d0036e60a29304029227f2d867b74aa156a516f4a16`

## Canary 前置条件

- preflight evidence 提交后不再修改 `lift_wire` 或 `gate1_code` identity 覆盖的文件；
- worktree clean，v7 Swiglu recovery output 尚不存在；
- predecessor 必须是 SHA
  `21fb0a2e548f75c619e0339664092defb554dacef5b6a132b54d5db0aed04b9d` 的 v6
  `lift_failed` receipt；
- fresh v7 最多一次 Lift；Lift 接受后最多一次 normal Replay；normal Gate1 hard-close 后才允许一次
  masked-J；
- 任一失败立即停止，不在 v7 内 retry，也不修复 response。
