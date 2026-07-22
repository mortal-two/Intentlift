# 退役实验控制面：formal hash/epoch protocol

本目录保存 2026-07-22 以前 formal batch v1–v7 的运行代码、manifest、测试、Task10 与实验
artifacts。该控制面依赖：

- Lift wire/file hash 锁；
- manifest digest、schema epoch 与 lineage receipt；
- 一次 normal Replay 加可选一次 masked-J 的固定预算；
- `masked hard-close -> LEAKAGE_REJECTED`；
- 独立 interface artifact 与旧 Replay packet。

这些规则已经与当前三份核心设计冲突，因此整体移出活动 Python 包、测试目录、`test/` 与
`artifacts/`。历史内容保留原貌，只用于追溯，不应被活动代码 import 或执行。

当前迁移要求见 [旧实现退役与首步重构](../../../spec/migration/legacy-removal.md)。下一实现应从：

```text
Lift_H(S) -> (J,C,pi)
TypeCheck(J,C,pi)
Replay = W_f(J,pi,MaskC(C);U_f)
```

重新建立 runner，而不是修改本目录中的旧 protocol。
