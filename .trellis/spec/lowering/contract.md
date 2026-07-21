# Lowering 合同

## Intent MLIR

J 到 Intent MLIR 是确定性的 schema/dialect 编码，不负责从 C/π 推导 source realization，
也不执行 Kernel family 识别。

## Scalar semantic executor

`E_J` 直接消费完整 J，是 Gate2 的简单、无优化、backend-neutral 语义路径。其 primitive
实现、版本和 hash 属于 Gate2 TCB。

## Production backend

```text
Lower_target(J, C?, target) -> executable
```

backend 必须接收完整 J；C 是独立可选/可重调 hint。不能只接收 A，也不能依赖 Gate
verdict、provider、source path、dataset identity 或 π 来补语义。

同一 J 对象直接跨 Gate1、Gate2 与 production backend 传递。backend 若不能实现某个 B
relation，必须在 codegen 前列出 unsupported relation 并失败；记录 digest 但不实际检查或
实现不算 consumption。

C 只采用 canonical `{slots, values}` schema。slot 名是匿名 handle，`role` 是开放但
source-neutral 的纯调度名；frontend/backend 可用固定的 role→公共编译选项映射消费它，
不得用 kernel/source pattern 选择映射。

production correctness 通过独立 backend conformance 记录，不进入 Gate1/Gate2 verdict。
运行调用 binding 只能是 harness 的轻 positional slots，不保存 dtype/mode/rank 等第二份
语义接口，也不需要持久化 LogicalABI。
