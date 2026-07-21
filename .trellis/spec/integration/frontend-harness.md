# 开放前端最小合同

新增前端只要求一个薄 adapter：

```text
Build(source, specialization, C) -> artifact
Run(artifact, logical_inputs) -> execution
Observe(execution) -> outputs/inout/effects
OptionalDump(artifact) -> observer input
```

首次接入不要求：

- J→frontend lowerer；
- frontend API mapping table；
- source→π parser；
- RuleCard 或 StrategyFamily；
- 修改 Intent schema；
- 强 Level-2 compiler observer。

真实参数名、调用约定和设备对象只在 adapter 内部。对 J 使用有序匿名参数；相同逻辑
输入按位置供给 Source、Replay 和 `E_J`。

该 positional binding 必须核对 arity、tensor/scalar kind、rank、dtype、constexpr 位置和
C 值，但不保存一份带语义的 ABI schema。compiler-only C slot 由纯调度 `role` 连接固定
frontend option；连接表不按 kernel/op/pattern 变化。

`constexpr` 位置也不能自动归 C。冻结 invocation 用一份按源码声明顺序排列的 parameter
数组作为唯一物理调用真值；每个 constexpr 只标为 `binding` 或 `schedule`：

- `schedule` 才投影成 pure C；
- `binding` 使用匿名 `p*`，具体值仅供 source/replay 同一调用。Lift 根据源码把它的 semantic
  consequence 写入 J，或把 representation-only use 写入 pi。

该二分类只区分“是不是 C”，不在 harness 中建立 J/π 语义对象。参数数组是编译/运行所需的
一次调用记录，不是 `LogicalABI`，也不包含 kernel rule。

前端 observer 可以随后增强，但证书必须记录 frontend、observer level/version 和 dump
可用性。
