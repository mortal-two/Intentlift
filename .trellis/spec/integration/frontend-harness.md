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

前端 observer 可以随后增强，但证书必须记录 frontend、observer level/version 和 dump
可用性。
