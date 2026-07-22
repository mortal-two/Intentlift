# Lift 与 Replay 合同

> 状态：当前派生规范。上位设计见 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。

## Lift

```text
Lift_H(Source) -> (J,C,pi)
```

Lift 直接完成 whole-tensor semantic lifting、C 分离与 source realization π 的提出。proof-relevant
输出只有 J、C、π。diagnostic、location、provider receipt 等可以作为 proof-irrelevant sidecar
保存，但不得进入 Replay、Gate verdict 或 production lowering。

J 必须包含 logical signature；不允许 root sibling `interface` 成为独立对象。实现若需要接口，
只能按需计算 `InterfaceView(J)`。H 可以向 Lift 提供匿名、未分类的 bindable coordinates，但
不得预先替 Lift 做 J/C/π 语义分类。

## Replay 前检查

```text
TypeCheck(J,C,pi)
```

最小 TypeCheck 必须机械检查：

1. 所有 J/C 引用存在；
2. route operand slot 与类型兼容；
3. observable instance 锚定 J-owned node/access/effect；
4. π 不含语义原子、Source/API、C value 或可执行自由文本；
5. sensitive refinement 引用 J/B authorization，或属于系统级安全白名单。
6. 未知字段、未知 relation 与自由文本扩展 fail closed。

TypeCheck 不做 SMT、range proof 或一般语义等价证明。检查失败时当前 artifact fail closed，
不进入 Replay。它只支撑 `FreshLogicalSemDecl(C,pi | J)=empty` 的显式所有权结论，不证明 π
没有统计语义线索，也不证明 realization 行为正确。

## Replay

```text
ReplayLLM(J,pi,MaskC(C);U_f) -> complete source R_f
```

`U_f` 是版本化、source-neutral 的公共前端文档与工具上下文，不按当前 Kernel/op/pattern/失败
样本选择。Replay 看不到 Source、Harness、C values、source parameter name/position、真实
BindingMap、独立 interface sidecar、source/compiler fingerprint、旧候选、旧 Replay、Gate1 diff、
旧模型 conversation context 或 per-kernel RuleCard。`MaskC(C)` 只暴露 C 的匿名 schema/slot
视图，不暴露 valuation 与私有 provenance。

Replay 必须生成完整、可独立构建的当前前端源码；固定代码只做绑定和检查，不根据 π 或 compiler
diff 施工。

## Gate1 失败与 relift

Gate1 的结构化 diff 只返回 source-aware Lift。下一轮 Lift 可以读取 Source、上一轮完整
`(J,C,π)` 与累计 Gate1 diff，但必须重新输出完整新三元组。随后使用新三元组发起全新的盲
Replay。

禁止把 diff 交给 Replay、patch 旧 Replay、只输出 artifact 增量，或跨候选复用 Replay。
最多三轮正式候选；仍不闭合则 `NO_CLOSURE`。
