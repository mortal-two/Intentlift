# RefEval 与 Production Lowering 合同

> 状态：当前派生规范。上位设计见 [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)。

## 两条不同路径

Gate2 reference 与 production lowering 不是同一个消费者：

```text
RefEval      : J x logical_inputs -> observable_outputs
Lower_target: J x C x target      -> executable
```

`RefEval` 是简单、无优化、J-only 的语义路径。其函数类型不存在 C 或 π；它不能从 implementation
selection、source realization、Harness 或历史 artifact 补语义。

production lowerer 必须消费完整 J，并可消费或安全忽略 C。它不读取 π，因为 π 只证明 Source
realization 与 Replay closure，不是目标 backend 的施工输入。

## RefEval 要求

- 逐项消费 J logical signature、A semantics 与 B result relation；
- 对 unsupported op/relation 在执行前给出具体 capability reason 并 fail closed；
- 不使用 case-specific tolerance、Source/Replay 输出或 C value 决定结果；
- 输出只交给外部比较政策 Q 构造的 Gate2 comparator 裁决；Q 不进入 RefEval。

当前 scalar RefEval 对显式内存边界只实现保守的 `dense_functional_v1` 投影：

- `index_expression` 必须是与 J tensor rank 一致的直接 `[axis0, ...]`；tensor shape 本身定义完整
  logical domain，若 B 额外声明 `0 <= axis and axis < extent`，extent 必须与 shape 精确相同；
- load/store 必须分别覆盖全部 logical input/output，并与所属 J node 的 operand/result 对应；
- 每个 access 必须有唯一且一致的 read/write effect，output mapping 必须与 J return 一致；
- predicate/fill、affine/gathered access、update、alias、一般 mutable storage 仍显式 unsupported。

这不是忽略 access/effect 后直接运行 A：只有整个边界可机械投影到完整逻辑张量求值时才进入
Intent CPU 路径，其他关系继续 fail closed。后续每扩展一种 boundary relation，都必须同时增加
其 denotation、正反例和真实 canary。

## Production 要求

- C 的消费只能影响合法 implementation choice，不能改变 J denotation；
- lowerer 不理解某个 C slot 时可以采用保守实现或显式 unsupported，但不能从 π/Source 猜测；
- authorization、alias/effect、logical update 与 allowed-result obligation 必须实际实现或拒绝；
- correctness 由独立 backend conformance 记录，不替代 Gate1/Gate2。

Intent MLIR 可以作为 J 的确定性编码，但不得成为携带 interface、C/π 语义副本或 Gate verdict 的
第二 wire truth。
