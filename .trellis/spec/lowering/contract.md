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

当前 scalar RefEval 对显式内存边界实现三个保守投影：

- `index_expression` 必须是 rank-1 bare axis 或与 J tensor rank 一致的直接 `[axis0, ...]`；tensor
  shape 本身定义完整 logical domain，若 B 额外声明 `0 <= axis and axis < extent`，extent 必须与
  shape 精确相同；
- load/store 必须分别覆盖全部 logical input/output，并与所属 J node 的 operand/result 对应；
- 每个 access 必须有唯一且一致的 read/write effect，output mapping 必须与 J return 一致；
- `dense_functional_v1` 只接收无 predicate/fill 的上述关系；
- `logical_domain_mask_erasure_v1` 接收 pure functional identity 或 pointwise J：每个 tensor access
  必须是与 tensor rank 一致的 direct axes，所有被 mask 的 logical axes 在相关 load/store 上使用同一
  full-logical-domain predicate，predicate upper bound 与 domain/shape 精确一致，masked load 有显式
  fill，每个 output 独立 store 一次，且 observable store producer 的显式 result type 与 output 完全
  一致。logical scalar/guard 可以是 rank-0 J input，但 guard 必须真实进入 J dataflow。此时 fill 只
  存在于 logical domain 外且没有可观察 store，RefEval 可以机械擦除 mask/fill；
- `logical_domain_reduction_padding_v1` 只接收单 input、单 output、同 dtype、单轴 `sum`：一个 direct
  load 覆盖完整输入 domain，一个无 predicate 的 direct store 覆盖删去 reduction axis 后的输出，且
  access/effect/output mapping 均完整唯一；load predicate 只能约束被归约的 logical axis，masked-load
  fill、reduction identity 与 `masked_contribution` 必须都是 typed zero，单一物理 tile axis 必须是 0。
  J body 必须是 flat graph、以唯一 reduction 结束，所有前缀节点都进入 reduction slice。当前只接受
  closed `identity`/`mul` pointwise 前缀，并用常量传播机械检查 padding fill 经完整前缀后仍为 sum
  identity；不能证明 `h(fill)=identity` 时 fail closed；
- 不能满足上述 full-domain relation 的一般 pointwise padding、非 sum 或多轴 reduction、update
  padding、affine/gathered access、alias/inout 与一般 mutable storage 仍显式 unsupported。

这不是忽略 access/effect 后直接运行 A：mask erasure 必须先证明所有被擦除 lane 位于 J logical
domain 外且没有可观察写入；执行证据记录 predicated access/fill 数量。只有整个边界可机械投影到
完整逻辑张量求值时才进入 Intent CPU 路径，其他关系继续 fail closed。后续每扩展一种 boundary
relation，都必须同时增加其 denotation、正反例和真实 canary。

## Production 要求

- C 的消费只能影响合法 implementation choice，不能改变 J denotation；
- lowerer 不理解某个 C slot 时可以采用保守实现或显式 unsupported，但不能从 π/Source 猜测；
- authorization、alias/effect、logical update 与 allowed-result obligation 必须实际实现或拒绝；
- correctness 由独立 backend conformance 记录，不替代 Gate1/Gate2。

Intent MLIR 可以作为 J 的确定性编码，但不得成为携带 interface、C/π 语义副本或 Gate verdict 的
第二 wire truth。
