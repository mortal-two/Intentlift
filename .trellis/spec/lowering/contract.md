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

两条路径在分叉前必须共享唯一标准化前端：

```text
response JSON -> Decode/Elaborate -> J
                                    |
                                    v
                               Intent MLIR
                                |       |
                                |       `-> target lowering(J,C,target)
                                `-> scalar CPU RefEval(J)
```

Decode/Elaborate 将 access index、predicate/domain ContractExpr、唯一可推导 result type 与恒等 output
binding 变成 typed canonical form。RefEval 和 production backend 均不得重新读取或各自解释响应中的
字符串；`J -> Intent MLIR` 只有一个共享入口。未接受的响应 JSON 不进入 lowering，解析缓存也
不是另一份 J，不增加 artifact、Agent 或 Gate。

## 共享 Functional Boundary Projection

RefEval 与 production lowerer在发射 Intent MLIR 前，共用同一个确定性内部分析：

```text
P_B(J) = ProjectFunctionalBoundary(J)
```

`P_B(J)` 只把唯一 J 中已经存在的 typed domain/access/effect/output/reduction relation 投影为 consumer
可直接使用的 structured axes、predicate contract、broadcast axes 与 index-bounds handles。它不改写
J，不读取 C/π/Source/Harness/Gate evidence，不补 predicate 或 operator，也不单独序列化；它是一次
调用内的派生分析结果，不是 `CanonicalJ`、第四 artifact 或新 certificate。

当前 `P_B` 只有在完整 B 可机械表示为 J 的完整 logical-tensor execution 时成功：pure value、dense
functional、full-logical-domain pointwise mask erasure，以及文中定义的窄 sum-padding profile。任何
affine-as-dense、predicate/domain mismatch、effect mismatch、alias/inout 或未实现的 mutable relation
都在进入 codegen 前 fail closed。RefEval 与 CUDA analyzer 记录相同的 projection record，不能各自
保留第二套 wire parser 或 B-consumption 判定。

投影成功的准确含义只是：当前 logical Intent MLIR 可以完整代表这些 B relation；它不自动证明任意
目标物理调度安全。若 target lowerer后来把精确 logical domain 扩张为 padded launch，必须在自己的
lowering 中物化 bounds/mask，或给出等价的目标侧证明；不能仅凭 `logical_domain_mask_erasure_v1`
删除可能越界的物理 load/store。最终 target correctness 仍由独立 backend conformance 裁决。

## RefEval 要求

- 逐项消费 J logical signature、A semantics 与 B result relation；
- 对 unsupported op/relation 在执行前给出具体 capability reason 并 fail closed；
- 不使用 case-specific tolerance、Source/Replay 输出或 C value 决定结果；
- 输出只交给外部比较政策 Q 构造的 Gate2 comparator 裁决；Q 不进入 RefEval。

当前 scalar RefEval 对显式内存边界实现三个保守投影：

- 每个 access 满足 canonical fill 合同：`fill_value != null` 蕴含它是带非空 predicate 的 load；
  无条件 access 的 predicate/fill 均为空。orphan fill 是非法 J，必须由 TypeCheck 及消费者 fail closed，
  不得由 RefEval 静默删除或从 domain/index 猜测未声明 predicate；
- J 中 structured index 必须是与 J tensor rank 一致的 direct ordered axes；响应中的 bare、
  bracketed 与 unbracketed-comma surface 差异在此之前已经消失；tensor
  shape 本身定义完整 logical domain，若 B 额外声明 `0 <= axis and axis < extent`，extent 必须与
  shape 精确相同；
- load/store 必须分别覆盖全部 logical input/output，并与所属 J node 的 operand/result 对应；
- 每个 access 必须有唯一且一致的 read/write effect，output mapping 必须与 J return 一致；
- `dense_functional_v1` 只接收无 predicate/fill 的上述关系；
- `logical_domain_mask_erasure_v1` 接收 pure functional identity 或 pointwise J：每个 tensor access
  必须是与 tensor rank 一致的 direct axes，所有被 mask 的 logical axes 在相关 load/store 上使用同一
  full-logical-domain predicate，predicate upper bound 与 domain/shape 精确一致，masked load 有显式
  fill，每个 output 独立 store 一次，且 observable store producer 的 elaborated result type 与 output
  完全一致。该类型可以由 closed op signature 唯一物化，不要求 Lift 重复填写。logical scalar/guard
  可以是 rank-0 J input，但 guard 必须真实进入 J dataflow。此时 fill 只
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
- production analyzer 必须先成功取得同一个 `P_B(J)` 才能发射 logical Intent MLIR；projection
  失败不得继续 codegen；
- lowerer 不理解某个 C slot 时可以采用保守实现或显式 unsupported，但不能从 π/Source 猜测；
- authorization、alias/effect、logical update 与 allowed-result obligation 必须实际实现或拒绝；
- correctness 由独立 backend conformance 记录，不替代 Gate1/Gate2。

## J-facing executable CUDA boundary

production compile 与 execution 的正式接口是：

```text
L = LowerCUDA(LoweringRequest(J,C,target), concrete_extents)
O_target = ExecuteCUDA(L, logical_inputs)
```

`L` 已持有生成它的唯一 J、确定性 Intent MLIR 与已编译程序；execution 不再接受另一份 J 或
interface/BindingMap sidecar。调用者必须用 `J.intent.inputs` 的 logical handles 提供精确 input set；
缺失、额外或 legacy `argN` positional fallback 在进入底层 runtime 前拒绝。shape/dtype/device 由从
同一个 J 发射的静态 MLIR ABI机械校验。底层按 return 顺序产生的 `out0/out1/...` 只能在 backend
内部一一映射回 `J.intent.outputs` handles，不能泄漏为上层接口或由 Harness 猜测。

execution 必须复用 `L` 中已编译的 cubin，不得在运行阶段重新生成 J、替换 MLIR 或重新调用
compiler。其正确性关系独立写成：

```text
Close_Q(ExecuteCUDA(L,x), RefEval(J,x))
```

这是 backend conformance，不是第四条 Gate2 edge，也不改变 `SCRC_CERTIFIED`。当前正式 J-facing
CUDA API 对 alias/inout 继续 fail closed；若以后开放，destination/alias mapping 必须直接来自
J.signature/B，而不能复用 Source Harness binding。

Intent MLIR 可以作为 J 的确定性编码，但不得成为携带 interface、C/π 语义副本或 Gate verdict 的
第二 wire truth。
