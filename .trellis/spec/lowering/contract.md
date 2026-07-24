# RefEval 与 Contract-Preserving Multi-Lowering 合同

> 状态：当前派生规范。上位设计见
> [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与
> [后端最终设计](../../../doc/IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)。

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
system orchestration 只把 SCRC-accepted artifact 的 J+C 交给 production；backend API 不读取 Gate
verdict，也不能自行接受未通过上游协议的 response。

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

RefEval 与 production lowerer 在发射 Intent MLIR 前，共用同一个确定性内部分析：

```text
P_B(J) = ProjectFunctionalBoundary(J)
```

`P_B(J)` 只把唯一 J 中已经存在的 typed domain/access/effect/output/reduction relation 投影为 consumer
可直接使用的 structured axes、predicate contract、broadcast axes 与 index-bounds handles。它不改写
J，不读取 C/π/Source/Harness/Gate evidence，不补 predicate 或 operator，也不单独序列化；它是一次
调用内的派生分析结果，不是 `CanonicalJ`、第四 artifact 或新 certificate。

当前 `P_B` 只有在全部影响 RefEval/target logical execution 的 functional B facts 可机械表示为 J 的
完整 logical-tensor execution 时成功：pure value、dense
functional、full-logical-domain pointwise mask erasure，以及文中定义的窄 sum-padding profile。任何
affine-as-dense、predicate/domain mismatch、effect mismatch、alias/inout 或未实现的 mutable relation
都在进入 codegen 前 fail closed。π 专用的 `realization_authorizations` 不属于该 functional
projection。RefEval 与 CUDA analyzer 记录相同的 projection record，不能各自
保留第二套 wire parser 或 B-consumption 判定。

投影成功的准确含义只是：当前 logical Intent MLIR 可以完整代表这些 B relation；它不自动证明任意
目标物理调度安全。若 target lowerer后来把精确 logical domain 扩张为 padded launch，必须在自己的
lowering 中物化 bounds/mask，或给出等价的目标侧证明；不能仅凭 `logical_domain_mask_erasure_v1`
删除可能越界的物理 load/store。target correctness 来自 materializer、pass contract 与 target
lowerer 的 compiler contract；独立 backend conformance 只做 CI/debug/evaluation 审计。

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

- C 只影响已经由 pass contracts 判定为合法的 candidate 的初始化、搜索顺序、优先邻域与有限预算
  下的访问集合；不参与 legality，也不改变 J denotation；
- production analyzer 必须先成功取得同一个 `P_B(J)` 才能发射 logical Intent MLIR；projection
  失败不得继续 codegen；
- lowerer 不理解某个 C slot 时可以采用保守实现或显式 unsupported，但不能从 π/Source 猜测；
- alias/effect、logical update 与当前 `exact/tolerance/one_of_updates` obligation 中所有影响 target
  denotation、safety 或 ABI 的 backend-relevant B facts 必须实际实现或拒绝；
- `realization_authorizations` 的既有 consumer 是 SCRC TypeCheck/π；backend 只识别其 SCRC-only
  身份，不把它当 target permission，也不因其存在自动 unsupported；
- production backend 不读取 π、Gate/Q、Source、kernel name 或 per-kernel RuleCard；
- correctness 由 compiler contracts 支撑；独立 backend conformance 不替代这些合同，也不替代
  Gate1/Gate2。

## Deterministic materialization

对固定 materializer version `v`：

```text
M0 = Emit_v(J)
```

同一 J 与 v 必须得到同一 Intent MLIR。`Emit_v` 严格不读取 C，完整编码 A，并把 backend-relevant
B 投影为 structured dataflow、closed attrs、effects、verifier facts 或 ephemeral analyses。它必须
识别 SCRC-only realization authorization 的身份，但不把它 materialize 成 target permission。

得到 M0 后，backend 才执行：

```text
SearchSeed_C(M0) -> discardable C annotations / candidate ordering
```

该步骤只产生 backend internal search state，不修改 M0/J，也不产生新 artifact。

Intent MLIR 是 J 的确定性 typed encoding 加 compiler internal state，不是第二份 J。structured ops
必须保留到 implementation-family 选择完成；未接受 response、π、Source 或 Gate evidence 不得进入。

## Implementation judgement 与 pass contract

定义：

```text
Gamma_(J,t) |- M : Impl_t(J)
```

`Gamma_(J,t)` 只由 J、target capability 与版本化 compiler contract 构造。它是 ephemeral、
non-serialized analysis，没有独立语义所有权，不成为 backend 输入或 receipt truth，不进入 SCRC，
也不能反向改写 J。每个 transformation
`rho` 必须声明 matched A structure、required B facts、target/current-state precondition、转换结果、
preservation class 与可搜索参数，并满足：

```text
Gamma_(J,t) |- M : Impl_t(J)
and Pre_rho(J,M,t)
=> Gamma_(J,t) |- rho(M) : Impl_t(J)
```

缺少 pass 所需 B fact 时，该 pass 不适用。系统不得从 Gate2 tolerance、运行结果、Source/π 或
kernel identity 补足前提。`realization_authorizations` 服务 π，不自动成为 target permission。

## Generic implementation families

backend 可以按 A 的通用 op/region/graph pattern 识别：

- sequential/vector/warp/block/hierarchical reduction；
- producer-consumer fusion、fission 与 intermediate elimination；
- scan、contraction、conv 与 structured recurrence family；
- 有独立 preservation contract 的 algorithmic rewrite。

family 是 source-neutral compiler library，不是 named-kernel rule。non-trivial identity，例如 online
recurrence 到 ordinary two-pass，必须由单独 transformation 及 B preconditions 支撑，不能伪装成
schedule tuning。

## Search space 与 C ordering

定义 MLIR compiler-state 合法空间；M 只表示 Intent/lower-IR state，J 是不可变参数：

```text
Reach^MLIR_t(M0; Gamma_(J,t)) = {
  M | M0 ==rho1,...,rhok==> M
      and every transition preserves Impl_t(J)
}
```

下文简写为 `Reach_t(J)`。

C 不定义 `Reach_t(J)`，只提供：

```text
Order_C(Reach_t(J))
```

即 source-like 初值、第一批参数、优先邻域与搜索顺序。target-specific legal parameter domain 属于
implementation family/target capability；`required` C slot 不表示 target schedule fixed。搜索产生的
assignment 与 lineage 是 backend internal state，不产生 `C_target` 或新 Agent/artifact。

对有限预算 b：

```text
Visited_(t,b)(J;C) is a subset of Reach_t(J)
```

C 可以改变 `Visited` 的顺序和有限预算覆盖，但不能改变 `Reach` 中哪些状态合法。

搜索器可使用 enumeration、beam、e-graph、bounded superoptimization、cost model、profiling 或
autotuning。搜索顺序可以启发式，但 correctness 只取决于可达空间对 `Impl_t(J)` 闭合。

## Performance-only selection

对有限预算内实际访问的合法 candidate：

```text
M*_(b,C) = argmin_{M in Visited_(t,b)(J;C)} Cost_t(Lower_t(M))
```

Cost 可以来自静态模型或测量。运行只选择性能，不判断语义合法性。Q 不进入 production backend；
Gate2 `tolerance` 不授权改变 reduction tree、accumulation dtype、FMA、rounding 或 transcendental
implementation。

backend conformance：

```text
Close_Q(Exec(Lower_t(M),x), RefEval(J,x))
```

只用于 CI、debug、new-family regression 与 evaluation audit，不是 production legality oracle，也不是
SCRC 的第四条 Gate2 edge。

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
compiler。其独立 conformance 审计关系写成：

```text
Close_Q(ExecuteCUDA(L,x), RefEval(J,x))
```

这是 backend conformance，不是 compiler pass 的 legality oracle，不是第四条 Gate2 edge，也不改变
`SCRC_CERTIFIED`。当前正式 J-facing
CUDA API 对 alias/inout 继续 fail closed；若以后开放，destination/alias mapping 必须直接来自
J.signature/B，而不能复用 Source Harness binding。

Intent MLIR 可以作为 J 的确定性编码，但不得成为携带 interface、C/π 语义副本或 Gate verdict 的
第二 wire truth。
