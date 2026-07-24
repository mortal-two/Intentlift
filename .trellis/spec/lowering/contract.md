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
                                |       `-> Attach_C(M0,C,t) -> target search/lowering
                                `-> scalar CPU RefEval(J)
```

Decode/Elaborate 将 access index、predicate/domain ContractExpr、唯一可推导 result type 与恒等 output
binding 变成 typed canonical form。RefEval 和 production backend 均不得重新读取或各自解释响应中的
字符串；`J -> Intent MLIR` 只有一个共享入口。未接受的响应 JSON 不进入 lowering，解析缓存也
不是另一份 J，不增加 artifact、Agent 或 Gate。

## 共享 Functional Boundary Projection

RefEval 与 production lowerer 在发射 Intent MLIR 前，共用同一套 deterministic
functional-boundary core 与 closed rules，但不是同一函数签名或同一次 analysis invocation：

```text
P_B^prod(J) = ProjectFunctionalBoundary(J)
P_B^ref(J,x,d) = ProjectFunctionalBoundaryForRefEval(J,x,d)
```

production 只能调用 value-free 的 `P_B^prod(J)`。它把唯一 J 中已经存在的 typed
domain/access/effect/output/reduction relation 投影为 consumer
可直接使用的 structured axes、predicate contract、broadcast axes 与 index-bounds handles。它不改写
J，不读取 C/π/Source/Harness/Gate evidence，不补 predicate 或 operator，也不单独序列化；它是一次
调用内的派生分析结果，不是 `CanonicalJ`、第四 artifact 或新 certificate。

主 SCRC runner 负责先按 pre-Gate2 evidence 选定 winner，再把该 Gate2 invocation 的 J logical
inputs 与完整 closed dimension mapping 交给 scalar RefEval 的 `P_B^ref(J,x,d)`；projection/RefEval
library 本身不选择 winner，也不取得跨 invocation 调用权限。该入口只用 J 自身 rank-0 logical
`i1`/integer values 逐 access 证明 J predicate 在完整 logical domain 恒真。结果仍是
invocation-local ephemeral analysis，不修改 J、不读 Source/C/π、不进入 Lift feedback。production
API 没有 `x,d` 参数，也不得读取或复用 v3 projection 中的 runtime-proof subreceipt。

shared core 只有在全部影响 RefEval/target logical execution 的 functional B facts 可机械表示为 J 的
完整 logical-tensor execution 时成功：pure value、dense
functional、full-logical-domain mask erasure，以及由每个 reduction contract 闭合的通用 padded
functional graph。任何
affine-as-dense、predicate/domain mismatch、effect mismatch、alias/inout 或未实现的 mutable relation
都在进入 codegen 前 fail closed。π 专用的 `realization_authorizations` 不属于该 functional
projection。RefEval 与 CUDA analyzer 使用同一 protocol/core output structure；v10 RefEval record
可以额外含 invocation-local runtime-proof subreceipt，production record 必须为空。二者不能各自
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
  完全一致。该类型可以由 closed op signature 唯一物化，不要求 Lift 重复填写。rank-0 logical
  scalar/guard 仍可构成合法 J；value-free production projector 对 guard-bearing access predicate
  继续显式 unsupported。RefEval v10 只在当前 invocation 中按 J predicate 的真实极性、closed axis
  与 dimension extent 证明其全域恒真后才可擦除；operand membership 不能替代该 proof；
- `logical_domain_reduction_padding_v2` 不再把合法图限制为单 input、单 output、唯一单轴 `sum`。
  每个带 physical predicate 的 direct access 必须为自身每个非常量 `0` 的 axis 显式声明完整零起点
  logical domain；所有仍携带某个 masked axis 的 load/store 必须声明同一个 full-domain predicate，
  predicated load 必须显式给出 fill，且 access/effect/output relation 仍完整唯一。被 reduction 消去的
  axis 不要求出现在降秩 output store 上；仍存在于 output 的 masked axis 必须继续由 store predicate
  保护；
- padded graph 中每个 logical `reduce` 必须恰有一个 `ReductionBoundary`。`identity` 若存在必须是该
  reducer 的代数单位元，`masked_contribution` 必须显式存在且为单位元；当前闭合集合包括
  `sum/prod/max/min/any/all`，其中 max/min 接收封闭的 JSON-safe infinity sentinel。若 J-only
  access/dataflow 能证明被 mask 的 axis 最终由 `mean` 消去，则因缺少 logical count contract 继续
  unsupported。`masked_contribution` 按 J 的既有定义记录经过任意普通
  pointwise/broadcast dataflow 后到达该 reduction 的有效 masked contribution，因此 projector 不从
  Source、C、hidden input 或 Kernel 公式重新推导它；
- 每个 pointwise operand 必须按 closed value type 满足普通 NumPy broadcast，或消费 J 中显式的
  non-trailing `boundary.broadcasts` relation。于是 max → sub → exp → sum → div 等多 reduction 图和
  普通 pointwise graph 走同一个规则，而不是 Softmax/named-op 分支；
- masked-axis provenance 只沿当前显式建模的 unary/cast、binary/select、transpose 与 reduce 规则传播。
  一旦已有 provenance 进入 contraction 或其他尚未建模的 executor op，projector 必须在该 op 处
  consumer-unsupported，不能静默停止传播后让下游 `mean` 或 output 被当成无 mask；
- 不能满足上述 full-domain/access/reduction/broadcast relation 的 incomplete padding、update padding、
  affine/gathered access、alias/inout、一般 mutable storage与 padded mean 仍显式 unsupported。

这不是忽略 access/effect 后直接运行 A：mask erasure 必须先证明所有被擦除 lane 位于 J logical
domain 外、没有未保护的可观察写入，且每个 reduction 的 masked contribution 为显式中性元素；
执行证据记录 predicated access/fill 数量。只有整个边界可机械投影到
完整逻辑张量求值时才进入 Intent CPU 路径，其他关系继续 fail closed。后续每扩展一种 boundary
relation，都必须同时增加其 denotation、正反例和真实 canary。

active `typecheck.v7` 已复用 J-only `functional_reduction_padding_issues(J)`，但只在下列 closed
seed 上传播 padded provenance：predicated direct load 不含 logical guard；index/domain 是完整零起点
矩形；predicate upper bound 与同轴 typed tensor extent 相等。只有该同轴 provenance 经已建模
pointwise/broadcast 路径被 `mean` 消去时才映射为 `padded_mean_count_unproven`。该必要条件与
consumer capability 仍分离：guard-bearing 与 cross-axis/general predicate 不成为 TypeCheck padding
seed，未知 op 会停止 TypeCheck 的窄传播；value-free projector 对 guard/cross-axis 访问或 masked
provenance 进入未知 op 均显式 consumer-unsupported。v10 只增加上述 invocation-local、可证明全真的
guard/runtime-bound 例外；fixed code 不补写 count、domain、predicate 或 reduction contract。

当前 shared projection receipt 为 `intentir.j-functional-boundary.v3`，scalar RefEval 为
`j_scalar_cpu_mlir_v10`。v10 完整继承 v9 的唯一 node-less logical-input load 与通用 multi-reduction
padding closure，只新增当前 Gate2 invocation 的 runtime predicate tautology proof。v3 projection 中
对应的 invocation-local proof subreceipt 只含
protocol、access id、J predicate digest、logical input refs 与 domain axes，不含 raw runtime value 或
value digest；这是 audit-data minimization，不是 confidentiality/privacy proof。旧 v8/v9 Gate2
receipt、functional-boundary v1/v2 与
`logical_domain_reduction_padding_v1/v2` 记录保持历史解释，不由 v10/v3 追认或重判。

CPU host pipeline 另在共享 `J -> Intent MLIR -> Intent-to-Linalg` 之后调度
`expand-rsqrt-for-libm`，只将 `math.rsqrt(x)` 展开为 `1 / math.sqrt(x)`，以使用标准 host libm
sqrt/sqrtf ABI。这是 CPU-only compatibility pass；共享 Intent lowering 与 GPU/target-neutral pipeline
均不调度它，因此不改变 production/GPU 的 rsqrt lowering 选择。

## Production 要求

- C 只影响已经由 pass contracts 判定为合法的 candidate 的初始化、搜索顺序、优先邻域与有限预算
  下的访问集合；不参与 legality，也不改变 J denotation；
- production analyzer 必须先成功取得 value-free `P_B^prod(J)` 才能发射 logical Intent MLIR；projection
  失败不得继续 codegen；该入口不接收 runtime logical values/dimensions，也不能消费 RefEval 的
  invocation-local proof subreceipt；
- lowerer 不理解某个 C slot 时可以采用保守实现或显式 unsupported，但不能从 π/Source 猜测；
- 可选 `subject_ref/axis_ref` 只定位 prior；局部 slot 缺失/含混 attachment 时必须安全忽略，不能
  依靠 role-first、π 或 Source 猜测作用点；
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
S_C = Attach_C(M0,C,t)
BaseIR(S_C) = M0
```

`S_C` 只包含不可变 M0、discardable C annotations、初始参数与 candidate ordering。若实现产生
decorated working MLIR `M_C`，必须满足 `Erase_C(M_C)=M0`，且不改变 A/B denotation、pass
applicability 或 target ABI。只有 `Attach_C` 可以产生 C-derived attr；parser、`Emit_v`、pass default
或 kernel recognizer 不得冒充 C 来源。该步骤不修改 J，也不产生新顶层 artifact。

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

这是规范级、条件化 compiler contract，不是当前已经完成的机器检查形式证明。支持 claim 只覆盖
当前 release 实际实现、登记并经过 admission 的 `Emit_v`、transformation 与 target lowerer；roadmap
中的 family 不进入当前 `Reach_t(J)`。compiler TCB 至少包括 `Emit_v`/B projection、
`ContractView`/target capability query、实际 transformation 及其 Pre/preservation/admission 检查、
transition admission 与 target lowerer。search/cost model 只有在不能绕过 admission 或直接构造
candidate 时才不承担语义裁决。

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

只用于 CI、release preflight/smoke、debug、new-family regression 与 evaluation audit，不是
production legality oracle，也不是 SCRC 的第四条 Gate2 edge。

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
