# RefEval 与 J-Grounded、Q/D-Certified Lowering-State Search 合同

> 状态：当前派生规范。上位设计见
> [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与
> [后端最终设计](../../../doc/IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)。

## 两条不同路径

Gate2 reference 与 production lowering 不是同一个消费者：

```text
RefEval      : J x logical_inputs -> observable_outputs
CoreLower   : J x C x target x compiler_state -> executable
SearchClose : J x C x target x Q x D_search x D_cert x opaque_oracles
              -> executable + receipt
```

`RefEval` 是简单、无优化、J-only 的语义路径。其函数类型不存在 C 或 π；它不能从 implementation
selection、source realization、Harness 或历史 artifact 补语义。

core lowerer 必须消费完整 J，并可消费或安全忽略 C。它不读取 π，因为 π 只证明 Source
realization 与 Replay closure，不是目标 backend 的施工输入。
system orchestration 只把 SCRC-accepted artifact 的 J+C 交给 core lowerer；generic search/closure
controller 另取得预先冻结 Q、声明的 D 与 opaque execution harness。core API 不读取 Gate verdict，
controller 也不能自行接受未通过上游协议的 response、修改 J 或选择 comparator。

两条路径在分叉前必须共享唯一标准化前端：

```text
response JSON -> Decode/Elaborate -> J
                                    |
                                    v
                     canonical J + functional projection
                         |                         |
                         |                         `-> RefEval(J): scalar Intent MLIR
                         |                                           or closed structured reference
                         `-> Emit_v(J) -> Intent MLIR
                                      -> Attach_C(M0,C,t)
                                      -> hard-safe target states
                                      -> Q/D filter + performance search
                                      -> final executable closure
```

Decode/Elaborate 将 access index、predicate/domain ContractExpr、唯一可推导 result type 与恒等 output
binding 变成 typed canonical form。RefEval 和 production backend 均不得重新读取或各自解释响应中的
字符串；production 的 `J -> Intent MLIR` 只有一个共享入口，RefEval 的 closed structured path 也只
读取同一个 canonical J 与同一 functional projection。未接受的响应 JSON 不进入 lowering，解析
缓存或 structured execution state 都不是另一份 J，不增加 artifact、Agent 或 Gate。

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
projection。RefEval 与 CUDA analyzer 使用同一 protocol/core output structure；v11 RefEval record
可以额外含 invocation-local runtime-proof subreceipt，production record 必须为空。二者不能各自
保留第二套 wire parser 或 B-consumption 判定。

当前 fresh artifact 的 typed relations 同样进入这一唯一消费边界：v4 的
`value_dimension_relations` 与 `consumer_extensions`、v5 的 `access_guard_relations` 全部由 Lift LLM
显式写入 J。shared core 只能验证并投影已声明 relation；不得从相等 runtime value、reducer/fill、
Source、C、π 或 target pattern 补造 relation。对任一 active relation，consumer 必须产生明确 receipt
或返回 unsupported/error，不能在发射 A 时静默丢弃 B。

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

当前 RefEval 对显式内存边界与 typed J graph 实现以下保守投影：

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
  继续显式 unsupported。RefEval v11 继承 v10，只在当前 invocation 中按 J predicate 的真实极性、closed axis
  与 dimension extent 证明其全域恒真后才可擦除；operand membership 不能替代该 proof；
- `logical_domain_reduction_padding_v2` 不再把合法图限制为单 input、单 output、唯一单轴 `sum`。
  每个带 physical predicate 的 direct access 必须为自身每个非常量 `0` 的 axis 显式声明完整零起点
  logical domain；所有仍携带某个 masked axis 的 load/store 必须声明同一个 full-domain predicate，
  predicated load 必须显式给出 fill，且 access/effect/output relation 仍完整唯一。被 reduction 消去的
  axis 不要求出现在降秩 output store 上；仍存在于 output 的 masked axis 必须继续由 store predicate
  保护；
- padded graph 中每个 logical `reduce` 必须恰有一个 `ReductionBoundary`。legacy v1-v3 artifact 的
  `identity` 若存在必须是该 reducer 的代数单位元，`masked_contribution` 必须显式存在且为单位元；
  v4+ 的逐 consumer authority 改由下述 `consumer_extensions` 承担。当前 legacy 闭合集合包括
  `sum/prod/max/min/any/all`，其中 max/min 接收封闭的 JSON-safe infinity sentinel。若 J-only
  access/dataflow 能证明被 mask 的 axis 最终由 `mean` 消去，则因缺少 logical count contract 继续
  unsupported。legacy `masked_contribution` 按 J 的既有定义记录经过任意普通
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

artifact.v4/v5 的新增 closed profile 继续遵守同一规则：

- `value_dimension_relations` 在 invocation 时检查已声明 integral rank-0 J value 与 symbolic dimension
  valuation 精确相等；不从 valuation coincidence 反向生成 relation；
- `consumer_extensions` 是 `sum|max|min` reduce/scan 的逐 consumer 域外 typed two-sided identity，
  必须闭合 access provenance、operand ordinal、local tile axis 和 dtype。artifact.v4+ 不再使用 legacy
  `ReductionBoundary.masked_contribution` 作为另一份 authority；
- 每个 `access_guard_relations` 必须被消费或显式 unsupported。当前 RefEval 只执行 Layer A 已经
  显式写出的 guarded-store `select`：relation 提供该 select guard 的唯一 broadcast axes 和 audit
  receipt，不做隐式 merge。目标必须是精确 input/output/inout tensor；loop carry 的 init/current/next
  或 non-loop return/output mapping 必须已经由 J/TypeCheck 闭合。guarded load/update 的 access-local
  pre-execution materialization 尚未实现，当前 RefEval 明确 unsupported；
- 多个无 predicate point store 可以形成 `unit_width_direct_store_partition`，但只在它们引用同一个
  producer 与完整 observable typed result、沿一个静态 axis 用常量 unit slices 无重叠无缺口覆盖
  全部 extent、其余 axes 完整一致时成立。multi-axis、symbolic、wide、predicated 或不同 producer
  不会被猜测成 partition；
- 当 common scalar Intent-to-Linalg 当前不能消去一个 typed contraction 时，Gate2 可以整图选择 closed
  structured J-only reference。该 executor 只接受 closed op table 和 binary einsum contract，核对
  operand/result rank、label extent、declared shape/dtype，并按 declared result/accumulator dtype 执行；
  不支持或不闭合的图 fail closed，不能在图中途 fallback。

最后两项是通用 **Gate2 reference consumer capability**：direct-store receipt 证明 RefEval 可以恢复
完整 logical output，mixed contraction path 使当前 J denotation 可执行。它们都不规定 production
CUDA 应使用 point stores、NumPy einsum 或某个 contraction algorithm，也不构成 pass admission、
target permission 或 kernel-specific lowering recipe。

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
provenance 进入未知 op 均显式 consumer-unsupported。当前实现保留上述 invocation-local、可证明全真的
guard/runtime-bound 例外；fixed code 不补写 count、domain、predicate 或 reduction contract。

当前 shared projection receipt 为 `intentir.j-functional-boundary.v14`，scalar RefEval 为
`j_scalar_cpu_mlir_v15`，closed structured reference 为
`intentir.j-structured-cpu-reference.v7`。这些 active receipt version 覆盖唯一 node-less logical-input
load、通用 multi-reduction padding closure、当前 Gate2 invocation 的 runtime predicate tautology proof、
上述 typed relations、direct-store partition 与 closed typed contraction consumer。runtime
predicate 对应的 invocation-local proof subreceipt 仍只含
protocol、access id、J predicate digest、logical input refs 与 domain axes，不含 raw runtime value 或
value digest；这是 audit-data minimization，不是 confidentiality/privacy proof。旧 v8/v9 Gate2
receipt、functional-boundary v1/v2 与
`logical_domain_reduction_padding_v1/v2` 记录保持历史解释，不由当前 consumer 追认或重判。

修复上述 RefEval capability 后，可以对 parent 中 `replay_closed` 的精确 artifact/Replay 做零模型调用
Gate2-only recheck；它写独立 evidence 且 `changes_parent_verdict=false`。这只证明当前 reference
consumer 如何执行同一个 J，不更新旧 receipt，也不提供 production legality 或 lowering recipe。

CPU host pipeline 另在共享 `J -> Intent MLIR -> Intent-to-Linalg` 之后调度
`expand-rsqrt-for-libm`，只将 `math.rsqrt(x)` 展开为 `1 / math.sqrt(x)`，以使用标准 host libm
sqrt/sqrtf ABI。这是 CPU-only compatibility pass；共享 Intent lowering 与 GPU/target-neutral pipeline
均不调度它，因此不改变 production/GPU 的 rsqrt lowering 选择。

## Production 要求

- C 只影响 hard-safe candidate 的初始化、搜索顺序、优先邻域与有限预算下的访问集合；不参与
  hard legality、Q/D feasibility，也不改变 J；
- production analyzer 必须先成功取得 value-free `P_B^prod(J)` 才能发射 logical Intent MLIR；projection
  失败不得继续 codegen；该入口不接收 runtime logical values/dimensions，也不能消费 RefEval 的
  invocation-local proof subreceipt；
- lowerer 不理解某个 C slot 时可以采用保守实现或显式 unsupported，但不能从 π/Source 猜测；
- 可选 `subject_ref/axis_ref` 只定位 prior；局部 slot 缺失/含混 attachment 时必须安全忽略，不能
  依靠 role-first、π 或 Source 猜测作用点；
- alias/effect、logical update 与当前 `exact/tolerance/one_of_updates` obligation 中所有影响 target
  denotation、safety 或 ABI 的 backend-relevant B facts 必须实际实现或拒绝；
- 当前 production J+C 路径复用 shared functional projection 的 `access_guard_relations` receipt：只有
  relation 已由 J-authored explicit select/access-local projection 完整物化，且每一条 authored relation
  都有精确 consumed ref 时才可继续。缺任一 receipt 或 guarded update 等未实现形式仍必须以
  `access_guard_materialization_not_implemented`/projection error fail closed；不能仅从图中看见 select
  就猜测 B 已消费；
- `realization_authorizations` 的既有 consumer 是 SCRC TypeCheck/π；backend 只识别其 SCRC-only
  身份，不把它当 target permission，也不因其存在自动 unsupported；
- core lowerer 与 transformations 不读取 π、Gate/Q、Source、kernel name 或 per-kernel RuleCard；
- generic search/closure controller 只通过 opaque harness 执行 Source/Replay/RefEval/target，不读取其
  文本来构造 state；所有 Kernel 复用同一 Q/D evaluator 与 receipt schema；
- hard correctness 由 compiler verifier/contracts 支撑；可能改变浮点观察值的 state 由 Q/D
  executable closure 支撑。后者不替代 Gate1/Gate2，也不修改 SCRC verdict。

## Certified corpus intake

截至 2026-07-26，严格 SCRC closure 为 16/28；当前 `corpus/production-jc-v1` 已 content-addressed
冻结并实际消费全部 16 项，verified intake queue 为 0。新增四项的 snapshot digest 为：

```text
xFormers RMSNorm-add:      e8e5451fa837443024a435b34b125f5c9f2261bdaf178842f80898875cf727be
Liger Sparsemax:           ca974aed6b5954b197a063f0e96d62652d569b5728608818adcdbfbc3cb11fda
Liger KLDiv:               b39a3df6d64782be475b5b459f6b7dc01efdb3356fa85abe22a75334f296df7a
Flash one-pass LayerNorm:  76ffde5e8eb4e53d0fca5afad8bb1d44ad4ba3f6c9df764dc67b21aa95402480
```

入库必须复制 exact accepted J/C 与 closure receipt，核验 artifact/J/C/receipt digest，并产生新的
content-addressed snapshot/catalog entry。不得重新 Lift、固定代码补 J、只按 case_id 构造 snapshot，
或在 artifact 未冻结时把 `/tmp` 路径当 production truth。历史 12-kernel correctness/performance report
保持历史口径；新增四项的 materialization/codegen smoke 或单 case Q/D closure 不得重标成 16-kernel
performance aggregate。

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
必须保留到所有依赖其语义结构的 transformation 完成；未接受 response、π、Source 或 Gate evidence
不得进入。

## Hard implementation judgement 与 pass contract

定义：

```text
Gamma_(J,t) |- M : HardImpl_t(J)
```

`Gamma_(J,t)` 只由 J、target capability 与版本化 compiler contract 构造。它是 ephemeral、
non-serialized analysis，没有独立语义所有权，不成为 backend 输入或 receipt truth，不进入 SCRC，
也不能反向改写 J。每个 transformation
`rho` 必须声明 matched A structure、required B facts、target/current-state precondition、转换结果、
preservation class、`exact-preserving|observation-changing` 与可搜索参数，并满足：

```text
Gamma_(J,t) |- M : HardImpl_t(J)
and HardPre_rho(J,M,t)
=> Gamma_(J,t) |- rho(M) : HardImpl_t(J)
```

这是规范级、条件化 compiler contract，不是当前已经完成的机器检查形式证明。支持 claim 只覆盖
当前 release 实际实现、登记并经过 admission 的 `Emit_v`、transformation 与 target lowerer；roadmap
中的未实现 transformation 不进入当前 `ReachHard_t(J)`。compiler TCB 至少包括 `Emit_v`/B projection、
`ContractView`/target capability query、实际 transformation 及其 Pre/preservation/admission 检查、
transition admission 与 target lowerer。search/cost model 只有在不能绕过 admission 或直接构造
candidate 时才不承担语义裁决。

缺少 hard pass 所需 B fact 时，该 pass 不适用。系统不得从 Gate2 tolerance、运行结果、Source/π 或
kernel identity 补足 memory/effect/ABI 前提。reduction tree、FMA、内部精度或 approximate math
只要保持 hard boundary，可以标记为 `observation-changing` 后进入统一 Q/D filter；这不把
`realization_authorizations` 或 result tolerance 解释成静态 target permission。

## Generic transformations 与 lowering derivation

backend 登记通用 MLIR transformations，并按 A 的 op/region/graph pattern 匹配、按 B/current-state/
target/pass contract 准入，例如：

- map reduction to vector/subgroup/workgroup、split partial/finalize；
- producer-consumer fusion、fission 与 intermediate materialization/elimination；
- scan、contraction、conv 与 structured recurrence transformations；
- 有独立 preservation contract 的 algorithmic rewrite。

compiler 不发射 implementation family。候选必须是由具体 transformation trace
`p=(rho_1,...,rho_k)` 得到的具体 MLIR state `M_p`。registry key 只能来自 transformation identity、
typed A match、B/current-state/target preconditions 与版本化 pass contract，不能来自 kernel/provider/
Source 名称或手写 family ID。non-trivial identity，例如 online recurrence 到 ordinary two-pass，必须
由单独 transformation 及 hard preconditions 支撑，不能伪装成 schedule tuning；若改变数值观察，
还必须经过 Q/D。

实现可以从 canonicalized candidate IR 派生擦除数值 schedule 参数后的 `topology_digest` 用于去重、
receipt 和实验分组；该 digest 不得参与 semantic admission、替代 MLIR state 或选择 emitter。

## Hard search space、Q/D feasibility 与 C ordering

定义 MLIR compiler-state 合法空间；M 只表示 Intent/lower-IR state，J 是不可变参数：

```text
ReachHard^MLIR_t(M0; Gamma_(J,t)) = {
  M | M0 ==rho1,...,rhok==> M
      and every transition preserves HardImpl_t(J)
}
```

下文简写为 `ReachHard_t(J)`。编译和执行后另定义：

```text
Feasible_(t,Q,D)(J) = {
  M in ReachHard_t(J) | QDAdmissible_(t,Q,D)(M,J)
}
```

C 不定义 `ReachHard_t(J)` 或 `Feasible`，只提供：

```text
Order_C(ReachHard_t(J))
```

即 source-like 初值、第一批参数、优先邻域与搜索顺序。target-specific legal parameter domain 属于
transformation/schedule/target contract；`required` C slot 不表示 target schedule fixed。搜索产生的
assignment 与 lineage 是 backend internal state，不产生 `C_target` 或新 Agent/artifact。

对有限预算 b：

```text
VisitedHard_(t,b)(J;C) is a subset of ReachHard_t(J)
```

C 可以改变 `VisitedHard` 的顺序和有限预算覆盖，但不能改变 hard-safe universe、Q/D policy 或
`Feasible` 的定义。

搜索器可使用 enumeration、beam、e-graph、bounded superoptimization、cost model、profiling 或
autotuning。搜索顺序可以启发式，但 correctness 取决于 transition 对 `HardImpl_t(J)` 闭合，以及
最终 executable 通过冻结 Q/D。

## Transformation 分类与当前 B 边界

所有 transformation 共用同一个 registry。下列分类只用于 contract 声明和实验归因，不建立 compiler
object hierarchy：

- `schedule-preserving`：固定 structural trace，只改变 tile、vector width、threads/warps、unroll、
  pipeline 或 launch 参数；
- `B-conditioned-local`：消费 B 事实，但不改变主 graph/materialization/reduction/synchronization
  topology，例如 scalar capture、read-only load motion、dead ABI input elimination；
- `structural`：改变 fusion partition、materialization points、reduction/synchronization topology、
  data-movement decomposition 或 kernel-stage graph。

memory placement 若只映射同一 logical buffer 的 address space，可以是 schedule；若插入/删除
materialization、改变同步或跨 kernel 生命周期，必须标记为 structural。分类由 transformation contract
与 before/after IR 共同审计，不由候选名称决定。

当前 accepted J 与 production `ContractView` 允许直接开发和搜索：

- 在 effect/alias completeness 可证明时的 producer-reduction-epilogue fusion；
- intermediate materialization/copy elimination；
- RVV whole-graph Linalg fusion 及其独立 schedule search；
- scalar capture、load motion、predicate specialization 等 local transformations；
- hard-safe 的 reduction tree、FMA/contraction、内部精度与 approximate states；这些数值类必须标记为
  `observation-changing` 并进入 Q/D evaluator。

当前 `reduce.assoc=associative|float_nondeterministic` 与 Gate2 `tolerance` 不得自动解释成这些机制在
D 外静态等价，也不得让 target pass 绕过 Q/D；但当前执行域路线不要求为每个 mechanism 增加
`allow_*` B relation。现有 `exact|tolerance` 继续选择 closed comparison class，具体阈值由冻结 Q
独占，controller 不得把 exact 改成 tolerance。floating atomic state 还必须满足 J update/effect hard
contract，并把重复执行与调度 seed 纳入 D；一般 `one_of_updates` 在当前 Q protocol 未实现前不进入
numeric search。

`ContractView` 不得仅凭 `boundary.aliases=[]` 或 `signature.alias_of=null` 推导全参数 no-alias，除非
artifact/schema version 明确规定 alias relation set complete。每个 derived capability 必须记录 exact J
carriers、使用的 completeness rule 与 derivation version；缺少 completeness 语义时返回
`missing_contract`。具体 benchmark inputs 恰好 disjoint、负例测试通过或 downstream pass 未观察到
alias 均不能补足该事实。

当前 artifact.v7 闭合批次不因 backend search 切换 schema。若未来研究目标改成脱离 D/Q 的静态
deployment theorem，才另行设计 versioned reduction-order、FMA、precision 或 atomic numeric
relations，并让使用新 relation 的 J 重新认证；该未来方向不是当前实现依赖。历史 J 保持原解释。

## Candidate 与 transition receipt

每个被枚举或拒绝的 structural/local transition 至少记录：

```text
j_digest
target/toolchain digest
input MLIR digest
transformation id + normalized parameters
matched A/IR carriers
required and consumed B facts
current-state/target preconditions
admission or rejection reason
output MLIR digest (if admitted)
transform trace
schedule assignment
C order score/order position
lowered IR/object digest (if compiled)
numeric observation class
D_search/Q filter result (if required)
correctness and performance record (if measured)
```

`topology_digest` 可以作为派生审计字段，但不是 admission truth。search/cost model 不得绕过 transition
admission 直接构造 compiler state。只有固定 final winner 生成正式 backend closure receipt；其他
candidate record 是 search log，不增加 proof artifact。

final receipt 必须额外绑定 J/C digest、target/compiler/toolchain version、实际 c0、Q/protocol、D 定义
与 realized challenge digest、final MLIR trace/schedule、binary digest，以及 target--Source、
target--Replay、target--J 对所有 observable outputs 的直接 verdict。performance record 与 correctness
record 分栏；任一 leg 缺失都不是 Q/D-closed executable。

## 实验投影

规范实验使用：

```text
E0  Source frontend implementation
E1  canonical deterministic target lowering + default schedule
E2  fixed structural + fixed numeric trace + schedule-preserving search
E3  hard-safe structural/numeric states + independent schedules,
    filtered by the same Q/D, with C
E4  exactly the E3 ReachHard/Q/D/budget with Erase_C

L0  fixed hard-safe structural state set; B-conditioned local transforms disabled
L1  same state set/budget; B-conditioned local transforms enabled
```

E3 candidate set 必须包含 E2 全部候选。L0/L1 均完整消费 J=A+B，关闭的只是额外 local
transformation。E2/E3 不得被 capture/load-motion 等 local-only 差异污染；E3/E4 只能改变 ordering，
不能改变 hard universe、Q/D、correctness policy 或 total budget。

数值机制消融使用同一个 registry/controller：

```text
N0  observation-changing numeric states disabled
N1  N0 + reduction-tree states
N2  N1 + FMA/contraction states
N3  N2 + target-supported precision/approximation states
```

E2/E3 必须共享 J、target、compiler revision、terminal target pipeline、profiler、Q、D_search 与
D_cert。报告同时给出 equal-total-budget anytime/time-to-best 与小空间 exhaustive/oracle 结果；
E3/E4 在共享 latency map 的 full budget 下必须收敛到相同最优候选。所有 Kernel 都进入汇总，
不适用的 transformation 记录 `not_matched/rejected`，不能只报告 winner。

上述 arm 由 repository-root 的 `spec/lowering/backend-search-ablation-v1.json` 唯一冻结并被 runner
直接读取；当前 canonical digest 为
`60ca646b34c29ce3542d3c4e6a9e08c39155bc9e2a27324844ab8fad80f050a7`。loader 必须同时校验 scope、
Q/D direct-edge 规则、D_search/D_cert digest-disjoint 规则、C ordering-only 角色、equal-total-budget
规则、arm 全集与字段类型。报告必须写入 digest；未知/重复 arm、额外字段或 correctness protocol
漂移均 fail closed，禁止 runner 内另存一套同名实验定义。

## Q/D-constrained performance selection

对有限预算内实际访问、hard-safe 且通过 `D_search` 的 candidate：

```text
Measured = VisitedHard_(t,b)(J;C) intersect Feasible_(t,Q,D_search)(J)
M*_(b,C) = argmin_{M in Measured} Cost_t(Lower_t(M))
```

Cost 可以来自静态模型或测量。Q/D feasibility 与性能 measurement 是两个不同 predicate，即使一次
execution 同时产出 correctness output 和 timing 也必须分开记录。core lowerer/pass 不读取 Q；generic
controller 不得让更快 candidate 覆盖 Q/D failure。

搜索期 filter：

```text
Close_Q(Exec(Lower_t(M),x), RefEval(J,x))
```

最终 winner 固定后在独立 `D_cert` 上直接检查 target--Source、target--Replay 与 target--J 三边；不能
依赖 tolerance 传递性，也不能把 hidden result 回流候选生成或 comparator。失败则本次 backend
certification 失败；下一候选必须遵循预先声明的独立 certification protocol。

这是 post-SCRC backend executable closure，不是 SCRC 的第四条 Gate2 edge，不修改 J/Gate2 verdict，
也不声称 D 外的静态 contract proof。

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

该边是通用 `D_search` filter 的组成部分；正式发布还必须具有上一节的 `D_cert` direct-edge receipt。
它不是第四条 Gate2 edge，也不改变 `SCRC_CERTIFIED`。当前正式 J-facing
CUDA API 对 alias/inout 继续 fail closed；若以后开放，destination/alias mapping 必须直接来自
J.signature/B，而不能复用 Source Harness binding。

Intent MLIR 可以作为 J 的确定性编码，但不得成为携带 interface、C/π 语义副本或 Gate verdict 的
第二 wire truth。
