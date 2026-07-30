# Gate2：Source / Replay / J 三方语义验证

> 状态：当前派生规范。上位设计见 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。
> 数学与 claim 边界见 [SCRC 最终数学总述](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。

## 三腿与权限

对 logical input x 和实际绑定 `c0 = Bind_H(C)`：

```text
O_S = Exec(Source, c0, x)
O_R = Exec(Replay, c0, x)
O_J = RefEval(J, x)
```

Source 与 Replay 使用同一个 C 对象和实际 valuation `c0`。该证书是 instance-wise 的，不自动
覆盖所有 admissible C。`RefEval` 的接口中不存在 C 或 π，也不读取
Source/Replay。所有 output、inout、alias/writeback 与声明的 observable effect 都必须裁决。

Gate2 不直接消费未接受的 Lift response。固定 Decode/Elaborate 成功后只产生一个 typed J；
RefEval 与 production lowering 使用同一个 canonical-J elaboration 与 functional-boundary core；
production 走共享 `J -> Intent MLIR` materialization，RefEval 则可在 scalar Intent-MLIR、closed
structured J-only reference 与 typed ContractIR indexed reference 之间按全局 capability 选择。于是 `k1,k0` 与 `[k1,k0]`、可唯一推导的
result type、冗余 identity output mapping 不会由消费者再次私自解释；general affine、缺
predicate/effect、类型冲突或不在 closed reference op table 内的图仍在准确边界 fail closed。两条
reference 执行路径都不是 production target lowering，也不产生第二份 J。

RefEval 与 CUDA production analyzer 共享同一个 deterministic functional-boundary core，但入口权限
不同：production 只可调用 value-free 的 `ProjectFunctionalBoundary(J)`；当前 Gate2 winner 的 RefEval
可在该次 invocation 内调用
`ProjectFunctionalBoundaryForRefEval(J, logical_inputs, closed_dimensions)`。后者只为 J 自身 rank-0
logical `i1`/integer values 上的 access predicate 做窄 tautology proof，不把 runtime value 变成 J、
artifact 或 production lowering 输入。两种投影都只是 ephemeral structured result，π 专用 realization
authorization 不在其中；它们不改变 `RefEval(J,x)` 的函数类型，也不把 target conformance 并入
Gate2。

## 比较关系

```text
G2 = Close_Q(O_S,O_R)
     and Close_Q(O_S,O_J)
     and Close_Q(O_R,O_J)
```

Q 是实验协议或 benchmark correctness policy 预先确定的可信比较政策；B 只声明封闭 comparison
class 与必要语义合同。Q 根据经验证的 class、dtype、reduction extent、reassociation 权限等
构造 comparator，并验证 tolerance 上限、NaN/Inf 等全部参数。B 不提供任意可执行 comparator；
不支持的类别、非法参数、缺腿或执行不完整均 fail closed。Q 不是 Lift artifact，也不进入 Replay
或 RefEval。

在 Gate2 之前，TypeCheck 必须验证 comparison class 与 output dtype、A 依赖图和 B numeric contract
相容；它只拒绝，不自动把 `exact` 改成 `tolerance`。当前使用最终执行流程第 6.1 节的
`RequiresTolerance_v1` closed dependency rule；该规则命中时，显式或默认 `exact` 都被拒绝，v1
没有自由文本或“更强合同”例外。其 reassociation 输入专指 J/B 中锚定 output dependency 的
source-realization authorization，只触发 SCRC compatibility check，不成为 backend permission。
Lift 仍显式声明 relation，Q 仍独占具体阈值。

`Close_Q` 必须是确定性、对称的谓词；若底层 `isclose` 有方向性，Q 必须显式对称化或执行双向
检查。第三条 Replay–J 边不增加执行，只比较已经产生的三个输出；两条 Source-centered 近似边
不能依靠传递性推出它。

Gate2 relation 只服务 Source/Replay/J 的观察闭合。production backend 不读取 Q；`tolerance` 不能
作为改变 reduction tree、accumulation dtype、FMA、rounding 或 approximate transcendental 的
lowering permission。

当前主认证域只覆盖 deterministic / contract-deterministic Kernel，即内部调度可变化，但 J/B
在协议可观察层面给出唯一合同结果。真正集合值的 Kernel 留给未来 `Accept_{J,Q}(x,o)` 的
per-observation conformance；有限次 membership 不能被宣称为完整的行为集合包含证明，两个不同
但都合法的集合值结果也不应继续强制三条 `Close_Q`。

Gate2 input precondition 直接检查 external J input 上可判定的 `index_bounds` 与 duplicate contract；
若 data/indices 是 J 内部值，则不要求把它们伪装成额外外部输入，而由 J-only structured
gather/scatter evaluator 在该 authored op 执行时对真实计算值检查 dtype、axis、bounds 与 duplicate
policy。检查失败仍使 J leg fail closed；该分工不从 Source、Replay、Harness、C 或 π 推导索引关系。

## Gate2 输入模式

多候选搜索先完成 pre-Gate2 wave，并仅依据 TypeCheck/Gate1 evidence 按固定
`(branch_id, formal_round)` 顺序选择一个 `(J,C,π,Replay)` winner。选择函数看不到 Gate2 输入、
输出或 verdict；winner 此后不再替换或修改，随后才产生 Gate2 输入。未选 sibling 不执行 Gate2，
winner 失败后也不能 fallback。两种输入口径必须明确区分。

### 隐藏随机输入

从预先声明的分布 D 独立同分布抽取 m 个对 Lift/Replay 隐藏的输入，Gate2 结果不回流本次实验。
令完整候选为 `a*=(J,C,pi,Replay)`，并令 `g2_Q(a*,x)` 表示三条边在 x 上全部通过：

```text
epsilon_a = Pr_{x ~ D}[not g2_Q(a*,x)]
```

若 `epsilon_a >= epsilon > 0`，该固定坏候选通过全部 challenge 的概率至多为：

```text
(1-epsilon)^m <= exp(-m*epsilon)
```

Source–J disagreement mass 是充分特例；“错误 J”本身不保证错误落在 D 的支持内或能被 Q 区分。
三条边不会把 m 变成 3m。

### 有限测试集合

若只使用预先给定的有限集合 T，只能声明：

```text
for every x in T:
  Close_Q(Source(x),Replay(x))
  and Close_Q(Source(x),RefEval(J,x))
  and Close_Q(Replay(x),RefEval(J,x))
```

此时不得引用隐藏随机 challenge 的概率上界。

每份 Gate2 记录必须包含 Q/protocol version、测试模式、D 或有限 T 的标识、实际 c0、三条边各自
结果与失败原因；这些是审计 metadata，不是新的 proof artifact 或 Replay 输入。

## 当前 RefEval 的标量与 logical-domain 支持边界

当前固定 scalar RefEval protocol 为 `j_scalar_cpu_mlir_v24`，shared functional projection receipt 为
`intentir.j-functional-boundary.v22`，structured reference 为
`intentir.j-structured-cpu-reference.v10`，relation-indexed pointwise reference 为
`intentir.j-contract-indexed-cpu-reference.v3`。这些版本在既有 source-neutral multi-reduction 与
invocation-local predicate proof 之上，消费当前 artifact.v4/v5 typed relations、闭合 direct-store
partition，并为 common scalar lowering 尚不能消去的 typed contraction 提供闭合 J-only reference。
旧 RefEval/projection 的 `REFERENCE_UNSUPPORTED`、receipt 和 Gate2 verdict 保持原解释，不由新
consumer 追认、覆盖或静默刷新。

当 shared functional projection 因一般 affine/sub-domain predicate 无法把 B 擦除为完整 tensor 时，
v22 只能尝试一个 closed ContractIR profile：flat、acyclic、relation-indexed pointwise/gather A graph；每个
被作者化的 load 在其 `node_ref`/operand edge 上按 typed index 与 predicate 取值，predicate 域外只用
该 access 的显式 fill；每个 output 再由 typed store index/predicate 无重叠、无缺口地组装。所有
ContractExpr 都由 canonical typed evaluator 执行，不匹配公式或 Kernel 名。没有绑定 access 的
sibling operand 保持普通 Layer-A 语义，绝不继承另一 node 的 access。这样 accepted J 若遗漏关系，
Gate2 会得到可比较但与 Source/Replay 不同的 `O_J`；系统应 fresh relift，而不是给 consumer 写答案。

该profile允许多个J domain axis具有相同extent，但producer轴只能由该producer自身access index中唯一
出现的axis绑定，不能按shape相等猜测。未直接绑定boundary access的普通gather，其`index_bounds`也
必须与该gather的data/indices/axis逐项一致，并由typed structured gather在执行时检查真实内部index。
terminal `concat`只在它是唯一output binding、每个operand与该output的唯一indexed store一一对应时
成为output-assembly receipt；每个store还必须在当前closed invocation上证明非concat轴逐点恒等、
concat轴从0开始连续且按operand顺序平移，所有切片无重叠并穷尽output。否则明确unsupported/error。
这只是从完整J/B确定性执行consumer-local sub-domain，不读取Source/Replay/C/π，也不允许普通
full-tensor concat被静默跳过。

当前 consumer 对新 relation 的权限是“逐项消费或显式 unsupported”，不是忽略：

- 对`J.dimension_relations`，RefEval以调用方提供的base dimension valuation求值closed identity/
  const-arithmetic DAG，补出缺失derived dimensions，并核对显式提供的result valuation；它不读取
  Harness、grid、C、π、Source或TTIR，也不从concrete shapes猜relation；
- 对 `J.value_dimension_relations`，RefEval 在 invocation 上同时检查 integral rank-0 logical value 与
  closed symbolic dimension valuation 的已声明 `eq`；缺值、冲突或不可表示均在执行前拒绝。relation
  只能来自 J，consumer 不从相等的 runtime 数字反推 relation；
- 对 `J.boundary.consumer_extensions`，projection 检查 relation 精确锚定一个 predicate-bearing load
  到一个 `sum|max|min` reduce/scan operand，验证 provenance、local tile axis、operand dtype 与
  two-sided identity，再按该特定 consumer 的域外贡献执行。它不把 access predicate 提升为 A value，
  也不从 reducer/fill 猜缺失 relation；
- 对 `J.boundary.access_guard_relations`，projection 必须消费每一项并写出
  `authored_access_guard_relation` receipt，或在执行前返回 unsupported/error。当前 RefEval 支持的
  guarded store 已经由 Layer A 显式 `select` 完成 merge：目标是精确 inout tensor，loop/non-loop
  pre-state、observable result、polarity 和唯一 broadcast axes 均由 J/TypeCheck 闭合；projection 只
  将 relation 的 axes 提供给这个既有 select condition 并记录 receipt，不执行隐式 post-merge。
  guarded load/update 需要尚未实现的 access-local pre-execution materialization，当前显式
  unsupported；TypeCheck 另要求 guarded load 有 fill，并拒绝当前无 merge contract 的 guarded update。

production J+C 当前即使能读取 shared access-guard receipt，也没有通用 target materializer，必须以
`access_guard_materialization_not_implemented` fail closed；不能把 RefEval 对显式 select 的执行能力
当成擦除 B relation 的许可。

`intentir.j-functional-boundary.v9` 还接受一个窄、通用的 direct-store partition：多个 store 必须都
无 predicate、引用同一个 producer 与同一个完整 observable typed result，只沿一个静态 output axis
使用常量、无重复且无缺口的 unit-width point slices 覆盖全部 extent；非 partition axes 必须完整
覆盖且一致。成功时产生 `unit_width_direct_store_partition` receipt。多轴、symbolic extent、wide、
predicated、不同 producer、缺口或重叠不会被猜测修复，而是 contradiction 或 consumer unsupported。
该证明不检查 Kernel 名或 producer opcode，是 Gate2 对完整 logical output 的通用消费能力；receipt
不是 CUDA/production 的 store lowering recipe，也不授权 target 复制这一物理写法。

当 J 含 `loop|scan|tile`，或存在 common scalar Intent-to-Linalg 当前不能消去的 typed contraction 时，
RefEval 整图选择 closed structured J-only executor，不在图中途 fallback。mixed typed contraction 只
接受两个 operand、唯一 typed result 和 closed binary einsum `subscripts`；它机械核对 label/rank/
extent/result shape，将 numeric operands 按 J 已声明的 result/accumulator dtype 转换后执行，并验证
结果 dtype。integer result 不接受 floating operand，未知 op 或不闭合 graph 全部 fail closed。该
路径解决的是 `RefEval(J,x)` 的可执行 denotation，不修改 production compiler 的 contraction lowering，
也不把当前 CPU compatibility limitation 写成 J 或 target recipe。

主 SCRC runner 必须先按 pre-Gate2 规则选定唯一 winner，再为当前 Gate2 invocation 绑定 logical
inputs/dimensions 并调用 RefEval-only projection；projection/RefEval library 本身不选择 winner，也没有
跨 invocation 调用权限。输入映射必须精确覆盖 J 的 logical inputs，dimension mapping 必须精确覆盖 J
输入/输出 shape 中的 closed symbolic dimensions。proof 只接受 J 已声明的 rank-0 logical `i1` guard
及 `index`/integer upper bound，并逐 access 证明 conjunction 在完整 logical domain 上恒真：布尔极性
必须由该次值满足，零下界与 direct axis 必须闭合，runtime integer bound 必须不小于该 closed domain
extent。失败只产生当前 J leg 的 consumer unsupported/error；不修改 J、不进入 Lift feedback，也不
读取 Source、C 或 π。

v3 projection 内的 invocation-local runtime-proof subreceipt 只记录 protocol、access id、J predicate
digest、logical input refs 与 domain axes；不得保存 raw value 或 value digest。这是 audit-data
minimization，不是 confidentiality/privacy proof；Gate2 case 的输入访问控制与隐藏输入绑定仍由主
runner/外层 evidence 负责。value-free production projection 继续没有 runtime values，也不得复用该
subreceipt 来放宽 target lowering。

runtime scalar 或 bound semantic constexpr 若代表 J 的逻辑数据，可以锚定一个 rank-0 J input。
Harness 给 Source/Replay 绑定前端标量，同时给 `RefEval(J,x)` 提供相同值的 rank-0 logical input；
它不能因此被折叠成 C。只有表示 shape/domain extent 的 integral handle 才进入 dimension binding，
不得同时冒充 J 的数据输入。

`typecheck.v7` 在 RefEval 前检查 rank-0 logical input 是否到达 observable output dependency，或由
显式 boundary/shape semantics 使用；direct identity 与 boundary-only 合法路径保留。仅服务
representation/address 的 runtime scalar 不属于 RefEval logical inputs，必须显式为 rank-0
`logical_ref=null` carrier，并只可由 π realization 引用。checker 不读取 Source 名称/位置、不自动
解绑，也不把该窄规则冒充一般 tensor liveness proof；该 TypeCheck judgement 与 RefEval v11 的消费
能力仍彼此独立。

这个 null-carrier 例外不适用于 bound constexpr：只有 implementation-selection 后果的 constexpr
属于 C，其他 bound constexpr 必须有 rank-0 logical-data 或 `x*` extent denotation；两者都不满足时
Harness invalid，不能让 RefEval 忽略、让 π 单独消费。

但只有在 J 自身已经足以机械证明“无 logical guard 的 predicated direct load 具有完整零起点矩形
index/domain、predicate bound 与同轴 typed tensor extent 相等，且该 masked axis 经已建模 closed
pointwise/broadcast 传播后被 `reduce(mean)` 消去”时，当前 `typecheck.v14` 保留 v7 的窄必要条件，
并在 RefEval 前以
`padded_mean_count_unproven` 拒绝：`masked_contribution` 只闭合分子，当前 boundary 不表达 logical
valid denominator。dense/unpredicated mean、guard-bearing 或 cross-axis/general predicate、
domain/extent 不闭合、mask 位于该 reduction 未消去轴或未知 op 阻断传播时不触发 TypeCheck。
不能形成上述 invocation-local tautology proof 的 guard、cross-axis/unknown-op 保持 consumer
unsupported。此 rule 不读取 Source/C/π，不把一般 RefEval
unsupported 偷换成 J 非法，也不自动改图。

当前 pure functional pointwise 与 reduction graph 可以机械消去物理 logical-domain tail mask，但只在
以下条件同时成立时成立：

- observable output 是直接、无 update/alias/inout 的 functional value；
- memory access 与 J 的 logical tensor/domain 直接锚定；每个 predicated direct access 为全部非
  常量 `0` axis 显式声明完整零起点 logical domain，各 bound 与 typed shape 一致；
- rank-0 `i1` guard 可以是合法 J input；v11 继承 v10，只在当前 winner invocation 中按 J predicate 的实际极性
  证明它在完整 logical domain 恒真。value-free production projection 与无法完成该 proof 的 RefEval
  仍显式 unsupported；operand membership 不能区分 `flag`/`not flag`，也不能替代该证明；
- direct singleton index `[0]` 只允许对应 logical extent 确为 1；
- 所有仍携带某个 masked axis 的 access 使用同一个 full-domain predicate；predicated load 有显式
  fill，output store 若已由 reduction 删除该 axis 则不需要伪造 predicate；
- padded graph 的每个 logical reduction 恰有一个 J-owned `ReductionBoundary`。legacy v1-v3 中
  `masked_contribution` 与可选 `identity` 必须是 reducer 的中性元素；artifact.v4+ 则由逐 consumer
  的 `consumer_extensions` 唯一声明 `sum|max|min` 的 typed two-sided identity，不能与 legacy 字段
  并存。若上述窄 rectangular access/dataflow provenance 能证明 masked axis 由 `mean` 消去，因缺少
  logical count contract 继续 unsupported；
- 普通 trailing broadcast 由 closed value type 决定，non-trailing broadcast 只能消费 J 中显式的
  `boundary.broadcasts`，缺失、错轴或未消费 relation 均在执行前 fail closed。
- masked provenance 若进入 contraction 或其他当前 provenance analysis 未建模的 executor op，必须在
  该 op 处 `REFERENCE_UNSUPPORTED`；不能静默丢弃 provenance 后放行下游 `mean`。

因此 `logical_domain_reduction_padding_v2` 可以消费任意由上述通用构造组成的 pointwise/broadcast/
multi-reduce DAG，例如 max → sub → exp → sum → div；实现不检查 Kernel 名称，也不含 Softmax
公式。legacy `masked_contribution` 或 v4+ `consumer_extensions` 都必须已经由 J 声明，projector 只
验证其 consumer-local identity 与 provenance，不从 Source、C、π、runtime/hidden input 或 named op
重新推导。update、一般 alias/inout、不一致 bound、
未锚定 guard、padded mean 或其他未闭合 topology 必须明确返回 `REFERENCE_UNSUPPORTED`/fail closed。

active `typecheck.v14` 保留 J-only `functional_reduction_padding_issues(J)`，用于精确报告上述窄
rectangular seed 下已建模 provenance 的 masked-axis → `mean` 缺口。它不把 guard-bearing、
cross-axis/general access 或未知-op consumer capability blanket 判成非法 J；当前 v9 projector 对这些
无法消去的 topology 明确拒绝；v11 继承 v10，只为可按当前 J logical valuation 完整证明的
predicate 中 logical guard/runtime
bound 新增上述 invocation-local 例外，并继续在 masked provenance 首次进入未知 op 时拒绝。
CandidateCheck 不另写一套规则，fixed code 也不补造 count contract。

## 失败与认证

Gate2 失败结束本次实验，不在同一实验中把结果反馈给 Lift/Replay 做局部修补。实验结束后可以
聚合失败、修改系统，再从新实验开始。

若 parent evidence 已经证明某一精确 `(J,C,π,Replay)` 为 `replay_closed`，而失败被归类为当前
RefEval/functional-boundary 的通用 capability 缺口，可以在修复 consumer 后执行
`intentir.scrc.gate2-capability-recheck.v1`：它必须核对 artifact 位于 parent evidence 内、Replay
bytes 与 receipt digest 一致、case/source lineage 匹配，然后以 **零模型调用**复用同一 artifact 和
同一 Replay，只重新执行当前 Gate2 三腿。输出写入独立 `gate2-recheck-NNN` evidence，并显式记录
`reuses_exact_gate1_artifact=true`、`reuses_exact_gate1_replay=true`、`model_calls=0` 与
`changes_parent_verdict=false`。它不 relift、不修 J、不重做或伪造 Gate1，也不覆盖 parent 的既有
verdict；新的结果只是在所记录 consumer/Q/test/environment 版本下，对该 exact parent artifact 的
独立 Gate2 capability evidence。

```text
SCRC_CERTIFIED
iff TypeCheck(J,C,pi) and Gate1 and Gate2
```

masked-J/J-authority 状态单独报告，不进入上述合取。Production CUDA/RVV 等 backend conformance
也独立于 Gate2，不能替代 Source–Replay、Source–J 或 Replay–J 任一腿。
