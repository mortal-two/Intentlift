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
RefEval 与 production lowering 使用同一个 `J -> Intent MLIR` 入口。于是 `k1,k0` 与
`[k1,k0]`、可唯一推导的 result type、冗余 identity output mapping 不会由 RefEval 再次解释或拒绝；
general affine、缺 predicate/effect 或类型冲突仍在准确边界 fail closed。

RefEval 在发射 MLIR 前调用共享的 `ProjectFunctionalBoundary(J)`，只消费其 ephemeral structured
结果；CUDA production analyzer 调用同一函数。该投影不是新的 J/Gate/artifact，只负责证明全部
影响 logical-tensor evaluation 的 functional B facts 已被消费；π 专用 realization authorization 不在
该投影内。它不改变 `RefEval(J,x)` 的函数类型，也不把 target conformance 并入 Gate2。

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

当前固定 scalar RefEval protocol 为 `j_scalar_cpu_mlir_v9`，shared functional projection receipt 为
`intentir.j-functional-boundary.v2`。v8 相对 v7 扩展唯一 node-less logical-input boundary load；v9
完整继承该能力，并把旧的单 input/单 output/唯一 sum padding profile 提升为 source-neutral 的
multi-reduction functional closure。旧 v7/v8 的 `REFERENCE_UNSUPPORTED` 与 Gate2 verdict 保持原
解释，不由 v9 追认或覆盖。

runtime scalar 或 bound semantic constexpr 若代表 J 的逻辑数据，可以锚定一个 rank-0 J input。
Harness 给 Source/Replay 绑定前端标量，同时给 `RefEval(J,x)` 提供相同值的 rank-0 logical input；
它不能因此被折叠成 C。只有表示 shape/domain extent 的 integral handle 才进入 dimension binding，
不得同时冒充 J 的数据输入。

`typecheck.v7` 在 RefEval 前检查 rank-0 logical input 是否到达 observable output dependency，或由
显式 boundary/shape semantics 使用；direct identity 与 boundary-only 合法路径保留。仅服务
representation/address 的 runtime scalar 不属于 RefEval logical inputs，必须显式为 rank-0
`logical_ref=null` carrier，并只可由 π realization 引用。checker 不读取 Source 名称/位置、不自动
解绑，也不把该窄规则冒充一般 tensor liveness proof；该 TypeCheck judgement 与 RefEval v9 的消费
能力仍彼此独立。

这个 null-carrier 例外不适用于 bound constexpr：只有 implementation-selection 后果的 constexpr
属于 C，其他 bound constexpr 必须有 rank-0 logical-data 或 `x*` extent denotation；两者都不满足时
Harness invalid，不能让 RefEval 忽略、让 π 单独消费。

但只有在 J 自身已经足以机械证明“无 logical guard 的 predicated direct load 具有完整零起点矩形
index/domain、predicate bound 与同轴 typed tensor extent 相等，且该 masked axis 经已建模 closed
pointwise/broadcast 传播后被 `reduce(mean)` 消去”时，`typecheck.v7` 才会在 RefEval 前以
`padded_mean_count_unproven` 拒绝：`masked_contribution` 只闭合分子，当前 boundary 不表达 logical
valid denominator。dense/unpredicated mean、guard-bearing 或 cross-axis/general predicate、
domain/extent 不闭合、mask 位于该 reduction 未消去轴或未知 op 阻断传播时不触发 TypeCheck。
guard/cross-axis/unknown-op 保持 consumer unsupported。此 rule 不读取 Source/C/π，不把一般 RefEval
unsupported 偷换成 J 非法，也不自动改图。

当前 pure functional pointwise 与 reduction graph 可以机械消去物理 logical-domain tail mask，但只在
以下条件同时成立时成立：

- observable output 是直接、无 update/alias/inout 的 functional value；
- memory access 与 J 的 logical tensor/domain 直接锚定；每个 predicated direct access 为全部非
  常量 `0` axis 显式声明完整零起点 logical domain，各 bound 与 typed shape 一致；
- rank-0 `i1` guard 可以是合法 J input，但当前 functional consumer 对任何 guard-bearing access
  predicate 都显式 unsupported；operand membership 不能区分 `flag`/`not flag`，也不能替代 observable
  control-dependency proof；
- direct singleton index `[0]` 只允许对应 logical extent 确为 1；
- 所有仍携带某个 masked axis 的 access 使用同一个 full-domain predicate；predicated load 有显式
  fill，output store 若已由 reduction 删除该 axis 则不需要伪造 predicate；
- padded graph 的每个 logical reduction 恰有一个 J-owned `ReductionBoundary`，显式
  `masked_contribution` 与可选 `identity` 都是 reducer 的中性元素；`sum/prod/max/min/any/all` 进入
  当前闭合集合；若上述窄 rectangular access/dataflow provenance 能证明 masked axis 由 `mean` 消去，因缺少 logical count
  contract 继续 unsupported；
- 普通 trailing broadcast 由 closed value type 决定，non-trailing broadcast 只能消费 J 中显式的
  `boundary.broadcasts`，缺失、错轴或未消费 relation 均在执行前 fail closed。
- masked provenance 若进入 contraction 或其他当前 provenance analysis 未建模的 executor op，必须在
  该 op 处 `REFERENCE_UNSUPPORTED`；不能静默丢弃 provenance 后放行下游 `mean`。

因此 `logical_domain_reduction_padding_v2` 可以消费任意由上述通用构造组成的 pointwise/broadcast/
multi-reduce DAG，例如 max → sub → exp → sum → div；实现不检查 Kernel 名称，也不含 Softmax
公式。`masked_contribution` 是 J 中已有的有效贡献合同，projector 只验证它与 reducer neutral 一致，
不从 Source、C、π、runtime/hidden input 或 named op 重新推导。update、alias/inout、不一致 bound、
未锚定 guard、padded mean 或其他未闭合 topology 必须明确返回 `REFERENCE_UNSUPPORTED`/fail closed。

active `typecheck.v7` 已复用 J-only `functional_reduction_padding_issues(J)`，用于精确报告上述窄
rectangular seed 下已建模 provenance 的 masked-axis → `mean` 缺口。它不把 guard-bearing、
cross-axis/general access 或未知-op consumer capability blanket 判成非法 J；v9 projector 对这些
无法声度擦除的 topology 明确拒绝，并在 masked provenance 首次进入未知 op 时拒绝。
CandidateCheck 不另写一套规则，fixed code 也不补造 count contract。

## 失败与认证

Gate2 失败结束本次实验，不在同一实验中把结果反馈给 Lift/Replay 做局部修补。实验结束后可以
聚合失败、修改系统，再从新实验开始。

```text
SCRC_CERTIFIED
iff TypeCheck(J,C,pi) and Gate1 and Gate2
```

masked-J/J-authority 状态单独报告，不进入上述合取。Production CUDA/RVV 等 backend conformance
也独立于 Gate2，不能替代 Source–Replay、Source–J 或 Replay–J 任一腿。
