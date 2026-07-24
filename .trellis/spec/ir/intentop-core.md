# J / IntentOp 语义核心

> 状态：当前派生规范。历史实现审计见 [IntentOp 健壮性审计](../../../doc/source-notes/IntentOp_健壮性审计_2026-07-21.md)，但审计材料不具有规范权威。

## 目标

J 使用小型、可组合、可执行的语义构造表达 Kernel denotation，而不是枚举 Kernel family 或
前端 API。当前目标构造包括：

1. elementwise expression；
2. logical index/map；
3. aggregate；
4. semantic recurrence；
5. logical update。

支持 claim 必须同时来自真实 parser/TypeCheck consumer、J-only RefEval、Gate1/Gate2 与相关
backend conformance；schema 能保存字段不等于系统已经支持该能力。backend conformance 在这里是
实现覆盖的 regression evidence，不是 pass applicability 或 candidate legality 的定义。

## 所有权

- logical computation、index relation、observable state transition 与 signature 属 J；
- source fusion、materialization、barrier、atomic instruction、async copy 与 pipeline topology 属 π；
- tile、vector、stage、prefetch、buffer 等可替换选择属于 C；
- conflict/order/rounding/tolerance 等会改变允许结果的事实属于 B。

具体 atomic、CAS、privatization 或 frontend spelling 不得进入 J。B 可以授权某种 realization，
但 π 只能引用该 authorization，不能复制其语义。

## Typed boundary

当前语义核心保留以下通用关系方向：

```text
IndexBounds(node, data, indices, axis)
LogicalUpdate(node, base, indices, values, axis,
              combiner, duplicate_policy, order)
ResultRelation(output,
               exact
               | tolerance
               | one_of_updates(update))
```

`IntentFunction.reduce.attrs.axes` 始终是 whole-tensor logical axes；与该节点对应的
`ReductionBoundary.tile_axis` 是每个 program 内物理 tile 的 frontend reduction axis。Replay 在
公开前端 reduction call 使用后者，RefEval 用前者解释 J 的逻辑函数。`tile_axis` 不是 tile extent、
C valuation 或任意公式；identity 与 masked contribution 仍由同一个 B reduction contract 约束。

当前 canonical binary `contraction` 只使用 `attrs.subscripts` 保存 NumPy-style 二元 einsum 关系；
`lhs_contract_axes/rhs_contract_axes` 不是别名，也不会被 fixed code 翻译。TypeCheck 在已有 typed J
能够机械决定的范围内检查两个 operand rank、label extent 与 result shape；不从 Source、π 或 C
补出 subscripts。frontend 能否把一个合法 contraction 直接拼写为 `tl.dot` 是版本化 `U_f` 与 π
stage multiplicity 的消费问题，不改变这个 J denotation。

`semantic recurrence` 不是“最终结果看起来等价”即可省略的提示。若 Lift 观察到 compiler-visible
loop-carried value dataflow，它必须使用现有 structured `loop` 明确给出 trip/scan、每个 carry 与
init、body 内有序 update，以及一一对应的 next/yield/result。loop body 内 contraction 的 canonical
subscripts 和 result-axis orientation 必须直接匹配 update 消费；只有真实 logical transformation
才能插入 transpose。π 没有公式所有权，不能把缺失的 recurrence、init/yield 或 contraction
orientation留给 Replay 恢复。

这些关系必须由 RefEval、comparator 或 production backend 实际消费。缺少必要 boundary、consumer
不支持、关系参数非法或 authorization 不完整时必须 fail closed，不能退回 case-specific Python
oracle 或旁路 metadata。

当前 SCRC 主认证只接收 deterministic / contract-deterministic 的可观察合同。当前真实 enum 是
`exact/tolerance/one_of_updates`；`one_of_updates` 只是受限 update relation，不表示主 Gate2 已支持
一般集合值语义。一般 `allowed_result` 与逐观察 `Accept_{J,Q}` 留给未来，不得从有限次 membership
外推完整行为集合包含。

## Comparator class

B 只能从系统支持的封闭 comparison class 中选择并声明必要语义合同。外部比较政策 Q 根据
dtype、reduction extent、reassociation 权限等构造确定性、对称的 comparator，并约束 tolerance、
NaN/Inf 等参数。Lift 不得写入任意 comparator 或自行放宽及格线。

comparison class 只定义观察比较类别，不是 backend transformation permission。实际 operand/result/
cast dtype 属于 A typed graph；B 可以规定 order、reassociation、rounding、tie、duplicate 与允许观察
关系；Q 只实例化 comparator。`tolerance` 不能授权 backend 改 reduction tree、accumulation dtype、
FMA 或 transcendental 实现。

## Intent MLIR 中的 A/B/C 投影

production materializer 先对唯一 J 做严格 J-only 的确定性编码：

```text
A -> structured Intent ops/regions/value graph
B -> closed attrs/effects/verifier facts/ephemeral analyses
```

得到 `M0=Emit_v(J)` 后，backend 只能通过独立后置步骤：

```text
S_C = Attach_C(M0,C,t)
BaseIR(S_C) = M0
```

将 C 投影成 discardable schedule annotations、search seed 与 candidate ordering。若使用 decorated
working MLIR `M_C`，必须满足 `Erase_C(M_C)=M0`。parser、`Emit_v`、pass default 或 kernel
recognizer 不得产生冒充 C-derived 的 attr；该步骤不修改 J，也不产生新顶层 artifact。

Intent MLIR 可以同时携带三类投影和 compiler internal state，但它不是第二份 J。所有权由信息来源
决定，不由 attr 物理附着在哪个 op 上决定：

- `reduce.assoc` 可以是 B-derived closed attr；
- `intent.loop.tile_hint` 只有在 canonical Emit、LoopOp denotation/verifier 与 pass applicability 均不
  依赖它，并且仅由 `Attach_C` 生成、满足 erase invariance 后，才可作为 C-derived annotation；
- 禁止恢复任意 per-op `layer_b` 或让 C annotation 反向改变 J denotation。

structured ops 必须保留到 generic implementation-family 选择完成。Op.td description 只写
extensional semantics、typed operands/results、logical axes/order 与 closed contract，不写 NVVM、
`arith.select`、Hillis--Steele、bitonic、shared-memory/barrier、one-block-per-row 或其他 target recipe。
target recipe 属于 production pass library。

当前实现尚未对齐这两条规则：`IntentOps.td` 仍含多处 target recipe；`emit_mlir.py` 总是发射
`loop.tile_hint`，而 `ConvertIntentToLinalg.cpp` 用它决定 slice size/trip count、缺失时默认 1。
这些是后端 B0 的已知清理项；在解除依赖前不得把当前 `tile_hint` 宣称为可删除的纯 C attr。
清理时仍须保留真正改变 denotation 的 extensional equation、tie/order/bounds、result type 与
region/yield contract。

backend 根据 A 识别通用 implementation family，根据 B 和 target capability 检查 pass precondition，
根据 C 排序候选。缺少某个 pass 所需的 B fact 时，该 pass 不适用；不能从 Source、π、kernel name、
Gate2 tolerance 或运行结果补出前提。

## 禁止恢复的旧旁路

- per-op `layer_b` 或第二份 B；
- per-op `out` 绕过函数级 result/destination relation；
- J 中的 schedule tile、atomic/barrier/async realization；
- 以 named op、RuleCard 或 Kernel recognizer 代替可组合语义；
- 只记录“已消费”而实际忽略 B relation。

新增语义构造必须先给出 denotation、typed operands/results、授权边界、RefEval 行为与 fail-closed
能力，再进入代表性 canary；不得根据单个失败样本直接扩充 ontology。
