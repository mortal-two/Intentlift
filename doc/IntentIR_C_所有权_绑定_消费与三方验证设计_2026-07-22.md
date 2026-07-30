# IntentIR C：所有权、绑定、消费与验证设计

> 状态：新架构核心设计（2026-07-22）
>
> 修订依据：[高级 AI 对 C 设计的纠正](source-notes/高级ai对c的纠正.md)、[高级 AI 对 C 设计的自省](source-notes/高级ai对c的自省.md)
>
> 联合复审：[三份核心设计复审与四项修正](source-notes/高级ai对三份核心设计的复审与四项修正_2026-07-22.md)
>
> 数学与 claim 总述：[IntentIR / SCRC 最终数学、执行流程与论文表述](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)
>
> Claim 收紧复审：[最终数学与三份设计的局部收紧建议](source-notes/高级ai对最终数学与三份设计的局部收紧建议_2026-07-22.md)
>
> 范围：定义 C 在 Lift、Replay、Gate1、Gate2、reference semantics 与直接 production lowering 中的最小合同
>
> 核心不变量：**证明相关 artifact 严格只有 `(J,C,pi)`；Replay 只看 `J + pi + MaskC(C)`。**

## 1. 最终判断

本版撤回此前草案中两个不属于当前系统的扩张：

1. 不再把 `CBindingMap`、portable hints、Generic Resolver 写成 Lift 的额外产物；
2. 不再引入 LoweringAgent、C 生成 Agent、retune agent 或额外 `C_target` artifact。

正确主架构是：

```text
Lift_H(Source) -> (J, C, pi)

TypeCheck(J,C,pi)

Replay = ReplayLLM(
    J,
    pi,
    MaskC(C),
    public frontend context
)

Gate1:
    Observe(Compile_H(Source, C))
    ==
    Observe(Compile_H(Replay, C))

Gate2:
    Close_Q(Source(c0), Replay(c0))
    Close_Q(Source(c0), RefEval(J))
    Close_Q(Replay(c0), RefEval(J))

Production:
    Lower_t(J, C)
```

其中：

- Harness 是预先存在的隐藏求值环境，不是 IR，也不是 Lift 的第四输出；
- C 是一个对象，不拆成多个顶层 artifact；
- logical signature 属于 J；任何工程 `InterfaceView` 都只能是 `J` 的确定性投影；
- Replay view、Gate view 和 lowering view 都是 C 的确定性投影；
- `RefEval` 在接口层面禁止访问 C；
- production lowerer 可以消费或忽略 C，但不读取 pi；
- backend cost model 与 retune 不进入 SCRC 核心设计或当前 28-kernel 认证验收；其 production
  合同由 [第五份后端设计](IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)
  独立定义。

## 2. 严格三对象因子化

### 2.1 Lift codomain

证明相关的 Lift 输出严格定义为：

```text
Lift_H(S) -> (J, C, pi)
```

不能扩张为：

```text
Lift(S,H) -> (J,C,pi,binding_map,hints,resolver_output,...)
```

这里下标 `H` 表示 Lift 在既有 Harness/调用环境下工作；H 是外部环境，不是 Lift 新生成的对象。

### 2.2 Replay 因子化

对 frontend `f`，Replay 定义为：

```text
R = W_f(J, pi, MaskC(C); U_f)
```

其中 `U_f` 是固定、公共、与当前源码无关的 frontend 文档和工具上下文。

禁止存在额外的当前源码条件化输入：

```text
Z = phi(Source, Harness)
R = W(J,pi,MaskC(C),Z)
```

除非可以证明 `Z` 是 `J/pi/MaskC(C)` 的确定性函数，否则它就是未受约束的第四信息通道。

### 2.3 条件信息隔离

Replay 必须满足：

```text
R independent of
    Source,
    Harness,
    concrete C values,
    source/replay binding
given
    J,
    pi,
    MaskC(C),
    fixed public frontend context U_f.
```

也就是说：

```text
Source -> (J,C,pi) -> Replay
```

是当前研究必须保住的因子化命题。

### 2.4 Logical signature 属于 J

输入、输出、runtime scalar、destination、dtype、rank、shape、alias、inout 与 observable effect
都属于 J 的 logical signature 或 J/B 边界。它们不能形成独立的第四个 Lift artifact。

如果前端适配器或实现代码需要一个接口视图，只允许定义：

```text
InterfaceView(J)
```

`InterfaceView(J)` 是 J 的确定性投影，不具有独立语义，不单独序列化为证明对象，也不作为
Replay 的额外输入。C slots 仍通过 `MaskC(C)` 单独提供；Harness 只保存 source/replay 的机械
调用坐标。

Lift 可以从 H 获得匿名、未分类的 bindable coordinates，以便形成 J signature 和 C；该机械输入
不是 `InterfaceView` artifact，也不能替 Lift 预先完成 J/C 分类。

### 2.5 Proof-irrelevant sidecars

实现可以保存 audit、location、diagnostic、provider receipt、compiler receipt 等 sidecar，但它们
必须是 proof-irrelevant：

- 不进入 Replay；
- 不进入 Gate verdict 的功能输入；
- 不改变 production lowering；
- 不成为从源码到 Replay 的额外通道。

LLM response 在进入证明相关路径前必须经过确定性的 Decode/Elaborate，成功后才构造出 typed J。
它将 closed ContractExpr surface sugar（例如 `k1,k0` 与 `[k1,k0]`）解析为同一 structured axes，
并物化唯一可推导类型、消去可机械证明的恒等表示。接受后只有一个 J；不存在 `J_wire` 与
`CanonicalJ` 两个 artifact。解析 AST/axes 只是 J 的私有派生缓存，不读取 Source、C/π、Harness
concrete value 或 Gate 结果，也不得猜测缺失语义。Replay、RefEval 与 production lowering 只消费 J。

#### 2.5.1 Lift 内的只读 CandidateCheck

为减少复杂 artifact 在静态 JSON Schema 通过后才暴露 dependent type/reference 错误，同一个
branch-local source-aware Lift owner 可以在累计 active wall-time 内，对自己已经写出的完整 candidate 调用只读、
非认证的 `CandidateCheck`：

```text
ctx = {
  anonymous_compiler_view,
  schema_version,
  decode_version,
  typecheck_version
}
CandidateCheck_v(candidate_bytes, ctx) -> PASS | diagnostics
```

该工具共享正式 Runner 的同一 source-free strict parser、窄 Decode/Elaborate、anonymous-view
consistency 与 `TypeCheck(J,C,pi)` callable core；正式 Runner 另执行 source-identifier privacy、
receipt/provenance，并在提交后重新运行该 core。禁止在 worker 中维护第二套近似 checker。该工具
还必须满足：

- 只检查当前完整 candidate；可以构造 ephemeral validation view，但不得合成、补写、分类、修改或
  返回新的 J/C/π candidate/artifact；
- diagnostics 只包含 closed `protocol/status/rule_code/path/observed_refs_or_types`，不返回建议值、
  canonical artifact、patch 或 corrected artifact；
- 不读取 Replay、Gate1、Gate2、RefEval、compiler observation 或历史 sibling；
- 调用次数与诊断只进入 proof-irrelevant audit；
- tool `PASS` 不签发证书，正式提交后 Runner 必须重新执行权威检查；
- agent 修改后仍提交完整 `(J,C,pi)`，不能只交局部 patch。

为使完整候选不会因 worker 忘记显式调用而绕过这一机械诊断边界，提交动作对尚无本
session `PASS` 的精确 candidate bytes 先调用同一 CandidateCheck。失败只返回 closed
diagnostics 并留在当前 worker 中，不提交也不修改 artifact；只有与某次 `PASS` 对应的精确 bytes
完全相同时才可复用，任何不同 bytes 都必须重检。
该前置检查与显式调用共享同一 diagnostics path，且不取消提交后 Runner 的独立重检。

当前 branch-local Lift owner 不以低的 model-turn/CandidateCheck 次数作为能力停止条件；
`step_limit=0` 与 check limit 0 表示 time-governed，默认每 branch 3600 秒累计 active wall time。
receipt 仍按 `explicit`/`submission` trigger 记录 calls/evaluations，并记录 model/physical calls、
elapsed 与 terminal reason。submission preflight 的 `PASS` 仍非认证，正式 Runner 必须重跑
TypeCheck。若一次 pause 前没有提交候选，只允许把最后一次**实际执行**的 source-free checker
诊断中最多 20 条 `rule_code/path` 摘要追加给同一 branch owner continuation；owner 自己的 Source
conversation 与受限 workspace 均保留，host 只原子刷新 closed `feedback.json`，当前完整 draft 不被
历史 artifact 覆盖；不得新增 observed hidden value、fixed semantic patch 或 sibling 信息。旧的大型
tool argument/observation 可在下一 model packet 中替换为仅含 hash/长度的 deterministic receipt，精确
内容仍从同一 workspace 重读。首个 owner packet 可精确内嵌不超过 64 KiB 的、Lift 本来就有权读取
的 Source/Harness/feedback 与 source-neutral contract/preflight/minimal-example 原始文本；超限时只给
bytes/hash/read command，不由 fixed code 摘要。
LLM 可以用通用 exact-text patch 修改自己明确给出的唯一字符串片段，
但固定代码不得解析 J/C/π、分配 handle、推断字段或返回 corrected patch。这个反馈只是延续已有机械
拒绝，不给 Harness/runner 生成 J/C/π 的权限。Gate1 diff 也只回到
该 branch 的同一 Lift owner；Replay 每个 formal round 仍全新且不可见该 conversation。

为消除“写完整文件后再用一个高 effort 回合选择 submit”的纯编排延迟，LLM 可在一个 function
action 中提供完整 `candidate_text`；runtime 原样原子落盘并立即调用上述既有 submission preflight。
同理，LLM-authored exact-text patch 可显式请求替换后立即 preflight。两条路径都不选字段、不修改
bytes、不改变 CandidateCheck/Runner 权威；FAIL 仍只返回 closed diagnostics，PASS 后外部 Runner 仍
独立重跑。

`ctx` 精确等于该 agent 原本已经获准读取的匿名机械 Harness compiler view 与公共协议版本；
CandidateCheck 禁止读取未声明的 ambient host state 或 Source text，也不增加新的 source-conditioned
信息通道。
该工具扩大的是即时裁决能力，不是固定代码的语义生成权；Harness 仍不得替 Lift 完成 J/C 分类。
CandidateCheck 与新增 dependent rules 必须随下一 TypeCheck protocol version 实现并写入 receipt；
在实现和版本同步前不视为现有 worker 能力，也不重判历史证书。

当前已落地的 fresh-run 合同为
`artifact.v8 / decode.v12 / typecheck.v20 / candidate-check.v1 / replay-check.v3`。Lift LLM 仍完整生成全部
`(J,C,π)`：artifact.v4 的 `J.value_dimension_relations` 与
`J.boundary.consumer_extensions`、artifact.v5 的
`J.boundary.access_guard_relations`、artifact.v6/v7 的 typed `effect_if/value_if` 都是 J-owned
authored facts；artifact.v8 新增的 `pi.physical_iterations` 与
`pi.physical_loop_carriers` 是 π-owned source-realization facts。Decode、TypeCheck 与 CandidateCheck 只
解析、验证、拒绝和诊断，不补造 relation、axes、polarity、pre-state、inout、Layer-A op 或 physical
time。v1-v7 仍按各自 wire/receipt 解释，不因当前 schema 增加空数组或被重标。

对 v5 guarded store，Layer A 必须显式包含 `select`；目标 tensor 必须精确同时作为 J input/output，
并由唯一 `mode=inout` parameter 绑定。tensor handle 按 inputs 后 outputs 的首次声明连续编号；同一
inout tensor 在 outputs 中复用 input 的 `t*`，不分配新 handle，也不借 `alias_of` 表达逻辑 inout。
loop 的 `carry.init_value` 必须是该 tensor，inactive branch
必须是 current body argument，select result 必须成为 `next_value`；非 loop 的 inactive pre-state
就是该 tensor，return/output mapping 必须指向 select result。relation 的 `axes` 是同一 select guard
的唯一 broadcast authority，不能在 `boundary.broadcasts` 重复。TypeCheck 不会因 dtype/shape 相同
而接受另一个 input 代替 pre-state，也不会插入 merge。

`typecheck.v20` 完整继承 v18/v19、v14-v17 typed-control 合同以及下面 v7 及其前序合同，并把
floating `log2` 纳入 closed tolerance requirement。v7 完整继承 v6；v6 继承 v5，v5 继承 v4 的 rank-0 closed signature、
Layer-A total typing 与显式 non-trailing broadcast 合同：canonical graph 的每个 Layer-A result 都必须
具有 closed total type，`map_binary` 默认严格服从 NumPy broadcast，非 trailing 轴只能消费
Lift 已在 `J.boundary.broadcasts` 中显式声明的 relation。v5 新增已知 `reduce` 与 binary
`contraction` 的 canonical fail-early；别名、缺失/非法 attrs 以及可机械判定的
axis/rank/shape 矛盾直接拒绝，fixed code 不猜测、翻译或补齐。v5 还拒绝将
`opaque(mode=value,dtype=none)` absent ABI carrier 用作 π instance semantic ref、realization
authorization subject 或 refinement carrier，不赋予它 logical denotation。

v6 将 RefEval/production 已共享的 `FunctionalBoundaryProjection` 中一个严格 J-only 必要条件前移到
同一 TypeCheck/CandidateCheck core：对可识别的零起点矩形 logical-domain profile，direct access
index、padding predicate、`boundary.domains` 与 typed tensor extent 必须显式一致。形如
`k0 < x0` 的 padding predicate 必须直接索引 `k0`，并由 J 自身声明匹配的
`0 <= k0 and k0 < x0` domain；checker 不得从 Source、C、π、tensor shape 的“显然含义”、runtime
value 或 hidden input 补造缺失 domain。该 rule 只拒绝此封闭 profile 内的机械矛盾；general
affine/交叉轴 logical predicate 仍是合法 J，不因当前 RefEval 尚未实现就被 TypeCheck 一概禁止。
合法 masked/padded access 的 predicate、fill 与一一对应 effect 保持原样，checker 不删除、不改写。

v7 再收紧一个可机械决定且 source-free 的 rank-0 logical-input liveness 合同。凡 signature 通过
`logical_ref=t*` 声明的 rank-0 J input，必须进入某个 observable output 的反向 value-dependency
闭包，或被 J 自身的显式 boundary/shape semantic relation 使用；direct identity/`returns`/
`output_mapping` 路径与合法 boundary-only guard 均计为使用，dead A node 或仅在 π 中 route 不计。
当前规则刻意不推广到一般 tensor liveness，以免把尚未证明的全图性质变成拒绝条件。

同一 v7 还加入独立的 `padded_mean_count_unproven` 必要条件，但只在无 logical guard 的 predicated
direct load、完整零起点矩形 index/domain、predicate bound 与同轴 typed tensor extent 相等，且该
masked axis 经已建模 closed pointwise/broadcast 位置传播后确由 `reduce(mean)` 消去时拒绝。零
`masked_contribution` 只能闭合分子，当前 B 没有 logical valid-count/denominator contract，因而不能
证明 mean 的除数。dense/unpredicated mean、guard-bearing 或 cross-axis/general predicate、
domain/extent 不闭合、掩码位于该 mean 未消去轴或传播进入未知 op 而无法证明者不触发 TypeCheck；
guard/cross-axis/unknown-op 留在 consumer boundary 并明确 unsupported。checker 不读取 Source/C/π、
不把 `mean` 改写成别的图；若 Source 本来是 masked sum 后显式除以逻辑计数，Lift 必须仍完整生成
`reduce(sum) + explicit div` 两阶段。

只有明确的 `kind=scalar, mode=value, rank=0, logical_ref=null` 才表示 public-dtype
representation-only ABI carrier：它不创建 J input，不得参与 J output/boundary/shape semantics，但可由
π 引用以保留 address/representation realization。分类由 Lift 根据 Source use 完整提出，checker
不读参数名/位置，不从 Source 猜分类，也不自动清空 `logical_ref`、删除 input 或重编号 tensor。
因此 v6/v14 及更早 receipt 仍保持原 verdict，必须以 v7/v15 fresh run 得到新结论。

该 wire form 只允许 runtime scalar；`bound_constexpr_values` 不是另一种 representation-only carrier。
bound constexpr 必须由 Lift 根据它在 Source 中的全部 use 选择 J logical denotation：
logical data/predicate/result 绑定 dtype-compatible rank-0 J input，integral extent-only 绑定
declared `x*`。`constexpr` 身份本身不证明它是 graph selector；只有所有语义 use 都严格
限于 compile-time control/discrimination，且 concrete value 不进入当前 active graph 的
shape/index/mask/arithmetic/boolean/result，才可保留
`kind=constexpr,mode=value,rank=0`、compatible dtype、无 `logical_ref` 的 p* callable carrier。
参与 active graph 的 constexpr 即使被 constant-fold 也必须走 t*/x*；同时是 data 与精确 shape
dimension 时使用 t* 及 Lift 作者化的 `value_dimension_relations`。纯 selector 的值只属于
Harness `bound_constexpr_values`，不写入 J/C/π；Lift 作者化完整 active graph。fixed
Decode/CandidateCheck 只检查已作者化的 wire/type 形式，不分类、不根据值选图、补图或修改
artifact。Replay 保留 constexpr ABI，RefEval 与 production 不消费纯 selector 值；string
constexpr 只能在纯 selector 情形使用这一 Harness specialization 形式，不支持的 logical string data 应拒绝。
只影响 layout/tile/stage 等 implementation
selection 的 constexpr 应由 Harness 分类为 C slot；static specialization carrier 不得进入 π topology。

Gate2 scalar RefEval 当前使用 `j_scalar_cpu_mlir_v24`，shared functional projection 使用
`intentir.j-functional-boundary.v22`，typed relation-indexed reference 使用
`intentir.j-contract-indexed-cpu-reference.v2`。它们完整继承 v14/v13、v12/v11、v10/v3 的唯一 node-less logical-input load 与严格
J-only 通用 padded functional graph 消费：每个 predicated direct access 的完整 logical domain、
每个 logical reduction 的唯一中性 outside-domain contract，以及 pointwise operand 的 closed
broadcast relation 必须同时闭合；于是多 input/output、max+sum 等多 reduction 与普通 pointwise DAG
不再被旧 single-sum topology 人为拒绝。

artifact.v11的J-owned `dimension_relations`只声明logical dimensions之间的closed identity/常数
arithmetic关系。它不属于C：Lift从Source作者化，RefEval与production按J求值；Harness concrete
valuation、grid、C与Source TTIR都不得补造或覆盖该关系。

v10/v3 只新增当前 Gate2 winner invocation 的 runtime predicate tautology proof：主 SCRC runner
先选定 winner，再把该次 J logical inputs 与 closed dimensions 交给 RefEval-only 入口；projection
library 本身不选择 winner。该入口只消费 rank-0 logical `i1`/integer values，逐 access 按真实
布尔极性、direct axis、零下界和 closed extent 证明 predicate 在完整 logical domain 恒真。它不修改
J、不进入 Lift feedback、不读取 Source/C/π；v3 projection 中的 invocation-local proof subreceipt 只
保存 access id、J predicate digest、logical input refs 与 domain axes，不保存 raw value 或 value
digest。这是 audit-data minimization，不是 confidentiality/privacy proof。production 仍调用没有
runtime value 接口的 value-free projection，不能消费该 subreceipt。

value-free projector 不读取 runtime/hidden input，也不生成 padding 公式；上述窄 rectangular
provenance 可证明 masked axis 被 `mean` 消去却没有 logical count contract、
缺 domain/fill/reduction/broadcast、update/alias 等继续 fail closed。
无法形成上述全域证明的 guard-bearing access 仍 consumer unsupported（避免把 dead operand
membership 当控制证明）；masked provenance 进入 contraction/其他未建模 op 时也在该 op 处拒绝，
不能静默丢弃后放行下游 mean。这两项不把合法 J 改判为 TypeCheck 非法。CPU host 为标准 libm ABI
在共享 Intent-to-Linalg 之后单独展开 `rsqrt -> 1/sqrt`；共享/GPU lowering 不调度该 pass。
既有 typecheck.v3/v4/v5/v6、`j_scalar_cpu_mlir_v7/v8/v9` 与 functional-boundary v1/v2 receipt 及
Softmax/RMS 历史 verdict 保持原解释，不由 v10/v3 追认或重判。

v11/v9 进一步逐项消费 v4/v5 J relation：检查 value/dimension `eq`，验证并应用
consumer-specific typed outside identity，并对每个 access guard 产生 receipt 或明确 unsupported。
当前 guarded store 只执行 A 中已有的显式 `select`，不由 relation 隐式 merge；guarded load/update
尚无 RefEval pre-execution materializer。v9 还可证明同 producer、同完整 observable typed value、
单静态轴、常量 unit-width、无 predicate、无重叠无缺口的 direct-store partition。v11 的 closed
structured J-only reference 可执行 common scalar lowering 尚不支持的 mixed typed binary
contraction。后两项只扩展 Gate2 reference consumer，不是 production target recipe 或 permission。

v12/v11 移除对 logical inout 的旧 blanket rejection：只有同一 `t*` 已以相同类型同时声明为 J
input/output、由唯一 `mode=inout` 参数绑定，并且 pre-state load、post-state store、effect、
canonical output binding 全部由 J 显式闭合时，RefEval 才分别消费 input binding 的旧值与产生 output
binding 的新值。`alias_of` 与 `boundary.aliases` 表达的物理别名仍明确 unsupported；consumer 不插入
merge、不复制 buffer，也不从 Harness/π 猜测 inout。

v13/v12 进一步只从 J 识别显式 inout post-state reload：唯一 inout handle 已有唯一 output mapping，
先由同 producer store，再以完全相同的 index/predicate load，且 boundary/scoped effect order 均证明
store-before-load，后继 A node 又直接消费映射后的 post-state value。consumer 只把该物理 reload 投影回
J 已有 SSA value，并产生 receipt；不读取 Source/C/π/Harness，也不补 state edge。

v14/v13 只在可由 J 证明 outside-domain value 不可观察时允许 predicated load 省略 fill：该 load 必须
绑定 pointwise node，所有依赖路径只能经过 `map_unary/map_binary/select/cast`，所有可观察 destination
都由完全相同 predicate 的 store 支配；reduce/scan/loop/transpose/pad、不同 predicate 或未匹配 effect
仍 fail closed。canonical output destination 可由 function `returns` 或 boundary `output_mapping` 任一
唯一给出；两者同时出现必须一致。shared J→Intent MLIR 入口与 structured reference v9 消费同一个 deterministic
`JElaboration.output_bindings`，不要求 LLM 重复序列化同一事实，也不新建第二份 J。

历史 `j_scalar_cpu_mlir_v20` / `intentir.j-functional-boundary.v19` 完整继承 v17/v16 的 product-domain mixed-radix bijection、
精确 axis-permutation transpose mask provenance、`log2` 与 IEEE special literal 均是共享 J consumer 能力。
它们要求完整 factor/digit/stride、精确 authored permutation 与同 predicate，不能由 C slot、C value
或 schedule role 补证；不满足时仍 fail closed。因而 C 既不获得新语义所有权，也不成为该 proof 的
membership oracle。
v19/v18 对多个 closed inout store→reload stage 与 `context_domain` 的消费也不改变 C 所有权：前者
完全由 J producer/dataflow、coordinates/predicate、effect order 与最终 output binding 决定；后者
是未参与 logical access/predicate 的矩形 J domain receipt，可供 π 锚定 program context，但 C slot/
valuation 都不能补 domain、选择 program axis 或把中间 store 变成 observable output。
v18/v17 再把两条既有 J-only 规则做严格组合：若 predicate-bearing whole-input load 没有
`node_ref`，只有当它是该 logical tensor 的唯一 load，且 canonical J 中该 tensor 恰有一个直接
pointwise consumer 时，missing-fill provenance 才从该 consumer 开始传播。零个或多个 direct
consumer、非 pointwise consumer、不同 predicate、未闭合 broadcast/effect/output 均继续 fail
closed。consumer 只解析 J 已有 dataflow，不补 `node_ref`、不修改 artifact，也不读取 Source、C、π
或 Harness。
v20/v19/v8 的 edge-scoped padding 与 conditional effect full-store 也完全不读取 C：前者只接受 J
中匹配 `index_bounds` 的独立 gather edge，后者只消费 J `effect_if` arm producer 和 Layer-B
access/effect destination。C slot/value 不得提供 predicate、branch value、output pre-state或补造
未执行 arm 的结果。
dynamic iota/div range 与 bounded-gather footprint同样不能从 C tile 推导：`x*`、runtime integer
carrier、`index_bounds`、consumer extension 与 access edge必须已在 J/B 闭合；C 不提供 divisor
nonzero proof、dimension identity或 node-less/node-bound load ownership。
v21/v20 的 direct singleton 与 sibling-`effect_if` full-store control partition，以及 v22 的
ContractIR indexed pointwise reference，都不让 C 获得语义权力。v22 只执行 J 已写出的 consumer-local
index/predicate/fill/store；一个 node 的 access 不传播给 sibling，C tile/value 也不能补 relation。
缺失 relation 会在 Gate2 形成数值分歧并要求 fresh relift，不能以 C 调参或 RefEval repair 追认旧 J。

当前 Lift knowledge v69/v65 继承 v68-v42 对 `fill_value` 的要求：它是 Source 对应 load 的实际 `other`，
不是由 C tile/role 或 downstream neutral value 重写的量；MiniSWE Replay v12 / Codex Replay v6 继承既有 typed rank/reshape fidelity
同样不读取 C value 来选择 shape。v56 的 bound-constexpr/shape-coordinate binding 仍是 J-owned
equality：adapter 只暴露 source-grounded、valuation-matched 的匿名 Harness relation，Lift 完整选择
direct `x*` 或 authored `t*=x*`；C 不拥有、也不能补证该关系。两项只约束 LLM authoring，不转移
任何语义所有权给 C 或 fixed code。
Lift/schema/preflight 对应升为 `intentir.lift-knowledge.v21`：LLM 仍须完整选择至少一个 authored
binding；该版本只删除互相重复的双写要求，不允许 fixed code 从 store、Source 或地址关系猜 value。
`intentir.lift-knowledge.v22` 再明确：每个 semantic reduce/inclusive-scan 都由 Lift 写自己的 local-axis
algebra boundary；predicate-bearing load 的 physical outside lanes 若沿 closed pointwise path 到达
sum/max/min reduce/scan，Lift 还须逐 consumer/operand/access 写 typed `consumer_extensions`。load fill
或 reduction identity 不能替代该关系，checker/consumer 仍不自动推导。

### 2.6 显式语义所有权投影

定义：

```text
SpecProj(J,C,pi) = J
```

该式只表示 artifact grammar 中 operator、predicate、result relation、access/effect contract 等
显式逻辑语义声明由 J 独占。它不是实现行为集合的等式，不能推出 Replay 已正确实现 J、C/π
与语义统计独立，或所有 admissible C 都保持 J 行为。若 `SemDecl` 只计 closed-world schema 中
直接创建的显式逻辑语义声明，则 TypeCheck 的目标性质是：

```text
TypeCheck(J,C,pi)
    -> SemDecl(J,C,pi) = SemDecl(J)
```

行为结论仍必须由实际绑定实例上的 TypeCheck、Gate1 和 Gate2 联合支持。

## 3. J、C、pi 与 Harness

| 对象 | 数学身份 | 所有内容 | 禁止内容 |
|---|---|---|---|
| `J=A+B` | 唯一 denotational semantic object | logical signature、value/state relation、domain、predicate、fill、effect、alias、result relation | schedule、frontend API、source topology |
| `C` | denotation-preserving implementation-selection object | anonymous slots、binding kind、value type、concrete valuation、source-neutral role，以及可选的 J-owned subject/axis attachment | J 语义副本、predicate/formula、kernel-specific executable policy、transformation permission |
| `pi` | source realization witness | instance、region、route、slot、order、sharing、placement、representation topology，以及只引用 J-owned dimension/axis 的 closed program-coordinate mapping | C value、J 语义副本、Source/target recipe、production target policy |
| `H` | 隐藏求值环境 | inputs、C values、source/replay invocation coordinates、grid、device、compiler options、receipts | 新 IR、语义分类、Replay construction |

最重要的关系是：

```text
J：算什么；
C：选择哪个保持 J denotation 的实现参数；
pi：原源码怎样 realization J 与 C；
Harness：给匿名坐标赋本次真实值并执行。
```

## 4. C 的数学定义

### 4.1 Denotation-preserving implementation selection

设：

```text
[[J]](x)
```

是 J 对输入 `x` 允许的可观察结果集合。设 target `t` 在 implementation selection `c` 下生成：

```text
I_t(J,c)
```

合法选择集合为：

```text
Adm_t(J,H,device_t,toolchain_t)
```

C 的纯调度性质是：

```text
for every c in Adm_t(J,H,device_t,toolchain_t):
    Obs(Exec(I_t(J,c),x)) belongs to [[J]](x)
```

这是实现族的设计合同或未来 backend conformance 目标，不是一次 SCRC 证书已经证明的全称定理。
当前在线证书只针对 Harness 实际绑定的：

```text
c0 = Bind_H(C)
```

因此论文与实验报告应写成 **instance-wise certification under the observed c0**。若测试多个
配置，只能报告被实际测试的有限集合；不得从单个 c0 外推到所有 admissible C。

C 可以改变 loop、tile、vector width、thread/warp mapping、pipeline、buffer、memory placement 与
允许范围内的浮点 association，但不能改变 `[[J]]`。

不同 C 的结果不要求逐位相等，只要求满足同一个 J/B relation。

### 4.2 J 完整，C 可稀疏

必须满足：

```text
J 必须完整；C 的缺失或 target 不理解只能影响性能或 source fidelity，不能补充 J 语义。
```

Gate2 的 J reference 不读取 C，正是为了检测“把语义事实错误塞进 C”这一类所有权错误。

## 5. C 是一个对象及其确定性视图

### 5.1 Canonical C

定义：

```text
C = (Sigma_C, nu_C)
```

其中：

- `Sigma_C`：匿名参数化 slot schema；
- `nu_C`：Harness 持有、解析并提交的 concrete valuation。

当前真实 canonical slot vocabulary 是：

```text
name, kind, role, value_type, required
```

其中 `role` 已是必填 source-neutral schedule metadata，并非仍待决定的可选字段。真实多-slot
artifact 已出现两个 `role=tile_extent` 对应不同结构位置的情况；production backend 又不能读取
Source/π，因此 role-only 不能一般性消歧。

`artifact.v2` 已在上述五个字段之外增加两个可选 closed attachment，当前 `artifact.v8` 原样保留：

```text
subject_ref : optional member of CSubject_v1(J)
axis_ref    : optional (non-negative integer | declared symbolic dimension x*)
axis_ref => subject_ref is present

CSubject_v1(J)
  = NodeIds(n*) union DomainIds(d*) union AccessIds(a*)
    union ScopeIds(q*) union StorageIds(b*)

axis_ref in AxisRefs_v1(J, subject_ref)
```

`subject_ref` 只回答“该 prior 作用于 J 的哪个稳定结构”；`axis_ref` 只回答该 subject 内的 logical
axis。它们不允许引用 π/C/source handle、realization authorization 或物理 thread/lane/warp axis，
也不授权某个 implementation family、参数值或 transformation。全局 compiler prior 可以省略
subject；局部 prior 缺失或含混时 backend 必须安全忽略，不能从 Source/π 猜测。

`tile_extent` 与 `reduction_tile_extent` 是当前 closed role vocabulary 中用于 structured J loop 的
`pi.iterations` missing-totality 的两个保留值；它们不是判断一个已经显式声明的 iteration relation
是否合法的 hidden type tag。显式 relation 可以引用 custom-role slot，只要该 slot 是 required
positive integer、attachment 指向同一 J subject/axis 且满足对应 dependent checks。反过来，固定核
只有在协议明确给出可枚举的保留-role pair 时才能机械报 missing；custom role 的误标、以及整个
Source physical region 被 Lift 遗漏，仍靠 Gate1 topology 暴露。

`AxisRefs_v1` 是 closed dependent domain，不由 consumer 自行解释：

- `reduce n*`：该 A node 的显式 reduction input-axis ordinals；
- `loop n*`：其唯一 trip axis 的 ordinal `0`，以及显式 `trip.axis=x*`；两种 spelling 等价地指向
  同一 trip axis，均可供 C attachment 与 iteration dependent check 使用；
- `domain d*`：canonical ContractExpr parser 给出的 ordered logical-axis ordinals；若某一轴显式绑定
  declared symbolic dimension `x*`，该 `x*` 也是同一轴的合法 ref；
- `access a*`：canonical structured index vector 的 axis ordinals；若对应 logical extent 显式为
  declared `x*`，该 `x*` 也是合法 ref；
- `scope q*`、`storage b*`：空集；它们只能承载无 `axis_ref` 的 scoped prior；
- 其他 `n*`：v1 仅在该 closed A op contract 显式声明 logical axes 时返回这些 ordinals，否则为空。

整数始终表示 subject 自己的 canonical logical-axis ordinal；`x*` 必须已经由该 subject 的 closed
typed relation 显式关联，不能从 shape、role、Source、π 或“唯一可能轴”猜测。TypeCheck judgement
因此精确写成 `axis_ref in AxisRefs_v1(J,subject_ref)`；集合为空、越界或无法唯一解析均拒绝。

物理 loop 的 induction basis 不属于 C。即使一个 `tile_extent/reduction_tile_extent` slot 已通过
`subject_ref/axis_ref` 无歧义地绑定到 structured J loop，C 仍只回答 tile 选择，不回答 Source 是
`[0,Trip)`、step=Tile 的 extent loop，还是 `[0,ceil_div(Trip,Tile))`、step=1 的 ordinal loop。
该 source-realization 差异由 `artifact.v3` 的 `pi.iterations` 以 closed
`extent_step|tile_ordinal` relation 表达；C 不增加 basis、bound、formula 或 frontend 字段。
production backend 仍只消费 J+C，并在 `Impl(J)` 内自行选择合法 lowering；它不得读取 π 来复刻
Source basis。因而这次扩展没有污染 C，也没有把 SCRC fidelity constraint 变成 production strategy。

`artifact.v8` 将同一所有权原则推广到 **J 中没有 structured loop、但 Source 用物理 tile loop
realize 一个既有 J reduce/contraction 或 boundary access/domain traversal** 的情形。C 仍只给出
attached tile；π 另行作者化：

```text
physical_iterations = {
  region, body_instance, tile_instance,
  induction_basis = extent_step | rounded_extent_step | tile_ordinal,
  direction = forward | reverse
}

physical_loop_carriers = {region, init, state, next, result}
```

`body_instance` 可以锚定 J 已有的 reduce/contraction，或同 subject/axis 的 boundary access/domain；
`tile_instance` 只能锚定同一 subject/axis 上的 required positive-integer C slot。后一种只表达 output
epilogue 或 transfer traversal 的 physical time。carrier 只声明 init/state/next/result 的 typed temporal
topology，不能携带初始化或更新公式。该 relation 不在 J 中制造 loop，也不向 C 增加 basis、direction、
bound 或 permission。TypeCheck 只能检查 closed vocabulary、引用、可见性、axis attachment、类型与 carrier
拓扑；它不读取 Source，因而不能证明 direction 的事实真实性或发现一条完全未作者化的 Source loop。
这些真实性/完整性义务仍由 fresh Replay 消费和 Gate1 observer comparison承担。SCRC 后 π 停止，
production backend 不读取上述字段。

当前 wire 示意（假设 `n3` 是显式 `axes=[3]` 的 reduce node）：

```json
{
  "slots": [
    {
      "name": "c0",
      "kind": "constexpr",
      "role": "tile_extent",
      "value_type": "i32",
      "required": true,
      "subject_ref": "n3",
      "axis_ref": 3
    },
    {
      "name": "c1",
      "kind": "compiler",
      "role": "parallel_worker_count",
      "value_type": "i32",
      "required": true
    }
  ],
  "values": {
    "c0": 128,
    "c1": 4
  }
}
```

Lift LLM 随完整 C 显式产生 attachment；Harness、Decode/Elaborate、CandidateCheck 与 TypeCheck
均不得推断或补造，只能检查 ref 存在、`axis_ref in AxisRefs_v1(J,subject_ref)`，以及
`axis_ref => subject_ref`。不增加
`legal_domain/strength/retunable/fixed`：target 参数合法域仍属于 implementation family/capability，
C 按定义仍可忽略或 retune。schema 变更必须随下一 schema/TypeCheck protocol version 生效，旧
artifact 不重判。上述字段仍属于同一个 C，不能扩张成 portable-hint artifact、schedule ontology
或 LoweringAgent 输入协议。

### 5.2 Replay view

Replay 只接收：

```text
MaskC(C) = (Sigma_C, bottom)
```

示意：

```json
{
  "slots": [
    {
      "name": "c0",
      "kind": "constexpr",
      "role": "tile_extent",
      "value_type": "i32",
      "required": true,
      "subject_ref": "n3",
      "axis_ref": 3
    },
    {
      "name": "c1",
      "kind": "compiler",
      "role": "parallel_worker_count",
      "value_type": "i32",
      "required": true
    }
  ]
}
```

`MaskC(C)` 保留 `subject_ref/axis_ref`，因为它们只引用已经可见的 J，是 source-neutral attachment；
它只删除 concrete values 与 private source invocation provenance。

Replay 看不到：

- `values`；
- raw source parameter spelling（例如 Source 中的 `BLOCK_N`、`num_warps`）及其到 `c*` 的调用映射；
- source parameter position；
- source/replay invocation binding；
- Harness；
- 独立 interface sidecar 或 BindingMap；
- source/compiler fingerprint；
- prior Replay、compiler diff 或旧模型 conversation context；
- per-kernel RuleCard 或其他 source-conditioned frontend context。

Replay 可以看到 schema 中的 closed public `role` token；即便某个合法 role token 恰与 Source 标识符
同名，其可见身份也只是协议 role，不能携带 source position、binding 或 provenance。

### 5.3 Gate view

Gate1 的 compile/run harness 使用完整 C：

```text
V_gate(C) = C
```

同一个 `nu_C` 被机械绑定到 Source 和 Replay 的各自调用坐标。

### 5.4 Lowering view

当前 production 接口直接写成：

```text
V_lower(C) = C
K_t = Lower_t(J,C)
```

backend 可以消费、部分消费或忽略 C；当前设计不新增 `C_target`、LoweringAgent 或 cost model。
如果未来 backend 内部 retune，它是 production 实现细节与独立 conformance 问题，不改变核心
artifact `(J,C,pi)`。

`V_lower(C)` 中可供 backend 使用的结构定位只有 C 自身的 source-neutral
`subject_ref/axis_ref`；private source invocation provenance 不属于 lowering view。backend 只能通过
11.3 的后置 `Attach_C` 消费这些 attachment，缺失或含混时安全忽略。

### 5.5 不存在额外顶层对象

本设计删除以下额外 Lift 产物：

- 独立 `CBindingMap`；
- `optional portable C hints`；
- `Generic Resolver` 输出；
- `C_source -> C_target` artifact；
- target schedule lineage 作为当前核心证明对象。

若实现必须记录 source coordinate 到 C slot 的关系，它只能是：

1. Harness 已有的机械调用关系；或
2. C 内部的 private provenance，且在 `MaskC(C)` 中被确定性擦除。

它不能成为第四个 Lift artifact，也不能进入 Replay packet。
这里的 source coordinate provenance 与 `subject_ref/axis_ref` 不同：后者只引用 J 的 canonical
subject/axis，保留在 `MaskC(C)` 中；前者引用 Source/Harness 调用位置，必须擦除。

## 6. Harness 是隐藏求值环境

### 6.1 环境语义

Harness 可以形式化为：

```text
rho_H : {logical input, C slot} -> {concrete tensor/scalar/value}
```

例如：

```text
rho_H(c0) = 128
rho_H(c1) = 4
```

它还持有两个机械调用关系：

```text
rho_H^S(c0) = Source.BLOCK_N := 128
rho_H^R(c0) = Replay.c0       := 128
```

这些关系只负责把同一个 C value 送到两个程序的正确调用位置。

### 6.2 Harness 可以做什么

- 枚举参数、constexpr 与 compiler option 的机械坐标；
- 保存 concrete inputs 与 C values；
- 将相同 C valuation 绑定给 Source 和 Replay；
- 保存 grid、device、toolchain、compiler options 与 receipts；
- 执行 compile/run/measure。

### 6.3 Harness 禁止做什么

- 判断一个事实属于 J 还是 C；
- 创建 C 的语义 role；
- 生成 Replay use-site；
- 修改 Replay AST；
- 增加 loop、mask、address、reduction 或 sharing；
- 向 Replay 暴露 source name、position 或具体 value；
- 生成第四份 source-conditioned artifact。

### 6.4 Harness 不整体进入模型输入

正确关系是：

```text
                    Source
                       |
                       v
                    LiftLLM
                       |
                    (J,C,pi)
                       |
           ReplayView(J,MaskC(C),pi)
                       |
                       v
                  ReplayLLM
                       |
              Parameterized Replay
                       |
       Harness privately binds full C
       to Source and Replay invocations
```

Lift 可以读取 H 提供的极小、匿名化机械坐标；最终 logical signature 必须进入 J。Replay 若需要
接口信息，只能读取 `InterfaceView(J)` 这一确定性投影。H 不进入 Replay，也不能产生额外
source-conditioned object 再送给 Replay。

### 6.5 参数化 shape identity 不是 C，也不是 grid 语义

Harness 可以同时保存某次 concrete tensor `shape` 与 source/launcher 已明确给出的匿名
`shape_contract`。其中正整数表示真正 static axis，`h*` 只表示跨 tensor/scalar 共享的 logical extent
identity；runtime scalar 可以显式关联 `shape_coordinate=h*`，integral semantic constexpr 也可以由
adapter 以 `bound_constexpr_shape_coordinates` 显式关联，grid 只以
`grid_shape_coordinates` 记录 realization。Lift 必须把每个 `h*` 显式、双射地作者化为 J 的 `x*`；
scalar/constexpr 同时是 logical data 时，还必须作者化 rank-0 `t*` 与
`value_dimension_relations(eq)`。

固定 Harness/CandidateCheck 只核对 source-grounded identity、concrete valuation 与映射一致性，不能
从 grid、当前数值、参数名或 tensor position 生成 `x*`、J shape、derived capacity formula 或 relation。
`h*`、grid 与 concrete valuation 都不进入 J/C/π、Replay 或 production。compile-time variant selector
仍由 Harness 选择一个 active program；不同 selector value 是不同证书实例，不把 inactive family
分支塞进当前 J，也不把 selector 伪装成 C 或 runtime predicate。

## 7. 谁决定一个 source coordinate 属于 C

### 7.1 枚举与分类分离

Harness 可以机械枚举：

```text
q0 = 第 0 个参数
q1 = 第 1 个参数
q2 = 第 2 个 constexpr
q3 = 第 0 个 compiler option
```

但不能预先声明：

```text
q2 是 schedule
q1 是 logical domain
```

语义分类由 Lift 完成，并直接体现在 J 和 C 中：

- logical domain、predicate、shape、result consequence 进入 J；
- 纯 tile/warp/pipeline 等 implementation selection 进入 C。

### 7.2 所有权按事实用途，不按源码名字

同一个 source parameter 可能有多种用途。例如 `HEAD_DIM` 同时决定逻辑 extent 与 source
vectorization preference：

- logical extent 必须进入 J；
- 可分离的纯 schedule use 可以在 C 中体现。

这仍然不需要第四个 binding artifact。分类结果已经由 J 和 C 的内容表达。

### 7.3 两种干净实现

方法 A：Harness 在 Lift 前机械匿名化 bindable coordinates，Lift 直接在 C 中采用该 handle。

方法 B：若必须保存 `q2 -> c0`，将其作为 C private provenance，Replay view 确定性擦除。

两种方法都必须保持：

```text
Lift_H(S) = (J,C,pi)
```

## 8. Replay

### 8.1 唯一输入

Replay 的当前 kernel 条件化输入严格是：

```text
J + pi + MaskC(C)
```

另有固定、公共、与当前源码无关的 frontend context `U_f`。

不允许增加：

- binding map；
- source span/location；
- complete C values；
- Harness view；
- 独立 interface sidecar；
- selected RuleCard；
- prior candidate；
- source/compiler diff 或 fingerprint；
- 旧模型 conversation context。

### 8.2 参数化生成

Replay 生成：

```python
@triton.jit
def scrc_k0(..., c0: tl.constexpr):
    lane = tl.arange(0, c0)
    ...
```

它知道存在 `c0`，但不知道 source 名、source position 和具体 value。

pi 负责表达 C slot 在 source realization topology 中的 use；Injector 不得在生成后补 use-site。

### 8.3 生成后绑定

Replay source 生成完成后，Harness 执行：

```text
Bind_H(Source,C)
Bind_H(Replay,C)
```

Injector 只允许 argument/option binding，禁止修改 Replay AST。

## 9. Gate1

设 frontend observer 为 `F_f`：

```text
F_source = F_f(Compile_H(Source,C))
F_replay = F_f(Compile_H(Replay,C))
```

Gate1：

```text
G1 = [F_source == F_replay]
```

Source 与 Replay 使用同一个完整 C，因为 constexpr 和 compiler option 会改变 IR shape、address
topology、reduction tree 与浮点 realization。

Gate1 不能从额外 binding artifact、RuleCard 或 source-conditioned sidecar 获得 construction 信息。

当前 active Triton profile 是 `triton-ttir-3.6-gate1-v12`。v12以全局图切片保留observable value、
predicate/boolean、reduction/control/effect、region/loop/carrier及program/pointer/transfer anchors，
仅将无J/C/π所有者、只服务physical scheduling/address/rank/bound的known-pure公式收缩成typed、
carrier-origin-preserving representation edge。C concrete values仍作为同一Harness schedule binding
比较；quotient不吞掉C carrier使用、program-axis/transfer数量、predicate/fill或effect topology。
closed cache modifier 与 closed eviction
spelling 均作为不由 J/C/π 拥有的 frontend presentation noise 归一；π 仍拥有 distinct transfer
instance 的数量、顺序与 route，Replay 可用 public spelling 防止同址 transfer 被 compiler CSE。
它保留既有对已建模 op 与 reachable
static structured acyclic `tt.call` graph 的 interprocedural comparison rules，并新增 typed
single-block、encodingless
`tt.scan`；axis/reverse、combine arguments/return、recurrence 与 region topology 都是 hard evidence。
operation/effect classification 为全局 closed profile。v9–v11 进一步支持 closed value/effect-only
structured `scf.if` 与 exact operand-free terminal binary `cf.cond_br`：predicate、arm role、member/
dataflow、effect order、terminator 与 parent context 都进入 hard evidence；physical block printing order
不进入。unknown op、malformed generic/body line、超出 closed profile 的
multi-block CFG/region 与未支持 control/region 均 fail closed。Source 侧能力缺口仍由 batch v3 报
`observer_unsupported`；这不声称 v5→v6 acceptance 单调，历史 v5 evidence 也不重判。

## 10. Gate2

### 10.1 J reference 接口中没有 C

若语义参数 `N` 被错误放进 C，而 Source、Replay 和 `Lower(J,C)` 三腿共同读取 `C.N`，三角可能
错误通过，即使 J 没有表达 logical domain。

因此：

```text
RefEval : J x runtime inputs -> observable outputs
```

函数类型中不存在 C。最强隔离不是“把 C 给它后声明 ignored”，而是它没有访问 C 的接口。

### 10.2 正式 Gate2

对测试输入 `x`：

```text
O_S = Exec(Source, c0, x)
O_R = Exec(Replay, c0, x)
O_J = RefEval(J, x)
```

Gate2：

```text
G2 = Close_Q(O_S,O_R)
     and Close_Q(O_S,O_J)
     and Close_Q(O_R,O_J)
```

其中 Q 是实验协议或 benchmark correctness policy 预先确定的可信比较政策。J/B 只声明封闭的
comparison class 与必要语义合同；Q 根据经验证的 class、dtype、reduction extent、reassociation
权限等构造 comparator，并检查 tolerance 上限、NaN/Inf 合同等全部参数。不支持的类别或非法
参数直接 fail closed。Lift 不得在 J/B 中写入任意可执行 comparator，也不得用任意大的
`rtol/atol` 或“所有输出均可接受”来定义自己的及格线。

`Close_Q` 必须是确定性且由协议定义为对称的比较谓词；若底层 `isclose` 有方向性，Q 必须显式
对称化或执行双向检查。第三条 Replay–J 比较不增加执行，只比较已经产生的输出；它防止非传递
近似关系下两条 Source-centered 边同时通过、但 Replay 与 J 彼此超出政策范围。

当前主认证只覆盖 deterministic 或 contract-deterministic Kernel，即内部执行顺序可以变化，但
J/B 在协议可观察层面给出唯一合同结果。真正集合值的输出语义留给未来
`Accept_{J,Q}(x,o)` 的逐观察 conformance；有限次 membership 不能被表述成完整行为集合包含证明。

因此这里的所有权分工是：

```text
B declares a closed comparison class and its required semantic contract
Q validates all parameters and builds the trusted comparator
```

而不是候选 J 可以自由定义比较程序。

### 10.3 同一次实验，不同权限

三腿可以共享：

- 同一 logical inputs；
- 同一组 J/C/pi 与 Replay；
- 同一 Q 与 comparator；
- 同一 runtime shape/condition；
- 同一实验记录。

但 `RefEval(J,x)` 不得从 C 恢复 logical domain、predicate、fill 或 result relation。

RefEval 的实验记录可以关联当前 J/C/pi，但其函数接口必须保证：

```text
c_access = forbidden_by_interface
```

该记录只说明接口权限，不向 `RefEval` 提供 C。

如果 exact parent artifact 与 Replay 已经 Gate1 closed，而失败来自 RefEval 的通用 capability，允许
零模型调用的 Gate2-only exact-artifact recheck。它复用同一 J/C/π 与 Replay、写入独立 evidence，
并显式声明 `changes_parent_verdict=false`；不得把 consumer 修复伪装成 relift、刷新旧证书或改写
parent verdict。

## 11. Production lowering

### 11.1 当前唯一接口

当前 production 接口只定义为：

```text
K_t = Lower_t(J,C)
```

这里的 `Lower_t(J,C)` 只是 orchestration 层的抽象签名，不表示 materializer 可以同时读取
`J` 与 `C` 来决定基准 IR。其内部因子分解必须满足 11.3：先由 `J` 独立得到
`M0 = Emit_v(J)`，再让 `C` 只影响搜索 seed / ordering。

不引入：

- LoweringAgent；
- C-generation agent；
- retune agent；
- 新的 LLM schedule phase；
- `C_target` 顶层 artifact；
- 当前任务必须完成的 per-backend cost model。

### 11.2 Lowering 与 RefEval 分离

production lowerer 是被验证对象，不能同时充当 J 的独立 semantic oracle。

两条路径共用同一 deterministic functional-boundary core 与 closed rules，但入口权限不同：
production 只能调用 value-free `ProjectFunctionalBoundary(J)`；主 SCRC runner 选定 Gate2 winner 后，
才可为当前 invocation 调用 RefEval-only
`ProjectFunctionalBoundaryForRefEval(J,x,d)`。projection library 不选择 winner；`x,d` 不进入
production。两种 view 都不改写 J、不读取 C/π/Source、不单独序列化，也不是第二个 J 或新的 proof
artifact。投影成功只说明相应 Layer-B relation 已被 logical Intent MLIR 完整表示；若目标 lowering
引入 padded physical execution，仍须自行物化 mask/bounds 或给出目标侧等价保证，不能把该投影或
runtime-proof subreceipt 冒充 production correctness 证明。

当前 production J+C 路径对任何 `access_guard_relations` 都显式 fail closed：shared projection 可以
验证 relation 并形成 receipt，但 target path 尚无通用 access-local guard materializer；A 中已有
`select` 不能授权 backend 擦除 B relation。Gate2 的 direct-store partition 与 mixed typed
contraction reference 同样只是通用 RefEval 消费能力，不规定 CUDA 的 store 形式、contraction
算法或搜索候选。

若当前阶段验证 backend：

```text
M in Visited_(t,b)(J;C) subseteq Reach_t(J)
O_target = Exec(Lower_t(M), x)
G_backend = Close_Q(O_target, RefEval(J,x))
```

该 conformance 与 Gate2 主三角分开记录。

### 11.3 C 在 lower IR 中的生命周期

直接 production 路径必须保持如下职责顺序：

```text
J
  -> M0 = Emit_v(J)
C
  -> S_C = Attach_C(M0,C,t)
  -> BaseIR(S_C) = M0
  -> M in Visited_(t,b)(J;C) subseteq Reach_t(J)
  -> target lowering
  -> tile/vector/thread/pipeline/memory decisions
  -> lower target IR
  -> erase consumed schedule annotations
  -> executable
```

这里不是把 `J+C` 合并成新的后端语义输入。`Emit_v` 只读取 `J` 并构造基准
Intent MLIR；独立的 `Attach_C` 随后只初始化可擦除 annotation、搜索顺序、候选优先级和代价
模型先验。若实现产生 decorated working MLIR `M_C`，必须满足 `Erase_C(M_C)=M0`；parser、
`Emit_v` 与 pass default 都不得冒充 C 来源。搜索得到的
`M` 必须属于由 `J` 定义的语义保持可达域，不能因 `C` 改写合法性、结果关系或
可观察语义。

production lowerer可以消费、部分消费或忽略 C；但不能读取 pi 或 source-specific hidden binding。
若它忽略未知 C，必须仍能从完整 J 走 generic correctness fallback。

production execution 也不能重新接收一份 J 或独立 interface sidecar。已编译 lowering 持有唯一 J
及由它生成的 target program；runtime input 必须以 J logical handles 精确绑定，底层顺序 output
只能在 backend 内部确定性映射回 J outputs。缺失/额外/positional fallback 均 fail closed。该路径的
正确性合同由 materializer、pass 与 target lowerer 支撑；
`Close_Q(Exec(Lower_t(M),x),RefEval(J,x))` 只作为独立 CI/debug/evaluation conformance，不向
Gate2 增加第四条边，也不成为 production legality oracle。

### 11.4 Retune 与 cost model 的 production 扩展

完整 multi-lowering、retune 与 cost-model 合同由
[第五份后端设计](IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)
定义。其核心是在 backend 内部选择：

```text
M0 = Emit_v(J)
S_C = Attach_C(M0,C,t) and BaseIR(S_C) = M0
M in Visited_(t,b)(J;C) subseteq Reach_t(J)
Lower_t(M) models J
```

若实现内部仍使用 `sigma_target`，它只表示某个 `M` 所携带的 schedule / layout /
precision 配置分量，而不是脱离 MLIR 的完整实现候选，也不是由 `C` 授权的语义变体。
`C` 可以决定先从哪些候选开始、先测试哪些 tile / vectorization / fusion 组合；
但 `C` 不定义 `Reach_t(J)`，也不证明某个候选合法。合法性来自 `J` 与后端变换合同。

`sigma_target` 是 backend internal schedule state，不是新的 C artifact。该扩展不改变核心 artifact
`(J,C,pi)`，也不产生新的系统角色。

面对“每个 backend 是否都要重建 cost model”的 reviewer 问题，当前回答是：

- correctness 架构不依赖 cost model；
- backend 可以消费、忽略或以 C 排序候选；
- target-specific cost model 是 production 性能层，不是 Lift/Replay/Gate 正确性的前置条件；
- 28-kernel SCRC canary 与 backend multi-lowering/performance 实验继续分开验收。

## 12. C-purity 与多值实验

C 的数学定义要求合法 C 不改变 `[[J]]`，但基础在线 Gate1/Gate2 只认证实际绑定
`c0 = Bind_H(C)`；多值 intervention 不进入每个 artifact 的基础在线 Gate2。

并行候选搜索不会产生“候选集合 C”或新的 proof artifact。每个 branch 都独立产生自己的
`(J_b,C_b,pi_b)`，Harness 机械检查并为所有 branch 绑定同一次运行的实际 `c0`。sibling 的 C、
artifact、Replay 或 diff 不得相互消费；最终证书只属于 pre-Gate2 选中的单个 branch，而不是对
五个候选投票或对所有 admissible C 的证明。

原因：

- source 未必支持重新绑定；
- admissible value 难以统一生成；
- 不同 C 可以合法改变浮点 association；
- 成本随 slot 数量增长；
- 当前 Gate 已认证 observed C 下的 artifact。

它适合作为：

- classifier/schema CI；
- 高风险 slot 抽样；
- 论文消融；
- future backend retune conformance。

正确判据是：

```text
for every verified admissible c in test set T_C:
    Obs(I(J,c),x) belongs to [[J]](x)
```

不是不同 C 的输出逐位相同。

## 13. 反泄漏边界

### 13.1 Replay 可见

- J；
- pi；
- anonymous C slot schema；
- value type 与 binding kind；
- fixed public frontend context。

### 13.2 Replay 不可见

- C concrete valuation；
- source/replay invocation mapping；
- Harness；
- source name/path/span/position；
- audit/location/diagnostic sidecar；
- prior Replay/compiler diff；
- RuleCard/Registry；
- future retune/cost-model state。

### 13.3 Injector 可做与禁止

Injector 可以：

- 将 C values 绑定到已存在的 Replay parameters/options；
- 建立 compile/run invocation；
- 保存 receipt。

Injector 禁止：

- 修改 Replay AST；
- 新增 C use-site；
- 生成 loop、mask、address、reduction；
- 根据 Source 或 diff 修补 Replay。

## 14. 当前实现映射

### 14.1 已经符合的部分

当前 `ImplementationConfig` 已保存：

```text
slots + values
```

当前 `ImplementationConfig.masked_view()` 只返回 slots，并明确将 concrete values 视为 secret。Gate1 在
Replay 生成后，由 Harness 使用同一 C 编译 Source 与 Replay。这正是 `C=(Sigma_C,nu_C)` 与
`MaskC(C)` 的基本实现，应保留。

当前 strict slot schema 已要求 `name/kind/role/value_type/required`，并已支持可选
`subject_ref/axis_ref`；二者由 Lift 显式生成，由同一 TypeCheck/CandidateCheck core 检查。role-only
历史 artifact 仍按原版本解释，不由当前 schema 静默增加 attachment 或刷新 receipt。

### 14.2 需要修正或审计的部分

1. formal manifest 是否预先替 Lift 完成了 J/C 语义分类；
2. Lift 是否被要求复述 Harness 已分类的 schedule values；
3. 当前 anonymous coordinate 是否足以让 Lift 直接在 C 中采用 handle，而不生成 BindingMap；
4. 任意 binding/private provenance 是否在 Replay view 中被确定性擦除；
5. Replay packet 是否严格只有 `J + pi + MaskC(C) + fixed U_f`；
6. Gate2 RefEval 类型是否机械禁止 C；
7. production lowerer是否仍保持直接 `Lower_t(J,C)`，且不读取 pi；
8. production 对含混或未消费 C attachment 的 receipt 是否始终保持安全 ignore/fail-closed，而不由
   Harness/normalizer 猜测作用点。

## 15. 当前阶段明确不做

本节约束 SCRC artifact、Replay 与在线 acceptance；它不禁止第五份后端设计在 production 内部实现
不进入 SCRC 的 generic search/cost model。

- 不新增 `CBindingMap` artifact；
- 不新增 portable-hint artifact；
- 不新增 Generic Resolver 作为证明对象；
- 不新增 LoweringAgent、retune agent 或 C-generation agent；
- 不设计每个 backend 的 cost model；
- 不新增 `C_target` 顶层对象；
- 不把 future retune 写进当前 Gate1/Gate2；
- 不把多 C intervention 变成当前在线基础 Gate；
- 不为 target schedule 建立新的 LLM 调用阶段。

当前任务只需闭合：

```text
Lift_H(S) = (J,C,pi)
Replay(J,pi,MaskC(C))
Gate1(Source(C),Replay(C))
Gate2(Source(C),Replay(C),RefEval(J))
```

## 16. 禁止的退化方案

1. Lift 输出 `(J,C,pi)` 之外的 source-conditioned proof object。
2. 独立 interface/BindingMap/portable hints/Resolver output 进入 Replay 或 Gate verdict。
3. Harness 根据 `tl.constexpr` 或参数名做 J/C 语义分类。
4. Harness 整体进入 Replay。
5. Replay 看到 C values 或 source binding。
6. Injector 修改 Replay AST 来新增 C use-site。
7. RefEval 接收 C 后自报 ignored。
8. `Lower_t(J,C)` 同时充当 production implementation 与 J semantic oracle。
9. production lowerer读取 pi、source path、RuleCard 或 compiler diff。
10. 凭空引入 LoweringAgent/Autotuner/C_target 并作为当前系统组成。
11. 将未来 backend cost model 设为当前正确性验证前提。
12. audit/location/diagnostic sidecar 改变 Replay、Gate verdict 或 lowering。

## 17. 建议验证标准

### 17.1 Artifact factorization

- proof-relevant Lift codomain 只有 J、C、pi；
- logical signature 属于 J，`InterfaceView(J)` 只能是确定性投影；
- 额外 sidecars 不进入 Replay/Gate/lowering；
- Replay request schema 不存在第四个 kernel-conditioned payload；
- historical source/binding names在 Replay packet 中不可搜索到。

### 17.2 C 与 Harness

- canonical C 只有一个对象，保存 slots 与 values；
- Replay view 确定性删除 values 和 private provenance；
- Replay view 保留 source-neutral `subject_ref/axis_ref`；
- Harness 只枚举/绑定坐标，不做 J/C 分类；
- Source/Replay 使用同一个 C 对象和同一份 valuation；
- Injector 只绑定，不修改 AST。

### 17.3 Gate1

- Replay 在 value-free C 下生成；
- compile 前才注入完整 C；
- source/replay observer 在同一 specialization 下比较；
- Gate1 不读取额外 BindingMap、RuleCard 或 source-conditioned sidecar。

### 17.4 Gate2

- Source 与 Replay 使用相同 C；
- 主认证只覆盖实际绑定 c0 下的 deterministic / contract-deterministic Kernel；
- RefEval API 没有 C 参数；
- RefEval 实验记录可注明 `c_access=forbidden_by_interface`；
- J/B 只声明系统允许的 comparison class；外部 Q 决定 comparator 模板与参数上限；
- Source–Replay、Source–J 与 Replay–J 三条边均直接执行 `Close_Q`；
- production backend conformance 与主 Gate2 分开。

### 17.5 Production boundary

- 当前接口是直接 `Lower_t(J,C)`；
- lowerer 不读取 pi；
- `Emit_v(J)` 不读取 C，只有后置 `Attach_C(M0,C,t)` 可以产生可擦除 C-derived search state；
- 未理解的 C 可被忽略而不破坏 generic correctness；
- future retune/cost model 不进入当前 acceptance。

## 18. 后续实现中需要确定的通用接口

1. Harness 的匿名 bindable coordinate 如何机械形成，避免预先做 J/C 分类？
2. source coordinate 到 C slot 的关系优先采用同 handle，还是 C private provenance？
3. 一个 source parameter 的多用途如何同时体现在 J 与 C，而不新增第四对象？
4. Lift 最小可见的 H 机械坐标包含哪些结构事实，以及如何确定性形成 `InterfaceView(J)`？
5. 如何从类型/API 层机械证明 RefEval 无法访问 artifact C？
6. production lowerer 对 C 的 consume/ignore/ambiguous receipt 需要做到什么粒度？
7. future backend cost model 在 reviewer response 中如何表述而不形成当前实现承诺？

这些开放项不能通过新增 proof-relevant artifact 或 LLM agent 静默解决。

## 19. 最终规范

```text
Lift_H(S) = (J,C,pi)

TypeCheck(J,C,pi)

Replay = W_f(J,pi,MaskC(C);U_f)

Gate1:
    F_f(Compile_H(S,C)) == F_f(Compile_H(R,C))

Gate2:
    Close_Q(Obs_H(S,c0,x), Obs_H(R,c0,x))
    and Close_Q(Obs_H(S,c0,x), RefEval(J,x))
    and Close_Q(Obs_H(R,c0,x), RefEval(J,x))

Production:
    K_t = Lower_t(J,C)
```

> **Harness 负责给既有符号赋值，不负责产生新的符号系统；Replay 只看 `J + pi + MaskC(C)`。当前证明相关 artifact 始终只有 `(J,C,pi)`；production multi-lowering、retune 与 cost model 不属于 SCRC，并由第五份后端文档单独约束。**
