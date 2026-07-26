# IntentIR π：依赖类型实现见证与 Replay 安全设计

> 状态：新架构核心设计（2026-07-22）
>
> 思想来源：[高级 AI 对 π 的思想](source-notes/高级ai对于π的思想.md)
>
> 联合复审：[三份核心设计复审与四项修正](source-notes/高级ai对三份核心设计的复审与四项修正_2026-07-22.md)
>
> 前置规范：[IntentIR C：所有权、绑定、消费与验证设计](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
>
> 数学与 claim 总述：[IntentIR / SCRC 最终数学、执行流程与论文表述](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)
>
> Claim 收紧复审：[最终数学与三份设计的局部收紧建议](source-notes/高级ai对最终数学与三份设计的局部收紧建议_2026-07-22.md)
>
> 范围：在 artifact 严格只有 `(J,C,pi)`、Replay 不接收额外 per-kernel 输入的约束下，定义足以闭合 Replay、但没有显式语义副本和独立语义权威的 π
>
> 核心结论：**π 可以完整，但不能语义闭合；masked-J 测量 J 的操作性必要性，不再独自裁决 artifact 的语义合法性。**

## 1. 问题与最终判断

当 J、C 已经确定，且系统坚持：

```text
Lift_H(Source) = (J,C,pi)

Replay = W_f(J,pi,MaskC(C);U_f)
```

Replay 不再接收 RuleCard、BindingMap、Harness、source diff、旧候选或其他 source-conditioned
输入；同样不得接收 source parameter name/position、interface sidecar、source/compiler
fingerprint、旧模型 conversation context 或 per-kernel frontend context。此时，所有不属于
J/C、但 strict Gate1 要求恢复的 source realization topology，只能由 π 表达。

因此：

```text
π 太弱  -> Replay topology 在信息上欠定；
π 完整  -> Replay 可以按 witness 恢复 source realization；
π 复制 J -> 形成非法第二语义权威。
```

正确方向不是继续零散增加 kernel/mechanism rule，而是把 π 定义为：

```text
一个由 (J,C) 索引的、typed、anonymous、relational realization witness
```

它可以接近：

```text
normalized compiler topology with J/C semantic holes
```

但不得成为：

- 第二份 J；
- frontend source；
- RuleCard catalog；
- deterministic replay plan；
- production lowerer IR；
- 可脱离 J/C 独立执行的程序。

## 2. 与 C 规范的硬兼容条件

本设计不得修改 C 文档已经确定的不变量。

### 2.1 Proof artifact 仍只有三件东西

```text
Artifact = (J,C,pi)
```

π 的增强不能引入：

- selected rule artifact；
- source binding map；
- Harness view；
- compiler diff；
- source-conditioned frontend context；
- 新 Agent 输出；
- 第四 proof-relevant sidecar。

### 2.2 Replay 输入不变

```text
Replay = W_f(J,pi,MaskC(C);U_f)
```

其中：

- `MaskC(C)` 只有匿名 slot schema，无具体值与 source binding；
- `U_f` 是版本化、公共、与当前源码无关的 frontend 上下文，不能按 Kernel/op/失败样本选择；
- Harness 不进入 Replay；
- π 可以引用匿名 C slot，但不能包含 C value。

### 2.3 Gate 边界不变，Gate2 谓词收紧

Gate1：

```text
F_f(Compile_H(Source,C)) == F_f(Compile_H(Replay,C))
```

Gate2：

```text
Close_Q(Obs_H(Source,c0,x), Obs_H(Replay,c0,x))
and Close_Q(Obs_H(Source,c0,x), RefEval(J,x))
and Close_Q(Obs_H(Replay,c0,x), RefEval(J,x))
```

`RefEval` 接口中没有 C，也不读取 π。Q 是实验协议预先确定的可信比较政策；B 只声明封闭
comparison class 与必要语义合同。Q 构造确定性、对称的 `Close_Q`，并检查 tolerance 上限、
NaN/Inf 合同等全部参数；不支持的类别或非法参数直接 fail closed。第三条 Replay–J 边是直接
比较，不依赖近似关系的传递性。

### 2.4 Production lowering 不读取 π

```text
K_t = Lower_t(J,C)
```

该式是 production orchestration 的抽象签名；其内部必须先执行 J-only 的
`M0 = Emit_v(J)`，再让 C 仅初始化搜索，具体合同由第五份后端权威文档定义。

π 只证明 source realization 与 Replay closure。production lowerer 读取 π 会使 backend 退化为
模仿 source frontend，违反 J/C 与 π 的所有权边界。

## 3. π 的数学身份：依赖类型

### 3.1 类型判断

定义：

```text
Gamma_(J,C) |- pi : RealizationWitness(J,C)
```

这里的 J 是固定 Decode/Elaborate 从模型 response 中构造出的唯一 typed J。该步骤只做 closed
grammar 的确定性消糖、handle/type resolution 与恒等表示消除，不读 π 来决定 J，也不根据
Source/Gate 结果修补语义；无唯一解释时在 TypeCheck/Replay 前 fail closed。它不产生另一份 J，
也不扩充 `Ids(J)` 或改变 π 的数学身份。

π 不是独立 JSON 图；它只有在给定 J、C 的声明环境后才有类型和意义。这里
`Ids(J)` 已经包括 J logical signature 中的 input、output、runtime scalar、destination 以及
node/access/effect/value handles。

这意味着：

- `semantic_ref` 必须解析到 J/C 中已声明的 handle；
- operand slot 必须与被引用 J carrier 的 arity 相容；
- effect realization 必须由 J/B 中已存在的 effect 授权；
- C use 只能引用 `MaskC(C)` 中存在的 slot；
- sensitive representation refinement 必须引用 J/B 中已经存在的显式 authorization，或者属于
  系统安全白名单；
- `J.signature` 中精确的 `opaque(mode=value,dtype=none)` 只表示缺席 ABI carrier，不能成为 π
  instance anchor、authorization subject 或 refinement carrier；该限制不泛化到其他 opaque dtype；
- 删除 J/C 后，π 不能获得独立 denotation。

π schema 是 closed-world 的：未知字段、未知 relation、可执行自由文本和未声明的 semantic
constructor 必须 fail closed。若 `SemDecl` 只计显式逻辑语义声明，则 TypeCheck 的准确 claim 是：

```text
TypeCheck(J,C,pi)
    -> FreshLogicalSemDecl(C,pi | J) = empty
```

等价地，在把对 J 的合法引用解析回其定义后，可以写成：

```text
SemDecl(J,C,pi) = SemDecl(J)
```

这只保证 artifact grammar 中没有新的显式逻辑语义声明和未授权的 sensitive refinement；不证明
π 与 J 在信息论上独立，不证明 J 对 Source 正确，也不证明 realization 行为等于 `[[J]]`。

### 3.2 J 的 carrier 与 semantic labels

为定义“π 没有语义权威”，逻辑上将 J 分为：

```text
J = (K, lambda)
```

其中：

- `K`：logical signature 以及 node/access/effect/value 的类型、arity、引用与 carrier；
- `lambda`：operator、predicate、fill、reducer、index formula、result relation 等具体语义。

π 的结构锚定可以依赖 `K`，并可以引用 J/B 中不透明的 realization authorization handle；
TypeCheck 的合法性取决于第 4.4 节定义的 `RealSig(J)`。π 不能把 `lambda` 或 authorization
所保护的具体语义复制进自身。

### 3.3 Realization graph

定义：

```text
G_pi = (Regions, Instances, Routes, Order, Sharing, Placement, Representation, ScheduleRefs)
```

并有锚定映射：

```text
mu : Instances -> Ids(J) union Ids(C)
```

本设计优先要求每个 instance 都具有 opaque anchor。region、order、sharing 等纯关系无需伪造
新的 semantic node。如果工程实现需要接口视图，只能使用 `InterfaceView(J)`；它是 J 的确定性
投影，不增加新的 anchor namespace，也不是 Replay 的额外输入。

## 4. π 的硬安全条件

### 4.1 无语义原子

必须静态满足：

```text
SemAtoms(pi) = empty
```

π 禁止包含：

- operator/function/reducer；
- predicate/fill/index formula；
- semantic constant；
- logical shape/result relation；
- output/destination/conflict semantics；
- frontend API；
- source text/name/path/span；
- kernel/family/rule ID；
- C concrete value；
- 可执行自由文本。

非法示例：

```json
{"kind": "native_reduce"}
```

```json
{"fn": "max"}
```

```json
{"kind": "row_softmax"}
```

```json
{"fill": "-inf"}
```

正确形式只引用 J：

```json
{
  "id": "i4",
  "ref": "n2",
  "region": "r1"
}
```

`n2` 究竟是 reduce、dot 或 elementwise，只能从 J 读取。

### 4.2 所有可观察行为必须由 J 授权

任何可能产生可观察 value/effect 的 realization instance 必须锚定 J 中已存在的 node/access/effect：

```text
Observable(i) => mu(i) in Ids(J)
```

锚定 J signature/C 的 instance 只能提供：

- binding；
- placement；
- schedule shell；
- representation path；
- source sharing/materialization；

不能新增 J 中不存在的 operator、store、predicate 或 effect。

### 4.3 π 只能细化 J，不能新增逻辑图

将 representation-only scaffolding 折叠，并将锚定同一 J ref 的 realization instances 按 `mu`
商掉后，应恢复 J 的 logical carrier：

```text
Quotient_mu(G_pi) = Carrier(J)
```

工程实现可以先采用较弱但可检查的图同态：

```text
if a pi route connects two J-anchored instances at operand slot k,
then the corresponding J refs must have a compatible carrier edge at slot k,
or the route must be a validated representation-only path.
```

因此 π 可以：

- 为一个 J node 增加多个 materialization stage；
- 决定同一纯节点共享还是 recompute；
- 决定 placement、region 与 source order；
- 在同一 J access 周围增加 address/representation stages；

但不能：

- 增加 J 中不存在的 max/sub/store；
- 删除 J 中存在的 semantic edge；
- 交换 non-commutative operand slots；
- 新增 predicate 或 observable effect；
- 用 representation 路径改变 J/B 允许结果集合。

这里的 placement 只回答“instance 位于哪个 region”，不隐含某条前端指令。例如
`placement=program` 不等于 Source 读取了 program/grid coordinate。若 Source 确实包含
`program_id`、grid offset 等 compiler-visible stage，Lift 应使用额外的 generic `realize` instance
和 routes 表达该独立阶段；不需要扩充 mechanism enum。若没有这个独立 instance，Replay 不得仅
根据 placement、常见 kernel 模板或 C 自行生成 program coordinate、乘加或 affine address。

该 generic relation 必须有机械可解释的 base/lane 区分：在 program region 中锚定对应 J domain
的 `realize` instance，若其 scalar dependency path 先进入 `address_base` 再进入 transfer，则表示
program-coordinate stage；若 domain/C dependency path 进入 `address_lane`，则表示 lane-varying
stage。Replay 保持该拓扑区分，公共前端上下文 (U_f) 再决定使用何种公开 API。π 本身仍不保存
`program_id`、axis 公式、前端文本或新的 mechanism tag。

program axis 的 identity 来自 J domain anchor，不来自 instance 数量。公共 Triton `U_f` 按 distinct
domain `semantic_ref` 的首次声明顺序分配 axis；同一 `d*` 上的多个 realization instance 仍是多个
compiler-visible stage，必须分别 materialize，但都读取同一 program axis，不能把第二个 instance
擅自解释为 `program_id(1)`。只有 distinct domain anchor 才能分配下一条 axis。这个规则不合并
instance，也不替 Lift 修复错误 multiplicity；若 Source 只有一次 coordinate stage，下一轮 Lift 仍
必须只声明一次并将其共享到全部消费者。

更一般地，一个 Source SSA coordinate/address intermediate 只定义一次、被多个 access 使用时，π
必须声明一个 producer、多个 routed consumers 与 one-producer sharing。多个语法 use 不构成多个
producer instance；重复同一 semantic ref 只表示不同 compiler-visible definition/stage。Replay 不得
为满足错误 multiplicity 而构造 `base + (coordinate - base)` 一类代数恒等式；Lift 也不能把真实的
多个 stage 反向折叠成一个。

这里的 `address_base/address_lane` 是 address contribution 的角色边界；pointer/address operation
次数必须由 `J.signature` 的 carrier kind 与 π route 联合决定，不能从 base-to-lane edge 单独猜测。
若 pointer/tensor `p*` instance 直接进入 `address_base`，该 base 本身就是一次 pointer/address
operation；其 pointer 结果再 route 到 `address_lane` 时，Replay 必须保留第二次 lane
pointer/address operation。若 `address_base` 只接收 integer/domain/C contribution，而 pointer
carrier 绕过 base、直到 `address_lane` 或最终 address stage 才进入，则 Replay 保留两个 integer
stage、先合并 base/lane，再执行唯一 pointer addition。两种情况都不能因为当前 grid extent 为 1
把 program path 改成常量。只有 Source 根本不存在可分离的 program-uniform contribution 时，才
使用单独的 `address_lane`。

一个直接锚定 J signature handle 或 C slot 的单独 `realize` instance，可以由对应参数 binding 与
direct routed use 满足，不要求额外的 identity assignment。若同一 handle 声明多个 instance，
它们仍表示多个 compiler-visible use-site/stage，不能折叠。这个区分对 constexpr 尤其重要：公共
前端合同可以要求直接使用参数来保留编译期类型，而不是为了形式上的 materialization 破坏类型。

相同原则适用于 structured operator realization。对 canonical binary contraction，在 rank-2、单一
contracted label、floating tile/dtype/accumulation 均由 public `tl.dot` 支持时，一个锚定该 node 的
`realize` instance 由当前 `U_f v6`（继承 v5 的 direct-contraction 合同）映射为一个 direct `tl.dot`；不得以 outer-product multiply +
`tl.sum` 扩成未声明 stages。若 Source 确有多个 compiler-visible decomposition stage，Lift 应把
它们完整声明为同一 node 上的多个有序 instances；若不满足 direct `tl.dot` support 条件，`U_f`
不能强迫该 spelling。这里没有新增 J op、π mechanism 或 Source-specific rule。

这里的 `tl.dot`、program-axis 与具体 pointer/address 拼写都只属于 Triton `U_f`。Replay prompt 的
公共部分只保留 frontend-neutral ownership/topology 合同；CUDA、CuTe 与 TileLang 不接收 Triton
话术，各自只能由其版本化 `U_f` 决定公开 API。共享的 π mechanism grammar 仍可描述
`address_base/address_lane` 等 source-neutral tag，但不能借此指定 `tl.*` 实现。

π 同样不能填补 J 中缺失的 semantic recurrence。若 Source 的 compiler-visible value dataflow 是
loop-carried recurrence，J 必须用已有 structured `loop` 完整声明 trip/scan、carry/init、body update
与对应 next/yield/result；contraction 的 canonical subscripts 与 result-axis orientation 必须和 update
消费一致。π 只锚定这些 J carrier 的 realization topology，不能保存或让 Replay 推断 initialization、
update formula、yield 或 transpose/orientation。把 recurrence 压成最终 global formula 会使 Replay
欠定，不能靠增加 π 自由文本补救。

同一边界也适用于 pad。若 J/π 没有证明 symbolic input extent、literal pad 与 static output extent
之间的 dependent equality，π/U_f 都不能读取 Harness 的冻结 valuation 补出该关系，Replay packet
也不应携带 per-candidate pad-discharge fact。当前 `replay-input.v1` 不提供这种旁路；应由
下一次 Lift 修正 J，而不是在 Replay consumer 中建立小型 theorem prover。

### 4.3.1 物理迭代基底 relation

Linear 反例表明，structured J loop 与 attached C tile 仍不足以唯一决定 Source realization：

```text
extent_step:  lower=0, upper=Trip(J.loop),                 step=Tile(C.slot)
tile_ordinal: lower=0, upper=ceil_div(Trip(J.loop),Tile), step=1
```

两者可以实现同一 logical loop，却在 compiler observer 中具有不同 induction topology。旧 π 没有该
区分时，Replay 选择任一写法都不是“消费 artifact”，而是在猜 Source。`artifact.v3` 因此引入、当前
`artifact.v5` 继续要求：

```text
pi.iterations[*] = {
  region,
  loop_instance,
  tile_instance,
  induction_basis: extent_step | tile_ordinal
}
```

entry 只引用已声明 region/instance。`loop_instance` 锚定 J structured loop；`tile_instance` 锚定
required positive-integer C tile slot，且该 slot 的 `subject_ref/axis_ref` 必须指向同一 loop/trip axis。
这里 axis 可以写成该 loop 的 canonical ordinal `0`，也可以写成显式 `trip.axis=x*`；二者都是
`AxisRefs_v1` 对同一 trip axis 的 closed spelling。显式 iteration relation 的合法性不依赖 C slot
role：custom role 只要满足同一 dependent type 也可被引用。
两个 instance 必须在 relation region 或其 lexical ancestor 可见，relation region 必须
`placement=loop`。wire 不包含 lower/upper/step 字段、公式、concrete tile、Source 文本、operator 或
frontend API；上述两种 basis 的 denotation 是全局协议，而非 Triton 私有约定。各 `U_f` 只能把同一
relation 映射到自己的公开构造。

parser 对每个 declared physical region 至多一条 relation。若 Source 确有多个 physical loop region
复用同一 logical loop/tile pair，Lift 可以声明多个 region 与多条 relation。`tile_extent` 与
`reduction_tile_extent` 是两个触发机械 missing-totality 的保留 role：只有 attachment 闭合且使用这
两个 role 的 C-tiled J-loop/slot pair，`typecheck.v7` 才能在 relation 完全缺失时要求至少一条合法
relation；没有 C tile 的 loop 不必制造 relation。custom role 不使已经显式写出的 relation 非法，但
若 Lift 误标 role 并完全遗漏 relation，fixed checker 没有 missing-totality trigger。一个 Source
region 若被 Lift 整体遗漏，也不存在可供 fixed checker 量化的 handle。这两类 completeness 缺口仍
由 Lift 的完整性责任与 Gate1 的 Source/Replay topology comparison 检出，不能把局部 totality 写成
对所有 Source physical regions 的形式证明。

该字段只服务 SCRC Replay fidelity。SCRC 结束后 π 停止，production backend 仍只读 J+C；它不得用
`pi.iterations` 限制或选择自己的 `Impl(J)` lowering/search。

后续落地的 wire 仍遵守这条版本纪律：artifact.v4 增加 J-owned
`value_dimension_relations`、`boundary.consumer_extensions` 等 typed relations，artifact.v5 增加
J-owned `boundary.access_guard_relations`；当前 decode 为 v6。它们没有进入 π，也没有让 π 获得
value/dimension、consumer context 或 access-control 语义。旧 v1-v4 wire/receipt 仍按原字段集合读取，
不由当前 parser 附加空数组、刷新或追认。

### 4.3.2 Consumer context 与 access guard 仍属于 J

artifact.v4 的 `value_dimension_relations` 由 Lift 显式声明 integral rank-0 logical value 与 symbolic
dimension 的 `eq`；`consumer_extensions` 由 Lift 显式声明某个 predicate-bearing load 到某个
`sum|max|min` reduce/scan operand 的 typed two-sided outside identity。artifact.v5 的
`access_guard_relations` 则把一个 boundary access 绑定到一个支配它的 Layer-A `i1` guard，并声明
唯一 broadcast axes 与 polarity。这些都是 J/B fact，不是 π topology，也不能由 π route、C slot 或
当前 runtime valuation 补齐。

尤其 guarded store 的 merge 必须已经在 A 中显式写为三 operand `select`：目标 tensor 精确同时为
J input/output，并由唯一 `mode=inout` parameter 绑定；tensor handle 按首次声明连续编号，output
复用同一 input `t*` 而不分配新 handle 或滥用 `alias_of`；loop 的 `init_value/current/next_value` 或
non-loop 的 tensor pre-state/return/output mapping 必须精确闭合。relation axes 是同一 select guard
的唯一 authority，不能由 `boundary.broadcasts` 或 π 重复。TypeCheck/CandidateCheck 只拒绝、诊断，
不插入 select、不升级 inout、不选择 axes/polarity，也不接受仅同 dtype/shape 的替代 pre-state。

Replay 从 J 读取这些 relation，从 π 读取 realization；Gate2 RefEval 对每个 relation 必须消费或明确
unsupported。当前 RefEval 可执行 A 中已有 select 的 guarded store，但 guarded load/update 的
access-local pre-execution materialization 仍 unsupported。production backend 仍只读 J+C，且当前对
任何 access guard 显式 fail closed；它不得借 π 或 RefEval receipt 补 target materialization。

### 4.4 对 semantic labels 参数化

必须满足：

```text
RealSig(J) = (Carrier(J), Capabilities(J,B))

if RealSig(J1) == RealSig(J2),
then TypeCheck(J1,C,pi) == TypeCheck(J2,C,pi)
```

`Capabilities(J,B)` 只包含 realization authorization，例如允许 signed widening、允许
reassociation、允许 native atomic 或允许 shared materialization；它不复制具体 operator、
predicate、fill 或 reducer 语义。

这里 B 是 J 内部的 boundary/typed-effect contract 子结构，不是 Lift 的第四个输出或 Replay 的
额外输入。

只要 J1/J2 具有相同 carrier 和相同 realization authorization，π 的 TypeCheck 结果不应依赖其
semantic label 是：

- add 还是 mul；
- sum 还是 max；
- 某个合法 fill 的具体值；
- 某个 predicate 的具体表达式。

Replay 必须从当前 J 读取语义，而不是服从 π 中的历史 source 语义。

### 4.5 所有标识符完全不透明

允许：

```text
n0, v0, a0, d0, e0, q0, c0, p0, r0, i0
```

禁止：

```text
reduce_max_0
row_component
lane_component
variance
hidden_size
mask_predicate
```

opaque handle 按 declaration order 生成。名字不能携带 source、operator、axis 或用途暗示。

### 4.6 C 只提供 slot，不提供值

π 可以引用：

```text
c0
```

表示某 realization stage 消费一个 late-bound C slot，但不能包含：

```text
c0 = 128
BLOCK_N
num_warps = 4
```

C value 仅由 Harness 在 Replay source 生成完成后绑定。

## 5. 最小关系型 π schema

### 5.1 设计原则

π 必须完整描述关系，但不能重新命名 J 的语义类别。核心不应持续增加：

```text
native_reduce
masked_tile_load
atomic_update
row_address
softmax_stage
...
```

否则 π 会再次演化成第二套 RuleCard、compiler IR 或 frontend-independent lowerer ontology。

推荐核心只包含：

```text
regions
instances(ref)
routes(source,target,operand_slots)
order(before,after)
sharing(source,consumers)
placement
representation refinements
schedule_refs
```

“generic”仍表示封闭的 typed constructors 与 opaque J/C refs，不能接受任意 formula、自由文本或
可执行 relation。新增前端或 operator 应优先组合这些关系；不得用一个可表达任意语义的
`generic_relation` 逃避 SemDecl 与 authorization 检查。

### 5.2 示意 schema

```json
{
  "regions": [
    {
      "id": "r0",
      "parent": null,
      "placement": "root"
    },
    {
      "id": "r1",
      "parent": "r0",
      "placement": "loop"
    }
  ],
  "instances": [
    {"id": "i0", "ref": "p0", "region": "r0"},
    {"id": "i1", "ref": "q0", "region": "r0"},
    {"id": "i2", "ref": "a0", "region": "r1"},
    {"id": "i3", "ref": "n2", "region": "r1"},
    {"id": "i4", "ref": "e0", "region": "r1"}
  ],
  "routes": [
    {"source": "i0", "target": "i1", "slots": [0]},
    {"source": "i1", "target": "i2", "slots": [0]},
    {"source": "i2", "target": "i3", "slots": [0]},
    {"source": "i3", "target": "i4", "slots": [0]}
  ],
  "order": [
    {"before": "i2", "after": "i3"},
    {"before": "i3", "after": "i4"}
  ],
  "sharing": [
    {"source": "i2", "consumers": ["i3", "i4"]}
  ],
  "representation": [
    {
      "instance": "i1",
      "property": "uniformity",
      "value": "program"
    },
    {
      "instance": "i2",
      "property": "materialization",
      "value": "staged"
    }
  ],
  "schedule_refs": [
    {"instance": "i2", "slot": "c0"}
  ]
}
```

该示例不声明 `i3` 是 reduction 或 elementwise，也不声明 `i2` 是 load/store。具体语义来自
`n2/a0/e0` 在 J 中的定义。

### 5.3 Opaque subterm refs

为了表达 address/index decomposition 而不复制 formula，J 的 canonical parser 应提供匿名 subterm
handle：

```text
q0, q1, q2, ...
```

π 引用：

```json
{"id": "i1", "ref": "q1", "region": "r0"}
```

禁止：

```text
a0.index.row_component
a0.index.lane_component
row * stride
pid * BLOCK_N + lane
```

如果当前 J wire 仍以完整 expression string 保存 index，可以由 canonical parser 派生只读、稳定、
declaration-order subterm handles。该派生是 J 的确定性视图，不是第四个 Replay 输入。

### 5.4 Representation refinement 的准入

通常安全的候选：

- placement；
- sharing；
- materialization；
- source SSA order；
- program/lane uniformity；
- memory space，在不改变可观察状态时；
- representation-only staging。

需要 J/B 显式授权的 sensitive refinement：

- integer width；
- numeric cast；
- accumulation dtype；
- reassociation；
- saturating conversion；
- atomic/privatized update form；
- 会改变 rounding、overflow、conflict 或允许结果的任何属性。

TypeCheck 不负责现场证明某个 refinement 不改变 `[[J]]`，也不运行 SMT、range proof 或一般语义
等价检查。它只机械检查：

```text
Sensitive(r)
  => auth_ref(r) in Auth(J,B)
     or kind(r) in GlobalSafeRefinement
```

其中 `GlobalSafeRefinement` 是系统统一维护的安全白名单，不由 Lift 或单个 Kernel 修改。若两者都
不满足，TypeCheck 直接失败。这样整数宽度、numeric cast 等语义由 J/B 表达，π 只表达显式
materialization、placement、order 与 sharing。

#### 5.4.1 `numeric_cast` authorization 的 dependent type

当 authorization 的 `subject_ref` 指向显式 cast node：

```text
n = cast(x : Ts) -> Tt
u = numeric_cast(subject_ref=n, source_dtype=Ts, target_dtype=Tt)
```

其类型规则是：

```text
u.kind == numeric_cast
u.subject_ref resolves to n
OpKind(n) == cast
Arity(n) == one operand and one result
u.source_dtype and u.target_dtype are both present
u.source_dtype == Type(n.operand[0])
u.target_dtype == Type(n.result)
```

不能用一个通用 “`source_dtype == Type(subject result carrier)`” 规则检查所有 authorization；该规则
会把合法的 `f16 -> f32` cast 错误地比较为 `f16 != f32`。同理，accumulation、widen、atomic/update
等 kind 必须分别定义自己读取的 operand/result/effect carrier。

authorization 仍由 source-aware Lift 显式写入 J/B，π refinement 显式引用它。固定代码只能检查：

```text
refinement.kind == authorization.kind
authorization.subject_ref == instance.semantic_ref
auth_ref resolves to that authorization
```

每个 π instance 至多有一个 refinement；重复 refinement 必须在 parser/TypeCheck 前置边界 fail
closed，不能由工具或 Replay 选择其中一个。

Decode/Elaborate、TypeCheck 与 Replay 都不得从 op 自动创建缺失的 authorization，也不得要求
numeric-cast instance 改绑 cast 前的 tensor 而丢失显式 cast stage 的 J node anchor。

显式 `cast` stage 与 closed `map_binary` 自身的 typed promotion 必须区分：后者只消费 Lift 已声明的
operand/result types 和系统固定 promotion lattice，可由共享 projection 以内部 `intent.cast`
materialize；它不创建 J node、authorization 或 π refinement。只有 Source/J 中独立可见的 cast stage
才受上述 `numeric_cast` dependent type 约束，不能被 op-internal promotion 合并掉。

### 5.5 Effect realization

π 可以决定如何 realization 一个 J-owned effect，不能决定 effect 是否存在或具有什么 conflict
semantics。

禁止独立写：

```json
{"mode": "atomic"}
```

允许的方向是：

```json
{
  "instance": "i4",
  "effect_ref": "e0",
  "representation_form": "native"
}
```

且 validator 必须验证 `e0` 的 J/B contract 允许该 realization refinement。若需要区分 atomic 与
privatized，它是锚定 `e0` 的受约束 refinement，不是 π 自己声明 atomic semantics。

## 6. 从旧 RuleCard/Composer 迁移什么

旧系统之所以容易 Replay，是因为：

```text
LLM 选择 C.strategy
-> composer 展开 work items
-> RuleCard 提供 construction/API knowledge
-> Replay 接收有序 plan 与规则内容
```

新系统不恢复 RuleRegistry，而是只把**本次 kernel 已经选定的 realization 关系**写入 π。

| 旧信息 | 新所有权 |
|---|---|
| 选定步骤的实例数量 | π instances |
| construction dependency | π routes/order |
| operand slot | π route slots |
| region/preheader/body placement | π regions/placement |
| producer reuse | π sharing |
| source materialization/staging | π representation refinements |
| source C use-site | π schedule_refs，引用匿名 c* |
| operator/reducer/predicate/fill/index formula | J，不进入 π |
| tile/warp/pipeline concrete value | C，不进入 π |
| RuleCard ID/family | 删除 |
| frontend API mapping/example/pitfall | 固定 `U_f` 与 Replay 模型能力，不成为 per-kernel artifact |
| source text/name | 删除 |

关键转换是：

```text
旧：Replay(J,C,selected_plan,RuleRegistry)

新：Replay(J,pi,MaskC(C);U_f)
```

π 必须 self-describing 到足以恢复关系结构，但不携带 RuleCard codebook 或语义文本。

## 7. Replay 如何消费完整 π

Replay LLM 应遵守以下过程合同：

1. 从 J 读取所有 operator、predicate、fill、index、effect 和 result semantics；
2. 从 `MaskC(C)` 声明参数化 C slots，不读取 values；
3. 按 π regions 建立源码结构；
4. 为每个 π instance 建立一次 source realization；
5. 按 routes 和 operand slots 连接 producer/consumer；
6. 按 sharing 复用 producer，不得擅自复制或折叠；
7. 按 order 保留必要偏序；
8. 按 `iterations` 保留每个已声明 physical loop region 的 induction basis；
9. 按 representation refinements 保留 source materialization/placement；
10. 遇到 J semantic ref 时从 J 查询当前语义 label；
11. 使用固定公共 frontend 知识 `U_f` 选择合法 API spelling；
12. 输出完整、参数化、source-blind frontend source；
13. 不从 π 猜出后覆盖 J，也不从 compiler diff 修补。

当前 fresh receipt 组合为 `intentir.scrc.replay-input.v1`、Triton
`public_frontend_context_v7` 与 `intentir.lift-knowledge.v17`；旧 receipt 保持其原 verdict，不能由
当前 Replay consumer 或 Lift knowledge 静默升级。

当前新 Lift wire/decode 为 `intentir.scrc.artifact.v5 / intentir.scrc.decode.v6`；v3 历史上把上述
`pi.iterations` 变为 required closed field，v4/v5 再增加前述 J-owned typed relations。v1-v4 历史
artifact 保持原始序列化、digest 与字段集合，禁止静默附加空数组、重标或刷新证书。

固定代码仍只做：

- schema/type/reference validation；
- graph closure/acyclicity；
- `TypeCheck(J,C,pi)`；
- Replay code policy；
- compiler observation 与 Gate verdict。

固定代码不把 π 解码成源码，不建立 deterministic composer。

## 8. Gate1 completeness：没有第四个事实来源

对 Gate1 observer 检查的每个 feature，必须建立唯一所有权：

```text
feature belongs to
    Derive(J)
    or Derive(C)
    or Derive(pi)
    or NormalizeAway
```

示意：

| Gate1 feature | 唯一来源 |
|---|---|
| operator、reducer、predicate、fill、logical index | J |
| C slot 与编译时 concrete value | C/Harness binding |
| instance count、route、slot、order、sharing、placement | π |
| source materialization 与 representation staging | π，受 J/B 约束 |
| frontend API spelling | 固定 `U_f`，不是 per-kernel 输入 |
| compiler 偶然 SSA 名/无意义临时差异 | normalize away |

若 strict Gate1 检查某个 source-specific topology，而 J/C/π 都不表达它，也不归一化，那么 Replay
失败是数学欠定，不是模型能力不足。

当前 Triton observer 将 `tt.load` 的 closed `cacheModifier {ca,cg,cv}` 与 `tt.store` 的
`{wb,cg,cs,wt}` 归为 `NormalizeAway`：这些值是 target cache-policy hint，不改变 J
denotation、memory effect、predicate/fill、address/dataflow 或 transfer topology，且 J/C/π 都不拥有
frontend cache spelling。`triton-ttir-3.6-gate1-v8` 进一步把 closed
`evictionPolicy in {evict_first,evict_last}` 归入同一类 NormalizeAway：π 负责同址 transfer instance
的数量与顺序，Replay 可用 public eviction spelling 防止 compiler CSE，但具体 spelling 不由 J/C/π
拥有，也不得成为 source-blind reconstruction 的隐含答案。未知或方向不合法的 policy、mask、fill、
type、volatility 与其他 attributes 仍是 hard evidence。这一决策不在 π 中新增假装的 frontend API
mechanism，也不把 cache hint 当作 J 语义。

当前 active `triton-ttir-3.6-gate1-v8` 保留既有对已建模 op 与 public-entry reachable、静态已知、
structured 且无环 `tt.call` graph 的 interprocedural comparison rules；formal-to-actual、helper return SSA、
op/def-use、effect 与已支持 region 都是 hard evidence，π 仍不保存 helper 名或 frontend call API。

v6 只新增 typed、single-block、encodingless `tt.scan` observer：ranked tensor shape、axis/reverse、
ordered combine args、唯一 typed return、recurrence/region topology 都被观察。它同时使用全局 closed
operation/effect classification；unknown op 不能因位于 root/`scf.for`/combiner 而默认 pure。
malformed generic/body line、动态/未知 callee、递归 cycle、call signature/return 不一致、reachable
arbitrary multi-block CFG/region 或未支持 control/region 均 fail closed；Source 侧缺口由 batch v3 报
`observer_unsupported`，不能伪装成 π 缺字段、Gate1 hard mismatch 或有界 `NO_CLOSURE`。历史 v5
evidence 保持原解释；这不声称 v5→v6 acceptance 单调。上述只是当前 compiler observer 的实现
合同，不是 call graph/TTIR observer
完备性的形式证明。

完整性目标是：

```text
Obs_f(Source,C)
=
Obs_f(Replay_f(J,pi,MaskC(C)),C)
```

## 9. 三类“泄漏”必须分开

### 9.1 显式复制泄漏

π 出现 operator、predicate、formula、semantic constant、API、kernel name 等。这是静态非法，
必须直接拒绝。

### 9.2 语义权威泄漏

π 覆盖 J 的语义。例如 J 将 `n2` 定义为 sum，Replay 却因 π 中的暗示继续实现 max。这是非法，
需要 dependent typing 与 J-authority challenge 检出。

### 9.3 结构推断泄漏

模型从合法 topology 推测 kernel 语义，例如从低熵 load/store graph 猜到 copy。这是统计相关性，
不等价于 π 显式复制 J，也无法在要求 exact topology 的同时普遍消除。

系统应硬保证：

```text
无显式语义副本
+
无独立语义裁决权
```

不能声称：

```text
MutualInformation(pi, semantics) = 0
```

## 10. 主认证与 masked-J 证书分离

### 10.1 主认证

定义：

```text
SCRC_CERTIFIED = TypeCheck(J,C,pi) and Gate1 and Gate2
```

候选 fan-out 时，上式逐 branch、逐完整 `(J,C,pi,Replay)` 检查。每个 π 只能由本候选的 `(J,C)`
索引；不得引用 sibling candidate 的 handle、π topology、Gate1 diff、prior Replay 或对话。多个 π
不能拼接成一个 witness。并行只扩大 adaptive witness search 的预算，不改变 TypeCheck 或证书
谓词；Gate2 只检查在任何隐藏输入产生前按固定规则选中的一个 Gate1-closed witness。

它证明：

- π 满足 `(J,C)` dependent ownership contract；
- `J + pi + MaskC(C)` 可以 source-blind 构造 Replay；
- Source/Replay strict observer closure；
- Source、Replay 与 J-only RefEval 在 Gate2 relation 下闭合。

这是主系统结论。

### 10.2 J 必要性证书

定义：

```text
J_NECESSARY = masked-J hard mismatch
```

它额外说明：

```text
在当前模型、当前 π 与当前 topology 下，删除 J 后不足以恢复 source。
```

该证书更强，但不要求每个 kernel 都获得。
这里的 `J_NECESSARY=PROVED` 仅表示在记录的模型、解码配置、Replay packet 与 masked-J 干预下
观察到 J 的操作性必要性，不是对所有生成器或所有实现的全局不可替代性证明。

### 10.3 为什么 masked-J 不能继续绝对否决

若 π 足够完整，使：

```text
Topology(Source) = g(pi)
```

而低熵 kernel 的语义又可能满足：

```text
Semantics(Source) = h(Topology(Source))
```

则即使 π 没有非法字段，masked-J 也可能闭合。此时：

```text
Semantics(Source) = h(g(pi))
```

masked-J hard-close 只能否定“J 对此样本的操作性必要”，不能自动否定 `Source equivalent J`。
后者由 Gate2 的 Source–J 边认证。

### 10.4 masked-J 结果解释

| masked-J 结果 | 解释 | 主认证处理 |
|---|---|---|
| hard mismatch | 支持 J 对 Replay 的操作性必要性 | 主认证可通过，并附 `J_NECESSARY` |
| hard close 且 `TypeCheck(J,C,pi)` 失败 | π 存在显式/权威泄漏 | FAIL |
| hard close 且 `TypeCheck(J,C,pi)` 通过 | topology 对语义具有预测力；J 必要性未证明 | 不自动否定主认证；报告，必要时另行运行离线挑战 |
| generation/compiler/tool failure | inconclusive | 不颁发 J 必要性证书 |

这要求修改当前：

```text
masked hard-close -> LEAKAGE_REJECTED
```

的绝对状态机。

## 11. 条件式 J-authority challenge

该挑战是可选的离线消融，不进入主流程，也不在 masked-J hard-close 后自动触发。只有在下列条件
成立时，它才具有可解释性：

```text
normal Gate1 closed
and TypeCheck(J,C,pi) passed
and masked-J hard-closed
```

时作为条件诊断触发；它不进入每个 artifact 的基础在线流程。

### 11.1 构造

保持不变：

```text
pi, C, RealSig(J)
```

改变一个 realization-signature-compatible semantic atom：

```text
J -> J'
```

例如：

- add ↔ mul；
- sum ↔ max，并同步修改 identity；
- live fill 改为另一合法值；
- predicate 改为同 signature 的合法表达式。

然后：

```text
R' = Replay(J',pi,MaskC(C))
```

不要求 R' 与原 Source 一致，只验证：

```text
Exec(R',C,x) equivalent_(B') RefEval(J',x)
```

### 11.2 解释

- R' 跟随 J'：π 没有覆盖 J；masked-J closure 主要来自 topology prediction；
- R' 仍实现原 source semantics：非 J 通道具有语义权威，是真泄漏；
- J' 与 π signature 不兼容：换一个同 signature atom或记录 uninformative；
- generation/tool failure：inconclusive。

该挑战使用现有 RefEval(J') 作为 oracle，不新增 frontend lowerer、RuleCard 或 source repair。

## 12. Gate2 对完整 π 的 soundness

当前主认证域是 deterministic / contract-deterministic Kernel：内部执行顺序可以变化，但 J/B
在协议可观察层面给出唯一合同结果。真正集合值的语义留给未来 `Accept_{J,Q}(x,o)` 的逐观察
conformance；有限次 membership 不能被宣称为完整的 `Beh(P,x) subseteq Spec_J(x)` 证明。

Gate2 的 J-only reference 边不读取 π：

```text
O_J = RefEval(J,x)
```

因此，即使 π 对 source semantics 有很强统计预测能力，也无法绕过错误 J 的 Source–J 边。若采用
隐藏随机输入模式，Gate1 首次通过的完整候选 `a*=(J,C,pi,Replay)` 先选定且在本次 Gate2 中不再
修改，随后才从预先声明的分布 D 独立同分布地抽取 m 个对 Lift/Replay 隐藏的测试输入，且 Gate2
结果不回流本次实验。在预先确定的 Q 下定义完整三边谓词 `g2_Q(a*,x)` 及其错误质量：

```text
epsilon_a = Pr_{x ~ D}[not g2_Q(a*,x)]
```

若 `epsilon_a >= epsilon > 0`，则该固定坏候选通过全部 m 个 challenge 的概率上界是：

```text
(1-epsilon)^m <= exp(-m*epsilon)
```

Source–J disagreement mass 是该界的充分特例，因为 Source–J 是 Gate2 必须通过的一条边。该上界
不对所有“错误 J”无条件成立：只有错误在 D 的支持内、能被 Q 区分并具有至少 epsilon 的错误质量
时才适用；三条比较边也不会把 m 变成 3m。

若实验只使用预先给定的有限测试集合 `T`，则只声明：

```text
for every x in T:
    Close_Q(Source(x),Replay(x))
    and Close_Q(Source(x),RefEval(J,x))
    and Close_Q(Replay(x),RefEval(J,x))
```

此时不引用上述随机测试概率上界。两种实验口径必须明确区分，但都不需要新增 proof artifact、
额外 Gate 或其他运行设施。

因此需要区分：

```text
masked-J：representation/J-necessity audit
Gate2：semantic soundness
```

不能用前者替代或否定后者。

## 13. 当前实现的正确基础

当前实现已经具备：

- closed `regions/instances/routes/order/sharing` schema；
- declaration-order opaque `r*/i*/J/C` handles；
- raw handle 与 π graph instance namespace 分离；
- route operand slots；
- order/region acyclicity；
- sharing uniqueness；
- source text/API/formula 不进入 π；
- multiple instances 可以锚定同一个 semantic ref；
- Replay 使用 J/C/pi source-blind packet。

这些部分应保留并扩展，而不是恢复 composer。

## 14. 当前实现状态与剩余缺口

### 14.1 π parser 保持自闭合，系统 TypeCheck 已真正依赖 J/C

当前 `RealizationWitness` 主要验证 JSON 自身闭合：

- handle spelling；
- region/instance existence；
- route/order/sharing graph legality；
- mechanism enum。

当前活动系统已在 Decode/Elaborate 之后运行 `TypeCheck(J,C,pi)`，机械验证：

- anchor 是否真的存在于当前 J/C；
- observable instance 是否由 J node/access/effect 授权；
- route operand slot 是否与 J carrier 相容；
- representation refinement 是否属于系统封闭 schema，以及 sensitive refinement 是否具有
  J/B authorization 或属于系统安全白名单；
- observable instance 的 J ownership、route slot/type 与 sharing/order legality。

尚未宣称由固定核完整证明的部分是一般 quotient/homomorphism 与 `RealSig(J)` 的全量 mutation
性质；这些属于继续增强/测试的 claim 边界，不等于当前仍只有 `Parse(pi)`。

系统调用链已经从：

```text
Parse(pi)
```

升级为：

```text
TypeCheck(J,C,pi)
```

当前首版 TypeCheck 已机械完成：

1. 所有 J/C refs 存在；
2. route operand slot 与类型合法；
3. observable instance 锚定 J-owned node/access/effect；
4. π 不含语义原子、frontend API、source text 或 C value；
5. sensitive refinement 具有 J/B 显式 authorization，或属于系统安全白名单。
6. 未知字段、未知 relation 与自由文本扩展 fail closed。

这些项目已有聚焦正反例和真实 canary；其余图同态检查继续增强，但不能把尚未检查的性质报告为
已经由系统机械保证。模型 response 先经 Decode/Elaborate 构造唯一 typed J，使 TypeCheck 不依赖
consumer 私有字符串解析，也不增加第二个 J 对象。

有界、只读、非认证的 `CandidateCheck` 共享正式 Runner 的同一 source-free
parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行 source-identifier privacy、
receipt/provenance，并在提交后重新运行该 core。禁止维护第二套近似 checker；它向 source-aware Lift
agent 暴露当前完整 candidate 的 dependent diagnostics。该工具不得修改 candidate、
返回 patch、生成 authorization 或读取 Replay/Gate evidence；正式提交后 Runner 仍重新执行
TypeCheck。其显式输入、ephemeral validation view 与 closed diagnostic schema 以最终执行流程
第 6.4 节为准；禁止读取 ambient host state。因此它只缩短机械错误反馈，不改变上述 judgement、
π 的所有权或证书谓词。

若精确 candidate bytes 在本 Lift session 中尚无 CandidateCheck `PASS`，提交动作先调用同一
只读 checker。FAIL 不提交或修改对象；只有与某次 PASS 对应的精确 bytes 完全相同时才可复用，
任何不同 bytes 都必须重检；提交后的外部 TypeCheck 仍
必须重跑。这个提交前诊断边界没有获得任何 π 构造权。

当前 fresh worker 最多 24 个 model turns。CandidateCheck 采用总上限 8 的有界池：7 次普通共享
allowance 由显式调用与提前 submission preflight 共同消费，另有 1 次只允许 submission 使用的
reserve；receipt 分别记录 `shared_budget`/`submission_reserve` 以及每种 trigger 的 calls/evaluations。
若 worker 未提交候选，跨到同一 branch 下一 fresh Lift 的内容最多是最后一次实际 source-free
evaluation 的 20 条 `rule_code/path`；observed value、候选 patch、Source/Gate/hidden/sibling 信息均
不得携带。该摘要不能生成或修补 π，submission `PASS` 也不能替代外部 TypeCheck。

numeric-cast、result-relation compatibility 与当前 v4/v5 typed relations 的检查均已随版本化
TypeCheck 落地并写入 fresh receipt；旧 artifact/证书保持历史解释，不由新规则静默重判。

活动 fresh-run 使用 `typecheck.v18`。v18 完整继承 v14、v15-v17 typed-control 合同及下述 v7
前序合同，并检查 authored
value/dimension、consumer extension 与 access guard 的 dependent closure；它仍只拒绝和诊断，不补造
artifact。历史 v7 完整继承 v6；v6 继承 v5，v5 继承 v4 的 rank-0 closed signature、
Layer-A value typing total 与严格 NumPy broadcast compatibility；只有 J 中显式存在的 non-trailing broadcast relation
才能被消费。v5 又在 RefEval/emitter 前对已知 `reduce` 与 binary `contraction` 执行
canonical fail-early，拒绝别名、缺失/非法 canonical attrs 及可机械判定的 axis/rank/shape
矛盾；它同时拒绝 absent `opaque(mode=value,dtype=none)` 成为 π instance semantic ref、
realization authorization subject 或 refinement carrier。v6 又对可识别的零起点矩形 functional
logical-domain profile 机械检查 direct access index、padding predicate、`boundary.domains` 与 typed
tensor extent 的闭合；只读 checker 不从 π topology、C、Source、runtime/hidden input 或 shape
补缺失 relation，也不把 general affine/交叉轴 logical predicate 因 RefEval 能力不足而判为非法。
v7 又只对 rank-0 signature logical input 增加 source-free liveness：绑定 J input 的 `logical_ref`
必须进入 observable output dependency，或被显式 boundary/shape semantics 使用；identity/direct-output
和合法 boundary-only 路径保留，dead node 或 π route 不冒充 logical contribution。仅用于
representation/address 的 runtime scalar 必须显式取 `logical_ref=null`，不创建 J input、不进入 J
semantics，但其 p* carrier 可以由 π realization 引用。checker 不从名称/位置推断，不自动解绑或
修补；一般 tensor liveness 也不在 v7 中臆测。对应能力由当前 Lift knowledge v17 继承。上述拒绝不会给 π、CandidateCheck 或
elaborator 增加 artifact 构造权。

上述 representation-only null-carrier 只允许 runtime scalar。`bound_constexpr_values` 由 Lift 查看每个
Source use 后分类：logical data/predicate/result 绑定 dtype-compatible rank-0 J input，
integral extent-only 绑定 declared `x*`。只有全部语义 use 都是 compile-time
control/discrimination，且值不进入 active graph 的 shape/index/mask/arithmetic/boolean/result，
对应 constexpr p* 才只保留 callable carrier/type 并省略 `logical_ref`。参与 active graph 的
constexpr 即使 constant-fold 也必须显式走 t*/x*；data 与精确 dimension 重合时由 Lift 作者化
`value_dimension_relations`。纯 selector 值留在 Harness；π 不复制值、不选择 branch、不增加 realization
instance/route/refinement。Lift 自己作者化完整 active J，fixed checker 只检查 wire/type 形式，不代替 Lift 做该语义分类。
只影响 layout/tile/stage 的 constexpr 归入 C。representation-only bound constexpr 不允许以
`logical_ref=null` 伪装成 π-only carrier；static specialization carrier 是独立的 Harness-bound callable
情形，但同样不由 π 补语义。

v7 同时把 `padded_mean_count_unproven` 作为与 π 无关的 J-only necessary condition，但只在以下
closed profile 触发：direct load predicate 不含 logical guard；index/domain 是完整零起点矩形；每个
predicate bound 与同轴 typed tensor extent 相等；该 masked-axis provenance 经已建模 closed
pointwise/broadcast 位置传播后被 `reduce(mean)` 实际消去。`masked_contribution=0` 只闭合分子，
当前 J/B 没有 logical valid-count/denominator contract；π route 也不能补这个语义。dense/no-predicate
mean、guard、cross-axis/general predicate、domain/extent 不闭合、mask 位于未消去轴或未知 op 阻断
证明时不触发 TypeCheck；guard/cross-axis/unknown-op 由 functional consumer 明确 unsupported。
checker 不读取 Source/C/π且不自动变换图；Source 的显式 sum/div 必须由 Lift 原样写成
`reduce(sum)` 与 explicit division。

Gate2 scalar consumer 当前独立标为 `j_scalar_cpu_mlir_v16`，共享
`intentir.j-functional-boundary.v15`：它们继承 v14/v13、v12/v11 与 v10/v3；v10 已继承 v9 的 node-less input load，并按 J 自身的完整
logical domains、逐 access predicate/fill、逐 reduction 中性 `masked_contribution` 与 closed broadcast
relation 消费通用 padded pointwise/multi-reduce DAG。它不从 π route 推导 operator、padding 公式或
logical input。

v10/v3 只允许主 SCRC runner 在已选定 Gate2 winner 后，为当前 invocation 调用 RefEval-only 入口；
projection library 不自行选择 winner。该入口用 J 自身 rank-0 logical `i1`/integer values 与完整
closed dimensions，逐 access 按真实 guard 极性、direct axis、零下界及 runtime bound 不小于
domain extent 证明 predicate 全域为真。不改 J、不进入 Lift feedback、不读取 Source/C/π；v3
projection 内的 invocation-local proof subreceipt 不保存 raw value/value digest。字段最小化只服务
audit-data minimization，不是 confidentiality/privacy proof。production projection 仍无 runtime
inputs，π 也不能转交或复用该 subreceipt。缺失 relation，以及上述窄 rectangular provenance 可证明
masked axis 被 `mean` 消去却没有
logical count contract 时继续 unsupported。无法完成该 proof 的 guard-bearing access 与已有 masked
provenance 进入 contraction/其他未建模 op 仍在 consumer 入口 fail closed。这些是 Gate2/后端
projection 能力边界，不由 π 补证，也不扩大 TypeCheck legality。CPU-only `rsqrt -> 1/sqrt` host
libm compatibility pass 位于共享 Intent-to-Linalg 之后，不改变共享/GPU lowering。
既有 typecheck.v3/v4/v5/v6、`j_scalar_cpu_mlir_v7/v8/v9`、functional-boundary v1/v2 receipt 与
Softmax/RMS verdict 均不追认。

v11/v9 还逐项消费 v4/v5 J relation；可证明同 producer/observable value、单静态轴、常量
unit-width、无 predicate、无重叠无缺口的 direct-store partition，并通过 closed structured J-only
reference 执行 common scalar lowering 尚不支持的 mixed typed binary contraction。两者都只扩展
Gate2 reference consumer，不是 production recipe，也不让 π 获得 store/contraction 语义。修复这类
consumer 后可对 exact Gate1-closed artifact/Replay 做零模型调用的 Gate2-only recheck；结果写独立
evidence 且 `changes_parent_verdict=false`，不刷新旧 receipt。

v12/v11 只扩展 J 已显式闭合的 logical inout：同一 typed `t*` 同时是 input/output、由唯一 inout
参数绑定，pre-state load、post-state store、effect 和 canonical output binding 均来自 J。π 不声明也不
补 alias；`alias_of`/`boundary.aliases` 继续 unsupported。RefEval 分别消费 input/output binding，不
据 Harness 地址相等猜测语义。

v13/v12 的 post-state reload proof 还要求唯一 inout output mapping、同 index/predicate 的唯一 store、
后继 load 对 post-state SSA 的依赖以及 boundary/scoped store-before-load effect order。v14/v13 的
missing-fill proof 只接受全 downstream pointwise 且所有 observable store 被同 predicate 支配的情形；
reduce/scan/loop/transpose/pad 或不同 predicate 一律 unsupported。两种 proof 都只读 J 并形成 receipt，
π 不携带、不授权也不修补这些关系。

`j_scalar_cpu_mlir_v16` / `intentir.j-functional-boundary.v15` 进一步只消费 J 已写出的完整 product-domain
mixed-radix bijection，以及 pointwise provenance 后精确 `transpose` 的 axis permutation；后者还要求
store coordinates 与 load coordinates 的 permutation 完全一致并由同 predicate 支配。`log2` 与 IEEE
special literal 也由 shared materializer 直接消费。π route/instance/order 不能提供 factor、digit、perm、
fill 或常量语义，不能让不完整关系通过；这些能力不改变 π 的 dependent typing 或 Gate1 witness 含义。

当前 Lift knowledge v42 要求 `fill_value` 保留 Source load 的实际 `other`；MiniSWE Replay v4 要求
writer 按 J typed rank/axis 保留必要的 reshape/expand-dimension stage。π route 可以约束已有 instance
flow，却不能把一个 fill 换成“等价”neutral value，也不能授权 Replay 折叠 shaped reduction；两项仍由
各自 LLM 完整作者化，fixed checker 只拒绝和诊断。

canonical output binding 可以由 `returns` 或 `output_mapping` 唯一给出，同时存在时必须一致；共享
J→Intent MLIR 入口与 structured reference v5 消费 deterministic elaboration 后的同一 binding。该 elaboration 是 J 的私有
symbol-table view，不是另一个 artifact，也不使 π 获得 destination 语义。
该 authored 规则由 `intentir.lift-knowledge.v21` 明示；至少一个 binding 仍由 Lift 选择，fixed code
不会从 π route、store effect、Source 或 Harness 地址补造 destination value。
`intentir.lift-knowledge.v22` 同时明确 reduce/inclusive-scan 的 local-axis algebra 与
predicate-bearing load→sum/max/min consumer 的 typed outside identity 都由 J/B 的 Lift authoring
承担；π instance/route、C tile、load fill 或 reduction identity 都不能替代 `consumer_extensions`。

### 14.2 Mechanism enum 正在走向第二套 ontology

当前 mechanism 包含：

```text
realize
address_base
address_lane
transfer_direct
transfer_async
update_atomic
update_privatized
synchronize
```

其中 address/transfer/update 类别确实帮助了 canary，但继续按失败增加 mechanism 会重走
RuleCard/C.strategy 膨胀路径。

建议迁移方向：

- 保留 generic relation graph；
- 将可证明非语义的差异表达为受约束 representation refinement；
- access/effect/operator family 只从 J ref 获取；
- atomic/privatized 必须由 J/B effect 授权；
- 不再通过新增 kernel/mechanism event 名修一个 case。

### 14.3 Opaque subterm granularity 不足

当前已有 `q*` handle 能力，但需明确它们是否覆盖 J index/predicate/value expression 的稳定
canonical subterms。若不能，π 无法在不复制 formula 的情况下表达 source decomposition。

### 14.4 masked-J 状态机与完整 π 冲突

旧 Triton/CuTe orchestrator 曾在 normal closure 后执行：

```text
if masked hard-closed:
    status = LEAKAGE_REJECTED
```

该旧分支已从活动 Gate1 orchestrator 中移除；当前运行记录已经拆分：

```text
main_status = SCRC_CERTIFIED / NO_CLOSURE / ...
j_necessity_status = PROVED / NOT_PROVED / INCONCLUSIVE
typecheck_status = PASS / FAIL
```

`typecheck_status=FAIL` 使当前 artifact 无效。可选离线 J-authority challenge 若观察到非 J 通道
覆盖 J，只形成独立 authority-leakage 诊断，并用于下一轮系统改进；它不重新成为主在线 Gate。

因此运行关系必须保持为：

```text
TypeCheck(J,C,pi) = FAIL
    -> artifact FAIL

masked-J hard close and TypeCheck(J,C,pi) = PASS
    -> J_NECESSARY not proved
    -> does not automatically reject the main certification
```

## 15. 当前阶段不做什么

- 不恢复 deterministic composer；
- 不恢复 RuleCard/RuleRegistry；
- 不恢复 `C.strategy`；
- 不向 Replay 增加第四输入；
- 不给 Replay source、Harness、binding、diff 或 prior candidate；
- 不让 π 包含 C value；
- 不让 production lowerer读取 π；
- 不建立新的 frontend-independent executable lowerer ontology；
- 不要求所有 kernel 都获得 `J_NECESSARY`；
- 不把 J-authority challenge 放入每个 artifact 的基础在线预算；
- 不在本设计中引入 LoweringAgent、retune agent 或 backend cost model。

## 16. 建议迁移阶段

### 阶段 0：确认对象边界与 TypeCheck 合同，不调用 LLM

- 确认 `(J,C)` dependent type judgement；
- 建立 explicit semantics/authority/structural-inference 三类泄漏定义；
- 将主认证与 J 必要性证书分开；
- 保留当前历史实验记录，不静默重写旧 verdict。

### 阶段 1：Dependent validator

- 建立 `TypeCheck(J,C,pi)`；
- 验证所有 anchor/ref；
- 验证 observable instance J authorization；
- 验证 operand slot 与 J carrier；
- 验证 C refs 不含 values；
- 建立 representation refinement admission；
- 建立 quotient/homomorphism 检查。

### 阶段 2：Relational π schema

- 保留 regions/instances/routes/order/sharing；
- 收紧或迁移 mechanism enum；
- 增加最小 representation/schedule-ref 关系；
- 建立 opaque q* subterm contract；
- 禁止 semantic event names/free text。

### 阶段 3：Replay consumption

- prompt 明确 topology 由 π 约束、语义只读 J；
- Replay 必须保留 instance multiplicity、route slots、sharing、order、placement；
- C use 只读 MaskC(C) slots；
- frontend API 只来自固定 `U_f`。

### 阶段 4：Gate 状态拆分

- `TypeCheck(J,C,pi) + Gate1 + Gate2` 形成主认证；
- masked-J 单列 J necessity；
- 静态非法 π 仍 hard fail；
- masked hard-close + legal π 不再自动 leakage reject；
- 条件 J-authority challenge 作为可选诊断。

### 阶段 5：代表性 canary

在重新运行 28-case batch 前，至少离线/单次验证：

- 低熵 copy；
- elementwise fusion；
- reduction；
- address staging；
- masked access；
- logical update/effect；
- async transfer；
- sharing/recomputation。

目标不是每个 masked-J 都 mismatch，而是：

- normal Replay hard-close；
- Gate2 语义闭合；
- π 静态合法；
- J intervention 时 Replay 服从当前 J；
- π intervention 只改变 realization。

## 17. 验证矩阵

### 17.1 静态非法 π

- operator/reducer/predicate/fill/formula 注入；
- frontend API/source text/rule ID 注入；
- C value 注入；
- semantic handle name 注入；
- undeclared ref/instance/region；
- route slot 越界或交换；
- order/region cycle；
- J 中不存在的 observable effect；
- 未授权 atomic/cast/reassociation；
- quotient 后新增/删除 J carrier。

全部必须在 Replay 调用前 fail closed。

### 17.2 Parametricity

在相同 `RealSig(J)` 下替换 semantic label：

- add ↔ mul；
- sum ↔ max；
- predicate/fill 合法替换。

π 的 TypeCheck 结果不应因具体 label 改变；若 realization authorization 改变，则
`RealSig(J)` 已经不同，不适用这一要求。

### 17.3 J authority

固定 π/C，修改 J 语义：Replay 与执行结果必须跟随新 J。若继续实现旧 source semantics，则判定
非 J 通道具有非法语义权威。

### 17.4 π authority boundary

固定 J/C，修改 π 中一个合法 realization relation：

- sharing ↔ recomputation；
- placement；
- representation staging；
- order，在不违反 J/B 时。

Replay topology 应发生预期变化，但 Gate2 semantics 应继续满足同一 J/B relation。

### 17.5 Masked-J

分别记录：

- hard mismatch；
- hard close + legal π；
- hard close + illegal π；
- inconclusive。

不得将所有 hard-close 合并为同一种 leakage。

## 18. 论文与报告口径

主结果报告：

```text
TypeCheck(J,C,pi) pass rate
first-attempt TypeCheck pass rate
closure@1 / closure@2 / closure@3
Lift call count / Replay call count / total LLM call count
NO_CLOSURE failure categories
Gate2 pairwise Source/Replay/J pass rate among Gate1-closed candidates
SCRC_CERTIFIED rate
```

`closure@k` 是截至第 k 个正式 Gate1 轮次的累计闭合率，不是第 k 轮的独立成功率。证书绑定
实际找到的 Replay，不把有限预算搜索写成无成本的抽象存在式 `exists R`。

J 必要性单独报告：

```text
masked-J hard mismatch rate
masked-J hard-close with legal pi rate
masked-J hard-close with illegal pi rate
inconclusive rate
```

条件挑战单独报告：

```text
J-authority follow rate
J-authority violation rate
uninformative/inconclusive rate
```

不能把：

```text
masked-J hard-close
```

直接写成：

```text
J 与 Source 不等价
```

后者由 Gate2 Source–J relation 决定。

## 19. 后续实现中需要确定的通用接口

1. 当前 `mechanism` 字段是删除、保留 generic `realize`，还是迁移成 closed representation refinements？
2. q* subterm handles 的 canonical 派生与稳定性如何定义？
3. quotient/homomorphism 的首版检查应覆盖哪些 J carrier edge？
4. 哪些 representation property 可无条件进入 π，哪些必须由 J/B 显式授权？
5. auxiliary instance 是否允许 `ref=null`，还是要求全部锚定 J/C？
6. sharing 与 recomputation 如何在不携带 operator 的情况下完整表达？
7. compiler observer 哪些 incidental topology 应 normalize away，而不是继续扩 π？
8. `SCRC_CERTIFIED` 与 `J_NECESSARY` 在当前 receipt schema 中如何分离？
9. masked-J hard-close + legal π 是否默认只报告，还是必须触发一次条件 J-authority challenge？
10. 旧 `LEAKAGE_REJECTED` receipts 如何保留历史口径，避免静默重判？

这些问题必须通过全局、source-neutral 合同解决，不能按 Swiglu、LayerNorm、CuTe 或 kernel family
增加专门分支。

## 20. 最终规范

```text
Gamma_(J,C) |- pi : RealizationWitness(J,C)

Lift_H(S) = (J,C,pi)

Replay = W_f(J,pi,MaskC(C);U_f)

SCRC_CERTIFIED = TypeCheck(J,C,pi) and Gate1 and Gate2

J_NECESSARY = masked-J hard mismatch
```

> **π 应完整描述 source realization 的关系结构，但不得保存语义原子或取得语义裁决权。硬 schema 与 dependent typing 负责阻止显式复制和权威泄漏；masked-J 只测量在当前模型与 topology 下 J 是否具有操作性必要，而不再独自否定 Gate1+Gate2 已认证的 artifact。**
