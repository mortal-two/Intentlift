# J、C、π 与 TypeCheck

> 状态：当前派生规范。上位设计见 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## J：唯一语义对象

```text
J = A + B
```

J 拥有 logical values/state relation、domain、predicate、fill、logical index、effect、alias、
result relation 以及 logical signature。signature 中的 input、output、runtime scalar、destination、
dtype、rank、shape、alias/inout/effect 都有 J-owned opaque handle。

对绑定同一个 logical tensor handle 的 signature parameter，`mode=input` 要求该 tensor 只在 J
inputs 中声明，`mode=output` 要求只在 J outputs 中声明；同一 handle 同时出现在 inputs/outputs
时必须显式写 `mode=inout`。Decode/TypeCheck 不得把 input/output 自动升级为 inout，也不得让
Harness 或 Gate2 为错误 overlap 伪造输入值。

当前 fresh-run wire 是 `intentir.scrc.artifact.v9`，固定 Decode 是
`intentir.scrc.decode.v13`。Lift LLM 必须一次性显式写出完整 `(J,C,π)`，包括当前版本要求的空数组；
固定代码不从 Source、Harness valuation、C、π、Replay 或 Gate 结果补造 artifact 字段。`artifact.v4`
引入 J-owned `value_dimension_relations` 与 `boundary.consumer_extensions`；`artifact.v5` 再引入
J-owned `boundary.access_guard_relations`；v6/v7 增加 typed control，v8 增加 π-owned physical-only
time relations，v9 增加 π-owned terminal-control topology。旧 v1-v8 wire 仍按各自字段集合读取，
不通过附加空数组或新 relation 静默升级。

`opaque(mode=value,dtype=none)` 是一个精确的缺席 ABI carrier：它只证明调用位置必须原样接收
`None`，没有 logical denotation。Lift 必须显式生成这个 signature handle，并且不得把它作为 π
instance 的 `semantic_ref`、realization authorization 的 `subject_ref`，或 refinement 的 carrier；
TypeCheck 与共享同一 source-free core 的 CandidateCheck 对三种引用分别拒绝并报告，不删除引用、
不补造 dummy value，也不改写 artifact。该规则只由 `kind=opaque && dtype=none` 触发，其他 public
dtype 的普通 opaque handle 仍可按其明确合同参与 realization。

public-dtype runtime scalar 还可显式表示 representation-only ABI carrier，但唯一 wire form 是
`kind=scalar, mode=value, rank=0, logical_ref=null`：它不创建 J input，不得出现在 J
output/boundary/shape semantics，却可由 π 以 p* handle 引用来保留 address/representation realization。
这不是 C slot；C 仍只承载 implementation selection。Lift 必须按 Source use 自行分类，fixed code
不得按参数名/位置猜测或自动解绑。该 null form 只允许 runtime scalar；bound constexpr 若是
logical data/predicate/result 必须绑定 dtype-compatible rank-0 J input，若是 integral extent-only
必须绑定 declared `x*`。只影响 implementation selection 的 constexpr 进入 C；无法归入 J/C 的
bound constexpr 是 Harness invalid，不得留成 representation-only π carrier。

反之，scalar 一旦以 `logical_ref=t*` 创建 rank-0 J input，`typecheck.v7` 要求该 t* 到达 observable
output 的反向 value dependency，或出现在显式 J-owned boundary/shape semantic relation。direct
identity、function `returns`、`output_mapping` 以及合法 boundary-only use 都算贡献；dead A node、
signature 声明本身和 π route 都不算。v7 只检查 rank-0，不声称完成一般 tensor liveness proof。

B 同时拥有会改变允许结果关系的 typed/effect authorization 与 comparison class 声明。它不包含
任意可执行 comparator；外部验证政策 Q 根据 B 的封闭类别构造实际比较关系。

comparison class 必须与 output dtype、A 依赖图和 B numeric contract 相容。TypeCheck 只检查并
拒绝不相容声明，不替 Lift 选择或改写 relation。当前采用最终执行流程第 6.1 节的 closed
`RequiresTolerance_v1(J,output)`：从 output producer/return/mapping 沿 A operands 得到依赖闭包，
浮点 output 若依赖 closed transcendental 集、floating reduction 或其依赖节点具有
`kind=reassociation` 的 source-realization authorization，则显式或默认 `exact` 均被拒绝。v1 没有
自由文本或“更强合同”例外；具体 tolerance 阈值仍只来自 Q。该 authorization 只触发 SCRC
compatibility check，不成为 production backend permission。

J access 采用唯一的 canonical fill 合同：

```text
fill_value != null => kind = load and predicate != null
```

无条件 access 的 `predicate` 与 `fill_value` 均为空。parser/schema 可以提前报告该错误，但
`TypeCheck(J,C,pi)` 必须权威拒绝 orphan fill；任何消费者都不得通过静默删除或猜测 predicate
来归一化候选 J。

`typecheck.v6` 还复用 `FunctionalBoundaryProjection` 的 J-only necessary condition。对于 checker
可识别的零起点矩形 logical-domain profile，direct access index、padding predicate、
`boundary.domains` 与 typed tensor extent 必须闭合：`k0 < x0` 必须直接索引 `k0`，且 J 显式包含
同 extent 的 `0 <= k0 and k0 < x0` domain。Source、C、π、runtime/hidden input 或 tensor shape 的
“显然含义”都不能补一个缺失/不同的 domain；checker 也不能删除 predicate/fill/effect 来让它通过。
该 rule 不把 general affine/交叉轴 logical predicate 一概禁止；当前 RefEval 未实现的能力仍是
consumer capability，不被偷换成 J 语义非法。

`typecheck.v7` 还对 padded `mean` 执行一条独立、窄的必要条件。只有 direct load predicate 不含
logical guard、index/domain 构成完整零起点矩形、predicate bound 与对应 typed tensor extent 相等，
且同一 masked axis 经已建模 closed pointwise/broadcast 位置传播后被 `reduce(mean)` 消去时，当前
`ReductionBoundary.masked_contribution` 才只能证明分子中性、不能证明 logical valid count 作为
除数，因而以 `padded_mean_count_unproven` 拒绝。dense/unpredicated mean、guard-bearing 或
cross-axis/general predicate、domain/extent 不闭合、mask 位于未消去轴或未知 op 阻断传播时不触发
TypeCheck；guard/cross-axis/unknown-op 由 functional consumer 明确 unsupported。checker 不读取
Source/C/π，也不把 mean 自动重写为 sum/div。

### `artifact.v4` 的 value/dimension 与 consumer-context relation

当同一个 Source scalar 同时是 integral rank-0 logical data 和 symbolic shape extent 时，Lift 必须
分别保留 `t*` 与 `x*`，并显式声明：

```json
{"value_ref": "t0", "dimension_ref": "x0", "relation": "eq"}
```

`value_ref` 必须是 integral rank-0 J input，`dimension_ref` 必须是 J shape 中已声明的 `x*`，当前
relation 只允许 `eq`。TypeCheck、CandidateCheck、Replay、RefEval 与 production projection 只能
检查或消费这条已声明等式；不得因两者当前 valuation、dtype、位置或 shape 恰好相同而自动建立
relation。一个 extent-only scalar 仍直接绑定 `x*`，不需要伪造 rank-0 data value。

`boundary.consumer_extensions` 是 predicated load 到一个 `sum|max|min` reduce/scan operand 的
J-owned consumer-context relation：

```json
{
  "consumer_ref": "n4",
  "operand_index": 0,
  "access_ref": "a0",
  "tile_axis": 0,
  "outside": {"dtype": "f32", "value": 0.0},
  "identity_law": "two_sided"
}
```

它声明该特定 consumer 在被引用 access 的 predicate 域外看到的 typed two-sided identity；predicate
仍只由该 `BoundaryAccess` 拥有，不能被 fixed code 提升成 Layer-A SSA。TypeCheck 必须检查 consumer
种类与 operand ordinal、access provenance、local tile axis、dtype 和 identity law，并对需要而缺失、
重复或矛盾的 relation fail closed。`artifact.v4+` 中它是这一 consumer-specific 域外贡献的唯一
authority；旧 `ReductionBoundary.masked_contribution` 不得与它并存。

### `artifact.v5` 的 access-local computed guard relation

computed Layer-A boolean guard 与 boundary access 的关系由 Lift 显式声明：

```json
{
  "access_ref": "a2",
  "value_ref": "v7",
  "axes": [0],
  "polarity": "positive"
}
```

`value_ref` 必须是支配该 access 的 typed `i1` Layer-A declaration；`axes` 是严格递增且唯一的 target
broadcast axis ordinals；`polarity` 只取 `positive|negative`。一个 access 当前最多一条 relation。
relation-bound guard 不得同时复制进 access predicate；对于同一 `select` guard，
`access_guard_relations.axes` 是唯一 broadcast authority，`boundary.broadcasts` 不能重复声明，即使
轴完全相同也拒绝。

该 relation 不拥有 merge value，也不授权 fixed code 插入控制流。relation-guarded store 必须由
Layer A 自己显式写出三 operand `select`，且目标 tensor 同时是 J input/output，并由唯一
`mode=inout` signature parameter 绑定；TypeCheck 永不把 output 自动升级为 inout。非 loop store 的
精确 pre-state 是该 access tensor 本身，函数 return 与唯一 `output_mapping` 必须指向 select result。
loop store 的 `carry.init_value` 必须精确等于该 access tensor，inactive branch 必须是当前
`body_argument`，select result 必须精确成为 `carry.next_value`。`positive` 表示
`select(guard,candidate,current)`，`negative` 表示 `select(guard,current,candidate)`；不能仅凭相同
dtype/shape 接受另一个 input 充当 pre-state。guarded load 还必须显式声明 fill；v1 relation 对
guarded update 没有闭合 merge contract，因而 TypeCheck fail closed。

Gate2 functional projection 对每一条 access guard 必须产生明确 receipt 或返回 consumer
unsupported，不能忽略 relation 后只运行 A。当前 RefEval 支持上述显式-select guarded store；guarded
load/update 的 access-local pre-execution materialization 尚未实现。production J+C 当前对任何
access guard 以 `access_guard_materialization_not_implemented` fail closed，即使 shared projection
已经能检查 relation，也不能把 RefEval 能力当成 target permission。

## J 的构造与标准化

模型先返回尚未接受的 response JSON；固定 decoder/elaborator 成功后才构造出 J：

```text
response JSON -> Decode/Elaborate -> J | deterministic diagnostic
```

因此不存在 `J_wire` 与 `CanonicalJ` 两个对象。接受后的 typed archive 就是唯一 J；解析得到的
ContractExpr AST/axes 只是 J 的私有派生缓存，不能单独序列化或拥有身份。该阶段不读
Source、Harness concrete value、C、π、Gate diff/output 或模型历史，只能完成 closed grammar 中
唯一、语义不变的推导：

- `k0`/`[k0]` 与 `k1,k0`/`[k1,k0]` 分别归一为相同的 structured ordered axes；
- domain/predicate/index formula 解析为 ContractExpr AST，consumer 不再私自解析字符串；
- closed op signature 与 J value graph 唯一确定的 result dtype/shape 被物化；显式类型必须一致；
- 可机械证明的 direct same-handle output identity mapping 归一为唯一省略形式。

`ContractExpr v1` 的 integer/index sort 使用目标无关的数学整数语义，而不是隐含的 host
`int64`、target `index` 宽度或 Triton/C 整数回绕语义。`a // b` 是 floor division，
`a % b = a - (a // b) * b`，`ceil_div(a,b)` 是数学 ceiling division；这些除法只在
`b != 0` 时有定义。若 Source 的 observable relation 确实依赖某一有限宽度、signedness 或
overflow 行为，Lift 必须通过 typed J value/op relation 表达，不能借 access formula 的实现细节
补入。

因此，把 ContractExpr 发射为有限宽度 MLIR integer/index 运算本身不构成合法性证明。target
lowerer 必须对 J 已声明的完整 logical domain 逐项消解至少以下 compiler obligations：所有引用、
常量与中间/最终整数结果在所选机器类型中可表示，所有 divisor 非零。不能消解时必须
unsupported/fail closed；不得读取 Source、π、C concrete value 或 Gate sample 猜测，也不得把
目标宽度、overflow 假设自动写回 J。这些 obligations 是 `J -> M0` 的派生检查结果，不是新的
artifact、B relation、证书或 membership oracle。

canonicalizer 不得猜 operator、缺失 predicate/effect/alias/authorization，不得从 Source 修复 J，也
不得把 general affine expression 偷换成 direct axes。语法无唯一解释、类型冲突或需要语义选择时
fail closed。响应 schema 目前仍以 `index_expression` 文本保存，但 J 的 canonical dump 只写 bracketed vector；
内部消费者读取 structured AST/axes。

RefEval 与 production lowering 还可以从这个唯一 J 确定性计算一次 ephemeral
`FunctionalBoundaryProjection`，以共享 Layer-B consumption 判定。该值既不包含另一份 J，也不
进入 `(J,C,π)`、Replay packet 或证书身份；任何缓存或诊断记录都不得反过来改写 J。

因此接受条件准确写成：

```text
Decode/Elaborate(response) = (J,C,pi)
and TypeCheck(J,C,pi)
```

## C：一个 implementation-selection 对象

C 只包含在不改变 J denotation 时可替换的匿名 slot、binding kind、value type、concrete
valuation 与必要的 source-neutral selection metadata。C 不得包含 operator、predicate、formula、
logical shape、result semantics、source topology、frontend API 或 per-kernel executable policy。

`artifact.v2` 引入、当前 `artifact.v9` 保留的 slot 字段为
`name/kind/role/value_type/required`，其中 `role` 必填，并增加两个可选 attachment：

```text
subject_ref in CSubject_v1(J)
axis_ref    in (non-negative integer | declared x*)  only if subject_ref exists

CSubject_v1(J)
  = NodeIds(n*) union DomainIds(d*) union AccessIds(a*)
    union ScopeIds(q*) union StorageIds(b*)

axis_ref in AxisRefs_v1(J, subject_ref)
```

`AxisRefs_v1` 是 closed dependent domain：reduce node 使用其显式 reduction input-axis ordinals；loop
node 使用 trip-axis ordinal `0` 及其显式 `trip.axis=x*`，两种 spelling 都指向同一 trip axis；domain 使用 canonical ContractExpr ordered
axis ordinals，并只在显式绑定 declared `x*` 时接受该符号；access 使用 canonical structured-index
axis ordinals，并只在对应 extent 显式为 declared `x*` 时接受该符号；scope/storage 的集合为空；
其他 node 只有 closed A op contract 显式声明 axes 时才非空。不得从 shape、role、Source、π 或唯一
候选位置推导 axis。集合为空、越界或无法唯一解析均由 TypeCheck 拒绝。

attachment 只定位 source-neutral prior，不授权 transformation。Lift 必须显式生成；Harness、
Decode/Elaborate、CandidateCheck 与 TypeCheck 不得推断或补造。TypeCheck 只检查
`axis_ref in AxisRefs_v1(J,subject_ref)` 与 `axis_ref => subject_ref`。全局 prior 可以没有 attachment；
局部 prior 缺失、含混或 consumer 不支持时必须安全忽略，不能读取 Source/π 猜测。具体 legal
domain 属于 implementation family/target capability，不在 C 中增加
`legal_domain/strength/retunable/fixed`。

`tile_extent` 与 `reduction_tile_extent` 是触发 iteration missing-totality 的保留 role。它们只让
TypeCheck 能够在 attachment-closed J-loop/slot pair 完全缺 relation 时机械拒绝，不是显式
`pi.iterations` relation 的 hidden validity tag；custom-role slot 只要满足同一 dependent type 也可
被显式引用。custom/misclassified role 导致 relation 整体遗漏时，固定 checker 无法从 role 推断，
仍由 Gate1 topology 检出。

```text
MaskC(C) = erase(concrete values, private provenance)
```

Replay 只读取 `MaskC(C)`；Harness 将同一 C valuation 绑定给 Source 与 Replay。C 不拆成
`C_source`、binding map、portable hints 或其他顶层对象。
`MaskC(C)` 保留 `subject_ref/axis_ref`，因为它们只引用 J；source invocation provenance 仍被擦除。

## π：依赖类型实现见证

```text
Gamma_(J,C) |- pi : RealizationWitness(J,C)
```

π 可以表达 region、instance、route、operand slot、order、sharing、placement、materialization 与
受约束 representation refinement。其锚定关系为：

```text
mu : Instances -> Ids(J) union Ids(C)
```

不存在 `Ids(interface)`。需要接口信息时读取 `InterfaceView(J)`，但该投影不增加命名空间、
不序列化成独立 artifact，也不作为 Replay 的额外输入。

π 禁止携带：

- operator/function/reducer、predicate/fill/index formula 或 semantic constant；
- logical shape、result relation、destination/conflict semantics；
- Source text/name/path/span、Kernel/rule/family ID；
- frontend API、C concrete value 或可执行自由文本；
- 任何能在删除 J/C 后获得独立 denotation 的结构。

所有 observable realization instance 必须锚定 J-owned node/access/effect。representation-only
scaffolding 折叠后，π 不得增加、删除或改写 J carrier 的逻辑边。

一个 Source SSA coordinate/address intermediate 若只定义一次并被多个 access 使用，π 必须用一个
producer instance、多个 routed consumers 与 one-producer sharing 表达。多个语法 use 不是多个
producer；只有不同 compiler-visible definition/stage 才能重复同一 semantic ref。反过来，若 Source
确有多个 stage，也不能为了减少 Replay 工作而把它们折叠成一次。

`region.placement=program` 只说明 instance 位于单个 program 的 region 中，不表示 Source 读取了
program/grid coordinate。若 Source 的确存在 `program_id`、grid offset 等 compiler-visible stage，
Lift 必须用一个独立 `realize` instance 及 routes 记录；无需新增 mechanism enum。没有该独立
instance 时，Replay 不得因 placement、常见前端写法或 C 自行发明 program coordinate、乘加或
affine address stage。

该 generic relation 的机械区分是：program region 中锚定相应 J domain 的 `realize` instance，其
scalar dependency path 先进入 `address_base` 再进入 transfer，表示 program-coordinate stage；进入
`address_lane` 的 domain/C dependency path 表示 lane-varying stage。Replay 必须保持 base/lane
区分，`U_f` 再把它映射到公共前端 API。该规则不在 π 中保存 `program_id`、axis 公式或前端文本，
也不增加 mechanism enum。

`address_base/address_lane` 标记 address contribution 角色；pointer/address operation 的次数由
`J.signature` carrier kind 与 π route 共同决定，不能只看 base-to-lane edge。若 pointer/tensor 的
`p*` instance 直接进入 `address_base`，该 base 已经是一次 pointer/address operation；其 pointer
结果再 route 到 `address_lane` 时，Replay 必须保留第二次 lane pointer/address operation。若
`address_base` 只接收 integer/domain/C contribution，而 pointer carrier 绕过 base、直到
`address_lane` 或最终 address stage 才进入，则 Replay 保留两个 integer stage、先合并 base/lane，
再执行唯一 pointer add。两种情况都不能因当前 grid extent 为 1 将 program path 常量化；只有不
存在可分离 program-uniform contribution 时才允许 lone `address_lane`。

单个直接锚定 J signature handle 或 C slot 的 `realize` instance，可以由对应参数 binding 及其直接
routed use 满足，不要求为了“显式 materialize”增加恒等赋值。尤其 `c*` 为 constexpr 时，前端
合同可以要求直接 use 以保留编译期类型。若同一 handle 有多个 instance，它们仍必须对应不同的
routed use-site/stage，不能据此全部折叠。

π 不能补足 J 中缺失的 semantic recurrence。若 Source 的 compiler-visible value dataflow 是
loop-carried recurrence，J 必须用现有 structured `loop` 完整声明 trip/scan、carry/init、body update
以及对应 next/yield/result；contraction 的 canonical subscripts 与 result-axis orientation 必须和
update 实际消费的值一致。π 只记录这些 J carrier 的 placement/multiplicity/routes，不能携带或让
Replay 猜 initialization、update formula、yield 或 transpose/orientation。

### 物理迭代基底

J structured `loop` 拥有逻辑 trip/scan/carry 语义，C tile slot 只拥有 implementation-selection
数值；二者仍不能区分 Source 的物理 induction variable 是“逻辑 extent 坐标”还是“tile 序号”。
例如下列两种 source realization 可共享同一 J loop 与同一 C tile，却产生不同 compiler topology：

```text
extent_step:  for k in [0, Trip(J.loop)) step Tile(C.slot)
tile_ordinal: for q in [0, ceil_div(Trip(J.loop), Tile(C.slot))) step 1
```

因此 `artifact.v3` 引入、当前 `artifact.v9` 继续要求 π 包含 closed `iterations` 数组。每项只有：

```text
{region, loop_instance, tile_instance, induction_basis}
induction_basis in {extent_step, tile_ordinal}
```

`loop_instance` 必须锚定 structured J loop，`tile_instance` 必须锚定 attachment 指向同一 loop/trip
axis 的 required positive-integer C slot，axis 可写 canonical ordinal `0` 或对应 `trip.axis=x*`；显式
relation 的合法性不依赖该 slot role。二者在该 loop region 或其 lexical ancestor 可见。wire
不得携带 lower/upper/step 公式、concrete tile、Source 片段、operator 或 frontend API；两种 basis
的上述全局 denotation 由协议固定，各 frontend 的 `U_f` 只能忠实映射。

parser 对一个 declared physical region 最多接受一条 relation；同一 `(J loop,C tile)` 若在 Source
中确有多个不同 physical loop region，可以由多个 region 分别引用。`typecheck.v7` 的可判定 totality
边界是：每个 attachment 闭合且 role 为 `tile_extent|reduction_tile_extent` 的 C-tiled
J-loop/slot pair 至少存在一条合法 relation；untiled J loop 不被迫制造 relation。固定 checker 无法
从 `(J,C,π)` 发现 custom/misclassified role 下完全遗漏的 relation，也无法发现 Lift 完全遗漏、因而
没有任何 handle 的 Source physical region，所以 Source topology completeness 仍由 Lift 完整提出并
由 Gate1 对 Source observer 检出；不得把局部 missing-totality 夸写成“每 Source region 完整”。

该 relation 只消除 Replay 的 source-realization 欠定，不给 production backend 增加输入：SCRC 后
production 仍只消费 J+C，自行确定性 materialize/search `Impl(J)`，不得读取 π 来选择 loop basis。

### J 外物理 tile time

`artifact.v8` 对已有 J reduce/contraction 外的 Source physical loop 使用：

```text
physical_iterations = {
  region, body_instance, tile_instance,
  induction_basis: extent_step | rounded_extent_step | tile_ordinal,
  direction: forward | reverse
}
physical_loop_carriers = {region, init, state, next, result}
```

body/tile 必须分别锚定已有 J node 与同 node/axis 的 required positive-integer C slot；carrier 只保存
typed temporal topology。字段不含 bound/formula/value/API，不在 J 中创建 loop。TypeCheck 只验证
closed reference/type/scope/totality；Source 的真实方向、数量与完整性仍由 Gate1 判断。Replay 必须由
J+C+π 得到 active physical tile rank，不能把 logical axis ordinal 直接当 frontend tensor-rank axis。

### Terminal control topology

J `effect_if` 拥有 predicate、positive/negative arm 及臂内 value/effect。若 Source realization 的一臂
终止当前 invocation、另一臂承载 lexical continuation，Lift 可在 `pi.terminal_controls` 显式写：

```json
{"terminal_instance": "i1", "continuation_instance": "i0"}
```

两个 handle 必须是同一个 `effect_if` 的 sibling arm-scope `realize` instances；该 `effect_if` 必须是
lexical parent 的 final operation，所有后续工作位于 continuation arm。relation 只声明 roles，不包含
predicate、polarity、return/value/formula、Source 或 frontend API。无 relation 的 `effect_if` 是普通
显式双臂 control。TypeCheck 不从空臂、effect 或名称推导 relation；Replay 只按 authored relation
materialize terminal CFG；production backend 不读取 π。

## Authorization 与 parametricity

```text
RealSig(J) = (Carrier(J), Capabilities(J,B))
```

`Capabilities(J,B)` 只包含 realization authorization，不复制 operator、predicate、fill、reducer
等 semantic labels。在相同 `RealSig(J)` 下，替换 semantic label 不得改变 π 的 TypeCheck 结果。

对 integer width、numeric cast、accumulation dtype、reassociation、atomic form 等 sensitive
refinement：

```text
Sensitive(r)
  => auth_ref(r) in Auth(J,B)
     or kind(r) in GlobalSafeRefinement
```

`GlobalSafeRefinement` 是系统统一维护的封闭安全集合，不由 Lift 或单个 Kernel 修改。两项均不
满足时 fail closed；TypeCheck 不现场运行 SMT、range proof 或一般语义等价证明。

authorization 的 carrier typing 按 kind 定义。对：

```text
n = cast(x : Ts) -> Tt
u = numeric_cast(subject_ref=n, source_dtype=Ts, target_dtype=Tt)
```

必须检查：

```text
u.kind == numeric_cast
u.subject_ref resolves to n
OpKind(n) == cast
Arity(n) == one operand and one result
u.source_dtype and u.target_dtype are both present
u.source_dtype == Type(n.operand[0])
u.target_dtype == Type(n.result)
refinement.kind == u.kind
instance.semantic_ref == u.subject_ref
refinement.auth_ref == u.auth_id
```

不能将 `source_dtype` 与 cast result carrier 比较，也不能要求 subject 改绑 cast 前的 tensor。
authorization 和 refinement 都由 Lift 显式声明；decoder/TypeCheck 不得自动生成缺失项。
每个 π instance 至多一个 refinement；重复项 fail closed，不由 parser/tool/Replay 选择。

## 最小 TypeCheck

`TypeCheck(J,C,π)` 接收已经构造完成的唯一 J，并至少检查 J access 的 canonical fill 合同、refs、route slot/type、observable
J anchor、语义/API/Source/C-value 隔离以及 sensitive authorization。自闭合 `Parse(π)` 只能作为
语法解析器，不能冒充 dependent TypeCheck。

TypeCheck 的结构化错误可以返回 source-aware Lift；检查结果和 sidecar 不得进入 Replay 作为
新的信息通道。schema 为 closed-world：未知字段、未知 relation 与自由文本扩展均 fail closed。

`typecheck.v5` 对已知 `reduce` 与 `contraction` 在进入 RefEval/emitter 前执行 canonical fail-early
contract。`reduce` attrs 必须
使用 `fn + axes + assoc`，其中 `fn` 只允许 `sum|mean|max|min|prod|any|all`，`axes` 是非空整数数组且
归一化后不重复、不越过 operand rank，`assoc` 只允许 `associative|float_nondeterministic`。可选
`keepdims` 若存在必须为 boolean；它不取代 result type/shape，declared result 与 keep/collapse shape
仍由既有 elaboration/type logic 核对。`kind` 不是 `fn` 的别名；无论单独出现还是与 `fn` 冲突都必须
拒绝，fixed checker/emitter 不得翻译、补齐或修复 `fn/assoc/axes`。`contraction` 只接受 canonical
`subscripts`，并按显式 operand/result rank 校验其输入项、reduction labels 与输出 orientation；
`lhs_contract_axes`、`rhs_contract_axes` 等别名不能由 fixed code 翻译成 `subscripts`。

准确的 TypeCheck claim 是：

```text
TypeCheck(J,C,pi) -> FreshLogicalSemDecl(C,pi | J) = empty
```

在合法引用解析回 J 后，可简写为 `SemDecl(J,C,pi)=SemDecl(J)`；这不等于 π 没有统计语义
线索，也不证明 J 或 realization 的行为正确。

上述 C attachment、numeric-cast、result-relation compatibility、rank-0 closed signature、Layer-A total
typing 与显式 non-trailing broadcast 合同已由早期 TypeCheck 版本逐步建立；canonical reduce/
contraction fail-early、functional logical-domain closure、rank-0 logical-input liveness 与
representation-only scalar 合同也保持原有版本化历史。物理迭代基底 relation 曾使 fresh-run
wire/decode 窄升为 `artifact.v3 / decode.v3`；v4 随后增加 closed dimension relations、
`value_dimension_relations`、`consumer_extensions` 和 typed loop-carrier wire，v5 再增加
`access_guard_relations`，v6/v7 增加 typed control，v8 增加 physical-only time，v9 增加
terminal-control topology。当前 active 合同为
`artifact.v9 / decode.v13 / typecheck.v23 / candidate-check.v1`；v23 在 v22 上新增 canonical
output binding 的完整 dtype/shape exact dependent check，禁止用相同 concrete valuation 或相同元素
数掩盖 axis permutation。v22 增加 signature mode 与 J
input/output boundary partition 的 exact dependent check：input-only、output-only 与显式 inout overlap
不得混写。v22 继承 v21 的 terminal-control sibling/finality checks，以及 v20 对上述 relations 的
source-free dependent checks，尤其验证 authored guard 的支配、类型、唯一 axes authority、显式
select、精确 inout/pre-state 与 loop/non-loop observable merge。`candidate-check.v1` 的 closed envelope
本来就
允许 diagnostics 数组，因此诊断聚合不改变其 acceptance path 或 protocol：同一严格 checker 已经
判定出的独立机械错误按稳定顺序返回，统一最多 20 项且每段文本最多 2048 字符；checker 不为继续
检查而修补 artifact，也不在前置对象解析失败时猜测 dependent error。`candidate-check.v1` 共享
正式 Runner 的同一 source-free parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner
另执行 source-identifier privacy、receipt/provenance，并在提交后重新运行该 core。receipt 必须记录
这些版本；`artifact.v1`、`artifact.v2` 仍按各自旧 wire 读取且不得携带 `pi.iterations`，其中 v1
也不能携带 v2 引入的 C attachment；v1-v3 禁止 v4 字段，v1-v4 禁止 v5 access guard。旧
receipt/证书不由新 wire 静默重写、追认或刷新；v8/v9 的新增字段同样不附加到旧 wire。
