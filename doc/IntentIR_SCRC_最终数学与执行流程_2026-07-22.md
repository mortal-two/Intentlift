# IntentIR / SCRC：最终数学、执行流程与论文表述

> **文档定位：数学与论文 claim 总述。** 对象边界分别以 C、π 和最终执行流程三份核心设计为准；
> 本文统一它们的数学记号、证书谓词和可宣称范围，不引入新的 artifact、Gate 或运行阶段。
>
> 核心设计：[C 所有权](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
> [π 实现见证](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)、
> [最终执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。
>
> **项目定位：开放世界生成，封闭核心认证。**
> 本文给出 IntentIR / SCRC 的最终统一理解：保留 LLM-first、低前端工程量与严格信息边界，不把系统扩张成每前端 lowerer、形式语义或大型规则库；同时用一套明确的数学对象和逐 artifact 证书，说明 TypeCheck、Gate1、Gate2 与 relift 各自支撑什么。

---

## 0. 一页结论

给定源 Kernel `S` 与外部 Harness `H`，Lift LLM 产生且只产生：

\[
\operatorname{Lift}_H(S)=A=(J,C,\pi).
\]

其中：

- \(J\)：artifact 中唯一允许**显式声明程序语义**的对象；
- \(C\)：实现选择与调度参数；
- \(\pi\)：依赖于 \(J,C\) 的 source-realization witness；
- \(H\)：外部调用、绑定与执行环境，不是第四个 artifact。

Replay LLM 看不到 Source，只接收：

\[
\operatorname{View}_f(A)
=
\bigl(J,\pi,\operatorname{MaskC}(C),U_f\bigr),
\]

并实际生成一份 Replay：

\[
R=
\operatorname{Replay}_{\theta,f}
\bigl(\operatorname{View}_f(A)\bigr).
\]

最终证书由三个互不替代的义务组成：

\[
\boxed{
\mathsf{SCRC\_Cert}
=
\underbrace{\mathsf{TypeCheck}(J,C,\pi)}_{\text{所有权与引用合法}}
\land
\underbrace{\mathsf{Gate1}}_{\text{source-blind 构造闭合}}
\land
\underbrace{\mathsf{Gate2}}_{\text{测试域行为与语义闭合}}
}
\]

因此，SCRC 认证的准确含义是：

> **在声明的前端观察器、比较政策、实际 C 绑定和测试域下，一份显式语义所有权受约束的 \((J,C,\pi)\) 能够在 Source 隔离条件下重建一个与 Source 编译观察闭合的 Replay；Source、Replay 与 J 的可执行语义进一步在测试域内一致。**

它是一种 **formalized validation protocol / hybrid certification framework**，不是对任意 GPU Kernel 的全输入域形式等价证明。

---

# 1. 研究问题与设计哲学

GPU Kernel 生态具有开放、碎片化和快速变化的特点：

- Triton、TileLang、CUDA、CuTe 以及新的 DSL 持续出现；
- 同一逻辑语义存在大量 source realization；
- 如果为每个前端建立完整形式语义、J→frontend lowerer、专用 verifier 和每类算子规则，工程成本会随前端、算子与实现形态的乘积增长；
- 纯粹让 LLM 生成代码并做几个输出测试，又缺少结构约束、可审计性和错误定位能力。

项目选择的设计点是：

\[
\boxed{
\text{让 LLM 吸收开放生态的理解与生成复杂度；}
\quad
\text{让小型确定性核心限制错误。}
}
\]

具体分工是：

### LLM 负责开放世界任务

- 从任意前端 Source 中 Lift 出 \(J,C,\pi\)；
- 根据 \(J,\pi,\operatorname{MaskC}(C)\) 直接生成任意前端的完整 Replay；
- 理解新 API、新 DSL、新 source realization；
- 在 Gate1 差异反馈后重新 Lift 完整 artifact。

### 固定代码负责封闭世界裁决

- schema、类型、引用和所有权检查；
- Replay 输入隔离；
- C 的机械绑定；
- 编译与 compiler observation；
- 确定性比较；
- Gate2 执行与数值比较；
- 实验记录。

固定代码不负责：

- 把 J 翻译成 Triton/TileLang/CUDA API；
- 根据 kernel family 选择 RuleCard；
- 为每种 source topology 增加 StrategyFamily；
- 从 \(J,\pi\) deterministic 地拼装 Replay 源码。

---

# 2. 系统边界与符号

固定一次证书环境：

\[
\eta=
\bigl(
 f,p,H,U_f,\Phi_{f,p},Q,T,v_D,v_{TC},v_J,\mathcal E,\mathcal G
\bigr),
\]

其中：

- \(f\)：当前 Source/Replay 前端；
- \(p\)：observer profile，包括 observer 版本、compiler/dump stage 与 canonicalization 版本；
- \(H\)：Harness，持有实际输入、C 值、semantic-constexpr 调用绑定和调用坐标；
- \(U_f\)：公共、与当前 Source 无关的前端知识；
- \(\Phi_{f,p}\)：Gate1 在 profile \(p\) 下的 frontend observer；
- \(Q\)：由实验协议或 benchmark correctness policy 预先确定的比较政策；
- \(T\)：Gate2 测试集；
- \(v_D\)：Decode/Elaborate protocol version；
- \(v_{TC}\)：正式 TypeCheck protocol version；
- \(v_J\)：J-only evaluator 版本；
- \(\mathcal E\)：编译器、运行时、设备等环境；
- \(\mathcal G\)：模型、prompt、调用预算和实验流程配置。

Harness 对 C 的本次实际绑定写作：

\[
c_0=\operatorname{Bind}_H(C).
\]

Harness 中已经存在的 compile-time semantic specialization 子绑定记作：

\[
\rho_H^{\mathrm{static}}:P_{\mathrm{static}}\to\mathrm{ClosedScalar}.
\]

它就是 Harness/编译调用已有的 `bound_constexpr_values` 子映射，不是新增 artifact、J relation 或
production 输入。证书是 **instance-wise** 的：它限定本次 \(\rho_H^{\mathrm{static}}\) 与 \(c_0\)，
不自动证明同一 Source 的其他 JIT specialization 或其他 C 取值。

---

# 3. Artifact 因子化

## 3.1 Lift 的唯一输出

\[
\boxed{
\operatorname{Lift}_H(S)=(J,C,\pi)
}
\]

不允许把下列对象扩张成第四个 proof-relevant 输入：

- BindingMap；
- Harness view；
- portable hints；
- Resolver 输出；
- RuleCard 选择；
- compiler diff；
- 旧 Replay；
- source-conditioned frontend context。

实现可以保存 location、diagnostic、receipt 等 sidecar，但它们必须 proof-irrelevant：不进入 Replay，不改变 Gate verdict，不进入 production lowering。

---

## 3.2 J：唯一显式语义载体

\(J\) 可以表示为：

\[
J=(\Sigma,A,B),
\]

其中：

- \(\Sigma\)：逻辑签名，包括 tensor/scalar、输入输出、dtype、shape、destination、inout；
- \(A\)：operator、value relation、logical dataflow、reduction；
- \(B\)：domain、predicate、fill、typed index、alias、effect、conflict 与允许结果合同。

准确的 claim 是：

\[
\boxed{
J\text{ is the sole explicit semantic carrier in the artifact.}
}
\]

这表示 **artifact 的语法所有权**，不表示 Lift 产生的 J 自动正确。

为了避免把 artifact 声明与具体程序行为混在一起，定义规范投影：

\[
\boxed{
\operatorname{SpecProj}(J,C,\pi)=J.
}
\]

也可以将 J 的证书语义 operationally 定义为版本化 evaluator：

\[
\llbracket J\rrbracket_{v_J}(x)
:=
\operatorname{Eval}_{J}^{v_J}(J,x).
\]

其中 \(v_J\) 是系统预先声明的 J evaluator 版本。它不接收 C、π、Harness、Q 或测试结果；Q 只
比较 evaluator 已经产生的输出。这意味着证书相对于该版本的 J evaluator 成立；J evaluator 是
所有前端共享的可信基座，而不是每个前端各写一份。

---

## 3.3 C：实现选择对象

\[
C=(\Sigma_C,\nu_C),
\]

其中：

- \(\Sigma_C\)：匿名 slot schema；
- \(\nu_C\)：Harness 持有的具体值。

Replay 看到：

\[
\operatorname{MaskC}(C)=(\Sigma_C,\bot),
\]

即只看到 slot、类型与 binding kind，不看到具体值、源码名、位置或绑定关系。

C 可以声明：

- tile；
- warp/thread mapping；
- vector width；
- pipeline；
- prefetch；
- buffer 和 memory placement。

C 不能显式声明：

- operator；
- predicate/fill；
- logical shape/result relation；
- observable effect semantics。

设计目标是：C 属于 implementation selection。当前证书只验证实际绑定 \(c_0\) 的行为；对所有 admissible C 的 purity 是独立的多-C conformance 问题，不进入基础在线流程。

---

## 3.4 π：依赖型 realization witness

类型判断：

\[
\boxed{
\Gamma_{J,C}\vdash
\pi:\operatorname{RealizationWitness}(J,C)
}
\]

\(\pi\) 可以表达：

- regions；
- instances；
- routes 与 operand slots；
- 依附既有 route/C attachment/J domain 的 closed route-local physical extent transform；
- order；
- sharing / recomputation；
- placement；
- materialization；
- physical loop induction-basis relation；
- effect-control 中 terminal arm 与 lexical continuation arm 的角色关系；
- 受约束 representation refinement；
- C slot use-site。

\(\pi\) 不得显式声明：

- operator/reducer；
- predicate/fill/formula；
- semantic constant；
- logical result/effect；
- frontend API；
- source text/name/path；
- C value；
- 可执行自由文本。

为更准确地区分“声明”与“引用”，定义：

- \(\operatorname{SemDecl}(X)\)：X 在 closed-world artifact grammar 中直接创建的显式逻辑语义声明；
- \(\operatorname{SemRef}_{\mathrm{sem}}(X)\)：X 对既有逻辑语义对象的引用。

未知字段、未知 relation 和可执行自由文本必须 fail closed；否则 TypeCheck 不能支撑下面的
所有权结论。普通的 region、route、placement 或 C-slot 引用不属于
\(\operatorname{SemRef}_{\mathrm{sem}}\)。

TypeCheck 的核心静态性质是：

\[
\boxed{
\mathsf{TC}(J,C,\pi)
\Longrightarrow
\operatorname{SemDecl}(C)=
\operatorname{SemDecl}(\pi)=\varnothing
}
\]

以及：

\[
\boxed{
\operatorname{SemRef}_{\mathrm{sem}}(C)
\cup
\operatorname{SemRef}_{\mathrm{sem}}(\pi)
\subseteq
\operatorname{Ids}(J).
}
\]

因此得到语法级所有权结论：

\[
\boxed{
\mathsf{TC}(A)
\Longrightarrow
\operatorname{SemDecl}(A)=\operatorname{SemDecl}(J).
}
\]

这不等于：

\[
I(\pi;\operatorname{Sem}(S))=0.
\]

合法 topology 仍可能让 LLM 推测语义。系统硬保证的是无显式语义副本和无越权语义声明，不是消除所有统计线索或隐写可能。
`SpecProj(J,C,π)=J` 和上述 TypeCheck 结论也都只是 artifact grammar 上的所有权性质，不能推出
Replay 已正确实现 J，或所有 admissible C 都保持 J 行为。

其中 physical loop relation 的 closed 形态为
`{region,loop_instance,tile_instance,induction_basis}`，basis 仅取
`extent_step|tile_ordinal`。前者表示 `[0,Trip(J.loop))` 以 matching C tile 为 step，后者表示
`[0,ceil_div(Trip,Tile))` 的 unit-step ordinal；wire 不携带公式、concrete value、Source/API 或
operator。它只区分同一 J loop/C tile 下不同的 Source realization，不向 J 或 C 增加语义。

对 J 已显式作者化的 `effect_if`，π 还可用 closed
`{terminal_instance,continuation_instance}` 关系区分“当前 invocation 在该臂终止”与“另一臂承载
lexical continuation”。两项只能引用同一 `effect_if` 的两个 sibling arm-scope `realize` instance；
relation 不复制 predicate/polarity，不包含 return、value、formula、Source spelling 或 frontend API。
因此 control semantics 仍在 J，π 只补足 Source-realization 的 terminal CFG topology。

`artifact.v10` 还允许一项严格封闭的 `route_extent_transforms` relation：它只能把 required
positive-integer C slot 在一条已存在 route 的 target-local slot 上按 literal `divisor>1` 作
`floor|ceil` partition；source 必须锚定该 C slot，target 必须锚定该 slot 通过
`subject_ref/axis_ref` attached 的 J domain。它不创建 J dimension/domain/predicate/value/operator，
不修改 C raw valuation，也不传播到 sibling route；Lift 显式作者化，Replay 原样消费，最终
target-local topology 仍由 Gate1 裁决。SCRC 后 production 不读取 π。

---

# 4. TypeCheck 的准确职责

\[
\boxed{
\mathsf{TC}(J,C,\pi)=\operatorname{TypeCheck}(J,C,\pi)
}
\]

TypeCheck 不是 Gate0，也不证明 Source 与 J 等价。它只机械执行 artifact 合同。

## 4.1 J 检查

- node、value、tensor、access、effect 引用；
- operand arity 与类型；
- domain、predicate、fill、index relation；
- result/update relation；
- typed index、numeric conversion 等语义关系。

TypeCheck 还检查 closed comparison class 与 output dtype、A 依赖图和 B numeric contract 的相容性。
它不为 Lift 选择 relation，只执行版本化封闭判定 `RequiresTolerance_v1(J,output)`，具体 dependency
解析、transcendental 集、floating reduction 与 source-realization reassociation authorization 规则
见最终执行流程第 6.1 节。v1 中 `RequiresTolerance=true` 即拒绝显式或默认 `exact`，没有自由文本
或“更强合同”例外；阈值仍完全由外部 Q 决定。该 reassociation fact 只约束 SCRC comparison class，
不是 production backend permission。
当前 `typecheck.v30`增加participant mapping对exact J lane domain、route与matching C launch extent的
dependent check，并继承v29的program-axis completeness与v28的J dimension relation closed grammar、
引用、唯一性与acyclic检查；同时拒绝
同tensor多个不同却都无consumer锚点的node-less load views；不从valuation、Harness或compiler
observation生成relation，也不猜access consumer。v27只新增 route extent transform 的 existing-route、required positive-int C、
matching J-domain attachment/axis 与 realize-anchor dependent checks；checker 不生成 relation、
divisor 或 attachment。v26 继承 v25，只把 J direct `x* == 1`/`1 == x*` 作为 symbolic singleton
证据来消费同一 J 的显式 broadcast axes；缺失、非 1 或冲突 direct equality 拒绝。它不做
SMT/transitive inference，也不从 Harness valuation、C、π、Source 或 target shape补事实。
v25 继承 v24，并把 `reduction_linearization` 限定为一个 `realize` stage 经唯一
slot-0 route 进入一个 J reduce `realize` stage；它不创建 relation/reshape。v24 继承 v23，并新增
两条 closed dependent check：floating typed const 的 JSON
string 只允许 `inf|-inf|nan`；每个 transfer 必须从绑定同一 boundary tensor 的 pointer/tensor
signature carrier 经 authored π routes 可达。它不生成 constant、instance、route 或 address formula。
v23 在 v22 上新增 canonical output binding 的完整 dtype/shape exact dependent check；相同 concrete
extent、相同元素数或当前 singleton axis 都不能代替 J 显式表达 permutation。
v22 继承 v21；v21 继承 v20，v20 又继承 v18 将 `log2` 纳入同一 closed
transcendental 集的规则。v22 只新增 signature mode 与 J input/output/inout partition 的 exact
dependent check，不替 Lift 改写 mode。
floating `log2` output 仍须由 Lift 显式选择 `tolerance`，fixed checker 不代选 relation 或阈值。
v19-v21 的后续 dependent π、physical-time 与 terminal-control 检查也不改变该 comparison-class 合同。

v14 还检查 v4/v5 authored relations。`value_dimension_relations` 只允许 integral rank-0 J input 与
declared symbolic dimension 的 `eq`；`consumer_extensions` 必须闭合 predicate-bearing load 到
`sum|max|min` reduce/scan operand 的 provenance、local axis、dtype 与 two-sided identity。
`access_guard_relations` 必须锚定支配 access 的 typed `i1` Layer-A value，并以严格递增唯一 axes 和
closed polarity 声明；这些 axes 是同一 select guard 的唯一 broadcast authority。

guarded store 必须已经由 A 显式写出三 operand `select`，目标 tensor 精确同时是 J input/output 且由
唯一 `mode=inout` parameter 绑定。tensor handle 按 inputs 后 outputs 的首次声明连续编号；output
端复用同一 input `t*`，不新增 handle，且不使用物理参数的 `alias_of` 代替。loop 的 init/current/next 或 non-loop 的 tensor
pre-state/return/output mapping 都必须精确闭合；同 dtype/shape 的另一 input 不能替代 pre-state。
固定核不插入 select、不升级 inout、不选择 axes/polarity，也不从 Source/C/π/运行值修补 relation。

## 4.2 C 检查

- slot 匿名且唯一；
- binding kind 与 value type 合法；
- concrete value 来自 Harness 已有坐标；
- 当前 v2+ schema 的可选 `subject_ref` 属于 closed `CSubject_v1(J)`；可选 `axis_ref` 满足
  `axis_ref in AxisRefs_v1(J,subject_ref)`，且 `axis_ref => subject_ref`；
- 不包含 predicate、formula、logical result 等 J 语义。

这些 attachment 由 Lift 显式声明，固定代码只验证、不推导；它们只定位 prior，不构成
transformation permission，也不改变 `Reach_t(J)`。

## 4.3 π 检查

- 所有 ref 存在于 J/C 声明环境；
- instance、region、route、order、sharing 闭合；
- iteration region 为 loop placement，loop/tile instances 分别锚定同一 structured J loop 与 attached
  required positive-integer C tile；显式 relation 不要求该 slot 使用特定 role；
- route operand slot 与 J carrier 相容；
- `physical_iterations` 只以已有 J reduce/contraction 或 boundary access/domain 与同 subject/axis 的
  required positive-integer C slot声明物理 tile time，basis/direction 属于 closed vocabulary；
- `physical_loop_carriers` 只保存对应 region 内 typed `init/state/next/result` 时间拓扑，不含公式/value；
- `terminal_controls` 只连接同一 J `effect_if` 的两个 sibling arm instance，且该 `effect_if` 在当前
  lexical parent 中为 final；terminal/continuation role 不携带 predicate、polarity 或 return 语义；
- `route_extent_transforms` 只限定一条既有 route slot，并依赖 required positive-int C 与匹配的
  J-domain attachment/axis；divisor/rounding 属于 closed vocabulary，不形成一般表达式；
- `participant_mappings`只允许既有program-region lane-domain instance、物理axis、matching匿名C launch
  extent与`projection=cyclic`；logical period只来自exact J，不含公式/API/target recipe/C value；
- observable instance 锚定 J；
- 不新增 operator、predicate、fill、effect 或 semantic constant；
- sensitive representation refinement 有 J/B 授权；
- global-safe `reduction_linearization` 不带授权，只表示 Source 已有的全轴线性化，并经唯一 slot-0
  route 进入一个 J reduce；逻辑 axes/reducer 仍只在 J；
- 不含 source text、API、C value 或自由文本。

authorization 必须按 kind 做 dependent typing。对 `kind=numeric_cast`，`subject_ref` 必须解析到
真实 unary A cast node `n : Ts -> Tt`，`source_dtype/target_dtype` 必须同时存在并分别等于 operand
type `Ts` 与 result type `Tt`；不能把二者都与 node result carrier 比较。authorization 必须由 Lift
显式提出，固定核只检查其与 instance/refinement 的 kind、`auth_ref`、`semantic_ref` 和类型一致性。
每个 instance 至多一个 refinement。

同一 pair 可以因多个 declared physical regions 出现多条 iteration relations；parser 每个 region
最多一条。slot attachment 的 loop axis 可以用 canonical ordinal `0`，也可以用该 loop 显式
`trip.axis=x*`。`tile_extent` 与 `reduction_tile_extent` 是触发机械 missing-totality 的保留 role：
只有这些 role 对应的 attachment-closed J-loop/slot pair 在完全缺 relation 时由 TypeCheck 报
`iteration_witness_missing`；custom role 不使一个已经显式写出的 relation 非法。若 role 被误标且
relation 完全遗漏，或 Lift 完全遗漏 Source region、因而无 handle，fixed checker 都没有可量化对象；
该 completeness 由 source-aware Lift 提出、Gate1 observer 比较检查。这个边界必须保留，不能把局部
dependent checking 夸写成 Source topology 的形式证明。

首版不需要让 TypeCheck 成为 SMT 或完整图证明器。最小可行实现只需 fail closed 地完成上述局部检查。

当前合并协议为 `artifact.v13 / decode.v18 / typecheck.v30 / candidate-check.v1 / replay-check.v16`；
Replay packet envelope 为 `replay-input.v3`；Triton 公共上下文由当前固定公开字段直接给出，不再使用 context version、SHA 或 receipt 作为运行门禁；
Codex/MiniSWE Lift knowledge 为 v89/v67，MiniSWE Replay backend 为 v31，Codex Replay backend 为 v35。
Codex Replay v35在episode-local syntax-only Composer基础上消费v13 participant critical facts；Composer仍
不读取packet，只操作agent作者化源码，
支持stable node ID、局部编辑、source map与exact checkpoint/rollback；normal与Mask-J使用独立
writer/session/ workspace。Composer telemetry不进入artifact、TypeCheck、Gate或certificate，最终
接受集合仍只由exact Replay ingress、compile、Gate1与Gate2决定。
当前runtime把authored route extent transform与participant mapping纳入`critical_facts.v6`、
`replay-obligation-audit.v5`与source-blind static ingress；前者只检查closed floor/ceil-div spelling，
后者要求logical period从exact J解析而不能从C/reduction猜测，
不生成 relation/code，也不替代 Gate1；单独存在该 relation 时也可用可选 audit 并置当前 exact Replay bytes；audit 不作为 ReplayCheck、compile 或 Gate1 前置条件。
Codex v26 将每个 provider turn 的 900 秒上限与 fresh owner episode 的 1800 秒总预算分离，使超时后
只能沿原 resumable thread 继续；不跨 Replay episode 继承。v29/v25 历史 evidence 保持原 verdict。
v40 为 source-aware Lift worker 增加只读、分页的 Source control-flow
outline（function/loop/if/return 的行号与嵌套深度）；v41 只把 shared dialect/lowering 已支持的
`log2` 补入通用 `map_unary` vocabulary；v42 只要求 access `fill_value` 保留 Source 该 load 的实际
`other`，不允许用 downstream-neutral 等价值替换。Replay v4 只要求 source-blind writer 保留 J typed
rank/axis 与必要 compiler-visible reshape，不把 shaped reduction 压成 flattened reduction。这些版本
都不分类/补造 artifact 语义，也不生成 Replay；仍由 Lift 与 Replay LLM 分别完整作者化输出。
Replay workspace view v2 只把 packet 中已有的 typed reduction facts 无损并置；replay-check.v16 在
公共 source-blind synthetic compile 后核对 π 已声明 `transfer_direct` occurrence 的 compiled
load/store multiplicity，并核对 structured J loop 加 `pi.physical_iterations` 已声明的 compiled
`scf.for` occurrence，并核对对应 authored logical/physical carrier 的 block-argument arity；只在
signature carrier kind 与 π address routes 完全闭合时核对 `tt.addptr` occurrence；独立 integer
address contribution 可使用自身 semantic ref，address 与 transfer 也不必复用同一 ref spelling，但
每个 transfer 必须从匹配 boundary tensor 的 pointer/tensor signature carrier 可达，且每个
pointer-producing stage 必须流向某个 transfer；并要求 compiled distinct
`tt.reshape -> tt.reduce(axis=0)` dataflow 数不少于已作者化 `reduction_linearization` 数；并只在
authored route/carrier 与 Replay 自身 TTIR 都给出唯一同-region occurrence 对应时，核对
`pi.order` 已声明的 transfer 词法先后，歧义时不猜。它不重复
计算 `pi.iterations`，也不选择 loop bound、carrier、update、address formula、reshape、tile 或 spelling，不读取
Source/Harness value/Gate，不改变最终 SCRC 接受谓词。Codex Lift v5 只要求 LLM reconcile
syntax-only Source control-flow outline，固定代码仍不分类或补 artifact。单次 provider turn wall-time 后只允许同 invocation/episode 的 exact checked
bytes 或 resumable same thread 继续；禁止跨 formal round/Replay episode 继承候选。
Replay v30/v26 继承 v29/v25，并完整执行上述 route-local extent relation 的消费合同；Lift knowledge
保持 v68/v64，继续承担该 relation 的显式作者合同。Replay v28/v24 与 Lift knowledge v66/v62 明确同一 physical region 中 carrier relation 的 declaration
order 是跨 init/state/next/result 共享的 tuple ordinal；Lift 完整作者化，Replay 原样消费，fixed code
不按 dtype/ref/mechanism 重排。它不表示 event execution order，后者仍由 `pi.order` 拥有。
v28/v24 继承 v27/v23；v27/v23 继承 v26/v22，并把每条 J-authored broadcast 的 operand closed type、matching access、
named consumer/ordinal 作 exact source-blind 重索引；存在该 obligation 时，owner 必须把 ledger 与
自己的当前编号源码并置自审。工具不建立 relation-to-statement 映射、不选择 rank-lift spelling/次数、
不生成或修复 Replay，也不读取 Source/Gate。v26/v22 已固定：J.signature 的一个 `p*` 是一个既有 callable ABI carrier，同一 `p*` 的
多个 π instance 只定位或路由多处 use，不能产生新 carrier 或 `program_id`。source-blind construction
compile 从 program region 中沿作者化 route 到达 `address_base` 的 distinct J domain 投影合法
`tt.get_program_id` occurrence，并在 v12 按 `pi.program_axes{domain_ref,axis}` 核对精确 launch axis；
它不生成 coordinate、不从声明顺序/grid/C选择 axis，也不参与 Gate/认证。
v3 历史上引入 `pi.iterations`；v4 引入 J-owned `value_dimension_relations`、
`consumer_extensions` 等 typed relations；v5 引入 J-owned `access_guard_relations`；v6/v7 分别引入
typed `effect_if/value_if`；v8 增加 π-owned `physical_iterations/physical_loop_carriers`；v9 增加
π-owned `terminal_controls`，只为 J 已存在的 `effect_if` 指定 terminal/continuation arm topology；
v10 增加依附既有 C/J/route 的 closed `route_extent_transforms`，v11增加J-owned closed
`dimension_relations`。后者只含identity、常数加/乘、常数ceil-div与round-up的acyclic grammar，
consumer只能验证/求值，不从Harness/grid/C/π/TTIR生成。v12增加π-owned closed
`program_axes`，只把Source-visible program-coordinate domain绑定到axis `0..2`，不携带grid extent、
地址公式或前端文本。v13增加closed `participant_mappings`，只将公开物理participant axis循环投影到
exact J lane domain；不携带period/formula/API/target recipe/C value。每个当前字段都由Lift LLM 作为完整 `(J,C,π)` 的一部分显式作者，
Decode/TypeCheck/CandidateCheck 只解析、检查、拒绝
和诊断，不补造 relation 或 artifact。v1-v7 保持各自 wire bytes/digest 与字段集合，不得静默升级、
刷新或追认；v8/v9/v10 同样保持各自字段集合。SCRC 后 π 停止，
production backend 仍只消费 J+C；iteration witness 不进入 `Impl(J)` search。

## 4.4 持久 Lift owner 内的非认证 CandidateCheck

同一个 branch-local source-aware Lift owner 可以对自己已经产生的完整 candidate 调用只读的：

```text
ctx = (anonymous_compiler_view, schema_version, v_D, v_TC)
CandidateCheck_v(candidate_bytes, ctx)
```

提前获得机械 diagnostics。CandidateCheck 共享正式 Runner 的同一 source-free
parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行 source-identifier privacy、
receipt/provenance，并在提交后重新运行该 core。禁止维护第二套近似 checker。该调用：

- 只读取显式 `candidate_bytes,ctx`，禁止读取 ambient host state；
- 可以构造 ephemeral validation view，但不修改、补全或返回候选/artifact；
- 不生成 J/C 分类、authorization、result relation 或 π topology；
- 不读取 Replay、Gate、RefEval 或 compiler observation；
- 只返回 closed `protocol/status/rule_code/path/observed_refs_or_types`，不返回建议值或 patch；
- 不签发证书，也不替代提交后的正式 TypeCheck；
- 只作为 proof-irrelevant Lift audit 记录。

若当前完整 candidate 的精确 bytes 在本 session 尚无 `PASS`，提交动作先自动调用同一
CandidateCheck。FAIL 只把 closed diagnostics 返回 worker，不提交、修补或 canonicalize；只有与
某次 `PASS` 对应的精确 bytes 完全相同时才可复用，任何不同 bytes 都必须重检。该前置调用与显式
调用共享同一 diagnostics path，PASS 仍不进入证书，正式 Runner 仍
重跑 Decode/TypeCheck。

当前 active runtime 不设置低的 model-turn 或 CandidateCheck count capability cutoff：`step_limit=0`
与 check limit 0 表示 time-governed。一个 branch-local Lift owner 默认拥有 3600 秒累计 active wall
time；每 action 的 CPU/output/workspace 限制、provider transport/cost safety 与外部显式有限 `K` formal
round 预算仍然有效。receipt 记录每次 `explicit`/`submission` calls/evaluations、model/physical calls、
active elapsed 与 terminal reason。若一次 pause 前未提交候选，同一 owner 的 continuation 最多接收
最后一次实际 source-free evaluation 的 20 条 `rule_code/path` 摘要；它保留自己的 Source
conversation 与受限 workspace，host 原子刷新 feedback 而不覆盖当前完整 draft；没有 observed hidden
value、fixed semantic patch 或 sibling 信息。单次 provider request 上限为 900 秒，timeout 不连续重试，
但 transient transport failure 仍有有界恢复。旧大型 tool argument/observation 可在 API packet 中
确定性替换为 hash/长度 receipt，完整 trajectory 与 workspace 文件保留。LLM 可用 exact-text patch
原子应用自己明确给出的唯一旧/新字符串；runtime 不解析 JSON 语义、不推断字段。首个 user packet
及 Gate1 continuation 默认携带 exact Source/Harness/current candidate 与 deterministic bounded
`feedback_frontier`；完整 closed feedback 保留 bytes/hash/path，并按命名 category/path 查询。frontier
只重排 Gate1 已有 category、计数、direct examples/local frontier/effect-order witnesses，删除 bulk
recursive node fallout；它不解释 Source、不判定根因、不推荐字段或 patch。该诊断投影与
submission `PASS` 都不进入证书，也不授权固定代码修补 `(J,C,π)`。

Harness 的 concrete tensor shape 只是当前 valuation；source-grounded `shape_contract h*` 必须由 Lift
双射为 J `x*`。普通 integral scalar 的 `shape_coordinate` 与 adapter 显式声明且 valuation-matched 的
`bound_constexpr_shape_coordinates` 使用同一规则：extent-only carrier 可直接绑定 `x*`，同时承载
logical data 时必须由 Lift 写 rank-0 `t*=x*` relation。固定 Harness 只核对已声明 identity、valuation
和 authored mapping，不从相等数值、grid、参数名或 Source 自动建立 relation。

LLM 还可以在单个 direct-submit action 中给出完整 candidate text，或对现有文件给出唯一 old/new
text 并请求替换后提交；runtime 只原样落盘/替换并调用同一 submission preflight。这不压缩或重写
artifact 语义，不增加 acceptance path，正式 Runner 仍独立执行 Decode/TypeCheck。

因此它改变的是 finding a well-typed witness 的工程效率，不改变：

\[
\mathsf{SCRC\_CERTIFIED}
=
\mathsf{TypeCheck}\land\mathsf{Gate1}\land\mathsf{Gate2}.
\]

扩充 TypeCheck judgement 必须提升 `v_TC`，CandidateCheck 与正式 Runner 必须记录并使用同一版本；
新版本不静默重判历史证书。

当前 fresh-run 取 `v_TC = typecheck.v30`。v30增加participant mapping dependent check；v29增加program-axis completeness dependent check；v28增加上述dimension relation dependent check；v27增加route extent transform dependent check；
v26 增加上述 direct singleton broadcast dependent check；
v25 增加上述 reduction-linearization topology check；v24 增加上述 special-constant 与 transfer pointer-route
closed checks，并完整继承 v23；v23 增加 canonical output binding 的完整 dtype/shape equality；v22 增加 signature mode 与 J boundary partition 的 exact
dependent check，并完整继承 v21；v21 继承 v20/v18/v19、v14-v17 typed-control 合同及下述 v7
前序合同，并包含上述 v9 terminal-control dependent checks；v20 已增加上述 v4/v5
relation 的 source-free dependent checks；仍不赋予 fixed core artifact 构造权。历史 v7 完整继承 v6；
v6 继承 v5，v5 继承 v4 的 rank-0 closed signature、
Layer-A graph total typing 与显式 non-trailing NumPy broadcast 合同：closed elaboration 只物化唯一确定的
rank-0/普通 result type，缺失的 broadcast axis relation 必须拒绝，不能由 checker 猜测。
v5 又在 RefEval/emitter 前对已知 `reduce` 与 binary `contraction` 执行 canonical fail-early，
拒绝 `kind`/contract-axis 别名、缺失或非法的 canonical attrs 以及与已有 typed J 矛盾的
axis/rank/shape；fixed code 不翻译、补齐或修复这些字段。同一 v5 judgement 还拒绝将
`opaque(mode=value,dtype=none)` 这一 absent ABI carrier 用作 π instance 的 semantic ref、
realization authorization subject 或 refinement carrier；它没有 logical denotation。v6 对 J 自身
可识别的零起点矩形 functional logical-domain profile 增加同源必要条件：direct access index、padding
predicate、`boundary.domains` 与 typed tensor extent 必须显式闭合，CandidateCheck/TypeCheck 不从
Source、C、π、shape、runtime/hidden input 补造 relation。general affine/交叉轴 logical predicate
不因当前 RefEval 未实现而被一概拒绝，合法 predicate/fill/effect 也不被删除或改写。v7 对 rank-0
signature logical input 做窄的 J-graph liveness：它必须到达 observable output dependency，或被显式
boundary/shape semantics 使用；identity/direct-output 与 boundary-only 合法使用不丢失，dead node 和
π-only route 不计。representation/address-only runtime scalar 由 Lift 显式声明为
`kind=scalar,mode=value,rank=0,logical_ref=null`，不创建 J input、只可作为 π carrier；fixed checker
不看 Source 名称/位置、不自动解绑，也不把规则推广成未证明的一般 tensor liveness。

这个 representation-only null-carrier 例外仅限 runtime scalar。对 `bound_constexpr_values`，Lift 必须按
每个参数在 Source 中的全部 use 分类：logical data/predicate/result 使用 dtype-compatible
rank-0 J input，integral extent-only 使用 declared `x*`。`constexpr` 本身不证明 graph-selector
身份；只有所有语义 use 都严格限于 compile-time control/discrimination，且 concrete value 不进入
active graph 的 shape/index/mask/arithmetic/boolean/result，才可保留
`kind=constexpr,mode=value,rank=0` 的匿名 p* callable carrier、compatible public dtype 并省略
`logical_ref`。参与 active graph 的 constexpr 即使 constant-fold 也必须走 t*/x*；同时是 data 与
精确 dimension 时，使用 t* 与 Lift 作者化的 `value_dimension_relations`。纯 selector 的 concrete
value 只留在 \(\rho_H^{\mathrm{static}}\)，不写入 J/C/π。Lift 必须自己作者化完整 active Layer-A graph，
不生成 dummy/identity dataflow、inactive op 或 runtime `effect_if/value_if`。fixed code 只校验已作者化的
wire/type 形式，不代替 Lift 分类，不根据值选图、补图、清空 `logical_ref` 或伪造 liveness；Replay 保留 exact constexpr ABI，source-blind
ReplayCheck.v4 拒绝 kernel body 对这种 carrier 的任何读取，并检查 authored terminal CFG。Gate1 由 Harness 绑定本次值，RefEval 与
production backend 均不消费该值。
只影响 implementation selection 的 constexpr 必须进入 C。representation-only bound constexpr
不能伪装成 π-only realization carrier；上述 static specialization carrier 也不得进入 π topology。

v7 还独立检查 `padded_mean_count_unproven`，但只在一个窄 closed profile 内：predicated direct load
不含 logical guard；index/domain 是完整零起点矩形；predicate upper bound 与对应 typed tensor extent
相等；同一 masked axis 经已建模的 closed pointwise/broadcast 位置传播后被 `reduce(mean)` 消去。
此时当前 J 的中性 `masked_contribution` 只证明分子，不能证明 logical valid count 作为除数，故在
Replay 前拒绝。dense/unpredicated mean、guard-bearing 或 cross-axis/general predicate、domain/extent
未闭合、mask 位于该 mean 未消去轴、或未知 op 阻断传播证明时不触发 TypeCheck；这些未闭合
consumer topology 由 functional consumer 明确 unsupported。该判定只读 J，不读取 Source/C/π，也不
将 mean 自动改写；若 Source 是 masked sum 后显式除法，Lift 必须完整生成
`reduce(sum) + explicit div`。

Scalar RefEval 当前能力版本为 `j_scalar_cpu_mlir_v24`，shared projection 为
`intentir.j-functional-boundary.v22`，typed relation-indexed reference 为
`intentir.j-contract-indexed-cpu-reference.v2`。它们完整继承 v14/v13、v12/v11 与 v10/v3；v10 已继承 v9，v9 已继承 v8 的唯一 node-less
logical-input load，并把 padded reduction 从 single input/output/sum 特例改为 J-only 的通用闭合：
每个 predicated direct access 必须
覆盖自身完整 logical domain，仍携带 masked axis 的 access 使用同一 predicate；每个 logical reduce
恰有一个中性 `masked_contribution` contract；pointwise 的 trailing/显式 non-trailing broadcast 必须
由 closed type/J relation 决定。满足这些条件的 max → pointwise → sum 等多 reduction DAG 直接交给
既有 Intent MLIR scalar executor，也没有 Softmax/Kernel 分支。

v10/v3 只为 pre-Gate2 规则已经选定的 winner，在当前 Gate2 invocation 内增加 runtime predicate
tautology proof。主 SCRC runner 负责 winner 选择并把该次 inputs/dimensions 交给 RefEval-only 入口；
projection library 不自行选择 winner。该入口要求精确 J logical-input mapping 与完整 closed
dimensions，只消费 J 自身 rank-0 logical `i1`/integer values，逐 access 按真实 guard 极性、direct
axis、零下界及 runtime bound 不小于
domain extent 证明 predicate 在完整 logical domain 恒真。不修改 J、不进入 Lift feedback、不读取
Source/C/π。v3 projection 内的 invocation-local proof subreceipt 只含 access id、J predicate digest、
logical input refs 与 domain axes，不保存 raw value/value digest；这是 audit-data minimization，不是
confidentiality/privacy proof。production projection 仍是没有 runtime-value 参数的 J-only 分析。
上述窄
rectangular provenance 可证明 masked axis 被 `mean` 消去却没有 logical count contract，以及缺
domain/fill/reduction/broadcast 等继续 unsupported。所有既有 typecheck.v3/v4/v5/v6、
`j_scalar_cpu_mlir_v7/v8/v9`、functional-boundary v1/v2 evidence 以及 Softmax/RMS 历史运行保持原
verdict，新协议只用于 fresh run。无法完成上述 proof 的 guard-bearing access 仍由 RefEval consumer
fail closed；已有 masked provenance 进入 contraction/其他未建模 op 时也必须在该 op 处拒绝，不能靠
operand membership 或静默停止传播来证明 mask erasure。两者是 consumer 能力边界，不新增
TypeCheck 非法 J 条件。CPU host 只在共享 Intent-to-Linalg 之后调度 `rsqrt -> 1/sqrt` 的标准 libm
ABI compatibility pass，不改变共享/GPU lowering。

v11/v9 进一步要求 RefEval 对每个 v4/v5 relation 消费或明确 unsupported：检查 value/dimension
equality，应用 consumer-specific outside identity，并只对 A 已显式 select、精确 inout/pre-state 的
guarded store 形成 access-guard receipt；guarded load/update 当前无 access-local pre-execution
materializer。v9 可证明同 producer、同完整 observable typed value、单静态 axis、无 predicate、
常量 unit-width 且无重叠无缺口的 direct-store partition；v11 的 closed structured J-only reference
可执行 common scalar lowering 尚未支持的 mixed typed binary contraction。后两者只扩展通用 Gate2
reference consumer，不是 production recipe、pass permission 或 Kernel 特判。production J+C 当前对
任何 access guard 仍显式 fail closed。

v12/v11 进一步消费显式 logical inout：同一 `t*` 以相同类型同时出现在 inputs/outputs，由唯一
`mode=inout` parameter 绑定，且 J 已完整声明 pre-state load、post-state store、effect 与
canonical output binding。Gate2 用 input binding 执行旧值语义、用 output binding 比较新值；物理
`alias_of`/`boundary.aliases` 仍 unsupported，RefEval 不补 merge、不复制 J 字段。

v13/v12 只把 J 已证明的 physical inout post-state reload 投影回既有 post-state SSA：唯一 mapping、
唯一 producer store、同 tensor/index/predicate、后继 dataflow 与 boundary/scoped store-before-load effect
order 必须同时成立。v14/v13 只在 missing-fill load 的所有 downstream 都是
`map_unary/map_binary/select/cast`，且每个 observable store 都由同一 predicate 支配时证明 physical
outside-domain value 不可观察；reduction/scan/loop/transpose/pad、不同 predicate 与未知 effect 继续
fail closed。两者都不读 Source/C/π/Harness，也不修改 J。

历史 `j_scalar_cpu_mlir_v20` / `intentir.j-functional-boundary.v19` 完整继承 v17/v16，并继续只扩展 deterministic
consumer：显式 product domain 只有在 access coordinates 构成完整、唯一、逐 factor 的 mixed-radix
bijection 时才投影为 logical tensor axes；missing-fill provenance 只有在 shape-preserving pointwise
链后接精确 `transpose`，且 store coordinates 等于 load coordinates 的 authored permutation、predicate
完全一致时才允许穿过 transpose。它还消费 J 已写出的 `map_unary(log2)` 与 JSON-safe IEEE special
constant。所有 proof 都只读 J、产生 receipt；不做一般代数化简、不新增字段、不改变 TypeCheck、
不按 kernel 名称分支，失败仍显式 unsupported。

v18/v17 再允许唯一 node-less whole-input load 与 missing-fill pointwise proof 组合：同 tensor 必须只有
该 load，canonical J 必须只有一个直接 pointwise consumer，之后仍逐路径验证 typed broadcast、同
predicate observable store、effect 与 output binding。该确定性分析只解析 J 已有 edge，不回填
artifact；歧义、非 pointwise consumer 或任一未闭合路径继续拒绝，因此不改变 Lift codomain、
TypeCheck 或 SCRC 认证公式。

v19/v18 进一步消费同一 logical inout 上多个 closed physical store→reload stage：每一对都必须由
唯一 producer SSA、相同 coordinates/predicate、同 scope 的 store-before-load effect order 与无
intervening write 共同决定；只有最终 canonical output binding 对应 store 是 observable output。
未被 logical access/predicate 使用的矩形 J domain 作为 `context_domain` 保留在 projection receipt
中，允许 π 锚定 compiler-visible program context，但不参与 Layer-A 数值执行或生成
access/program formula。Gate2 Triton Source/Replay 两条 leg 共享
`triton_static_dot_transpose_v1`，只把已移除的静态 `trans_a/trans_b` 等价归一成 `tl.trans` 并记录
receipt，不按 kernel 名改写。

v20/v19/v8 只修正已有 J/B 的确定性消费。predicated load 的 padding provenance 现在绑定到其
`(node_ref,operand)` edge；同 tensor 上另一个 J `gather` 只有在匹配 `index_bounds` 已作者化时才作为
独立点访问。`effect_if` arm 内的单结果 full store 则由现有 access/effect destination 形成临时
`authored_effect_if_full_store` receipt，并仅在该 arm 实际执行时更新 output state。它不把
`effect_if` 改成 value control，不补另一 arm 的值或 pre-state；未定义输出继续 fail closed，故不
改变 J、B、π 所有权或 SCRC 谓词。
其 invocation input proof 还可在 current closed dimensions 上解释 `iota(n=x*)`，并仅对 divisor
interval 不含零的 signed integer `div` 按 toward-zero 语义传播区间；每步仍检查 finite-width
overflow。动态 gather 的 index shape 必须唯一绑定 logical domain axes。node-less padded load 只有
已有 consumer extension 才可与同 tensor 的显式 reload 共存，后者在自己的 operand edge 上覆盖
前者 provenance；这仍是 J/B consumer closure，不是从 Harness valuation 发明 shape identity。

v21/v20 继续只扩展 deterministic J/B consumer：J direct `x* == 1`/`1 == x*` 是 constant
coordinate 0 的唯一 symbolic singleton 证据；同一 `effect_if` 的正负 sibling arm 若分别完整写同一
output，则既有 conditional-store receipts 证明控制分区，无需伪造共同 producer。它们均不读
Source/Gate、不补 J relation。

v22 再增加 `intentir.j-contract-indexed-cpu-reference.v1`：当旧 functional projection 无法把一般
affine/sub-domain B 擦成 full-tensor 执行时，只对 flat、acyclic、shape-preserving pointwise A graph
从 canonical ContractIR 执行 typed index/predicate/fill 和穷尽、无重叠的 output stores。access 的
`node_ref` 是 consumer-local；sibling node 没有自己的 access 时保留普通 Layer-A operand 语义，
fixed consumer 不复制 relation、也不按 Kernel 名识别 half/split/rotation。于是 Gate1 已闭合但 J
漏写 consumer-local access 时，Gate2 会形成真实 Source--J/Replay--J 数值分歧，后续必须 fresh
relift，不能把 RefEval capability repair 伪装成 J repair。该 reference receipt 不自动赋予 production
target indexed-store lowering permission。

函数 `returns` 或 boundary `output_mapping` 任一均可唯一声明 canonical output binding，同时出现必须
一致。shared J→Intent MLIR 入口及 structured reference v9 消费 deterministic `JElaboration.output_bindings`，所以 Gate2 与
production materialization 使用同一个 J-owned destination；不要求 Lift 复制相同字段，也不从物理
地址反推结果绑定。
Lift/schema/preflight 同步为 `intentir.lift-knowledge.v21`；它仅取消同一 binding 的冗余双写，仍由
LLM 完整选择 `returns`/`output_mapping` 中至少一个非 identity authored relation，checker 不补值。
`intentir.lift-knowledge.v22` 再把已有 B grammar 的闭合义务写清：每个 reduce/inclusive-scan 拥有
local-axis algebra entry；predicate-bearing load 的 outside lanes 到达 sum/max/min consumer 时，逐
consumer/operand/access author typed `consumer_extensions`。fill、reduction identity、C/π 或 runtime
巧合都不构成替代证明。

---

# 5. Source-isolated Replay

Replay 的唯一 Kernel-conditioned 输入为：

\[
\boxed{
\operatorname{View}_f(A)
=
\bigl(J,\pi,\operatorname{MaskC}(C),U_f\bigr).
}
\]

实际生成：

\[
\boxed{
R=
\operatorname{Replay}_{\theta,f}
\bigl(\operatorname{View}_f(A)\bigr).
}
\]

Replay 不接收：

- Source；
- Harness；
- C 具体值；
- source parameter name/position 或真实 BindingMap；
- 独立 interface sidecar；
- Gate1 diff；
- source/compiler fingerprint；
- 旧 Replay；
- 旧候选代码；
- 旧模型 conversation context；
- per-kernel RuleCard 或其他 source-conditioned frontend context；
- 任何额外 source-conditioned sidecar。

\(U_f\) 必须是版本化、source-neutral 的公共前端知识，不能根据当前 Kernel、operator 或失败样本
选择内容。runner 可以在 Replay 之外保存 request/output、compiler fingerprint 与普通实验日志；
这些记录不得回流 Replay request，也不需要 hash、commitment 或签名门。

这里认证的是 **request-level source isolation**，而不是试图证明模型内部在统计意义上完全不依赖训练先验。

无需建设复杂 hash 或签名系统；只需 runner 实际构造的 Replay request 严格来自上述 View，并保存普通实验日志即可。

---

# 6. Harness 的职责

Harness 是隐藏求值环境：

\[
\rho_H:
\{\text{logical input},\text{C slot}\}
\rightarrow
\{\text{concrete value}\}.
\]

Replay 生成完成后，Harness 将同一份 \(\nu_C\) 分别绑定到 Source 和 Replay：

\[
\operatorname{Bind}_H(S,C),
\qquad
\operatorname{Bind}_H(R,C).
\]

Harness 可以：

- 枚举匿名参数和 compiler slots；
- 保存实际输入和 C values；
- 建立 Source/Replay compile/run invocation；
- 运行 Gate1 和 Gate2。

Harness 不可以：

- 判断事实属于 J 或 C；
- 生成 loop、mask、address、reduction、sharing；
- 修改 Replay AST；
- 在生成后新增 C use-site；
- 产生第四个 source-conditioned artifact。

J reference 的接口必须是：

\[
\operatorname{Eval}_J:J\times\text{runtime inputs}\rightarrow\text{outputs},
\]

接口中没有 C 或 \(\pi\)。

---

# 7. Gate1：observer-relative source-blind replay closure

## 7.1 Frontend observer

定义：

\[
\Phi_{f,p}(P,c_0)
=
\operatorname{Canon}_{f,p}
\left(
\operatorname{Observe}_{f,p}
\left(
\operatorname{Compile}_f(P,c_0)
\right)
\right).
\]

其中：

- Compile：前端已有编译器；
- Observe：读取当前前端可提供的稳定 compiler artifact；
- Canon：一次实验开始前由协议确定的全局规范化。

Gate1 不需要所有前端 lower 到一个公共 IR。每个前端只需提供自己的 \(\Phi_{f,p}\)，证书强度
显式参数化于 observer profile。

## 7.2 Gate1 公式

\[
\boxed{
G_1^{f,p}(S,R,c_0)
=
\mathbf 1
\left[
\Phi_{f,p}(S,c_0)=\Phi_{f,p}(R,c_0)
\right].
}
\]

等价地定义 observer-relative 类：

\[
P_1\sim_{\Phi_{f,p},c_0}P_2
\iff
\Phi_{f,p}(P_1,c_0)=\Phi_{f,p}(P_2,c_0).
\]

Gate1 通过意味着：

\[
\boxed{
[S]_{\Phi_{f,p},c_0}=[R]_{\Phi_{f,p},c_0}.
}
\]

即本次实际 Replay 在 Source 隔离条件下，被构造成与 Source 落在同一 frontend-observer 等价类中。

## 7.3 Gate1 支撑什么

Gate1 支撑：

- artifact 是否足以重建 compiler-visible realization；
- typed artifact 是否足以承载 observer 所要求的 source-specific realization 信息；若要隔离
  \(\pi\) 的独立贡献，必须另做消融，不能由一次 Gate1 closure 单独推出；
- LLM 是否能够直接生成目标前端代码，而无需 J→frontend lowerer；
- operator/dataflow/predicate/effect/representation 等 observer 能看到的差异是否闭合。

Gate1 不支撑：

- 所有输入上的语义等价；
- observer 未观察部分的相等；
- LLM 内部究竟“相信”J 还是根据 topology 猜测；
- TypeCheck 自身的完备性。

## 7.4 Observer 强度

若 \(\Phi_2\) 比 \(\Phi_1\) 区分更细：

\[
\Phi_2\succeq\Phi_1
\iff
\ker(\Phi_2)\subseteq\ker(\Phi_1),
\]

则：

\[
\Phi_2\succeq\Phi_1
\land
G_1^{\Phi_2}=1
\Longrightarrow
G_1^{\Phi_1}=1.
\]

这解释了多前端证书强度不同而协议相同：Triton TTIR observer 可以强，CUDA observer 可以较粗；无需给 CUDA 再建完整形式语义。

每份证书的外部记录至少要包含 `frontend`、observer profile/version、compiler/dump stage 与
canonicalization version。mutation test 可以评估 observer 的经验敏感性，但不构成 observer
完备性定理，也不进入主认证谓词。

当前 Triton active profile 是 `triton-ttir-3.6-gate1-v12`。除closed cache/eviction spelling外，v12
进一步按唯一所有权作compiler图切片：完整保留observable value slice、predicate/boolean树、直接
常量、reduction/control/effect、region/loop/carrier及program/pointer/transfer anchors；只将不进入
上述切片的known-pure physical scheduling/address/rank/bound公式收缩为typed、carrier-origin-preserving
edge。该quotient不改变`G1=[F_source==F_replay]`，而是定义当前`F`不观察J/C/π均不拥有的具体公式
拼写；错误语义、predicate/fill、reducer、effect、carrier/route/stage数量仍导致不等。
它将 closed cache modifier 与 closed
eviction spelling 统一视为不由 J/C/π 拥有的 frontend presentation noise；π 仍严格拥有 distinct
transfer instance 的数量、顺序与 route，Replay 只可用 public spelling 防止 compiler CSE。它保留
既有对已建模 op 的 closed load/store policy 规则，以及从 public entry 可达、静态已知、structured 且无环 `tt.call` 的
interprocedural comparison：call graph 仍 interprocedurally 展开，formal-to-actual、helper return
SSA、op/def-use、effect 与已支持
region 都进入同一 observation。

v6 新增 typed、single-block、encodingless quoted-generic `tt.scan`；ranked tensor shape、axis/reverse、
ordered combine args、唯一 typed return 与 recurrence/region topology 都是 hard evidence。全局 closed
operation/effect classification 禁止 unknown op 在 root、`scf.for` 或 combiner 中默认 pure；malformed
generic/body line、未知/动态 callee、递归/cyclic call、signature/return 不一致、reachable arbitrary
multi-block CFG/region 与未支持 control/region 一律 fail closed。若 Source 侧因此无法形成
`F_source`，`intentir.scrc.batch.v3` 报 `observer_unsupported`；它表示 observer 能力缺口，不是候选
非法、Gate1 mismatch 或 `NO_CLOSURE`。Replay 侧失败不能冒充 Source observer 缺口；历史 v5 evidence
与 batch v2 report 均保持原版本，不由 v6/v3 重判。这不声称 v5→v6 acceptance 单调。上述 profile
只是当前实现与测试覆盖的 compiler contract，不是 TTIR observer 完备性的形式证明。

---

# 8. Gate2：Source-anchored behavioral and semantic validation

## 8.1 外部比较政策 Q

比较政策 \(Q\) 必须由实验协议或 benchmark correctness policy 预先确定，而不能由 Lift 任意放宽。

Q 可以包含：

- exact；
- 固定 atol/rtol/ULP；
- dtype-specific policy；
- output/effect comparator。

J/B 可以声明所需的语义类别与合同，但不能自由产生一个任意宽松的 comparator。形式上，Q 是
受信任验证协议中从已验证 comparison class、dtype、extent 和合同参数构造比较器的确定性政策；
它不是 Lift artifact，也不进入 Replay。用于“两两闭合”时，\(\operatorname{Close}_Q\) 必须由
协议定义为对称谓词；若底层 `isclose` 有方向性，必须显式对称化或规定双向检查。

## 8.2 三个实际输出

对 \(x\in T\)：

\[
O_S(x)=\operatorname{RunObs}_f(S,c_0,x),
\]

\[
O_R(x)=\operatorname{RunObs}_f(R,c_0,x),
\]

\[
O_J(x)=\operatorname{Eval}_{J}^{v_J}(J,x).
\]

Source 与 Replay 使用相同实际 C 绑定；J evaluator 不读取 C 或 \(\pi\)。

## 8.3 当前主线：确定性或 contract-deterministic Kernel

当前证书域明确为：

\[
\mathcal K_{\mathrm{main}}
=
\{\text{deterministic or contract-deterministic kernels}\}.
\]

这里的 contract-deterministic 是协议可观察层面的性质：对每个被测输入，合法执行由 J/B 合同
给出唯一的外部观察要求。内部线程顺序可以变化；但不能仅凭一次运行稳定就把真正集合值的行为
归入该类。

定义由 Q 确定的近似比较：

\[
\operatorname{Close}_Q(o_1,o_2)\in\{0,1\}.
\]

Gate2 使用三条直接比较：

\[
\boxed{
\begin{aligned}
G_2^{\eta}(S,R,J,c_0,T)
=
\bigwedge_{x\in T}
\Big[&
\operatorname{Close}_Q(O_S(x),O_R(x))
\\
&\land
\operatorname{Close}_Q(O_S(x),O_J(x))
\\
&\land
\operatorname{Close}_Q(O_R(x),O_J(x))
\Big].
\end{aligned}
}
\]

第三条 Replay–J 比较不增加一次执行，因为三个输出本来已经产生，只多一次确定性比较；它避免近似关系不满足传递性时，两个 Source-centered 边都通过但 Replay 与 J 彼此偏离。

三条边分别回答：

1. Source–Replay：Replay 的实际行为是否保持 Source；
2. Source–J：Lift 得到的 J 是否与 Source 在测试域内一致；
3. Replay–J：Replay 是否同时符合 J 的可执行语义。

当 exact parent `(J,C,π,Replay)` 已有 `replay_closed` evidence，而失败被证明属于通用 RefEval
capability，可以在 consumer 修复后以零模型调用重跑唯一 Gate2。该
`gate2-capability-recheck.v1` 必须核对 artifact/Replay digest 与 case/source lineage，复用同一 bytes，
写入独立 evidence，并记录 `changes_parent_verdict=false`。它不 relift、不重新 Gate1、不刷新旧
receipt；新结果只是在记录的当前 evaluator/Q/test/environment 下对 exact artifact 的独立 evidence。

## 8.4 真正集合值的非确定性语义

“非确定性时使用 conformance”不是 claim 降级，而是集合值规范下正确的实现关系。

- 若内部调度不确定，但可观察结果唯一，仍属于当前 deterministic / contract-deterministic 主线；
- 只有同一输入确实允许多个不同可观察结果时，J 的语义才是集合值。

未来统一扩展可以定义：

\[
\operatorname{Accept}_{J,Q}(x,o),
\]

表示输出 \(o\) 满足 J 的允许结果合同；然后分别要求 Source 与 Replay 的本次观察结果都被接受。
是否还要求二者直接匹配，取决于未来集合值协议，而不是当前三边谓词的自动延伸。

该接口首先只支持“本次观察结果满足合同”的 per-observation conformance；有限次 membership
检查本身不能推出完整的
\(\operatorname{Beh}(P,x)\subseteq\operatorname{Spec}_J(x)\)。对于两个彼此可能不接近、但都
合法的集合值结果，未来一般化应分别检查 membership，而不能继续强制三条 `Close_Q`。

当前若 benchmark 和 evaluator 面向单值或 contract-deterministic Kernel，不需要为了理论最大一般性立即实现集合语义 checker、并发调度模型或 SMT membership checker。

---

# 9. 两个正交闭合

## 9.1 Factorized Intensional Closure（FIC）

以下定义的论域已经要求
\(R=\operatorname{Replay}_{\theta,f}(J,\pi,\operatorname{MaskC}(C),U_f)\)，即 R 确实由第 5 节
规定的 source-isolated View 生成；request isolation 是运行前提，不是第四个 Gate。

\[
\boxed{
\operatorname{FIC}_{\eta}
=
\mathsf{TypeCheck}(J,C,\pi)
\land
G_1.
}
\]

FIC 表示：

1. artifact 的显式语义所有权合法；
2. Replay request 不包含第四个 Source 通道；
3. typed artifact 能构造出一个与 Source 落在同一 observer class 的 Replay。

## 9.2 Extensional Semantic Closure（ESC）

\[
\boxed{
\operatorname{ESC}_{\eta}=G_2.
}
\]

ESC 表示 Source、Replay 和 J 可执行语义在测试域和 Q 下直接闭合。

## 9.3 为什么是正交的

“正交”不是概率独立，也不是二者互斥；它只表示证明义务、失败模式和诊断职责不同且互不蕴含：

\[
\boxed{
\operatorname{FIC}\nRightarrow\operatorname{ESC},
\qquad
\operatorname{ESC}\nRightarrow\operatorname{FIC}.
}
\]

反例一：J 错了，但 \(\pi\) 或模型先验使 Replay 恢复 Source，可能 FIC 通过、ESC 失败。

反例二：Source、Replay、J 在测试域行为一致，但 Replay 使用不同 compiler realization，可能 ESC 通过、严格 FIC 失败。

工程上先 Gate1 后 Gate2只是成本和流程顺序，不是逻辑蕴含关系。

---

# 10. 最终证书谓词

最终不必把 SCRC 定义成传统数学上的对称、传递二元等价关系；将其定义成逐 artifact 的 certificate predicate 更准确。

设：

\[
A=(J,C,\pi),
\qquad
c_0=\operatorname{Bind}_H(C).
\]

令 R 已经由规定的 source-isolated Replay call 生成。最终：

\[
\boxed{
\begin{aligned}
\mathsf{SCRC\_Cert}_{\eta}
(S,A,R)
\iff{};&
\mathsf{TypeCheck}(J,C,\pi)
\\
&\land
\left[
\Phi_{f,p}(S,c_0)=\Phi_{f,p}(R,c_0)
\right]
\\
&\land
\bigwedge_{x\in T}
\Big[
\operatorname{Close}_Q(O_S(x),O_R(x))
\\
&\qquad\qquad\land
\operatorname{Close}_Q(O_S(x),O_J(x))
\\
&\qquad\qquad\land
\operatorname{Close}_Q(O_R(x),O_J(x))
\Big].
\end{aligned}
}
\]

在当前确定性主线中，可将其解释为：

# **Replay-Closed Approximate Semantic Equivalence on the Test Domain**

更一般地，它是：

# **Observer-Relative Replay Closure with Test-Domain Semantic Certification**

---

# 11. 证书精确支撑的结论

若 \(\mathsf{SCRC\_Cert}_{\eta}=1\)，可以准确声称：

## 11.1 显式语义所有权

\[
\operatorname{SemDecl}(A)=\operatorname{SemDecl}(J).
\]

即 C 与 \(\pi\) 没有在 artifact schema 中直接创建新的语义声明。

不声称 \(\pi\) 完全没有可推断的语义信息，也不声称不存在所有 covert channel。

## 11.2 Observer-relative Replay closure

\[
\Phi_{f,p}(S,c_0)=\Phi_{f,p}(R,c_0).
\]

即：

\[
[S]_{\Phi_{f,p},c_0}=[R]_{\Phi_{f,p},c_0}.
\]

结论强度由 \(\Phi_{f,p}\) 明确限定。

## 11.3 Test-domain semantic agreement

\[
\forall x\in T:\quad
\operatorname{Close}_Q(O_S(x),O_R(x))
\land
\operatorname{Close}_Q(O_S(x),O_J(x))
\land
\operatorname{Close}_Q(O_R(x),O_J(x)).
\]

即 Source、Replay 与 J evaluator 在测试域和预先确定的 Q 下三条边都直接通过。

不声称：

\[
\forall x,\quad S(x)=\llbracket J\rrbracket(x).
\]

---

# 12. 随机隐藏测试的条件错误接受界

如果论文希望给出概率声度，设 \(\mathcal F\) 表示隐藏测试生成前的全部历史，包括 Lift、relift、Replay、Gate1 和最终候选。最终候选在 \(\mathcal F\) 下已经固定。

并行候选搜索时，令 branch `b=1,...,B`、branch 内正式候选 `k=1,...,K`：

\[
A_{b,k}=(J_{b,k},C_{b,k},\pi_{b,k},R_{b,k}).
\]

每个候选独立满足自己的 ownership、TypeCheck 与 source-blind Replay 合同。定义：

\[
\operatorname{PreClose}(A_{b,k})
=
\operatorname{TypeCheck}(J_{b,k},C_{b,k},\pi_{b,k})
\land G_1(S,c_0,A_{b,k}).
\]

令 \(\tau_{\mathrm{pre}}\) 为所有 branch 的 Gate2 前记录，并用预先确定且不读取 Gate2 的选择函数：

\[
A^*=\sigma(\tau_{\mathrm{pre}}),
\qquad
A^*\in\{A_{b,k}:\operatorname{PreClose}(A_{b,k})\}.
\]

当前实现采用 `first_observed_gate1_closure_cooperative_stop.v4`：第一个完整、已原子登记的 Gate1
closure 成为 winner，sibling 只观察 winner id 并在 provider retry 之外 cooperative stop；不读取
Gate2，也不共享 sibling candidate/diff。并发到达次序是 proof-irrelevant 调度事实，receipt 必须保存
实际 winner branch/formal round。候选集合不是第四 artifact；证书始终绑定这一个实际 winner：

\[
\boxed{
\operatorname{SCRC\_CERTIFIED}(A^*)
\iff
\operatorname{PreClose}(A^*)\land G_{2,Q}(S,c_0,A^*;T).
}
\]

随后独立采样：

\[
x_1,\ldots,x_m\overset{\mathrm{iid}}{\sim}\mathcal D,
\qquad
T\perp\mathcal F.
\]

令已经选定的完整候选为 \(a^*=(J,C,\pi,R)\)，并令 \(g_{2,Q}(a^*,x)\) 表示第 8.3 节
三条比较边在输入 \(x\) 上全部通过。定义该固定候选的 Gate2 分布错误质量：

\[
d_{\mathcal D,Q}(a^*)
=
\Pr_{x\sim\mathcal D}
\left[
\neg g_{2,Q}(a^*,x)
\right].
\]

若：

\[
\varepsilon>0,
\qquad
d_{\mathcal D,Q}(a^*)\ge\varepsilon,
\]

则：

\[
\boxed{
\Pr
\left[
\mathsf{SCRC\_Cert}=1
\land
d_{\mathcal D,Q}(a^*)\ge\varepsilon
\mid\mathcal F
\right]
\le
(1-\varepsilon)^m.
}
\]

Source–J disagreement mass 可以作为这一结论的一个充分特例，因为 Source–J 是 Gate2 的必要边。
“错误 J”只有在 \(\mathcal D,Q\) 下具有正的可检测错误质量时才受该界约束；落在分布支持之外或
被 Q 合并的错误不由该式发现。

这条定理不需要形式化前端语义，也不依赖 LLM 内部如何推理。它只依赖：

- 最终候选先确定；
- Q 在本次挑战采样前已经确定且非空洞；
- 隐藏样本独立；
- Gate2 结果不反馈到同一次实验继续修补。

若从有限池无放回采样，应使用相应的有限总体界，不能仍把样本写成 iid 后照搬该公式。

它不是通过后的后验正确概率，也不是全输入域证明；三条比较边不会把 \(m\) 变成 \(3m\)，多个
artifact 或重复选择的 batch 也不能不做额外错误率核算就直接复用单候选上界。

本实现只在 \(A^*\) 选定后生成隐藏输入，并且只执行一次 Gate2，因此搜索过 \(BK\) 个 pre-Gate2
候选不会给上界额外乘 `BK`。若另一实现对 `q` 个候选都执行 Gate2，并以任一通过为接受，则只能写：

\[
\Pr[\text{至少一个错误候选通过}]
\le
\sum_{i=1}^{q}(1-\varepsilon_i)^{m_i}
\le
\min\{1,q(1-\varepsilon)^m\}.
\]

不能写成 \((1-\varepsilon)^{mq}\)，也不能在 winner Gate2 失败后切换 sibling candidate。

---

# 13. Relift：固定验证器下的自适应 witness 搜索

第 \(r\) 轮：

\[
A_r=(J_r,C_r,\pi_r),
\]

\[
R_r=
\operatorname{Replay}
\bigl(J_r,\pi_r,\operatorname{MaskC}(C_r);U_f\bigr),
\]

\[
D_r=
\operatorname{Diff}
\bigl(\Phi_{f,p}(S,c_r),\Phi_{f,p}(R_r,c_r)\bigr).
\]

Gate1 失败后：

\[
A_{r+1}
=
\operatorname{Relift}(S,A_r,D_r).
\]

当前实现中的 `Relift` 由同一 branch-local source-aware owner conversation 继续执行：上一完整
candidate 与 closed diff 进入真实 user continuation，owner 继续修改并重新提交完整三元组。该
conversation 是 proof-irrelevant search state，不跨 branch、不进入 Replay，也不改变上述数学映射。

然后必须重新启动一份全新的 source-blind Replay：

\[
R_{r+1}
=
\operatorname{Replay}
\bigl(J_{r+1},\pi_{r+1},\operatorname{MaskC}(C_{r+1});U_f\bigr).
\]

差异、旧 Replay 和旧候选代码只进入 source-aware Lift，不进入下一轮 Replay。

因此 Relift 是：

# **Adaptive Witness Search**

它影响在有限预算中找到可接受 witness 的概率，不改变 certificate predicate 的含义。

证书绑定实际找到的 Replay \(R\)，不是不计搜索成本的抽象存在式 \(\exists R\)。运行必须分别
报告 branch 数 `B`、每分支正式预算 `K`、首轮 TypeCheck 通过率、`closure@branch/round`、
time-to-first-closure、logical/physical Lift、normal/masked Replay、transport retry、未完成 branch
与 `NO_CLOSURE_(B,K)` 分类。Gate2 只报告预选 winner 的三边结果。

一次 branch-local Lift owner session 内的 `CandidateCheck` 次数、agent model turns、active wall time
与 diagnostics 另行报告；它们
不增加正式 Gate1 round。只有 agent 最终提交的完整 candidate 经外部 TypeCheck 通过后，才进入
正式 witness search 计数。

一次实验中：

- schema、prompt、TypeCheck、observer、Q、compiler 和最大轮数保持不变；
- 只搜索新的 \((J,C,\pi)\) 候选；
- 当前 Task 14 聚焦闭合批次使用 `B=3`，每 branch 最多 `K=3` 个 TypeCheck-valid Gate1 rounds；
- 只有所有完整 branch 都耗尽本次声明的预算且未闭合，才是 `NO_CLOSURE_(3,3)`；
- Gate1 diff 只进入同 branch，sibling artifact/Replay/diff 永不共享；
- 只从 Gate1-closed 候选中按固定 pre-Gate2 规则选择一个 winner；
- Gate2 失败后本次实验结束，不再 relift。

实验结束后才修改系统，并重新开始一场新实验。

多个独立 case 可以由 `intentir.scrc.batch.v3` 在 `W/B/G/G2` 有界资源下并行运行；这些调度量只进
proof-irrelevant audit metadata。v3 新增的 `observer_unsupported` 只用于 Source observer 无法产生
可比较 observation 的能力缺口：fan-out 只有所有完整 branch 均因此停止才聚合为该类，混合结果仍
按既有 `NO_CLOSURE` 等分类收敛。它不进入上述认证公式，也不改变任何 batch v2 历史 verdict。

---

# 14. masked-J 的位置

masked-J 不属于主认证条件。

定义附加状态：

\[
\operatorname{J\_NECESSARY}
=
[\text{masked-J hard mismatch}].
\]

它只表示：

> 在当前模型、当前 \(\pi\) 与当前 topology 下，删除 J 后不足以恢复 Source。

若 masked-J hard-close，但 \(\pi\) 通过 TypeCheck，只能说明 topology 对语义有较强预测力；不能自动推翻 TypeCheck + Gate1 + Gate2。

因此状态应分离：

```text
main_status:
    SCRC_CERTIFIED / NO_CLOSURE / GATE2_FAILED

j_necessity_status:
    PROVED / NOT_PROVED / INCONCLUSIVE
```

J mutation / J-authority challenge 不进入主在线 Gate。若研究需要，可作为条件式离线消融，但不应增加主系统复杂度。
`j_necessity_status=PROVED` 也只表示在记录的模型、解码配置、Replay packet 和 masked-J 干预下
观察到 J 的操作性必要性，不是对所有生成器的全局不可替代性定理。

---

# 15. 与最初版本相比的关键修改

| 最初版本倾向 | 最终版本 |
|---|---|
| C.strategy 承载 source realization | C 恢复为纯 implementation selection；realization 进入 \(\pi\) |
| RuleCard / RuleRegistry / deterministic semantic composer 持续扩张 | Replay LLM 直接消费 typed relational \(\pi\) 与公共 \(U_f\)；可在 fresh episode 内用 syntax-only editing composer 维护自己作者化的源码 |
| Lift 输出 J/C/π 外还有 interface、BindingMap、Resolver 等 | proof artifact 严格只有 \((J,C,\pi)\) |
| Replay 可能看到 Harness、真实名字或额外 context | Replay 只看 J + \(\pi\) + MaskC(C) + 公共 \(U_f\) |
| \(\pi\) 是机制/施工 opcode 列表 | \(\pi\) 是 anonymous relational witness |
| masked-J hard-close 直接 leakage reject | masked-J 单列 J 的操作性必要性 |
| Gate1 被泛称为“证明语义住在 J” | Gate1 精确定义为 observer-relative source-blind replay closure |
| Gate2 只模糊写“三方一致” | Q 外置；确定性主线直接比较 Source、Replay、J 三条边 |
| 认证被包装为全域形式等价 | 定义为 observer-relative、test-domain-bounded certificate |
| 实验失败时边跑边增加字段 | 实验内只改候选；实验后才修改系统 |

这些修改没有改变项目的 LLM-first 哲学，而是去掉了会把系统重新做成传统 lowerer/规则库的部分。

---

# 16. 不增加前端和系统复杂度的原因

新增前端最低只需要：

\[
\boxed{
\mathcal A_f=
\bigl(
\operatorname{Compile}_f,
\operatorname{Run}_f,
\operatorname{Observe}_f
\bigr).
}
\]

其中 Observe 只返回稳定 compiler observation，不负责判断程序语义正确。

不需要新增：

- \(J\to f\) lowerer；
- frontend formal semantics；
- frontend-specific proof system；
- per-kernel RuleCard；
- LoweringAgent / retune agent；
- 公共语义 IR Q；
- 从 J/π 选择 target algorithm/API 的 deterministic Replay composer。fresh Replay agent 内部的
  syntax-only editing composer 只渲染 agent-authored nodes，不是 J→frontend lowerer，也不进入证书；
  具体边界见 [J 主导的 π_min 与 Episode-Local Composer Replay](IntentIR_J主导_pi_min与Episode_Local_Composer_Replay最终设计_2026-07-31.md)。

通用核心仍然只有：

\[
\boxed{
\mathsf{TypeCheck}
+
\text{request isolation}
+
\text{observer equality}
+
\operatorname{Close}_Q.
}
\]

---

# 17. 论文可正式保留的数学结果

## Lemma 1：Semantic Ownership Confinement

\[
\mathsf{TC}(A)
\Longrightarrow
\operatorname{SemDecl}(A)=\operatorname{SemDecl}(J).
\]

这是 closed-world artifact grammar 上的语法级性质，不需要形式化每个前端；它不声称消除了
topology 的统计线索，也不证明 realization 的行为正确。

## Proposition 1：FIC 与 ESC 互不蕴含

\[
\operatorname{FIC}\nRightarrow\operatorname{ESC},
\qquad
\operatorname{ESC}\nRightarrow\operatorname{FIC}.
\]

## Proposition 2：Observer Monotonicity

\[
\Phi_2\succeq\Phi_1
\land
G_1^{\Phi_2}=1
\Longrightarrow
G_1^{\Phi_1}=1.
\]

## Theorem 1：Conditional False-Accept Bound

\[
\Pr
\left[
\mathsf{SCRC\_Cert}=1
\land
d_{\mathcal D,Q}(a^*)\ge\varepsilon
\mid\mathcal F
\right]
\le
(1-\varepsilon)^m.
\]

## Corollary：Relift Invariance

在 certificate predicate 固定、每轮重新 Replay、隐藏测试不反馈的条件下，Relift 只改变找到 witness 的机会，不改变证书语义。

---

# 18. 图示建议

## 图 1：总流程

```text
Source S + external Harness H
              │
              ▼
          Lift LLM
              │
              ▼
       A = (J, C, π)
              │
       TypeCheck(A)
              │
              ▼
View(A) = J + π + MaskC(C) + U_f
              │
              ▼
 source-isolated Replay LLM
              │
              ▼
          Replay R
              │
        ┌─────┴─────┐
        ▼           ▼
      Gate1       Gate2
        │           │
        └─────┬─────┘
              ▼
       SCRC_CERTIFIED
```

在 Replay 旁边用红色叉号标明不可见：Source、Harness、C values、Gate1 diff、old Replay。

## 图 2：Gate1

```text
Source S ─ Bind same c0 ─ Compile_f ─ Observe/Canon Φ_(f,p) ─┐
                                                        ├─ exact equality ─ G1
Replay R ─ Bind same c0 ─ Compile_f ─ Observe/Canon Φ_(f,p) ─┘
```

标题：**Static, Observer-Relative Replay Closure**。

## 图 3：Gate2

```text
                    Source S(c0)
                    /           \
          Close_Q  /             \  Close_Q
                  /               \
         Replay R(c0) ── Close_Q ── Eval_J(J)
```

标题：**Source-Anchored Three-Way Behavioral Validation**。

三条边都直接比较；三个输出只执行一次。

## 图 4：FIC / ESC 正交二维图

- 横轴：FIC / replay closure；
- 纵轴：ESC / semantic closure；
- 右上角：SCRC_CERTIFIED；
- 标注 \(\operatorname{FIC}\nRightarrow\operatorname{ESC}\) 与 \(\operatorname{ESC}\nRightarrow\operatorname{FIC}\)。

## 图 5：实验内循环与系统改进外循环

```text
固定系统规则；B=5 个 branch 并行
      │
每 branch: Lift → TypeCheck → fresh Replay → Gate1
              ▲                               │
              └── branch-local diff ──────────┘
                   每 branch 最多本次声明的 K 轮

pre-Gate2 sigma 选择一个 Gate1 PASS winner
      → 生成隐藏输入 → Gate2 → CERTIFIED / FAILED

NO_CLOSURE or GATE2_FAILED
      │
      ▼
实验结束后的通用系统改进
      │
      ▼
开始新实验
```

---

# 19. 论文中心表述

英文：

> **SCRC is a frontend-parametric certification protocol for open-world GPU kernels. An LLM factors a source kernel into an explicit semantic carrier J, an implementation-selection object C, and a (J,C)-indexed realization witness π. A replay model, isolated from the source, reconstructs a frontend program solely from the typed artifact. A small deterministic core then checks artifact-level semantic ownership, equality under a declared compiler observer, and pairwise behavioral agreement among the source, replay, and the executable semantics of J over a fixed test domain. The resulting certificate is observer-relative and test-domain bounded; it does not require a formal semantics or a dedicated verifier for each frontend.**

中文：

> **SCRC 是一种面向开放世界 GPU Kernel 的前端参数化认证协议。LLM 将 Source 因子化为显式语义载体 J、实现选择对象 C 和依赖于 (J,C) 的 realization witness π；一个与 Source 隔离的 Replay 模型仅依据 typed artifact 重建前端程序。随后，小型确定性核心检查 artifact 级语义所有权、声明 compiler observer 下的结构闭合，以及 Source、Replay 和 J 可执行语义在固定测试域上的两两行为一致性。该证书相对于明确的 observer、比较政策和执行环境成立，不要求为每个前端重新定义完整形式语义或专用验证器。**

---

# 20. 能宣称与不能宣称

## 能宣称

- Artifact 的显式语义声明被限制在 J；
- Replay 请求不包含 Source、diff 或 C values；
- Source 与本次 Replay 在声明 observer 下闭合；
- Source、Replay、J evaluator 在测试域和 Q 下行为一致；
- 若采用隐藏独立测试，可给出条件错误接受上界；
- 新前端不需要新的 J→frontend lowerer 或完整形式语义。

## 不能宣称

- 对所有输入的全域形式等价；
- TypeCheck 排除了所有 \(\pi\) 隐写通道；
- \(I(\pi;\operatorname{Sem}(S))=0\)；
- masked-J 对所有 Kernel 必须失败；
- 当前证书证明所有 admissible C 都纯；
- Gate1 observer 是完整语义观察器；
- production lowerer 已自动通过主 Gate2。

---

# 21. 最终一句话

\[
\boxed{
\text{开放世界生成，封闭核心认证；}
\quad
\text{TypeCheck 管信息归属，Gate1 管能否重建，Gate2 管表达是否正确。}
}
\]

这是一种在**覆盖范围、前端工程成本和验证可信度**之间的主动设计选择，而不是一个“不完整的形式验证器”。
