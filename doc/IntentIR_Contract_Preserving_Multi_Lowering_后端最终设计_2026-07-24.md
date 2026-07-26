# IntentIR Contract-Guided Lowering-State Search：后端最终设计

> 状态：第五份权威设计（2026-07-24；2026-07-26 lowering-state 修订）
>
> 文件名为保持既有链接稳定而不变。本文不再把 `implementation family` 作为 compiler object、
> registry key 或 emitter 概念；旧材料中的 `multi-lowering` 统一按本文定义解释为
> **contract-guided structural lowering-path expansion**。
>
> 上位前端与认证设计：
> [C：所有权、绑定、消费与三方验证](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
> [π：依赖类型实现见证与 Replay 安全](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)、
> [Lift、Replay、Gate1、Gate2 最终执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)、
> [SCRC 最终数学与执行流程](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。
>
> 范围：本文只定义 SCRC 之后的 production backend。它不修改 Lift、Replay、TypeCheck、Gate1、
> Gate2 或 π 的协议，也不把 backend conformance 加入 SCRC 证书。

## 1. 最终判断

IntentIR 的 production backend 不是把 J 当作一段等待唯一 lower 的普通 IR，也不是在 JSON J 中
进行程序搜索。它把一次已接受 artifact 中的 J 作为不变语义接口，将其确定性编码为 Intent MLIR，
再在通用 compiler pass contracts 对 J 保持闭合的实现空间中搜索。

完整后端主线是：

```text
SCRC-accepted (J,C,pi)
          |
          |  orchestration 只交付 J+C；pi 到此停止
          v
M0 = Emit_v(J)                         deterministic materialization
          |
          v
S_C = Attach_C(M0,C,t)                 independent post-materialization step
          |
          |  BaseIR(S_C)=M0; C only supplies erasable hints/order
          v
Intent MLIR + B-derived contract facts + C-derived search state
          |
          v
registered MLIR transformations + lowering-state/path search
          |
          v
target lowering + performance-only selection
          |
          v
target executable
```

核心作用可简写为：

\[
\boxed{
A\text{ 暴露结构；}\qquad
B\text{ 筛除不合法实现；}\qquad
C\text{ 提高搜索效率。}
}
\]

这句话只描述 A/B/C 对后端的作用，不重新定义其语义所有权。A 与 B 共同组成唯一 denotational
对象 J；C 始终是可忽略、可替换的 implementation-selection prior。

## 2. 后端硬边界

### 2.1 唯一输入

production backend 的逻辑接口保持：

```text
Lower_t(J,C) -> target executable
```

系统 orchestration 只对 SCRC-accepted artifact 调用 production backend；backend 自身不需要读取
Gate verdict。backend 禁止读取：

- Source；
- Replay；
- π；
- Gate1 diff；
- Gate2 input/output；
- Q；
- Source compiler fingerprint；
- kernel name、source path 或 provider identity；
- per-kernel RuleCard、历史 candidate 或旧实验结果。

允许进入 backend 的只有：

- 唯一 J；
- 可选 C；
- target capability；
- backend 自身的版本化 generic pass library、cost model 与运行环境。

### 2.2 π 到此停止

π 只证明 Source realization 能被 source-blind Replay 重建。目标后端的任务不是模仿 Source
realization，因此：

\[
\pi\notin\operatorname{BackendInput}.
\]

Source 使用 warp reduction、shared memory、atomic、async copy 或某种 pipeline，不构成 target
backend 必须复用同一机制的理由。

### 2.3 J 在一次后端搜索中保持不变

搜索器不修改 J，不产生 `J_target`、`optimized_J` 或第二语义对象。所有候选都是同一 J 的 compiler
implementation state：

```text
J  --Emit_v-->  M0 --rho1--> M1 --rho2--> ... --rhok--> Mk
```

候选 MLIR、schedule assignment、cost record 与 lowering trace 都是 production 内部状态或审计
记录，不是新的 Lift artifact，也不进入 Replay 或 SCRC 证书。

## 3. J=A+B 在后端中的身份

## 3.1 Layer A：typed structured value/state program

A 是可执行、可组合的结构化计算，至少包括：

- logical operands/results；
- dtype、shape 与 value carrier；
- operator 与 dataflow；
- logical axes；
- map/select/cast；
- reduce/scan/contraction；
- gather/scatter 与 logical update；
- structured loop 与 recurrence；
- observable value/state transition。

实际 value dtype 属于 A 的 typed graph。例如：

```text
t0:f16 -> cast:f32 -> mul:f32 -> reduce:f32 -> t1:f32
```

这里的 cast、operand/result dtype 与 reduction result dtype不能在 B 中再复制出第二份真值。

A 不包含：

- CUDA/RVV pass 名；
- block/warp recipe；
- Hillis--Steele、bitonic 等实现算法名；
- shared-memory/barrier/atomic spelling；
- target-specific launch policy；
- `fusion_allowed`、`use_tensor_core` 等 pass permission。

## 3.2 Layer B：immutable semantic boundary

B 补全所有合法实现必须保持的边界事实：

- logical domain；
- predicate/fill；
- logical access 与 output mapping；
- alias、effect 与 observable destination；
- conflict、duplicate 与 update order；
- reduction ordering/reassociation；
- rounding、overflow、tie、NaN/Inf 与 determinism 等已声明数值合同；
- closed result-relation class；
- SCRC/π 所需的 typed realization authorization。

B 中只保存语义事实，不保存某个 backend transformation 的名字。正确形式是：

```text
no observable intermediate effect
no alias
reduction order is tree-permitted
duplicate updates are forbidden
```

而不是：

```text
fusion_allowed = true
online_to_twopass_allowed = true
use_tensor_core = true
gpu_strategy = warp
```

backend pass 从固定事实推导 applicability；它不能要求 Lift 直接选择 target recipe。

## 3.3 B 的单一所有权与多种物理投影

canonical J 中只有一个 B。禁止恢复任意 per-op `layer_b` 字典或第二份 boundary。

在 Intent MLIR 内，为便于 verifier 和 pass 使用，B 的 closed facts 可以被确定性投影为：

- op/region attrs；
- memory-effect interfaces；
- typed contract ops；
- dynamic legality predicates；
- ephemeral analysis facts。

这些物理位置不改变 B 的语义所有权，也不能反向改写 J。相同事实不能在多个位置拥有互相独立的
解释。

## 3.4 realization authorization 不充当 target permission

`realization_authorizations` 的既有职责是约束 π 中的 sensitive representation refinement。
production backend 不得把它直接解释为 target transformation permission。

target legality 始终来自：

```text
A semantics
+ B semantic facts
+ current MLIR state
+ target capability
+ pass contract
```

若未来一种关系确实需要 SCRC 与 backend 共同消费，必须定义一个新的、单一所有权的 closed semantic
relation，并同时指定所有消费者；不能通过重命名 authorization 偷换含义。

## 4. C：非绑定的 schedule prior

C 保留 Source realization 中可分离的 implementation-selection 信息，例如：

- tile extent；
- vector width；
- warp/thread count；
- pipeline stage；
- unroll；
- prefetch；
- memory placement；
- compiler option。

C 的完整作用限定为：

- 提供 canonical/source-like 初始 schedule；
- 给出第一批候选参数；
- 改变搜索顺序；
- 指定优先邻域；
- 帮助 cost model 冷启动。

C 不决定某条 transformation 或 lowering path 是否语义合法，不增加或删除 J 的 allowed behavior。

### 4.1 当前最小 C 足以启动后端

当前 slot、kind、role、value type、required 与 observed value 已能保存 source prior，但只靠 `role`
无法在多 loop、多 reduction 或多 tensor 的 J 中无歧义地指出消费位置。C 权威文档已经把下一
protocol 的最小、source-neutral `subject_ref/axis_ref` attachment 定义清楚；其 schema、Lift 与
TypeCheck 激活规则仍只属于 C/SCRC 文档，不由本文重定义。

本文只规定 backend 消费语义：`subject_ref/axis_ref` 若存在，`Attach_C` 可据此定位 prior；全局 prior
或历史 artifact 可以没有 attachment。局部 prior 缺少 anchor、anchor 含混或 backend 不理解时必须
安全忽略，不能读取 Source/π 猜测，也不能据此改变 `Reach_t(J)`。

特别地：

- `required` 表示本次 Source/Replay invocation 需要实际 valuation，不表示 target 必须沿用；
- target-specific parameter domain 由 transformation/schedule contract 和 target capability 定义；
- C 按定义可以被忽略或 retune，不需要重复声明 `retunable=true`；
- 若某个值不可替换且会改变逻辑结果，它不属于纯 C；
- 不增加 `legal_domain/strength/retunable/fixed`；候选合法域仍属于 pass/schedule/target contract；
- `subject_ref/axis_ref` 只解决“这个 prior 作用于哪里”，不授权该作用点上的 transformation。

### 4.2 不增加 C_target

target 搜索产生的 schedule assignment 是 compiler internal state：

```text
sigma_target in SearchState
```

它不是新的 proof artifact，也不产生：

- `C_target`；
- C-generation Agent；
- retune Agent；
- source-to-target schedule translation artifact。

搜索与 lineage 可以写入普通 backend receipt，但不进入 SCRC。

## 5. Deterministic J materialization

对 schema/materializer 版本 v：

\[
M_0=\operatorname{Emit}_v(J).
\]

同一 J 与同一 v 必须得到同一 canonical Intent MLIR。`Emit_v`：

- 不读取 Source、π、Gate 或 Q；
- 不调用 LLM；
- 不选择高性能实现；
- 完整编码 A；
- 将 backend-relevant B 投影为 dataflow、attrs、effects、verifier facts 或内部 analysis；
- 对无法完整消费、但会影响 target denotation、safety 或 ABI 的 J/B relation fail closed；
- 识别 `realization_authorizations` 的 SCRC-only 身份，不把它当 target permission，也不因其存在
  自动拒绝 J。

`Emit_v(J)` 严格不读取 C。materialization 完成后，backend 才可以通过独立入口执行：

```text
S_C = Attach_C(M0,C,t)
BaseIR(S_C) = M0
```

`S_C` 是一次 backend invocation 内的 search state，包含不可变 `M0`、可擦除 C annotations、初始
参数与 candidate ordering。若实现为工作副本 `M_C` 上的 attrs，必须满足：

```text
Erase_C(M_C) = M0
```

且擦除前后 A/B denotation、pass applicability 与 target ABI 不变。只有 `Attach_C` 可以把 artifact C
投影为 C-derived MLIR attr；parser、`Emit_v`、pass default 或 kernel recognizer 都不能冒充 C 来源。
该步骤不产生新的顶层 artifact，也不修改 J。

这里的 deterministic materialization 与 Lift/Replay 中被禁止的 deterministic composer完全不同：

- Lift/Replay 面对开放前端，语义理解与源码生成归 LLM；
- production materializer 面对已经 closed、typed、accepted 的 J，只执行 compiler encoding。

## 6. Intent MLIR 的正式角色

基准 Intent MLIR `M0` 是：

```text
J 的确定性 typed encoding
+ B-derived compiler facts
```

调用 `Attach_C` 后的 backend working state 才可以另外持有 C-derived schedule annotation 与其他
compiler internal implementation state；`BaseIR(S_C)` 仍是同一个 `M0`。

它不是：

- 第二份 J；
- J 的可变替代品；
- Replay 输入；
- 携带 Source/π/Gate 语义副本的 wire format。

structured Intent ops 必须至少保留到所有需要这些语义结构做 match/precondition 的结构性
transformation 完成，避免过早退化为 generic loops 而丢失
map/reduce/scan/contraction/recurrence 等结构。

## 7. Generic transformation library 与 lowering derivation

backend 登记 source-neutral、kernel-name-independent 的 compiler transformations。每个
transformation 根据 A 的 op、region、axis 与 graph pattern 做结构匹配，再由 B、当前 MLIR state、
target capability 与 pass contract 决定是否可应用：

```text
A/IR structural match
  + B semantic preconditions
  + current-state/target legality
    -> admitted transformation transition
```

示例 transformation 包括：

```text
reduce
  -> map-to-vector-tree / map-to-subgroup / map-to-workgroup
  -> split-partial-and-finalize

scan
  -> map-to-SIMD-prefix / map-to-subgroup-scan / map-to-workgroup-scan

contraction
  -> tile / vectorize / map-to-library / map-to-MMA

conv
  -> direct-to-loops / rewrite-to-implicit-GEMM / materialize-im2col

softmax-shaped graph
  -> fuse-producer-reductions-epilogue
  -> materialize-selected-intermediate
  -> map-row-reduction-to-subgroup/workgroup

structured recurrence
  -> direct recurrence lowering / registered equivalent decomposition rewrite
```

compiler 不“发射一个 family”。搜索候选始终是一个具体 MLIR state `M` 及产生它的
transformation trace：

```text
p = (rho_1, ..., rho_k)
M_p = rho_k(...rho_1(M0))
```

同一结构可以由不同 trace 到达；搜索器应通过 canonical IR digest 去重。实验和审计可以从结果 IR
派生擦除数值 schedule 参数后的 `topology_digest`，但它只用于去重、分组和展示，不成为 semantic
object、registry key、legality input 或 emitter selector。禁止按当前 kernel 名称、源码路径、
provider、历史 benchmark 或手写 `family_id` 选择隐藏 recipe。

## 8. B-conditioned pass contracts

对 target t，定义合同环境：

\[
\Gamma_{J,t}=\operatorname{ContractView}(J,t).
\]

`ContractView/Γ` 是由唯一 J、target capability 与版本化 compiler contracts 确定得到的 ephemeral、
non-serialized compiler analysis。它没有独立语义所有权，不成为 backend 输入或 receipt truth，不
进入 SCRC，也不能反向改写 J。

定义实现 judgement：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}_t(J).
\]

其含义是：M 是 target t 上满足 J 全部可观察合同的一个 compiler state。它覆盖：

- value/state behavior；
- bounds 与 memory safety；
- alias/effect；
- update/conflict；
- order/numeric/result relation；
- target ABI。

每个 transformation \(\rho\) 必须声明：

```text
matched op/region/graph
required A carriers
required B facts
target capability
current-state precondition
transformation result
preservation class
searchable parameters
```

应用规则是：

\[
\frac{
\Gamma_{J,t}\vdash M:\operatorname{Impl}_t(J)
\qquad
\operatorname{Pre}_{\rho}(J,M,t)
}{
\Gamma_{J,t}\vdash\rho(M):\operatorname{Impl}_t(J)
}.
\]

这是版本化 compiler contract 的规范级 judgement 与条件组合义务，不是已经完成的机器检查形式
证明。某个实现或 release 只能对当时实际实现、登记并经过准入检查的 `Emit_v`、transformation
`rho` 与 `Lower_t` 作支持 claim；第 7 节列出的尚未实现 transformation 以及第 20 节 roadmap
不能被计入当前 `Reach_t(J)`。

若当前 B 没有足够事实证明某个 precondition，该 pass 对该 J 不适用；系统选择保守实现或显式
unsupported，不能从 tolerance、Source 惯例或运行结果补足前提。

## 9. Reachable implementation space

定义由 J、target 与启用的通用 transformations 决定的 MLIR compiler-state 空间：

\[
\operatorname{Reach}^{\mathrm{MLIR}}_t(M_0;\Gamma_{J,t})=
\left\{
M\mid
M_0\xRightarrow[\Gamma_{J,t}]{\rho_1,\ldots,\rho_k}M
\right\}.
\]

正文可将其简写为 `Reach_t(J)`，但其中的 M 始终是 Intent/lower-IR compiler state；J 只是不可变
参数，不是搜索状态。

C 不改变该空间的语义边界，只定义搜索顺序：

\[
\operatorname{Order}_C\bigl(\operatorname{Reach}_t(J)\bigr).
\]

对有限搜索预算 b，定义实际访问集合：

\[
\operatorname{Visited}_{t,b}(J;C)
\subseteq
\operatorname{Reach}_t(J).
\]

C 可以改变 `Visited` 的顺序和有限预算下覆盖到的候选，但不改变 `Reach` 中哪些状态合法。

实际搜索可以是：

- enumeration；
- beam search；
- dynamic programming；
- e-graph/equality-saturation 风格搜索；
- superoptimizer-style bounded rewrite；
- cost-model-guided search；
- evolutionary search；
- profiling-guided autotuning。

搜索器可以是确定性的，也可以是启发式的。正确性不依赖候选顺序，而依赖所有生成 transition 都
保持 `Impl_t(J)`。

## 10. Structural lowering-path expansion 与 schedule tuning

production search 统一搜索合法 compiler states；下面的分类只服务 transformation contract、实验控制
与性能归因，不建立另一套 compiler hierarchy。

### 10.1 Schedule-preserving transformation

从固定的 canonical structural derivation 出发，只改变不会改变 compiler graph、fusion partition、
materialization points、reduction/synchronization topology 或 kernel-stage decomposition 的参数，例如：

- tile；
- vector width；
- threads/warps；
- unroll；
- pipeline；
- launch configuration。

memory placement 不能一概视作 schedule：若它只把同一个 logical buffer 映射到另一合法 address
space，可以归入 schedule；若它插入/删除 materialization、改变同步图或跨 kernel 生命周期，则属于
结构性 transformation。

### 10.2 B-conditioned local transformation

这类 transformation 使用 B 事实，但不改变主计算结构，例如：

- rank-0/singleton host-by-value capture；
- read-only load motion；
- dead ABI input elimination；
- 已有控制/谓词的局部 specialization。

它们可以带来真实性能收益，但必须单独报告为 `B-conditioned local transformation gain`，不能作为
结构 lowering-path 扩展的证据。

### 10.3 Structural transformation

允许改变实现结构：

- fusion/fission；
- producer-consumer fusion；
- intermediate materialization/elimination；
- sequential/warp/block/hierarchical reduction；
- direct/multi-stage decomposition；
- layout/data-movement decomposition；
- registered algorithmic equivalence transformation。

“结构改变”不是任意重写。每种非平凡等价变化都必须作为通用 compiler transformation 存在，
并具有明确的 A match、B preconditions、current-state/target legality 与 preservation contract。

例如 online recurrence 与 ordinary two-pass reduction 不能因为输出在 tolerance 下接近，就被视为
普通 schedule 变化。只有一条单独、已定义保持合同的 transformation 才能产生另一条合法
lowering derivation。

形式上可把已登记变换按实验目的记为 `R_sched`、`R_local` 与 `R_struct`。这三个集合都是同一个
transformation registry 的标签；candidate identity 始终来自具体 MLIR state、trace 与 schedule，
而不是来自预先声明的实现族。

## 11. Numeric correctness boundary

backend 必须分清三类信息：

```text
A typed graph:
  实际 operand/result/cast dtype 与显式运算

B numeric/result contract:
  order、reassociation、rounding、exception policy、允许观察关系

Q:
  SCRC/实验比较时使用的具体 comparator policy
```

production backend 不读取 Q。Gate2 的 `tolerance` 不能授权：

- 改变 reduction tree；
- 使用 approximate transcendental；
- 启用 FMA/contraction；
- 改变 accumulator dtype；
- 放宽 NaN/Inf、signed zero 或 overflow 行为。

这些 transformation 必须分别由 A/B 与 pass contract 支撑。

第一版 backend 只开放现有 A/B 足以证明前提的 transformation。某类通用 transformation 确实缺少
必要 numeric fact 时，才增加最小、closed、版本化的 B relation，并同时定义其 materializer、
RefEval/Q、TypeCheck 与 backend consumer；不能先添加无人消费的字段。

当前 accepted J corpus 与 production `ContractView` 的能力边界必须保守表述：现有
`reduce.assoc=associative|float_nondeterministic`、boundary reduction、effect/alias 与 result relation
足以支持结构匹配、mask/identity 物化，以及 exact preconditions 能从当前 schema 完整证明的
effect-safe transformation；但 `float_nondeterministic` 或 Gate2 `tolerance` 本身不自动授权任意 tree
reduction、FMA、近似数学、accumulator dtype 改变或 atomic update。缺少更强 closed fact 时这些路径
不进入 `Reach_t(J)`。

这里的“只读无别名”也必须有完整性依据。若当前 artifact version 只列出已知 alias relation，却没有
规定“关系缺失即 no-alias”或显式声明 relation set complete，则 `boundary.aliases=[]`、
`signature.alias_of=null` 不能单独推出全参数 no-alias。`ContractView` 的 capability 必须携带其 exact J
field、schema completeness rule 与推导版本；无法证明 completeness 时返回 `missing_contract`，不能把
具体 benchmark 恰好使用 disjoint buffers 当作语义证据。

本阶段不等待 B schema 扩展，也不修改任何已闭合 J。优先实现并实验现有 B 已可证明的路径：

- producer/reduction/epilogue 的 effect-safe fusion；
- intermediate materialization/copy elimination；
- RVV whole-graph Linalg fusion 与后续 vectorization；
- local scalar capture/load motion/predicate specialization；
- 对数值顺序不敏感或已具明确 bit-exact associativity 的 target mapping。

若未来扩展 reduction numeric relation，不能使用含混的 `tree_permitted`。最小 closed vocabulary 至少
区分：

- `fixed_order`：保持 A 定义的 canonical leaf order 与固定 parenthesization；
- `ordered_tree`：允许改变 parenthesization/tree shape，但保持 leaf order；
- `commutative_tree`：允许改变 tree shape 与 leaf permutation。

后两者都可能改变浮点结果，必须绑定 rounding/exception/result contract；accumulation dtype 若已由 A
的显式 cast/typed reduce 唯一确定，不在 B 重复第二份真值。FMA/contraction、approximate
transcendental、改变 accumulator dtype 或 atomic linearization 仍需要各自独立、最小的 closed numeric
relation，不能由上述 reduction order 或 tolerance 顺带授权。

任何新 permission 必须通过新的可选、版本化 B relation 与新 J digest 进入 SCRC；旧 artifact 缺失即
fail closed，不被静默重解释。当前 artifact.v7 闭合批次不为后端搜索中途切换 schema。后端先记录
真实 pass 的 `missing_contract` 和最小前提；只有需要这些 transformation 的旧 kernel 才在后续版本
产生新 J 并完整重认证，其他已闭合 artifact 与证书保持原解释。

## 12. IntentOp 清理与归属规则

### 12.1 Op description 只写语义

IntentOps.td 中的 summary/description 应保留：

- extensional semantics；
- typed operands/results；
- logical axes/order；
- region/yield contract；
- B-derived closed attr 的语义。

应移出的内容包括：

- NVVM approximate intrinsic；
- `arith.select` 等具体 lower op；
- Hillis--Steele/shared-memory/barrier；
- bitonic、one-block-per-row；
- CUDA load/store/atomic recipe；
- direct/no-im2col 等当前实现限制；
- 性能 tolerance 数值。

移走 recipe 不等于删除 backend 能力；它使同一 A/B 能进入多条 target lowering derivation。
当前 `include/Intent/IntentOps.td` 仍含上述多类 target recipe；因此这是 B0 的实际清理项，而不是
已经完成的架构事实。extensional equation、tie/order/bounds、result type 与 region/yield contract
仍属于 op semantics，不应因清理 recipe 一并删除。

### 12.2 不恢复 per-op layer_b

B 仍由 canonical J enclosing boundary 单一拥有。MLIR op 上只允许由 J/B 确定性投影的 closed attrs，
不得重新接受任意字典或 consumer-private semantics。

### 12.3 reduce.assoc

`reduce.assoc` 是 B-like numeric fact 在 reduce op 上的物理投影，不是 schedule。现有词汇若无法区分
fixed order、tree-permitted、浮点 reassociation 与 observation relation，应通过最小 closed relation
演进，而不能把 Gate2 `tolerance` 当成同一个 permission。

### 12.4 loop.tile_hint

canonical J/A 不拥有 schedule tile。`intent.loop.tile_hint` 只有同时满足以下条件时，才可以作为
C-derived MLIR annotation 保留：

- canonical `Emit_v(J)`、LoopOp 的 A denotation 与 verifier 都不依赖它；
- 只由独立的 `Attach_C` 后置生成；
- 删除它满足 `Erase_C(M_C)=M0`，并且不改变合法候选集合。

满足这些条件后，backend 才可以替换、retune 或删除该 annotation。当前实现尚未满足：
`emit_mlir.py` 总是发射 `tile_hint`，`ConvertIntentToLinalg.cpp` 用它决定 slice size 与 trip count，
缺失时还默认成 1。这是 B0 必须解除的旧依赖；解除前不能把当前 `tile_hint` 当作 ownership-compliant
C 投影的例证。

## 13. Attention 与复杂 Kernel 的后端解释

### 13.1 简单 reduction-shaped Attention 子图

对于：

```text
Out:f16 -> cast:f32
DO:f16  -> cast:f32
mul
reduce(head_dim)
Delta:f32
```

A 已经暴露 cast、elementwise producer 与 reduction。可选 lowering 包括：

- 分离或融合 cast/mul；
- producer-reduction fusion；
- warp/block/multi-stage reduction；
- accumulator placement；
- intermediate store elimination；
- vector/layout/tile 调优。

不需要增加 named FlashAttention op 或读取 π。

### 13.2 完整 online Attention

若 J/A 明确包含 online-softmax recurrence，canonical lowering path 可以直接 lower recurrence。
其他算法结构只有在 generic compiler library 中存在相应 contract-preserving transformation 时才进入
搜索。

不能为了扩大后端搜索空间先把 A 改写成模糊的 extensional attention，同时让 π 保存 online update
算术；这会破坏 J 的语义所有权和 source-blind Replay 边界。

## 14. Forge 内容的迁入方式

TianchenForge/历史 backend 中的内容可以按以下类别迁入当前工程：

1. 通用 MLIR transformation 或 lowering pass；
2. target lowering pass；
3. schedule parameter domain；
4. cost-model feature；
5. benchmark kernel、输入域与测量 harness；
6. pass 正反例与性能工程经验。

迁入后必须：

- 以 A pattern、B precondition 和 target capability 注册；
- 不读取 kernel name、Source 或 π；
- 不形成 per-kernel RuleCard；
- 由当前 `J -> Intent MLIR -> target` 路径生成候选；
- 在当前系统中重新记录 compile、performance 与 conformance evidence。

历史数值可以用于选择优先迁入的 transformation 和搜索范围，但不替代当前系统的运行记录。

## 15. Target lowering 与性能选择

若：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}_t(J),
\]

则 target lowerer 的合同是：

\[
\operatorname{Lower}_t(M)\models J.
\]

给定有限预算 b，最终候选选择为：

\[
M^*_{b,C}=\arg\min_{M\in\operatorname{Visited}_{t,b}(J;C)}
\operatorname{Cost}_t\bigl(\operatorname{Lower}_t(M)\bigr).
\]

Cost 可以来自静态模型、profiling、benchmark 或混合策略。候选选择只比较性能；运行结果不参与
候选是否合法的判定。

## 16. 条件组合保持命题

假设：

1. materializer 正确编码 J：

\[
\Gamma_{J,t}\vdash M_0:\operatorname{Impl}_t(J);
\]

2. 每个启用 transformation 保持实现 judgement：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}_t(J)
\land \operatorname{Pre}_{\rho}(J,M,t)
\Rightarrow
\Gamma_{J,t}\vdash\rho(M):\operatorname{Impl}_t(J);
\]

3. target lowerer 保持合同：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}_t(J)
\Rightarrow
\operatorname{Lower}_t(M)\models J.
\]

则：

\[
M\in\operatorname{Reach}_t(J)
\Rightarrow
\operatorname{Lower}_t(M)\models J.
\]

因此性能选择不会改变 correctness 边界：

\[
\operatorname{Lower}_t(M^*_{b,C})\models J.
\]

该命题的 compiler TCB 至少包括：`Emit_v` 及其 B projection、`ContractView`/target-capability
query、每个实际启用 transformation 的实现及其 `Pre`/preservation/admission 检查、transition
admission，以及 target lowerer。search/cost model 只有在不能绕过 admission、不能直接构造未登记
candidate 时才不承担语义裁决。该命题不声称 SCRC 已经证明 production compiler 的每个 pass，
也不把测试通过冒充上述条件的形式证明。

## 17. Backend conformance 的位置

可以独立运行：

```text
Close_Q(Exec(Lower_t(M),x), RefEval(J,x))
```

但它只用于：

- compiler regression；
- CI；
- release preflight / smoke test；
- debug；
- 实验审计；
- 新 structural transformation/lowering path 的工程验证。

它不是：

- SCRC 的第四条 Gate2 边；
- production search 的 legality oracle；
- 一个候选先猜语义、再通过有限测试获得合法性的机制。

## 18. 实验与消融设计

实验使用四个主要层级。这里的“固定结构”由 canonical structural transformation trace 决定，不由
`family_id` 或 named emitter 决定：

```text
E0  Source frontend implementation
E1  canonical deterministic target lowering + default schedule
E2  fixed-structure schedule search
E3  contract-guided structural lowering-path + schedule search, with C
```

额外消融：

```text
E4  same structural Reach and budget as E3, with Erase_C
```

对应问题：

- `E1 vs E0`：semantic interface 与 canonical lowering 的基础质量；
- `E2 vs E1`：只调 schedule 的收益；
- `E3 vs E2`：结构性 lowering-path expansion 的额外收益；
- `E3 vs E4`：C 作为搜索先验的作用，而不是 correctness 作用。

另设与 E0--E4 正交的 local-B 配对实验：

```text
L0  fixed structural Reach, registered B-conditioned local transforms disabled
L1  same Reach/budget, registered B-conditioned local transforms enabled
```

L1/L0 保留完整 J=A+B 和 fail-closed `ContractView`；关闭的只是额外 local transformation，不是 B
语义。rank-0 capture、singleton capture、load motion 与 predicate specialization 的收益只写入该轴，
不得混写成 E3/E2 的结构路径收益。

报告指标至少包括：

- absolute latency/throughput；
- 相对 Source 与 vendor baseline 的性能；
- candidate 数、compile 数与测量数；
- time-to-best；
- transformation match/admission/rejection reason 与 trace；
- fixed-structure schedule 与 structural-path expansion 的增量；
- C prior 对搜索效率的影响；
- compile failure 与 backend conformance failure；
- pass/library 的跨 Kernel、跨 frontend、跨 target 复用。

E2/E3 必须共享同一 J、target、compiler revision、terminal code generator、correctness policy 与
profiler。E3 的 candidate set 必须包含 E2 的全部候选。报告同时给出：

- equal-total-budget 的 anytime/time-to-best 曲线；
- 小候选空间 exhaustive/oracle best；
- 固定 latency map 下 E3/E4 的 full-budget convergence；
- all-kernel 与 transformation-applicable subset，禁止只挑 winner。

不同结构路径必然经过不同的高层 transformation trace；“同一 code generator”指共享同一 backend
revision 和 terminal target pipeline，不要求结构分支使用完全相同的高层 rewrite。

## 19. 版本与演进规则

后端可以快速演进，但每次演进必须保持对象边界清楚：

- 新 transformation/pass contract 只扩展 backend library，不改变 J；
- 新 schedule parameter 只进入 compiler internal state或现有 C 的安全消费；
- 新 B relation 必须有真实 consumer、closed schema 与完整版本升级；
- 修改 A denotation、B relation 或 materializer interpretation 时，必须升级对应 schema/materializer/
  evaluator 版本；
- 旧 artifact 不被静默解释成新语义；
- 不使用 hash、所谓冻结门或 kernel allowlist 决定合法性。

## 20. 实现阶段

项目执行上，B0--B4 在目标 Kernel 的 SCRC closure 与 accepted J corpus 形成后再独立展开；不要求
每闭合一个 Kernel 就紧邻实现其 backend。SCRC 阶段若真实证据暴露 J/TypeCheck/Replay/RefEval 的
通用缺口，可以修正对应 A/B 语义合同，但不提前混入本节的 IntentOp ownership/search 重构。

### 阶段 B0：IntentOp/ABC ownership 对齐

- 清理 Op.td target recipe；
- 标记 A semantics、B-derived attrs 与 C-derived annotations；
- 解除 `intent.loop.tile_hint` 对 canonical Emit、LoopOp denotation/verifier 与通用 loop lowering 的旧依赖；
- 审计每个 B 字段的 consumer；
- 保证 structured ops 不在结构 transformation 完成前过早消失。

### 阶段 B1：后端合同入口

- 固化 `Emit_v(J)`；
- 实现独立 `Attach_C(M0,C,t)` 与 `Erase_C` invariant；
- 建立 `ContractView(J,target)`；
- 让 materializer 对所有影响 target denotation、safety 与 ABI 的 backend-relevant B facts 完整消费
  或 fail closed；
- 扩展现有 generic compiler transformation registration/admission API，只以 typed A pattern、
  B precondition、current state 与 target capability 为 key；不另建 family registry。

### 阶段 B2：第一批 lowering-path expansion

- elementwise producer/reduction fusion；
- 在现有 B 足够时启用 producer/reduction/epilogue fusion、materialization/copy elimination；
- 把 RVV whole-graph fusion 从 schedule flag 分离为显式 structural transformation trace；
- reduction mapping 只开放当前 B 能证明的路径；tree/multi-stage 路径缺少 numeric permission 时
  fail closed，不等待或伪造新 B；
- schedule candidate 与 C-prior ordering；
- lowering trace、B facts used、topology/IR digest、rejection reason 与 performance receipt。

### 阶段 B3：算法结构与跨 target

- softmax-shaped graph；
- structured recurrence；
- contraction/conv 等通用 structural transformations；
- CUDA/RVV 或其他 target 的同一 J reuse；
- 迁入 Forge 通用 recipes。

### 阶段 B4：实验与消融

- E0--E4；
- search cost/time-to-best；
- transformation/path reuse；
- C prior；
- backend conformance 与性能结果分开报告。

## 21. 最终系统表述

> **IntentIR production backend consumes only J and optional C from an SCRC-accepted artifact. It deterministically materializes the immutable semantic program and boundary into Intent MLIR, expands concrete lowering derivations through registered source-neutral transformations, admits every transition only under A-matching, B- and target-conditioned compiler contracts, and uses C solely to prioritize finite-budget exploration. The compiler emits concrete lowered IR and target executables, never an abstract implementation family; performance measurement selects among already legal compiler states and never serves as a semantic oracle.**

中文：

> **IntentIR production backend 只消费 SCRC 已接受 artifact 中的 J 与可选 C。它将不变的语义程序和边界确定性编码为 Intent MLIR，通过已登记、source-neutral 的 transformations 扩展具体 lowering derivation，并根据 A match、B、当前编译状态与 target capability 判断每一步是否适用；C 只提高有限预算下的搜索效率。compiler 只发射具体 lowered IR 与 target executable，不发射抽象实现族；性能测量只在已经合法的 compiler states 之间选择，不承担语义裁决。**
