# IntentIR J-Grounded、Q/D-Certified Lowering-State Search：后端最终设计

> 状态：第五份权威设计（2026-07-24；2026-07-27 closed-18 横向消费、execution-domain search、TileLang-aligned 评测口径与 Roofline 修订）
>
> 文件名为保持既有链接稳定而不变。本文不再把 `implementation family` 作为 compiler object、
> registry key 或 emitter 概念；旧材料中的 `multi-lowering` 统一按本文定义解释为
> **J-grounded structural/numeric lowering-state expansion with Q/D closure**。
>
> 上位前端与认证设计：
> [C：所有权、绑定、消费与三方验证](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
> [π：依赖类型实现见证与 Replay 安全](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)、
> [Lift、Replay、Gate1、Gate2 最终执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)、
> [SCRC 最终数学与执行流程](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。
>
> 范围：本文只定义 SCRC 之后的 production backend 与 backend executable closure。它不修改
> Lift、Replay、TypeCheck、Gate1、Gate2 或 π 的协议；backend executable closure 使用同一冻结
> Q/D，但属于独立 backend receipt，不增加 SCRC Gate，也不修改已接受 J。

## 1. 最终判断

IntentIR 的 production backend 不是把 J 当作一段等待唯一 lower 的普通 IR，也不是在 JSON J 中
进行程序搜索。它把一次已接受 artifact 中的 J 作为不变、可执行的 reference interface，将其
确定性编码为 Intent MLIR，再搜索由通用 compiler transformations 产生的具体 lowering states。

本文的 correctness claim 与 SCRC 保持同一边界：observer-relative、实际 `c0`、预先声明的有限
测试集合 T 或分布 D、以及冻结比较政策 Q。对于 reduction tree、FMA/contraction、内部精度或
approximate math 等可能改变浮点观察值的候选，不要求先向 J 增加一套跨执行域的静态 permission；
候选必须通过统一 Q/D 执行闭合后才可进入性能选择。shape、dtype、ABI、domain、bounds、alias、
effect、update 与 observable state 等 hard semantics 仍由 J 和静态 compiler verifier fail closed，
不能由 tolerance 或测试结果放宽。

完整后端主线是：

```text
SCRC-accepted (J,C,pi) + recorded Q/D scope
          |
          |  core lowerer 只取得 J+C；pi 到此停止
          v
M0 = Emit_v(J)                         deterministic materialization
          |
          v
S_C = Attach_C(M0,C,t)                 independent post-materialization step
          |
          |  BaseIR(S_C)=M0; C only supplies erasable hints/order
          v
Intent MLIR + B-derived hard facts + C-derived search state
          |
          v
registered MLIR transformations + lowering-state/path search
          |
          v
static hard verification + target lowering
          |
          v
generic Q/D candidate filter + performance measurement
          |
          v
held-out Q/D final-winner closure
          |
          v
target executable + backend closure receipt
```

核心作用可简写为：

\[
\boxed{
A\text{ 暴露结构；}\quad
B\text{ 固守 hard boundary；}\quad
Q/D\text{ 裁决数值候选；}\quad
C\text{ 提高搜索效率。}
}
\]

这句话只描述各对象对后端的作用，不重新定义其所有权。A 与 B 共同组成唯一 J；Q/D 是预先冻结
的外部观察与执行域；C 始终是可忽略、可替换的 implementation-selection prior。Q/D 不进入 J，
搜索也不产生 `optimized_J` 或 numeric-permission sidecar。

## 2. 后端硬边界

### 2.1 唯一输入

后端明确拆成两个通用接口：

```text
CoreLower_t(J,C,target,compiler_state) -> target executable

SearchAndClose_t(J,C,target,Q,D_search,D_cert,opaque_oracles)
  -> target executable + backend closure receipt
```

`CoreLower_t` 只对 SCRC-accepted artifact 工作，不读取 Gate verdict、Q/D 或任何 Source realization。
它和每个 transformation 均禁止读取：

- Source；
- Replay；
- π；
- Gate1 diff；
- Gate2 hidden input/output；
- Source compiler fingerprint；
- kernel name、source path 或 provider identity；
- per-kernel RuleCard、历史 candidate 或旧实验结果。

允许进入 core lowerer 的只有：

- 唯一 J；
- 可选 C；
- target capability；
- backend 自身的版本化 generic pass library、cost model 与运行环境。

`SearchAndClose_t` 是一个跨所有 Kernel、所有 concrete MLIR states 共用的 compiler/autotuning
controller。它可以通过 opaque harness 执行 Source、Replay、`RefEval(J)` 与 target executable，并
调用冻结的 Q；但不得读取 Source/Replay 文本、π、kernel 名或历史结果来生成 transformation、修改
候选或选择 tolerance。workload adapter 只做 signature-driven 输入绑定与执行，不成为 per-kernel
lowering recipe。

因此“backend 使用 Q/D”不等于让 target pass 读取 Q：core compiler 仍是纯 J/C/target；Q/D 只在
外层统一 evaluator 中形成候选 feasibility 与最终 executable closure。

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

### 2.4 符号 extent valuation 不是新 J

当 J 显式声明符号 extent `x*` 时，production invocation 可以提供一个具体 valuation
`nu` 来 materialize 该执行实例。严格写法是 `M0 = Emit_v(J; nu)`；在当前 `c0` 已唯一
确定 `nu` 时，正文继续简写为 `Emit_v(J)`。相同 `(J,v,nu)` 必须得到相同 `M0`。

该 valuation 只能绑定 J 已经声明的符号维，因此：

- J digest 与 denotation 不变，但 invocation/materialized-MLIR digest 必须绑定 `nu`；
- `nu` 不是 C；C 仍只是 schedule prior，与 logical shape valuation 正交；
- workload profile 可以同步更新 Source harness 的 shape/stride/grid 与 C valuation，但不得
  创造 J 中没有的符号维；
- J literal 不可由 backend profile 改写；把 literal 变成符号维会修改 J digest，必须由
  frontend 产生新 artifact 并重走 Lift/TypeCheck/Gate1/Gate2；
- 评测必须报告 exact `j_digest + nu + workload_profile`，不得把一个符号轴上的放大
  写成所有 shape 的泛化证明。

例如，当前 Flash preprocess J 的载体为 `[1,x0,33,16] -> [1,x0,64]`；只有
`heads=x0` 可重新 valuation，`batch=1`/`sequence=33`/`head_dim=16`/`rounded=64` 都是
literal。当前 Softmax 和 Liger RMSNorm 的外层 rows 同样是 literal `2`；后端必须保留
这个边界，而不是为了跑大规模实验自行参数化。

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
- closed result-relation class；
- artifact 当前确实声明的 order、rounding、overflow、tie、NaN/Inf 或 determinism 事实；
- SCRC/π 所需的 typed realization authorization。

B 不需要枚举 backend 可能尝试的 reduction tree、FMA 或 target precision。当前 J 没有某个 numeric
mechanism permission，不妨碍 hard-safe candidate 进入 Q/D evaluator；字段缺失也不能被 target pass
解释成任意静态许可。

B 中只保存语义事实，不保存某个 backend transformation 的名字。正确形式是：

```text
no observable intermediate effect
no alias
duplicate updates are forbidden
output relation uses the closed tolerance class
```

而不是：

```text
fusion_allowed = true
online_to_twopass_allowed = true
use_tensor_core = true
gpu_strategy = warp
```

backend pass 从固定事实推导 hard applicability；它不能要求 Lift 直接选择 target recipe。可能改变
浮点观察值的 state 由 Q/D controller 裁决，而不是回写 B。

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
安全忽略，不能读取 Source/π 猜测，也不能据此改变 `ReachHard_t(J)`、Q/D 或 candidate feasibility。

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

对 schema/materializer 版本 v 与 J-declared extent valuation `nu`：

\[
M_0=\operatorname{Emit}_v(J;\nu).
\]

同一 J、同一 v 与同一 `nu` 必须得到同一 canonical Intent MLIR。`Emit_v`：

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

定义静态 hard-safety judgement：

\[
\Gamma_{J,t}\vdash M:\operatorname{HardImpl}_t(J).
\]

其含义是：M 是 target t 上完整编码 J，并满足所有不能由数值 tolerance 放宽的 compiler state。
它覆盖：

- typed value/state graph 与 observable carrier；
- bounds 与 memory safety；
- alias/effect；
- update/conflict；
- predicate/domain/output mapping；
- target ABI。

它不声称一个会改变浮点观察值的 transformation 已经对 D 之外的所有输入保持 J denotation。该类
候选的最终 admissibility 由第 11 节的 `QDAdmissible` 单独给出。

每个 transformation \(\rho\) 必须声明：

```text
matched op/region/graph
required A carriers
required B facts
target capability
current-state precondition
transformation result
preservation class
numeric observation class: exact-preserving | observation-changing
searchable parameters
```

应用规则是：

\[
\frac{
\Gamma_{J,t}\vdash M:\operatorname{HardImpl}_t(J)
\qquad
\operatorname{HardPre}_{\rho}(J,M,t)
}{
\Gamma_{J,t}\vdash\rho(M):\operatorname{HardImpl}_t(J)
}.
\]

这是版本化 compiler contract 的规范级 judgement 与条件组合义务，不是已经完成的机器检查形式
证明。某个实现或 release 只能对当时实际实现、登记并经过准入检查的 `Emit_v`、transformation
`rho` 与 `Lower_t` 作支持 claim；第 7 节列出的尚未实现 transformation 以及第 20 节 roadmap
不能被计入当前 `ReachHard_t(J)`。

若当前 J 没有足够事实证明某个 hard precondition，该 pass 对该 J 不适用；系统选择保守实现或
显式 unsupported，不能从 tolerance、Source 惯例或运行结果补足 memory/effect/ABI 前提。反之，
reduction tree、FMA、内部精度或 approximate math 只要不破坏 hard boundary，可以标记为
`observation-changing` 后进入统一 Q/D filter；它们不要求 J 预先枚举实现机制。

## 9. Reachable implementation space

定义由 J、target 与启用的通用 transformations 决定的 hard-safe MLIR compiler-state 空间：

\[
\operatorname{ReachHard}^{\mathrm{MLIR}}_t(M_0;\Gamma_{J,t})=
\left\{
M\mid
M_0\xRightarrow[\Gamma_{J,t}]{\rho_1,\ldots,\rho_k}M
\right\}.
\]

正文可将其简写为 `ReachHard_t(J)`，但其中的 M 始终是 Intent/lower-IR compiler state；J 只是
不可变参数，不是搜索状态。经过编译和 Q/D evaluator 后，定义：

\[
\operatorname{Feasible}_{t,Q,D}(J)=
\left\{M\in\operatorname{ReachHard}_t(J)\mid
\operatorname{QDAdmissible}_{t,Q,D}(M,J)\right\}.
\]

C 不改变该空间的语义边界，只定义搜索顺序：

\[
\operatorname{Order}_C\bigl(\operatorname{ReachHard}_t(J)\bigr).
\]

对有限搜索预算 b，定义实际访问集合：

\[
\operatorname{VisitedHard}_{t,b}(J;C)
\subseteq
\operatorname{ReachHard}_t(J).
\]

C 可以改变 `VisitedHard` 的顺序和有限预算下覆盖到的候选，但不改变 hard-safe candidate universe、
Q/D policy 或最终 `Feasible` 集合的定义。

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
保持 `HardImpl_t(J)`，且被选择的 executable 通过冻结 Q/D final closure。

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

“结构改变”不是任意重写。每种变化都必须作为通用 compiler transformation 存在，并具有明确的
A match、hard B preconditions、current-state/target legality、preservation class 与
`exact-preserving|observation-changing` 标记。

例如 online recurrence 与 ordinary two-pass reduction 不能伪装成普通 schedule 变化。它必须由
一条显式 structural transformation 产生，保留 hard semantics，并在可能改变数值观察时经过 Q/D
filter。Q/D 通过证明的是当前声明执行域内的候选可接受性，不把这条 transformation 升格成跨执行域
等价定理。

形式上可把已登记变换按实验目的记为 `R_sched`、`R_local` 与 `R_struct`。这三个集合都是同一个
transformation registry 的标签；candidate identity 始终来自具体 MLIR state、trace 与 schedule，
而不是来自预先声明的实现族。

## 11. Numeric correctness boundary

backend 必须分清四类信息：

```text
A typed graph:
  实际 operand/result/cast dtype 与显式运算

B hard/result boundary:
  domain、predicate、effect、update、output ABI 与 exact/tolerance comparison class

Q:
  预先冻结的具体 comparator policy

D_search / D_cert:
  候选过滤域与最终 winner 的独立认证域
```

core lowerer 与 target pass 不读取 Q/D。外层 `SearchAndClose_t` 使用 Q/D，但不能修改 J、选择更宽
tolerance 或从测试结果补 memory/effect/ABI 前提。对于以下可能改变浮点观察值的 compiler choices：

- 改变 reduction tree；
- 使用 approximate transcendental；
- 启用 FMA/contraction；
- 改变 accumulator dtype；
- 在 contraction 上选择 IEEE/TF32/其他内部精度；
- 改变 floating atomic accumulation order。

本文不要求先为每一种机制向 J 增加 `allow_*` 或 reduction-order permission。只要 transformation
保持 hard boundary、输出 ABI 与可执行性，它可以作为 `observation-changing` concrete MLIR state
进入统一 evaluator。其 admissibility 定义为：

\[
\operatorname{QDAdmissible}_{t,Q,T}(M,J)
\iff
\bigwedge_{x\in T}
\operatorname{Close}_Q(\operatorname{Exec}(\operatorname{Lower}_t(M),x),
\operatorname{RefEval}(J,x)).
\]

上式定义 realized finite set T 上的闭合。若 D 是隐藏抽样分布，则 `T~D^m`，只使用 SCRC 文档定义
的 disagreement-mass 概率表述。不得把二者写成全输入域 theorem。

搜索期可以在预先声明的 `D_search` 上做低成本 `target--J` filter。最终 winner 必须在独立、隐藏的
`D_cert` 上固定后再执行直接闭合：

```text
Close_Q(Target, Source)
and Close_Q(Target, Replay)
and Close_Q(Target, RefEval(J))
```

三条边分别计算，不能依靠 tolerance 传递性。`D_cert` 结果不得回流 transformation 生成或 tolerance；
若固定 winner 失败，本次 backend certification 失败，不能在同一 hidden sample 上依次试到通过。
下一候选必须使用预先声明的独立 certification protocol。

这不是“每个 Kernel 一套认证”。系统只有一个 signature-driven evaluator：所有 Kernel 共用同一
compile/execute/compare 状态机、同一 Q protocol 和同一 receipt schema。可以对所有候选统一执行，
也可以让 `exact-preserving` transformation 只做静态检查和最终 winner closure；只有最终 executable
获得正式 backend closure receipt，其余状态只是 search log。

这里的“只读无别名”也必须有完整性依据。若当前 artifact version 只列出已知 alias relation，却没有
规定“关系缺失即 no-alias”或显式声明 relation set complete，则 `boundary.aliases=[]`、
`signature.alias_of=null` 不能单独推出全参数 no-alias。`ContractView` 的 capability 必须携带其 exact J
field、schema completeness rule 与推导版本；无法证明 completeness 时返回 `missing_contract`，不能把
具体 benchmark 恰好使用 disjoint buffers 当作语义证据。

本阶段不等待 B schema 扩展，也不修改任何已闭合 J。优先实现并实验：

- producer/reduction/epilogue 的 effect-safe fusion；
- intermediate materialization/copy elimination；
- RVV whole-graph Linalg fusion 与后续 vectorization；
- local scalar capture/load motion/predicate specialization；
- reduction sequential/subgroup/workgroup/hierarchical tree states；
- FP32 FMA contraction；
- contraction 的 IEEE/TF32 等 target-supported precision states；
- approximate transcendental 和 accumulator precision states。

后四类都必须标记为 `observation-changing` 并经 Q/D，而不能借 `float_nondeterministic`、Source
惯例或 target 默认值静态宣称全域等价。若未来项目目标改成脱离 D/Q 的静态 deployment theorem，
可以另行设计版本化 numeric relations；这不是当前 backend 路线，也不阻塞当前 18 个 J 的消费。

现有 `exact|tolerance` 仍有约束力：`exact` 输出不能由 controller 私自改成 tolerance；`tolerance`
只选择既定 Q class，具体阈值继续由外部 Q 独占。缺少有效 Q、空 D、非法 comparator 或未覆盖的
observable output 一律 fail closed。floating atomic 候选还必须由 J 的 update/effect hard contract
覆盖，并把重复执行次数与调度 seed 纳入 D；当前主线不支持一般 `one_of_updates` 集合值搜索，因此
该类 candidate 在相应 Q protocol 实现前保持关闭。

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

`reduce.assoc` 是当前 J 中已有 reduction classification 的物理投影，不是 target schedule，也不要求
Lift 枚举 target reduction tree。core pass 不能把它或 `tolerance` 当成全域等价 permission；具体
tree state 保持 hard boundary 后由 Q/D controller 裁决。只有未来追求 D 外静态 theorem 时才需要
另行演进更细的 closed relation，当前 backend 不等待该扩展。

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
\Gamma_{J,t}\vdash M:\operatorname{HardImpl}_t(J),
\]

则 target lowerer 必须保持 hard boundary 并产生可执行 candidate；若 M 是 `observation-changing`，
该事实本身不宣称数值候选已经可接受。给定有限预算 b，性能选择域为：

\[
\operatorname{Measured}_{t,b,Q,D_{search}}(J;C)=
\operatorname{VisitedHard}_{t,b}(J;C)\cap
\operatorname{Feasible}_{t,Q,D_{search}}(J).
\]

搜索期 winner 为：

\[
M^*_{b,C}=\arg\min_{M\in\operatorname{Measured}_{t,b,Q,D_{search}}(J;C)}
\operatorname{Cost}_t\bigl(\operatorname{Lower}_t(M)\bigr).
\]

若定义域为空则本轮无候选。固定 `M*` 随后必须通过 `D_cert` final closure 才能发布。Cost 可以
来自静态模型、profiling、benchmark 或混合策略；性能值不能覆盖 hard verifier 或 Q/D failure。

## 16. 条件组合保持命题

假设：

1. materializer 正确编码 J 的 hard boundary：

\[
\Gamma_{J,t}\vdash M_0:\operatorname{HardImpl}_t(J);
\]

2. 每个启用 transformation 保持 hard judgement：

\[
\Gamma_{J,t}\vdash M:\operatorname{HardImpl}_t(J)
\land \operatorname{HardPre}_{\rho}(J,M,t)
\Rightarrow
\Gamma_{J,t}\vdash\rho(M):\operatorname{HardImpl}_t(J);
\]

3. target lowerer 保持 hard boundary 且 final winner 通过冻结 Q/D：

\[
\operatorname{HardSafe}(\operatorname{Lower}_t(M),J)
\land
\operatorname{BackendClose}_{Q,D_{cert}}(M,J).
\]

则：

\[
M\in\operatorname{ReachHard}_t(J)
\land \operatorname{BackendClose}_{Q,D_{cert}}(M,J)
\Rightarrow
\operatorname{Lower}_t(M)\text{ 在记录的 target/Q/D 范围内可发布}.
\]

该命题的 TCB 至少包括：`Emit_v` 及其 B projection、hard verifier、target-capability query、实际
transformation、target lowerer、Q implementation、D protocol、opaque execution harness 与 final
receipt writer。它明确是 execution-domain closure，不冒充跨 D 的 compiler-pass 等价证明。

## 17. Backend conformance 的位置

backend executable closure 统一运行：

```text
search filter:
  Close_Q(Target, RefEval(J)) on D_search

final winner:
  Close_Q(Target, Source)
  and Close_Q(Target, Replay)
  and Close_Q(Target, RefEval(J)) on D_cert
```

它：

- 是一个由所有 Kernel 复用的 backend evaluator，不是 per-kernel mechanism；
- 是数值 observation-changing candidate 的执行域 feasibility oracle；
- 只给最终 winner 产生正式 backend closure receipt；
- 不修改 SCRC verdict、J digest 或 Gate2 receipt；
- 不授权 memory/effect/ABI 错误，也不允许 backend 调整 Q。

它不是 SCRC 的第四条 Gate2 边，也不声称 candidate 在 D 之外 contract-proven。CI、release preflight、
debug 和 regression 可以复用同一机制，但不能用更弱的 smoke set 冒充正式 `D_cert`。

final receipt 至少绑定：J/C digest、target 与 compiler/toolchain version、实际 `c0`、Q/protocol、
D 定义和 realized challenge digest、完整 MLIR transformation trace、schedule、lowered object/binary
digest、所有 observable outputs 的三条 direct-edge verdict，以及独立 performance record。receipt
缺任一 correctness leg 时不得发布为 Q/D-closed executable。

## 18. 实验与消融设计

实验使用四个主要层级。这里的“固定结构”由 canonical structural transformation trace 决定，不由
`family_id` 或 named emitter 决定：

```text
E0  Source frontend implementation
E1  canonical deterministic target lowering + default schedule
E2  fixed structural + fixed numeric trace, schedule search
E3  hard-safe structural/numeric lowering-state + independent schedule search,
    filtered by the same Q/D, with C
```

额外消融：

```text
E4  same ReachHard/Q/D and budget as E3, with Erase_C
```

对应问题：

- `E1 vs E0`：semantic interface 与 canonical lowering 的基础质量；
- `E2 vs E1`：只调 schedule 的收益；
- `E3 vs E2`：结构性与数值 lowering-state expansion 的额外收益；
- `E3 vs E4`：C 作为搜索先验的作用，而不是 correctness 作用。

另设与 E0--E4 正交的 local-B 配对实验：

```text
L0  fixed hard-safe structural state set, registered B-conditioned local transforms disabled
L1  same state set/Q/D/budget, registered B-conditioned local transforms enabled
```

L1/L0 保留完整 J=A+B 和 fail-closed `ContractView`；关闭的只是额外 local transformation，不是 B
语义。rank-0 capture、singleton capture、load motion 与 predicate specialization 的收益只写入该轴，
不得混写成 E3/E2 的结构路径收益。

E3 内再做机制消融，均复用同一个通用 search controller，不建立 named family：

```text
N0  E3, observation-changing numeric states disabled
N1  N0 + reduction-tree states
N2  N1 + FMA/contraction states
N3  N2 + target-supported precision/approximation states
```

N0--N3 的 hard verifier、Q、D、target pipeline 与总搜索预算保持一致；某个 Kernel 不匹配某类
transformation 时记录 `not_matched`，不得从 aggregate 中删除。

结构机制还必须单独做“瓶颈切断”对照，避免 E3/E2 的平均数掩盖 J 打开的最大结构机会：

```text
G0  best admitted materialized state：保留 producer/reduction/epilogue 中间值与 dispatch 边界
G1  best admitted whole-graph state：消除对应 materialization/dispatch，融合后再独立调 schedule
```

G0/G1 必须来自同一 J、同一 Q/D、同一 target pipeline 和同一份候选测量回执；两侧都独立搜索
schedule，只在真实 post-pass MLIR 同时观察到 `kernel_count>1` 的 materialized state 与
`kernel_count=1` 的 whole-graph state 时计入 applicable subset。该对照回答的不是“多开一个 pass
是否更快”，而是 J 的 typed producer--consumer graph 与 B hard boundary 能否合法删除一次或多次
全张量物化、内存往返和 dispatch。它应当作为结构创新的首要机制结果；L0/L1 的 scalar capture、
load motion 等局部收益只作次级证据。G0/G1 是 E3 同一候选回执上的派生对照，不是新的搜索 arm，
因此不修改 v2 machine spec 或其 digest，也不为它另造 candidate universe。

这些实验臂不是文档中的自由文本约定。实现必须从
`spec/lowering/backend-search-ablation-v2.json` 读取 E0--E4、L0/L1、N0--N3 的开关、预算类别、
concrete-state identity 与计时合同，
校验 protocol、字段全集和类型，并把 canonical spec digest 写入每份实验报告。当前 v2 spec digest 为
`28c2117f35e1360fa4f9b4c85f4af27edf343630d0d5be000eef0b3b4e9c8f9a`。runner 不得在代码中重新定义
同名实验臂；spec 缺失、未知 arm、重复 arm 或非布尔开关均须 fail closed。这样论文中的 E2/E3、
C 消融和数值机制消融都与实际执行的 candidate universe 绑定，而不是事后按 winner 改口径。

v2 中一个预算单位是一个不同的 canonical post-pass MLIR state；相同 MLIR digest 的重复编译仍留在
audit log，但不消耗预算。无法形成 state 的不同 exact candidate compile failure 消耗一个预算单位。
finalist 不混用初次排序样本，而按正反交替 traversal 做 fresh round-robin remeasurement；最终 Source
对比使用等数量、顺序交替的 Source/target paired batches。正式确认前，Source 与 target prepared
thunk 按相同声明时长做 round-robin untimed warm-up；计时只覆盖 prepared launch/graph，allocation、
cache eviction 与 correctness synchronization 位于 timed closure 之外。

### 18.1 TileLang-aligned 主评测、缓存状态与 Roofline

E0--E4 只定义 candidate-space 因果轴，不足以定义一个可复现的高性能 Kernel
实验。缓存状态与 Roofline 另由 `spec/lowering/backend-performance-measurement-v1.json`
冻结；当前 canonical JSON content digest（不是文件原始字节 SHA-256）为
`7797827a12b0563084dcc8c838c3b3017b84ae9381c9ad87e6379210b72c4fcf`，bundle/report 必须绑定当时的
canonical spec digest。评测参考 TileLang/Triton 的
kernel benchmark 实践：代码、module、page allocation 与 GPU 频率先 warm-up，性能主数字则用
CUDA event 计时，并在每次 single-shot 前在 timed interval 之外管理/驱逐 L2。因此必须区分：

- `warm-up`：所有正式计时的必要准备，不是一个性能结果组；每个候选排序前单独稳定，最终
  Source/target 用相同时间预算 round-robin 稳定，并把实际轮数写入 receipt；
- `warm-cache`：同一 address 反复 graph/launch 的调试或 deployment-specific 结果，不进入主表 aggregate；
- `L2-controlled steady state`：预热后、每次计时前驱逐 L2、驱逐本身不计时；这是 CUDA
  Kernel 主表的 publication estimand；当前 runner 为保持 protocol 兼容仍将其记为 `cold-data`；
- `streaming`：轮转互不相交 replica 的压力/敏感性实验，只在 target 真实实现时放入附录。

机器 spec 仍分别记录三种数据状态，以便审计：

```text
warm:
  同一 prepared address 上重复执行；CUDA 可用 graph/event，报告 steady-state latency

streaming:
  轮转同一 certified invocation 的互不相交 replicas；实际触及 working set 必须严格大于 LLC

cold-data:
  每次 single-shot 前用严格大于 LLC/L2 的 target-side working set 驱逐；
  eviction、allocation、binding 与 synchronization 不进入 timed interval
```

`cold-data` 只描述数据缓存状态，不包含进程启动、module load、JIT 或首次 page allocation。
warm-cache graph per-invocation latency 与 L2-controlled single-shot event latency 是不同 estimand，禁止
放进同一个几何平均。当前 RVV runner 实现三种 regime；CUDA 实现 warm-cache 与严格
L2 eviction 的 `cold-data`。CUDA 不把无法真实转到大于 L2 的小 tensor 假写成
streaming。正式主表报告 p50/中位 latency 与全部 paired batch；warm-cache、raw distribution、
streaming/cold 敏感性与 compile time 可置于附录或 artifact，不为每个单元格子重复堆叠。

对 materialized J 定义逻辑算术强度：

\[
I_J=\frac{\operatorname{FLOP}(J)}
{\operatorname{BoundaryInputBytes}(J)+\operatorname{BoundaryOutputBytes}(J)}.
\]

分子只计算 J typed graph 中可审计的 floating operations；transcendental、comparison、conversion、
select 与 dynamic control 单独记数，不使用任意“等效 FLOP”权重。分母是 certified boundary 的逻辑
最小流量，不是 DRAM transaction。target 端另实测 sustained copy bandwidth 与 TF32-disabled FP32
compute ceiling；有权限时再把 H100 的 DRAM/L2 counters 作为独立事实加入，不能用 logical bytes
冒充 hardware traffic。

低算术强度只决定 asymptotic Roofline 在 bandwidth 一侧，不能单独证明当前运行已撞到 memory wall。
当实测 latency 快于校准的 logical DRAM Roofline lower bound（容差后仍小于 `0.95x`）时，
这只能说明 cache reuse 或 logical-traffic/ceiling 不匹配，必须禁止 DRAM memory-wall claim。
当 latency 超过 lower bound `2x` 时，不再称为 roof-approaching；超过 `10x` 时还必须显式
写 `latency_or_unmodeled_work_dominated`。只有 J cost complete、不含未建模 SFU、低算术强度且
latency 位于 `[0.95,2.0]x` logical lower bound 内时，logical Roofline 才能单独支撑 memory-wall
claim。当前 closed-18 的正式 closure case
边界数据只有约 1--8 KiB，所以它们首先是 correctness/small-latency workload；不能因 Softmax、Norm
或 copy 的名字就写成 DRAM-bound。memory-wall 机制实验使用同一个符号 J 的独立大实例：首个冻结
profile 将 `masked_copy` 的 `x0` 实例化为 16,777,216，Source 的 N/grid、RefJ 与 target 同步绑定，
形成 128 MiB boundary。它不改变 J 或 compiler path，但把 D 扩展到同时超过 RTX 5090D 96 MiB L2
与 H100 50 MiB L2 的实际 workload。只有该类 large-shape/硬件-counter 证据才可支撑 bandwidth claim。

### 18.2 C 的同宇宙、同预算 target-side 配对

C-prior 与 Erase-C 不能只在 exporter 中各打印一份 requested order，而只编译/测量其中一份。portable
bundle 必须访问两条 order，分别按“不同 post-pass MLIR state 或 exact failure”消耗相同预算，运输
二者 concrete-state union，并用 MLIR digest 折叠重复 state。target runner 在同一次 candidate
correctness/timing receipt 上分别求两条 order 的 best；C gain 定义为
`Erase-C best latency / C-prior best latency`。这样 C 只改变有限预算下的访问顺序，候选宇宙、hard
admission、Q/D、实际 executable 和 profiler 均相同。

`Retile(C,target)` 的 score 必须建模已注册 lowering 的 schedule transition，而不是只算参数数值距离。
例如 tiny-row elementwise 在 `N/T>32` 时进入 flat grid-stride state；C 排序应优先仍跨过该边界的最大
合法线程数，使 target 获得足够 block-level parallelism。该逻辑仍只排序既有 candidate，不进入
transformation admission，也不改变 J/C、candidate ID 或 concrete-state universe。

报告指标至少包括：

- absolute latency/throughput；
- 相对 Source 与 vendor baseline 的性能；
- candidate 数、compile 数与测量数；
- hard rejection、`D_search` Q rejection 与 `D_cert` final result；
- correctness execution cost 与 performance measurement cost，二者分开统计；
- time-to-best；
- transformation match/admission/rejection reason 与 trace；
- fixed-structure schedule 与 structural-path expansion 的增量；
- C prior 对搜索效率的影响；
- compile failure 与 backend conformance failure；
- pass/library 的跨 Kernel、跨 frontend、跨 target 复用。

E2/E3 必须共享同一 J、target、compiler revision、terminal code generator、Q/D policy 与 profiler。
E3 的 hard-safe candidate set 必须包含 E2 的全部候选；equal-total-budget 访问顺序先走与 E2 相同、
同样由 C 排序的 fixed-structure schedule spine，再用剩余预算访问结构/数值扩展状态。这样
`E3 vs E2` 不会混入“扩大搜索后反而忘记固定路径强 schedule”的伪回退，同时仍把额外路径的编译、
失败和测量计入同一总预算。所有性能比较只在通过相同 `D_search` filter 的候选间进行，最终 winner
使用相同 `D_cert` protocol。报告同时给出：

- equal-total-budget 的 anytime/time-to-best 曲线；
- 小候选空间 exhaustive/oracle best；
- 固定 latency map 下 E3/E4 的 full-budget convergence；
- all-kernel 与 transformation-applicable subset，禁止只挑 winner。

微秒级 Kernel 不得用两个独立进程的 best latency 直接归因于 E3。primary marginal gain 必须在同一
E3 candidate timing receipt 中，从同一批 fresh finalist 样本分别选择 schedule-spine best 与
expanded best；独立 E2/E3 报告仍保留 absolute latency、Source 配对和 cross-report ratio，但后者只作
复现实验漂移诊断。若两侧选择同一 executable，same-receipt 增量自然为 `1.0x`，不能把时钟噪声写成
lowering-state 收益。

不同结构路径必然经过不同的高层 transformation trace；“同一 code generator”指共享同一 backend
revision 和 terminal target pipeline，不要求结构分支使用完全相同的高层 rewrite。

### 18.3 closed-18 横向矩阵，不以代表 Kernel 代替全量实验

主实验的基本单位不是“选几个算子讲故事”，而是一个显式 coverage cell：

```text
(frozen j_digest,
 target,
 exact J-declared extent valuation nu,
 workload profile,
 experiment arm,
 compiler/toolchain digest,
 Q/D protocol)
```

当前主矩阵必须从 `corpus/production-jc-v1` 的 **18/18** 条目出发，并在 RTX 5090D、H100
与 RVV4 上横向扩展。对每个 target：

- 18/18 都要 materialize、compile 并给出 Target--RefJ correctness 或精确的 fail-closed 状态；
- CUDA/H100 在 Source 可执行时，18/18 都进入 Source--RefJ 与 Source--Target 主性能表；
- E0--E4、L0/L1、N0--N3 对全 18 条目执行；不适用的 transformation 记为 `not_matched`，
  不从 all-corpus aggregate 中删除；
- 每个 J 除 closed valuation 外，还要运行所有已登记且只绑定 J-declared symbols 的
  medium/large valuation；当前 18/18 都至少有一个可绑定 symbol，但 `rows=2`、Flash 的
  batch/sequence/head_dim 等 literal 必须原样保留；未来完全没有符号轴的 J 才记
  `literal_shape_only`，不伪造大 shape；
- 报告同时给出 all-corpus、structure-applicable subset 与每 Kernel 数字，禁止只保留 winner
  或只报加速的 case；
- target 缺 Source/Replay、J 缺某符号轴、pass 不匹配、compile failure 与 Q/D failure 都是
  矩阵内的显式状态，不是被静默跳过的空格。

只有开销极高、用于解释机制而非建立性能事实的证据，例如 Nsight/CUPTI 详细 counters、
Roofline 分解和单变量 instruction-level 剖析，可以在每类结构上选 1--2 个代表 valuation。
这只是降低“解释证据”成本，不缩小 18-J 主性能、correctness、E2/E3 或 C 消融的横向覆盖。

## 19. 版本与演进规则

后端可以快速演进，但每次演进必须保持对象边界清楚：

- 新 transformation/pass contract 只扩展 backend library，不改变 J；
- 新 schedule parameter 只进入 compiler internal state或现有 C 的安全消费；
- reduction tree、FMA、内部精度与 approximate candidate 的 Q/D 搜索不要求新增 B relation；
- 只有未来改变 J 自身 denotation/比较类别或追求 D 外静态 theorem 时，新 B relation 才必须有真实
  consumer、closed schema 与完整版本升级；
- 修改 A denotation、B relation 或 materializer interpretation 时，必须升级对应 schema/materializer/
  evaluator 版本；
- 旧 artifact 不被静默解释成新语义；
- 不使用 hash、所谓冻结门或 kernel allowlist 决定合法性。

## 20. 实现阶段

项目执行上，B0--B4 在目标 Kernel 的 SCRC closure 与 accepted J corpus 形成后再独立展开；不要求
每闭合一个 Kernel 就紧邻实现其 backend。SCRC 阶段若真实证据暴露 J/TypeCheck/Replay/RefEval 的
通用缺口，可以修正对应 A/B 语义合同，但不提前混入本节的 IntentOp ownership/search 重构。

### 20.1 当前闭合与 production intake 基线（2026-07-26）

当前严格 SCRC 独立覆盖是 **18/28**，J-search 分支已将 18 项全部 content-addressed 冻结到
`corpus/production-jc-v1` 并实际消费：DYT、Flash backward preprocess、Liger fused-add RMSNorm、
GeGLU、JSD、Liger LayerNorm、Liger RMSNorm、PolyNorm、Softmax、SwiGLU、TVD、xFormers RMSNorm，
以及下列新增六项：

| 已导入 case | frozen snapshot digest | 当前真实后端 consumer |
|---|---|---|
| xFormers `_rms_norm_add_kernel` | `e8e5451fa837443024a435b34b125f5c9f2261bdaf178842f80898875cf727be` | inout/post-state、reduction、epilogue、多输出 |
| Liger Sparsemax | `ca974aed6b5954b197a063f0e96d62652d569b5728608818adcdbfbc3cb11fda` | access-local guard、scan + 双 reduction、logical select/and、generic scan baseline |
| Liger KLDiv | `b39a3df6d64782be475b5b459f6b7dc01efdb3356fa85abe22a75334f296df7a` | loop scan/carry full-axis step、log/reduction、CUDA/RVV state search |
| FlashAttention `_layer_norm_fwd_1pass_kernel` | `76ffde5e8eb4e53d0fca5afad8bb1d44ad4ba3f6c9df764dc67b21aa95402480` | runtime value-if、双 reduction、unused ABI preservation、多输出 half epilogue |
| SCRC canary `masked_copy` | `10d67ee0b4f061e1ce28b3e6b1ebfc371be2825692671c927e80b8e743e4ad49` | direct input/output binding 确定性物化为 `intent.map_unary identity`；generic C++ MLIR 单 cubin；exact Q/D closure |
| SCRC canary `masked_row_square_sum` | `31208517fc91dca13e48573198090ca46c6905e2c8d814109f0149cd72d9c892` | `map_binary mul -> reduce sum`；row-reduction topology search；tolerance Q/D closure |

当前 18 项都没有 typed rank-2/3 contraction；名称含 `do_o_dot` 的 Flash preprocess 仍是 FP32
elementwise multiply + row sum。因此这批的首要数值搜索轴是 reduction tree、FP32 FMA、fusion 与
materialization，不是 TF32。TF32 只在未来闭合的 `tl.dot`/matmul/完整 Attention contraction 上按
target capability 枚举，并与 IEEE state 使用同一 Q/D 做配对。

因此：

```text
SCRC-closed corpus         = 18
frozen production catalog = 18
verified intake queue      = 0
```

不得把历史 12/16-kernel 性能数字重标为 18-kernel 结果。导入过程只复制 exact accepted artifact、C、
Source、Replay 与 closure receipt 并交叉核验 digest，没有重新 Lift、修 J 或按 case name 选择 emitter。后续新 closure
继续使用同一 append-only intake 工具；同名不同 digest 必须 fail closed，未冻结 `/tmp` 路径不能成为
长期 truth。

当前实现基线为：18/18 可从冻结 J deterministic materialize；18/18 可经通用
Intent→Linalg→CUDA cubin；17/18 至少有一个注册 C++ whole-graph path，`masked_copy` 由 generic
C++ MLIR path 生成单 cubin。RTX 5090D 的 v2 equal-budget=32 正式 E2/E3 已覆盖全 18 项：两臂
各 18/18 完成 opaque Replay、互斥 `D_search/D_cert` 与 held-out Target–Source/Replay/RefJ 三条直接
闭合；E3/E2 几何平均为 **1.1302x**，E3/Source 为 **1.0928x**，17/18 达到或超过 Source，
15/18 的 E3/E2 超过 1%。这组结果替代此前不得重标的 closed-12/16 aggregate。

第一性原理的结构结果来自同一正式回执内的 G0/G1：6 个同时产生 materialized multi-kernel 与
whole-graph MLIR state 的 J，消除中间物化与 dispatch 的几何平均增益为 **17.1258x**，其中
GeGLU 10.7662x、LayerNorm 19.8063x、masked row square-sum 39.7813x、PolyNorm 45.0139x、
Softmax 15.4004x、SwiGLU 4.2902x。它证明 J 的 typed graph 与 B hard boundary 能打开编译图级
搜索空间；不能把该结果归因于 kernel 名、Python emitter 或局部 peephole。

数值阶梯进一步给出因果分解：N0→N1 的 reduction-tree state 在全 18 项上为 **1.0409x**、12 个
适用 J 上为 **1.0622x**；N1→N2 的 FMA 只有 **1.0003x**（适用子集 1.0019x），必须作为负结果；
N2→N3 的 precision/approximation state 在全体为 **1.0676x**、2 个适用 J 上为 **1.1271x**。
该 ladder 自身只有 `D_search`，正式 winner closure 由上述 E3 报告承担。budget=8 的 role-aware C
ordering 相对 Erase-C 提高低预算 best latency **1.0195x**；C 不改变候选空间。正式 L0/L1 则在
10 个真实 post-pass MLIR 观察到 local-B transformation 的 J 上为 **1.0295x**，全 18 项为
**1.0162x**；这只是次级局部收益。

跨 target 结果使用同一 18-J catalog：H100 E3/E2 为 **1.1196x**，E3/Source 为 **1.0539x**，
18/18 通过 Source–RefJ 与 Target–RefJ，但缺 Replay 与 held-out 三边，因此不写成 formal backend
closure；RVV4 的真实 MLIR materialized/fused 搜索 18/18 Target–RefJ correct，E3/E2 为
**1.1817x**、15/18 winner 使用 B-admitted transformation，同样因缺 Source/Replay `D_cert` 只作为
同 J/target 的内部因果证据。完整压缩报告与 SHA-256 固化在
`doc/results/production-backend-closed-18/`。

上述数字首先是 **closed valuation 上的全量消费与因果基线**。它们不等于已完成
production-shape 论文主表：多数原始 closure workload 仅有 KiB 级 boundary，warm-cache 小延迟
不能用来主张 DRAM wall。新的 horizontal matrix 必须按第 18.1--18.3 节重新运行
L2-controlled、J-symbolic production valuations，并将 literal-only 格子显式保留。不得把旧的
closed-valuation aggregate 重命名为新主实验。

横向 valuation intake 已先完成：一个 target-neutral、versioned、fail-closed 的
`HarnessBindingSpec` 对 18/18 同步绑定 buffer axes、stride/scalar、semantic constexpr、C 与 grid；
它不进入 compiler path selection。`symbolic-medium-v1` 与 `symbolic-large-v1` 均逐项保持原
`j_digest` 并通过 `build_workload_jc_suite`，随后两档各 18/18 在 `sm_120a` 真实形成 canonical
post-pass MLIR 与 cubin。
其中 `masked_copy` 的 expanded valuation 因 Source 缺 `program_id` 明确标为
`Target--RefJ-only`。这些是 implementation/coverage smoke，不提前充当正式 latency 表。

第一项 horizontal lowering 已落地为通用 tiled-frontier multi-reduce concrete state：同一依赖层的
独立规约共享 feature-tile traversal，依赖层从 workgroup-cached frontier 继续遍历，各 partial
先在 warp 内 shuffle，再以每 warp 一个 compact shared partial、每依赖层一个 barrier 完成 block fold；
epilogue 也从 frontier 取值。large profile 上它按 A/ContractView 真实准入 7/18 项；其余格子
fail closed 保留。7 项均已形成 true post-pass MLIR/cubin 并通过 Target--RefJ，但当前 5090D
搜索没有一项由该状态胜出；这是提示继续改进 applicability/topology 的负结果，不伪装成收益。

RTX 5090D 的 `symbolic-large-v1` 横向主回执现已冻结：E2/E3 均 18/18 成功，17 项有 exact Source
baseline，`masked_copy` 为 Target--RefJ-only。E3 在同一候选回执中强制纳入 fixed-structure
schedule spine，并以相同 total budget=32、同一 fresh finalist tournament 比较 spine best 与
expanded best，因此 primary E3/E2 为 **1.0321337x**；独立进程 cross-report ratio **1.0062950x**
仅作诊断。E2/Source 为 **1.0857808x**，E3/Source 为 **1.1061843x**，C-prior/Erase-C 为
**1.0036794x**，primary winner 为 7/18。完整摘要、候选回执、spec/compiler digest 与 SHA-256
固化在 `doc/results/production-backend-symbolic-large-18-sm120a/`。该结果替换“等待正式计时”的
旧状态，但不覆盖 H100/RVV，也不与 closed-valuation aggregate 混算。

### 20.2 当前横向里程碑与下一 target

当前 18 个 J 按 A 结构覆盖四类横向路径：

| 结构类 | frozen J | 当前共性瓶颈 | 下一批真实 MLIR state |
|---|---|---|---|
| dense/masked elementwise | DYT、GeGLU、SwiGLU、masked copy | tail/vector width、grid 遍历、端到端内存合并 | masked vec2/vec4 与 target-derived grid-stride/vector layout |
| single reduction + epilogue | Flash preprocess、fused-add RMSNorm、KLDiv、Liger RMSNorm、masked row square-sum、xFormers RMSNorm、xFormers RMSNorm-add | producer 重读、中间物化、单一 reduction topology | producer/reduction/epilogue 单趟 state，subgroup/workgroup/multi-stage 独立搜索 |
| multiple reductions + epilogue | Flash one-pass LayerNorm、JSD、LayerNorm、Softmax | compact tiled frontier 已正确准入但尚无 winner；shared frontier/traversal 成本仍未打破现有 register/workgroup 路径 | subgroup/frontier/multi-stage topology 与 schedule 独立搜索，扩展 dtype/表达式并保留不匹配/拒绝回执 |
| scan/effect/contraction/gather | Sparsemax、TVD、PolyNorm | scan/update/control 或 contraction/gather 落到 generic/materialized path | registered prefix/segmented-scan、effect-safe update 与 contraction/gather tiling states |

实施顺序以“同一 compiler mechanism 横向覆盖一类 J”为原则，不以单个 benchmark 得分为目标：

1. **全量 coverage receipt 先行。** 任何新 pass 提交都要对 18/18 运行 structural match、hard
   admission、post-pass digest 与 correctness；`not_matched` 也是可审计结果。
2. **CUDA/5090D 与 H100 共用高层 transformation。** 先完成 tail-safe vector load/store、多 reduction
   单趟累积与 producer/epilogue fusion，再补 scan/effect/contraction 状态；只由 target capability
   选择 NVVM/MMA 终端能力，不分裂出 kernel-name recipe。
3. **RVV 与 CUDA 同步横向扩展。** workload-profile/symbolic valuation 与 per-kernel receipt 已由
   两个 portable exporter 共用；下一步实现 dynamic-VL strip mining、SEW/LMUL 状态、vector
   reduction/scan 和可审计的 multicore state，不把工作停在静态 tile 与 whole-graph fusion。
4. **搜索与 C 做横向收敛。** 每类 transformation 都进入同一 candidate-state universe；C 只重排
   全量状态。主表对 18/18 运行 E2/E3/E4，同时给 all-corpus 与 applicable subset。
5. **性能验收不代替 correctness。** 所有候选仍先 Target--RefJ；有 Source/Replay 的 target 再完成
   held-out direct edges。主 latency 用 L2-controlled event protocol，Nsight/Roofline 只解释每类机制，
   不代替全 18 表。

禁止为了某一 Kernel 的数字在 Python 中增加 target emission、按 case name 切换 compiler pass、
改写 frozen J literal 或只运行一个适用子集后声称全量提升。当前仅有一个 J 命中的结构
也可以实现，但 matcher、pass contract、negative test 与 receipt 必须仍然 source-neutral，并在
18-J matrix 中呈现其他 17 项的 `not_matched`。

### 阶段 B0：IntentOp/ABC ownership 对齐

**当前状态：核心入口已落地，继续以增量方式补齐 consumer。**

- 清理 Op.td target recipe；
- 标记 A semantics、B-derived attrs 与 C-derived annotations；
- 解除 `intent.loop.tile_hint` 对 canonical Emit、LoopOp denotation/verifier 与通用 loop lowering 的旧依赖；
- 审计每个 B 字段的 consumer；
- 保证 structured ops 不在结构 transformation 完成前过早消失。

### 阶段 B1：后端合同入口

**当前状态：已实现。** `materialize_j`、`ContractView`、`Attach_C/Erase_C`、统一
`SearchAndCloseController`、content-addressed 18-kernel catalog 与 fail-closed rejection 已进入
J-search 分支；旧 artifact 在缺少某个高级 consumer 前提时保留 generic/conservative path，不被静默
升级。

- 固化 `Emit_v(J)`；
- 实现独立 `Attach_C(M0,C,t)` 与 `Erase_C` invariant；
- 建立 `ContractView(J,target)`；
- 让 materializer 对所有影响 hard semantics、safety 与 ABI 的 backend-relevant B facts 完整消费
  或 fail closed；
- 扩展现有 generic compiler transformation registration/admission API，只以 typed A pattern、
  B precondition、current state 与 target capability 为 key；不另建 family registry；
- 建立唯一 `SearchAndClose_t`：signature-driven harness、冻结 Q、`D_search/D_cert`、candidate cache、
  direct-edge comparator 与 backend closure receipt；不按 kernel 名注册数值机制。

### 阶段 B2：第一批 lowering-path expansion

**当前状态：第一批已实现并完成 18-J、三 target 的真实消融。** CUDA
row/elementwise/padded/generic-Linalg lowering 与 RVV materialized/whole-graph states 均由真实
post-pass MLIR receipt 驱动。主收益来自 materialization/dispatch elimination、reduction topology 和
Q/D-admitted precision/approximation；FMA 的接近 1.0x 结果照录。下一步是把同一通用 transformation
library 扩到 contraction/完整 Attention 等新 A 图，而不是新增 family 抽象或继续堆无关局部 pass。

当前 CUDA 已登记并在 receipt 中区分：padded-row `serial_row/subgroup_feature`、row reduction
`gpu_all_reduce/workgroup_hierarchical/subgroup_rows`（含 `rows_per_block`）、FP32 FMA count，以及 DYT
`stable_tanh/native_tanh`。这些字段来自实际 C++ post-pass MLIR，不从 requested toggle 推断；2/4/8
rows-per-block 进入 topology digest 和 actual-state evidence。candidate budget 按不同 MLIR digest 消耗，
重复 state 只作为 search evidence 保留。

portable CUDA search bundle 已升级为 graph-capable v3：单 launch 继续使用紧凑 cubin+ABI plan；
多阶段 state 则冻结 lowered host MLIR、全部 cubin、launch/intermediate-buffer receipt，并在目标端复用
既有 orchestrator 预绑定临时 buffer与 launch sequence。真实 5090D smoke 已在同一候选集合执行并
验证 10-kernel/10-launch materialized graph。因而 partial-reduce/finalize、scan 或其他多阶段结构可以
进入与单 kernel state相同的 Q/D、Target--RefJ、计时与 finalist selection，不需要 Python target
kernel emitter，也不另建发射抽象。

- elementwise producer/reduction fusion；
- 在 hard boundary 足够时启用 producer/reduction/epilogue fusion、materialization/copy elimination；
- 把 RVV whole-graph fusion 从 schedule flag 分离为显式 structural transformation trace；
- 注册 sequential/subgroup/workgroup/hierarchical reduction states、FP32 FMA 与 target-supported
  precision states；保持 hard boundary 后以 `observation-changing` 标记进入 Q/D filter，不新增 J 字段；
- schedule candidate 与 C-prior ordering；
- lowering trace、hard facts used、topology/IR digest、static/Q rejection reason、closure 与 performance
  receipt。

### 阶段 B3：算法结构与跨 target

- softmax-shaped graph；
- structured recurrence；
- contraction/conv 等通用 structural transformations；
- CUDA/RVV 或其他 target 的同一 J reuse；
- 迁入 Forge 通用 recipes；
- contraction 上显式比较 IEEE/TF32 等 target state；非 contraction reduction 不错误套用 TF32。

### 阶段 B4：实验与消融

**当前状态：closed valuation 上的 E2/E3、E3/E4、L0/L1、N0--N3 与 G0/G1 已按
v2 spec 在 18-J corpus 上执行并归档；第 18.1--18.3 节的全 target、L2-controlled、
production-valuation 横向矩阵是下一个未完成里程碑。**

- E2/E3 正式 CUDA 主实验报告 search cost/time-to-best、all-corpus、applicable subset 与三边 closure；
- G0/G1 从同一候选回执报告 materialization/dispatch elimination，不以 winner 名称反推适用性；
- E3/E4 证明 C 的低预算 ordering 作用；L0/L1 隔离 local-B；N0--N3 隔离数值机制；
- H100/RVV 严格标明缺失的 Replay/Source 边，不把 Target–RefJ correctness 冒充 formal closure；
- 当前 18 J 与后续新 J 均复用同一 machine spec、receipt schema 与 all-corpus/not-matched
  口径；主表不因机制解释只取代表点而收缩。

## 21. 最终系统表述

> **IntentIR uses an SCRC-accepted J as the immutable, executable reference for lowering-state search. A pure core lowerer consumes only J, optional C, and target capabilities to emit concrete MLIR states and target executables; no implementation-family object or per-kernel recipe exists. Static verification preserves memory, effect, ABI, and other hard J boundaries. A single generic search-and-closure controller then applies a precommitted Q over declared search and held-out certification domains, admits observation-changing reduction, FMA, and precision states only within that recorded scope, and selects performance only among admitted candidates. C changes search order, never correctness; neither search nor closure mutates J or claims contract proof outside Q/D.**

中文：

> **IntentIR 以 SCRC 已接受的 J 作为 lowering-state search 的不变、可执行 reference。纯 core lowerer 只消费 J、可选 C 与 target capability，发射具体 MLIR state 和 target executable；系统不存在 implementation-family object 或 per-kernel recipe。静态 verifier 固守 memory、effect、ABI 等 hard J boundary；唯一通用 search-and-closure controller 再在预先冻结的 Q 与声明的搜索/隐藏认证域上，裁决可能改变观察值的 reduction、FMA 与精度状态，并只在通过者中选择性能。C 只改变搜索顺序，不改变 correctness；搜索与闭合均不修改 J，也不声称 Q/D 之外的 contract proof。**
