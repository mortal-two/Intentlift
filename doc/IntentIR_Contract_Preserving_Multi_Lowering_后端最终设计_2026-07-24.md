# IntentIR J-Grounded、Q/D-Certified Lowering-State Search：后端最终设计

> 状态：第五份权威设计（2026-07-24；2026-07-28 更新为 fresh Task14-14 首批正式输入、legacy
> 14 个 fresh accepted J 首批矩阵，并加入 actual-schedule/multi-stage 实施状态）
>
> 文件名为保持既有链接稳定而不变。本文不再把 `implementation family` 作为 compiler object、
> registry key 或 emitter 概念；旧材料中的 `multi-lowering` 统一按本文定义解释为
> **J-grounded structural/numeric lowering-state expansion with Q/D closure**。
>
> **2026-08-02 corpus 更新**：当前正式后端集合为 24 个 Triton 来源 accepted J（原 22 项加
> Llama4 RoPE 与 FlashAttention forward）。此前误混入的两个 TileLang 样本已删除，不进入任何 aggregate；
> 本文第 18.3 与 20.1 中的 fresh-14/legacy 数字保留为阶段历史，不再是当前论文分母。当前结果和剩余缺口以
> [已通过24个J：后端与论文证据总览](已通过24个J_三目标后端总览_2026-08-02.md) 为准。
>
> 上位前端与认证设计：
> [C：所有权、绑定、消费与三方验证](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
> [π：依赖类型实现见证与 Replay 安全](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)、
> [Lift、Replay、Gate1、Gate2 最终执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)、
> [SCRC 最终数学与执行流程](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。
>
> 范围：本文只定义 SCRC 之后的 production backend 与 backend executable closure。它不修改
> Lift、Replay、TypeCheck、Gate1、Gate2 或 π 的协议；backend executable closure 使用预先声明且
> 在实验中保持不变的 Q/D，返回普通运行结果，不增加 SCRC Gate，也不修改已接受 J。

## 1. 最终判断

IntentIR 的 production backend 不是把 J 当作一段等待唯一 lower 的普通 IR，也不是在 JSON J 中
进行程序搜索。它把一次已接受 artifact 中的 J 作为不变、可执行的 reference interface，将其
确定性编码为 Intent MLIR，再搜索由通用 compiler transformations 产生的具体 lowering states。

本文的 correctness claim 与 SCRC 保持同一边界：observer-relative、实际 `c0`、预先声明的有限
测试集合 T 或分布 D、以及预先声明的比较政策 Q。对于 reduction tree、FMA/contraction、内部精度或
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
target executable + backend closure result
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

这句话只描述各对象对后端的作用，不重新定义其所有权。A 与 B 共同组成唯一 J；Q/D 是预先声明
的外部观察与执行域；C 始终是可忽略、可替换的 implementation-selection prior。Q/D 不进入 J，
搜索也不产生 `optimized_J` 或 numeric-permission sidecar。

## 2. 后端硬边界

### 2.1 唯一输入

后端明确拆成两个通用接口：

```text
CoreLower_t(J,C,target,compiler_state) -> target executable

SearchAndClose_t(J,C,target,Q,D_search,D_cert,opaque_oracles)
  -> target executable + backend closure result
```

`CoreLower_t` 只对 SCRC-accepted artifact 工作，不读取 Gate verdict、Q/D 或任何 Source realization。
它和每个 transformation 均禁止读取：

- Source；
- Replay；
- π；
- Gate1 diff；
- Gate2 hidden input/output；
- Source compiler metadata；
- kernel name、source path 或 provider label；
- per-kernel RuleCard、历史 candidate 或旧实验结果。

允许进入 core lowerer 的只有：

- 唯一 J；
- 可选 C；
- target capability；
- backend 自身的版本化 generic pass library、cost model 与运行环境。

`SearchAndClose_t` 是一个跨所有 Kernel、所有 concrete MLIR states 共用的 compiler/autotuning
controller。它可以通过 opaque harness 执行 Source、Replay、`RefEval(J)` 与 target executable，并
调用固定的 Q；但不得读取 Source/Replay 文本、π、kernel 名或历史结果来生成 transformation、修改
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

- J 内容与 denotation 不变，materializer 直接消费 `J` 与 `nu`；
- `nu` 不是 C；C 仍只是 schedule prior，与 logical shape valuation 正交；
- workload profile 可以同步更新 Source harness 的 shape/stride/grid 与 C valuation，但不得
  创造 J 中没有的符号维；
- J literal 不可由 backend profile 改写；把 literal 变成符号维会修改 J 内容，必须由
  frontend 产生新 artifact 并重走 Lift/TypeCheck/Gate1/Gate2；
- 评测必须直接报告 J 所声明的 shape/dtype、具体 `nu` 与 `workload_profile`，不得把一个符号轴上的放大
  写成所有 shape 的泛化证明。

例如，当前 Flash preprocess J 的载体为 `[1,x0,33,16] -> [1,x0,64]`；只有
`heads=x0` 可重新 valuation，`batch=1`/`sequence=33`/`head_dim=16`/`rounded=64` 都是
literal。当前 Softmax 和 Liger RMSNorm 的外层 rows 同样是 literal `2`；后端必须保留
这个边界，而不是为了跑大规模实验自行参数化。

## 3. J=A+B 在后端中的作用

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

搜索只输出普通候选、运行与性能记录，不建立 lineage 或额外 artifact，也不进入 SCRC。

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
- 识别 `realization_authorizations` 的 SCRC-only 职责，不把它当 target permission，也不因其存在
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

这里的 deterministic materialization 与 Lift/Replay 中被禁止的 deterministic semantic composer完全不同。
Replay fresh episode 内的 syntax-only editing composer 也与二者不同：它只渲染 agent 已作者化的
源码结构，不读取 J/π 选择 lowering，不跨 episode，也不进入 production：

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

同一结构可以由不同 trace 到达；搜索器按显式枚举顺序逐候选编译和测量，不计算 canonical IR digest，
也不因两个候选偶然生成相同文本而建立内容身份或折叠预算。实验按实际 transformation trace、post-pass
结构字段、schedule 与 launch plan 直接解释结果。禁止按当前 kernel 名称、源码路径、provider、历史
benchmark 或手写 `family_id` 选择隐藏 recipe。

## 8. B-conditioned pass contracts

对 target t，定义合同环境：

\[
\Gamma_{J,t}=\operatorname{ContractView}(J,t).
\]

`ContractView/Γ` 是由唯一 J、target capability 与版本化 compiler contracts 确定得到的 ephemeral、
non-serialized compiler analysis。它没有独立语义所有权，不成为 backend 输入或持久化 truth，不
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
保持 `HardImpl_t(J)`，且被选择的 executable 通过固定 Q/D final closure。

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
transformation registry 的标签；候选由本次枚举中的具体 MLIR state、trace 与 schedule 直接描述，
而不是由内容身份或预先声明的实现族定义。

## 11. Numeric correctness boundary

backend 必须分清四类信息：

```text
A typed graph:
  实际 operand/result/cast dtype 与显式运算

B hard/result boundary:
  domain、predicate、effect、update、output ABI 与 exact/tolerance comparison class

Q:
  预先声明且所有实验臂固定不变的 comparator policy

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
compile/execute/compare 状态机和同一 Q protocol。可以对所有候选统一执行，
也可以让 `exact-preserving` transformation 只做静态检查和最终 winner closure；最终 executable
返回 closure result，其余状态只保留本次搜索所需的直接运行记录。

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
可以另行设计版本化 numeric relations；这不是当前 backend 路线，也不阻塞当前 certified J 与
后续 accepted J 的执行域搜索。

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

3. target lowerer 保持 hard boundary 且 final winner 通过固定 Q/D：

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
transformation、target lowerer、Q implementation、D protocol 与 opaque execution harness。它明确是
execution-domain closure，不冒充跨 D 的 compiler-pass 等价证明。

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
- 只给最终 winner 返回 backend closure result；
- 不修改 SCRC verdict、J 内容或 Gate2 结果；
- 不授权 memory/effect/ABI 错误，也不允许 backend 调整 Q。

它不是 SCRC 的第四条 Gate2 边，也不声称 candidate 在 D 之外 contract-proven。CI、release preflight、
debug 和 regression 可以复用同一机制，但不能用更弱的 smoke set 冒充正式 `D_cert`。

final result 直接包含：target、实际 J/C 内容、`c0`、Q/protocol、D 中的具名 case、完整 MLIR
transformation trace、schedule、launch/code-size、所有 observable outputs 的三条 direct-edge verdict，
以及独立 performance record。它不计算、校验或输出 J/C、case、MLIR、toolchain、object 或 binary 的
hash/digest。缺任一 correctness leg 时不得发布为 Q/D-closed executable。

## 18. 实验与消融设计

最终论文把外部竞争力与内部因果归因分开。外部 panel 固定为：

```text
GPU: Source / IntentIR
RVV: Native RVV / IntentIR
```

GPU baseline 固定为被 Lift 的原始 Source kernel。RVV 通常没有可执行的 Triton Source，故优先复用
严格匹配的 ggml/llama.cpp 等 native 实现，不存在时编写独立 RVV intrinsic kernel；Triton-CPU 只在
真实可执行子集作为补充。所有 baseline 必须严格匹配 operator、shape、dtype、layout、output relation、
timed boundary、cache policy、device 与 profiler，并与 IntentIR 搜索实现独立。

内部归因在所有 target 上只保留三臂。这里的“固定路径”由 canonical transformation trace 和真实
post-pass MLIR 决定，不由 `family_id`、named emitter 或 kernel 名决定：

```text
CF / C-Fixed:
  canonical lowering path
  + target 对真实 C tile/schedule 的机械合法投影
  + exactly one concrete candidate, no search

E2 / Schedule-Only:
  与 C-Fixed 相同的 canonical lowering path
  + target schedule search

E3 / IntentIR:
  immutable J
  + multiple hard-admitted structural/numeric lowering paths
  + schedule search within each path
  + the same Q/D filter
```

C-Fixed 与 Schedule-Only 的唯一变量是 target schedule：前者固定 C 投影，后者开放 tile、threads/warps、
vector width、unroll、pipeline等同结构参数。二者擦除 schedule 后必须拥有相同的 graph、fusion、
materialization、reduction topology和stage decomposition。C-Fixed投影只允许 role/axis解释、固定缺省值
和target legality/clamp；禁止用 workload shape、occupancy heuristic、cost model或历史性能替换 C。

主指标以 latency 记为：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_lowering = latency(Schedule-Only) / latency(IntentIR)
```

`E4 / IntentIR-Erase-C` 不进入第四根主柱。它只与 E3 在相同 J、相同 lowering/schedule universe、
相同总预算和同一 latency map 上切换遍历顺序，正文 C prior 小表只报固定预算下 best performance 与
达到最终最好性能 95% 所需候选数。

E0/E1、L0/L1、N0--N3 与 materialized/fused、reduction-tree、FMA/approximation 等仍可保留为补充
诊断，但不得变成主实验臂。它们必须复用 E3 的同一 controller、hard verifier、Q/D、target pipeline
与总预算；不匹配记 `not_matched`，不得从 all-corpus aggregate 删除。特别地，producer/reduction/
epilogue 的 read-once frontier、materialization elimination 和真正 whole-graph path 都只是 IntentIR
候选；只有 actual post-pass structure record 表明相应 producer 被融合或中间物化被删除时，才可使用该机制
标签。requested pass 开关本身不是证据。

机器实现从 `spec/lowering/backend-search-ablation-v3.json` 读取实验轴。一个预算单位是一个显式枚举
候选；即使两个候选偶然产生相同代码也不做基于内容的去重，无法形成 executable 的候选同样消耗一个
预算单位。finalist 不混用初次排序样本，而按正反交替 traversal 做 fresh
round-robin remeasurement；matched external baseline 与 target 使用一致的准备、cache 与计时规则。

### 18.1 TileLang-aligned 主评测、缓存状态与 Roofline

E0--E4 只定义 candidate-space 因果轴，不足以定义一个可复现的高性能 Kernel
实验。缓存状态与 Roofline 由 `spec/lowering/backend-performance-measurement-v1.json` 直接描述；
runner 和 report 记录实际采用的协议字段，不计算或绑定 spec digest。评测参考 TileLang/Triton 的
kernel benchmark 实践：代码、module、page allocation 与 GPU 频率先 warm-up，性能主数字则用
CUDA event 计时，并在每次 single-shot 前在 timed interval 之外管理/驱逐 L2。因此必须区分：

- `warm-up`：所有正式计时的必要准备，不是一个性能结果组；每个候选排序前单独稳定，最终
  Source/target 用相同时间预算 round-robin 稳定，并把实际轮数写入测量记录；
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
claim。当前 16-certified + 2-canary regression 的 closed-valuation case
边界数据只有约 1--8 KiB，所以它们首先是 correctness/small-latency workload；不能因 Softmax、Norm
或 copy 的名字就写成 DRAM-bound。memory-wall 机制实验使用同一个符号 J 的独立大实例：首个登记
profile 将 `masked_copy` 的 `x0` 实例化为 16,777,216，Source 的 N/grid、RefJ 与 target 同步绑定，
形成 128 MiB boundary。它不改变 J 或 compiler path，但把 D 扩展到同时超过 RTX 5090D 96 MiB L2
与 H100 50 MiB L2 的实际 workload。只有该类 large-shape/硬件-counter 证据才可支撑 bandwidth claim。

### 18.2 C 的同宇宙、同预算 target-side 配对

C-prior 与 Erase-C 不能只在 exporter 中各打印一份 requested order，而只编译/测量其中一份。portable
bundle 必须访问两条 order，按显式枚举候选消耗相同预算，并运输二者候选 union；不按 MLIR、object
或 binary 内容折叠。target runner 在同一次 candidate correctness/timing run 上分别求两条 order 的
best；C gain 定义为
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
E3 的 hard-safe candidate set 必须包含 E2 的全部候选。主三柱从同一次 complete-space target timing中
分别选择机械 C 投影、完整 fixed-structure schedule best和完整 global best；这三个性能包络不由 bounded
search budget截断。equal-total-budget只用于C-guided/Erase-C与辅助anytime诊断：其访问顺序先走与E2
相同的 fixed-structure schedule spine，再访问结构/数值扩展状态。所有性能比较只在通过相同
`D_search` filter的候选间进行，最终winner使用相同`D_cert` protocol。报告同时给出：

- complete-space 的 C-Fixed/Schedule-Only/IntentIR 三柱；
- equal-total-budget 的 C/no-C 与辅助 anytime/time-to-best 曲线；
- 固定 latency map 下 E3/E4 的 full-budget convergence；
- all-kernel 与 transformation-applicable subset，禁止只挑 winner。

微秒级 Kernel 不得用两个独立进程的 best latency 直接归因于 E3。primary marginal gain 必须在同一
E3 candidate timing run 中，从同一批 fresh finalist 样本分别选择 schedule-spine best 与
expanded best；独立 E2/E3 报告仍保留 absolute latency、Source 配对和 cross-report ratio，但后者只作
复现实验漂移诊断。若两侧选择同一 executable，同次测量增量自然为 `1.0x`，不能把时钟噪声写成
lowering-state 收益。

不同结构路径必然经过不同的高层 transformation trace；“同一 code generator”指共享同一 backend
revision 和 terminal target pipeline，不要求结构分支使用完全相同的高层 rewrite。

### 18.3 fresh-14 首批横向矩阵，不以 canary、legacy J 或代表 Kernel 代替全量实验

主实验的基本单位不是“选几个算子讲故事”，而是一个显式 coverage cell：

```text
(J-declared operator/shape/dtype/layout,
 target/device,
 exact J-declared extent valuation nu,
 workload profile,
 experiment arm,
 compiler/toolchain configuration,
 Q/D protocol)
```

当前主矩阵必须区分 fresh 正式输入与历史/开发输入，并在 RTX 5090D、H100 与 RVV4 上横向扩展：

```text
14 fresh Task14 certified high-performance J（首批正式 aggregate）
+ legacy production-jc-v1（16 high-performance + 2 canary，只作回归）
+ 其他历史/开发输入（仅作独立 regression）
```

对每个 target：

- fresh 14 个 high-performance 输入都要 materialize、compile 并给出 Target--RefJ correctness 或精确的
  fail-closed 状态；legacy/provisional/canary 另表，不进入 fresh aggregate；
- CUDA/H100 在 Source 可执行时，进入 Source--RefJ/RefDev 与 Source--Target 主性能表；
- C-Fixed、Schedule-Only、IntentIR 对 fresh 14 个 high-performance 输入执行；E0/E1、Erase-C、
  L0/L1、N0--N3 只进入外部 baseline、C 小表或补充诊断。不适用的 transformation 记为
  `not_matched`，不从 all-corpus aggregate 中删除；
- 每个 J 除 closed valuation 外，还要运行所有已登记且只绑定 J-declared symbols 的
  valuation；fresh Task14 J 已符号化的 rows/cols 及 Flash 逻辑轴可以绑定新 valuation，而仍为 literal
  的 shape/dtype/selected variant 不由 backend 改写；完全没有符号轴的 J 记为 `literal_shape_only`，
  不伪造大 shape；
- 报告同时给出 all-corpus、structure-applicable subset 与每 Kernel 数字，禁止只保留 winner
  或只报加速的 case；
- target 缺 Source/Replay、J 缺某符号轴、pass 不匹配、compile failure 与 Q/D failure 都是
  矩阵内的显式状态，不是被静默跳过的空格。

只有开销极高、用于解释机制而非建立性能事实的证据，例如 Nsight/CUPTI 详细 counters、
Roofline 分解和单变量 instruction-level 剖析，可以在每类结构上选 1--2 个代表 valuation。
这只是降低“解释证据”成本，不缩小 fresh high-performance 主性能、correctness、E2/E3 或 C
消融的横向覆盖；canary 只进入独立 regression 表。

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
- backend 不实现或保留 hash/digest/freeze/receipt 身份能力，也不使用 kernel allowlist 决定合法性。

## 20. 实现阶段

项目执行上，B0--B4 在目标 Kernel 的 SCRC closure 与 accepted J corpus 形成后再独立展开；不要求
每闭合一个 Kernel 就紧邻实现其 backend。SCRC 阶段若真实证据暴露 J/TypeCheck/Replay/RefEval 的
通用缺口，可以修正对应 A/B 语义合同，但不提前混入本节的 IntentOp ownership/search 重构。

### 20.1 当前闭合与 production intake 基线（2026-07-28）

frontend Task14 的当前 fresh 权威账本是 **14/28**：11 个原运行直接认证，3 个复用各自 exact
Gate1 artifact/Replay、以零模型 Gate2-only capability recheck 认证。它们构成下一版 production intake
的首批正式分母。`corpus/production-jc-v1` 的 18 个 entry 仍由 **16 个历史 high-performance + 2 个
canary** 组成，只保留作 legacy regression，不能与 fresh-14 混合统计。

三个 capability recheck（Flash one-pass LayerNorm、xFormers RMSNorm-add、SplitK reduce）与其余 fresh
accepted J 一样由普通 loader 直接读取。后端不复制、冻结或交叉核验 Gate receipt/digest；它只消费
accepted J/C 内容，并在自己的 Q/D、compile 与性能路径中产生直接运行结果。

当前 fresh 14 个 high-performance 输入都没有 typed rank-2/3 contraction；名称含 `do_o_dot` 的 Flash preprocess 仍是 FP32
elementwise multiply + row sum。因此这批的首要数值搜索轴是 reduction tree、FP32 FMA、fusion 与
materialization，不是 TF32。TF32 只在未来闭合的 `tl.dot`/matmul/完整 Attention contraction 上按
target capability 枚举，并与 IEEE state 使用同一 Q/D 做配对。

因此：

```text
fresh SCRC-certified high-performance pending v2 intake = 14
legacy v1 high-performance regression = 16
legacy compiler/correctness canary = 2
```

不得把历史 12/16-kernel 性能数字、16+2 canary 或开发结果重标为 fresh-14 结果。正式后端直接读取
选定的 fresh catalog entry；报告矩阵只按 kernel、J 声明的 shape/dtype、valuation、target 与实验臂组织，
不建立 append-only intake、content-addressed catalog 或 artifact identity gate。

当前 implementation regression 基线仍可让 catalog 的 18 个 entry（16 high-performance + 2 canary）
deterministic materialize，并由真实 C++/MLIR path 形成 cubin；这证明后端消费覆盖，不证明 18 个
high-performance J 已 SCRC closed。`doc/results/production-backend-closed-18/` 目录名与其中 aggregate
属于历史命名，发布时必须重标为 `16-certified-plus-2-canary development matrix`，canary 从论文
aggregate 排除。

该历史矩阵中的 materialized/whole-graph、reduction-tree、FMA、precision/approximation、local-B、
H100 与 RVV 数字仍可作为机制线索，但不得继续写成“18-certified formal closure”。其中几十倍的
materialized→whole-graph 比值衡量的是消除低效中间物化与 dispatch，不等于相对成熟 Triton Source
的端到端加速；H100/RVV 缺少相应 Replay/held-out 边时只能作为 Target--RefJ 或同 target 内部因果证据。

上述数字首先是 **closed valuation 上的全量消费与因果基线**。它们不等于已完成
production-shape 论文主表：多数原始 closure workload 仅有 KiB 级 boundary，warm-cache 小延迟
不能用来主张 DRAM wall。新的 horizontal matrix 必须按第 18.1--18.3 节重新运行
L2-controlled、J-symbolic production valuations，并将 literal-only 格子显式保留。不得把旧的
closed-valuation aggregate 重命名为新主实验。

历史横向 valuation 的 `HarnessBindingSpec` 仍可作为 legacy regression 输入，但不进入 compiler path
selection，也不进入 fresh-14 aggregate。后端不再维护 J_dev、dev-workload 或 digest-pinned lineage；
literal shape/dtype 不由后端修改，fresh 实验只使用 accepted J 已声明的 literal/symbolic 使用域。

第一项 horizontal lowering 已落地为通用 tiled-frontier multi-reduce concrete state：同一依赖层的
独立规约共享 feature-tile traversal，依赖层从 workgroup-cached frontier 继续遍历，各 partial
先在 warp 内 shuffle，再以每 warp 一个 compact shared partial、每依赖层一个 barrier 完成 block fold；
epilogue 也从 frontier 取值。历史 large regression 上它按 A/ContractView 真实准入 7/18 entry；其余格子
fail closed 保留。7 项均已形成 true post-pass MLIR/cubin 并通过 Target--RefJ，但当前 5090D
搜索没有一项由该状态胜出；这是提示继续改进 applicability/topology 的负结果，不伪装成收益。

`doc/results/production-backend-symbolic-large-18-sm120a/` 同样是 16 certified + 2 canary 的历史
regression，不是 fresh-14 主表。其同次测量 Schedule-Only/expanded、C/no-C 与 Source 数字只能用于
检查 runner 和搜索因果口径；正式 aggregate 必须以 fresh-14 的自然 shape/dtype 重跑。

首批自然 FP32 development slices 已在 RTX 5090D 跑通真实 Source、RefDev、MLIR/cubin 与全部计时
候选 Q/D：RMSNorm `M=2048,H=1024` 的同次测量 IntentIR/Schedule-Only 为 `1.0531x`、Source/Target
为 `1.0491x`；`H=4096` 分别为 `1.0000x` 与 `1.0174x`，multi-stage 因额外重读落败；DYT
`BT=4096,H=4096` 分别为 `1.0024x` 与 `1.0275x`，约 128 MiB 工作集已接近带宽。它们当前是 warm
机制校准，不是最终 cold-data/BF16 主表；尤其旧小输入的 1.57x 不得外推。

RVV 机器三臂与 C 小表现已由同一次 native-object timing run 执行：C-Fixed 是 canonical
materialized path 上 target 对 C 的唯一投影；Schedule-Only 搜完整固定路径 schedule spine；IntentIR
在相同总预算内先访问该 spine，再访问 producer-consumer structural expansion。runner 显式验证
C/no-C 访问同一显式候选序列与 budget，并可选择测完整 latency map 计算 candidates-to-95%。
2026-07-28 的 Softmax `M=N=4096, FP32` warm 开发运行中，修复后的真实 MLIR producer-consumer path
相对同次测量 Schedule-Only 为约 `1.11x`；C/no-C 最终 best 相同，达到 95% best
分别需要 26/35 个候选。该数字仅是协议与路径校准，不进入最终 single/16-thread、外部 baseline 主图。

### 20.2 当前横向里程碑与下一 target

当前 fresh Task14-14 已由普通 loader 直接进入 typed J→MLIR 路径；下一步据其 A 结构生成正式矩阵，
不能用 legacy J 或 canary 代替：

| 结构类 | accepted J | 当前共性瓶颈 | 下一批真实 MLIR state |
|---|---|---|---|
| dense/masked elementwise | DYT、GeGLU、SwiGLU、masked copy | tail/vector width、grid 遍历、端到端内存合并 | masked vec2/vec4 与 target-derived grid-stride/vector layout |
| single reduction + epilogue | Flash preprocess、fused-add RMSNorm、KLDiv、Liger RMSNorm、masked row square-sum、xFormers RMSNorm、xFormers RMSNorm-add | producer 重读、中间物化、单一 reduction topology | producer/reduction/epilogue 单趟 state，subgroup/workgroup/multi-stage 独立搜索 |
| multiple reductions + epilogue | Flash one-pass LayerNorm、LayerNorm、Softmax | compact tiled frontier 已正确准入但尚无 winner；shared frontier/traversal 成本仍未打破现有 register/workgroup 路径 | subgroup/frontier/multi-stage topology 与 schedule 独立搜索，扩展 dtype/表达式并保留直接 rejection reason |
| scan/effect/contraction/gather | Sparsemax、TVD、PolyNorm | scan/update/control 或 contraction/gather 落到 generic/materialized path | registered prefix/segmented-scan、effect-safe update 与 contraction/gather tiling states |

实施顺序以“同一 compiler mechanism 横向覆盖一类 J”为原则，不以单个 benchmark 得分为目标：

1. **横向 compiler mechanism 先行。** 每个新 pass 都对 fresh-14 运行 structural match、hard admission、
   compile 与 correctness；`not_matched` 作为普通矩阵状态保留，不额外建设 coverage receipt。
2. **CUDA/5090D 与 H100 共用高层 transformation。** 先完成 tail-safe vector load/store、多 reduction
   单趟累积与 producer/epilogue fusion，再补 scan/effect/contraction 状态；只由 target capability
   选择 NVVM/MMA 终端能力，不分裂出 kernel-name recipe。
3. **RVV 与 CUDA 同步横向扩展。** 两个 target 共用 workload-profile/symbolic valuation；下一步实现
   dynamic-VL strip mining、SEW/LMUL 状态、vector
   reduction/scan 和可审计的 multicore state，不把工作停在静态 tile 与 producer-consumer fusion。
4. **搜索与 C 做横向收敛。** 每类 transformation 都进入同一 candidate-state universe；C 只重排
   全量状态。主表对 fresh 14 个 high-performance 输入运行 CF/E2/E3，并用 E4 形成独立 C 小表；
   applicable subset 另报但不能替代 all-fresh-14，legacy/provisional/canary 单列。
5. **性能验收不代替 correctness。** 所有候选仍先 Target--RefJ；有 Source/Replay 的 target 再完成
   held-out direct edges。主 latency 用 L2-controlled event protocol，Nsight/Roofline 只解释每类机制，
   不代替全 fresh-14 表。

禁止为了某一 Kernel 的数字在 Python 中增加 target emission、按 case name 切换 compiler pass、
改写 accepted J literal 或只运行一个适用子集后声称全量提升。当前仅有一个 J 命中的结构
也可以实现，但 matcher、pass contract 与 negative test 必须仍然 source-neutral，并在
完整矩阵中呈现其他输入的 `not_matched`。

### 阶段 B0：IntentOp/ABC ownership 对齐

**当前状态：核心入口已落地，继续以增量方式补齐 consumer。**

- 清理 Op.td target recipe；
- 标记 A semantics、B-derived attrs 与 C-derived annotations；
- 解除 `intent.loop.tile_hint` 对 canonical Emit、LoopOp denotation/verifier 与通用 loop lowering 的旧依赖；
- 审计每个 B 字段的 consumer；
- 保证 structured ops 不在结构 transformation 完成前过早消失。

### 阶段 B1：后端合同入口

**当前状态：已实现。** `materialize_j`、`ContractView`、`Attach_C/Erase_C`、统一
`SearchAndCloseController` 与 fail-closed rejection 已进入 J-search 分支；fresh-14 由普通 loader 直接
消费，旧 artifact 在缺少高级 consumer 前提时保留 generic/conservative path，不被静默升级。

- 固化 `Emit_v(J)`；
- 实现独立 `Attach_C(M0,C,t)` 与 `Erase_C` invariant；
- 建立 `ContractView(J,target)`；
- 让 materializer 对所有影响 hard semantics、safety 与 ABI 的 backend-relevant B facts 完整消费
  或 fail closed；
- 扩展现有 generic compiler transformation registration/admission API，只以 typed A pattern、
  B precondition、current state 与 target capability 为 key；不另建 family registry；
- 建立唯一 `SearchAndClose_t`：signature-driven harness、固定 Q、`D_search/D_cert` 与 direct-edge
  comparator；候选逐个显式编译/测量，不保留 candidate performance cache 或 backend closure receipt。

### 阶段 B2：第一批 lowering-path expansion

**当前状态：第一批已实现并完成 16-certified + 2-canary regression；自然 workload 的正式全 target
矩阵仍在推进。** CUDA row/elementwise/padded/generic-Linalg lowering 与 RVV materialized/
producer-consumer paths 均由真实 post-pass MLIR 结构驱动。RVV fusion 从 terminal consumer 反向
遍历 producer，multi-use producer 停在 read-once frontier，避免把 `exp` 或 reduction 重复计算；普通
transformation record 直接记录 fused producer、multi-use rejection 与 requested/actual schedule。
下一步是横向扩展到 scan、dynamic-VL/multicore、contraction/完整 Attention，而不是新增 family 抽象
或继续堆无关局部 pass。

当前 CUDA 已登记并在 transformation record 中区分：padded-row `serial_row/subgroup_feature`、row reduction
`gpu_all_reduce/workgroup_hierarchical/subgroup_rows`（含 `rows_per_block`）、FP32 FMA count，以及 DYT
`stable_tanh/native_tanh`。这些字段来自实际 C++ post-pass MLIR，不从 requested toggle 推断；2/4/8
rows-per-block 直接进入 actual-state structure。candidate budget 按显式枚举候选消耗，不计算 MLIR 或
topology digest，也不按内容去重。

portable CUDA search bundle 已升级为 graph-capable v3：单 launch 继续使用紧凑 cubin+ABI plan；
多阶段 state 直接携带 lowered host MLIR、全部 cubin 与 launch/intermediate-buffer record，并在目标端复用
既有 orchestrator 预绑定临时 buffer与 launch sequence。真实 5090D smoke 已在同一候选集合执行并
验证 10-kernel/10-launch materialized graph。因而 partial-reduce/finalize、scan 或其他多阶段结构可以
进入与单 kernel state相同的 Q/D、Target--RefJ、计时与 finalist selection，不需要 Python target
kernel emitter，也不另建发射抽象。

- elementwise producer/reduction fusion；
- 在 hard boundary 足够时启用 producer/reduction/epilogue fusion、materialization/copy elimination；
- 把 RVV producer-consumer fusion 从 schedule flag 分离为显式 structural transformation trace，并以
  actual fusion/materialization structure 决定标签；
- 注册 sequential/subgroup/workgroup/hierarchical reduction states、FP32 FMA 与 target-supported
  precision states；保持 hard boundary 后以 `observation-changing` 标记进入 Q/D filter，不新增 J 字段；
- schedule candidate 与 C-prior ordering；
- lowering trace、hard facts used、actual structure、static/Q rejection reason、closure result 与
  performance samples。

### 阶段 B3：算法结构与跨 target

- softmax-shaped graph；
- structured recurrence；
- contraction/conv 等通用 structural transformations；
- CUDA/RVV 或其他 target 的同一 J reuse；
- 迁入 Forge 通用 recipes；
- contraction 上显式比较 IEEE/TF32 等 target state；非 contraction reduction 不错误套用 TF32。

### 阶段 B4：实验与消融

**当前状态：v3 machine spec、真实 C-Fixed、统一三臂、同 universe C/no-C 与 complete latency-map
能力已实现；第 18.1--18.3 节的全 target、L2-controlled、production-valuation 与外部 baseline 横向矩阵
仍未完成。**

- 正式内部主实验只报告 C-Fixed、Schedule-Only、IntentIR 与 `G_schedule/G_lowering`；
- materialization/producer fusion 从同次候选测量作补充机制解释，不以 requested 名称反推适用性；
- C/no-C 小表只报告 best-at-budget 与 candidates-to-95%；L0/L1、N0--N3 仅作补充机制诊断；
- H100/RVV 严格标明缺失的 Replay/Source 边，不把 Target–RefJ correctness 冒充 formal closure；
- fresh-14 与后续 accepted J 均复用同一 machine spec 与 all-corpus/not-matched 口径；主表不因机制
  解释只取代表点而收缩。

## 21. 最终系统表述

> **IntentIR uses an SCRC-accepted J as the immutable, executable reference for lowering-state search. A pure core lowerer consumes only J, optional C, and target capabilities to emit concrete MLIR states and target executables; no implementation-family object or per-kernel recipe exists. Static verification preserves memory, effect, ABI, and other hard J boundaries. A single generic search-and-closure controller then applies a precommitted Q over declared search and held-out certification domains, admits observation-changing reduction, FMA, and precision states only within that recorded scope, and selects performance only among admitted candidates. C changes search order, never correctness; neither search nor closure mutates J or claims contract proof outside Q/D.**

中文：

> **IntentIR 以 SCRC 已接受的 J 作为 lowering-state search 的不变、可执行 reference。纯 core lowerer 只消费 J、可选 C 与 target capability，发射具体 MLIR state 和 target executable；系统不存在 implementation-family object 或 per-kernel recipe。静态 verifier 固守 memory、effect、ABI 等 hard J boundary；唯一通用 search-and-closure controller 再用固定 Q 与声明的搜索/隐藏认证域，裁决可能改变观察值的 reduction、FMA 与精度状态，并只在通过者中选择性能。C 只改变搜索顺序，不改变 correctness；搜索与闭合均不修改 J，也不声称 Q/D 之外的 contract proof。**
