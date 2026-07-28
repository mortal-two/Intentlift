# IntentIR Contract-Preserving Multi-Lowering：后端最终设计

> 状态：第五份权威设计（2026-07-24）
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
再在通用 compiler pass contracts 筛选的实现空间中搜索，并以预先声明的 execution domain/comparator
完成候选语义闭合；本文不声称对所有未知输入已经取得无需运行验证的静态保持证明。

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
generic implementation-family lowering / rewrite search
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

Source 的 JIT semantic specialization 已在 Lift 前由 Harness 固定；J 只含对应 monomorphic instance
的完整 active graph。concrete selector value、inactive Source branch 与 Harness binding 均不进入 J/C
或 production backend。materializer 不恢复 variant family，也不把 callable constexpr carrier 当作
backend search knob。

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
- 已显式声明的 rounding、overflow、tie、NaN/Inf 与 determinism 等边界合同；
- closed result-relation class；
- SCRC/π 所需的 typed realization authorization。

矩形 domain 还有一个可由 J 自身确定的窄子类：若其 axis 不出现在任何 logical access index 或
predicate 中，它是 `context_domain`，只声明一个 source-grounded execution context（例如两个 logical
axes 的 flattened program context），不改变 Layer-A 输出函数。production projection 必须保留该
分类 receipt，但不得把它自动变成 loop、program/grid mapping 或地址公式；backend 只有在自身 pass
derivation 已证明合法时才可选择消费它。SCRC 的 π 可以用该 J domain 锚定 compiler-visible program
axis，但 production 仍不读取 π。若 domain axis 参与 access/predicate，它仍是普通 logical domain，
必须满足完整 extent/predicate closure，不能借 `context_domain` 逃逸。

B 中只保存语义事实，不保存某个 backend transformation 的名字。正确形式是：

```text
no observable intermediate effect
no alias
duplicate updates are forbidden
logical access is confined to the declared domain
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

C 不决定某个 implementation family 是否语义合法，不增加或删除 J 的 allowed behavior。

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
- target-specific parameter domain 由 implementation family 和 target capability 定义；
- C 按定义可以被忽略或 retune，不需要重复声明 `retunable=true`；
- 若某个值不可替换且会改变逻辑结果，它不属于纯 C；
- 不增加 `legal_domain/strength/retunable/fixed`；候选合法域仍属于 pass/family/target contract；
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

structured Intent ops 必须至少保留到 implementation-family 选择完成，避免过早退化为 generic loops
而丢失 map/reduce/scan/contraction/recurrence 等结构。

## 7. Generic implementation-family library

backend 根据 A 的 op、region、axis 与 graph pattern 匹配通用实现族：

```text
A structure + target capability
    -> candidate implementation families
```

示例：

```text
reduce
  -> sequential / vector-tree / subgroup / warp / block / hierarchical

scan
  -> scalar / SIMD prefix / subgroup scan / block scan

contraction
  -> generic / tiled / vector / library / MMA

conv
  -> direct / implicit GEMM / im2col + GEMM

softmax-shaped graph
  -> generic multi-pass / SIMD row / GPU warp / GPU block / hierarchical

structured recurrence
  -> direct recurrence lowering / registered equivalent decomposition
```

实现族必须是 source-neutral、kernel-name-independent 的 compiler library。禁止按当前 kernel 名称、
源码路径或历史 benchmark 选择隐藏 recipe。

## 8. B-conditioned pass contracts

对 target t，定义合同环境：

\[
\Gamma_{J,t}=\operatorname{ContractView}(J,t).
\]

`ContractView/Γ` 是由唯一 J、target capability 与版本化 compiler contracts 确定得到的 ephemeral、
non-serialized compiler analysis。它没有独立语义所有权，不成为 backend 输入或 receipt truth，不
进入 SCRC，也不能反向改写 J。

定义静态候选准入 judgement：

\[
\Gamma_{J,t}\vdash M:\operatorname{Admissible}_t(J).
\]

其含义是：M 是 target t 上由通用 pass contracts 判定为 well-typed、memory-safe、边界关系可消费的
compiler state。它覆盖可静态检查的：

- value/state behavior；
- bounds 与 memory safety；
- alias/effect；
- update/conflict；
- 显式 ordered loop/scan、dtype 与 result-relation plumbing；
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
\Gamma_{J,t}\vdash M:\operatorname{Admissible}_t(J)
\qquad
\operatorname{Pre}_{\rho}(J,M,t)
}{
\Gamma_{J,t}\vdash\rho(M):\operatorname{Admissible}_t(J)
}.
\]

这个 judgement 只做静态准入与安全剪枝，不声称数值候选已在未知输入上普遍保持 J。某个实现或
release 只能对当时实际实现、登记并经过准入检查的 `Emit_v`、transformation `rho` 与 `Lower_t`
作支持 claim；第 7 节列出的尚未实现 family 以及第 20 节 roadmap 不能被计入当前 `Reach_t(J)`。

若 bounds、alias、effect、update safety 等 safety-critical precondition 缺失，该 pass 对该 J 不适用。
FMA、TF32、reassociation、accumulation realization 等数值候选不要求 Lift 预先给出 backend permission；
它们可以进入通用候选空间，但必须通过第 15--17 节的 execution-domain closure 才能被选择。

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

C 可以改变 `Visited` 的顺序和有限预算下覆盖到的静态 admissible 候选，但不改变安全准入边界。

实际搜索可以是：

- enumeration；
- beam search；
- dynamic programming；
- e-graph/equality-saturation 风格搜索；
- superoptimizer-style bounded rewrite；
- cost-model-guided search；
- evolutionary search；
- profiling-guided autotuning。

搜索器可以是确定性的，也可以是启发式的。候选顺序不改变静态安全边界；语义接受最终由第 15 节
预先声明的 execution-domain closure 决定。

## 10. Algorithmic multi-lowering 与 schedule tuning

必须区分两类优化。

### 10.1 Same-family schedule tuning

保持 implementation family，仅改变：

- tile；
- vector width；
- threads/warps；
- unroll；
- pipeline；
- memory placement；
- launch configuration。

### 10.2 Contract-guided multi-lowering

允许改变实现结构：

- fusion/fission；
- producer-consumer fusion；
- intermediate materialization/elimination；
- sequential/warp/block/hierarchical reduction；
- direct/multi-stage decomposition；
- layout/data-movement decomposition；
- registered algorithmic equivalence transformation。

“算法结构改变”不是任意 kernel patch。每种非平凡候选都必须作为通用 compiler transformation
存在，并具有明确的 safety-critical A/B preconditions、target contract 与可审计 lowering trace。

例如 online recurrence 与 ordinary two-pass reduction 不是普通 schedule 变化；必须由单独登记的
通用 transformation 产生。它不需要 Lift 预先写 `allow_two_pass`，但只有在目标 execution domain
上通过冻结 comparator 后才能成为 accepted candidate。

## 11. Numeric correctness boundary

backend 必须分清四类信息：

```text
A typed graph:
  实际 operand/result/cast dtype 与显式运算

B numeric/result contract:
  order、reassociation、rounding、exception policy、允许观察关系

Q_SCRC:
  SCRC Gate2 的冻结比较政策

(D_exec,Q_exec):
  backend search/部署声明的执行域与独立 comparator
```

materializer/pass pipeline 不读取 `Q_SCRC`，backend search 也不能把 Gate2 evidence 当成 permission。
同时，J 不新增以下 backend permission 字段：

```text
allow_fma
reduction_order
allow_tf32
accumulator_precision_permission
```

实际 operand/result/cast/accumulation dtype 仍属于 A；Source 真正要求的 ordered loop/scan、effect、
NaN/Inf 或 overflow behavior 仍由现有 A/B/result relation 表达。FMA、TF32、tree realization、近似
transcendental与 mixed-precision accumulation 是 backend 候选选择，选中的实际配置写入 backend
receipt，而不是反向写进 J。

若 `D_exec` 是完整有限部署域，可以声称对该域闭合；若它只是采样集合，结论必须明确限定为该样本
政策，不能外推为所有未知输入上的形式证明。bounds、alias、effect、update safety 等不能通过有限
数值测试豁免。

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

移走 recipe 不等于删除 backend 能力；它使同一 A/B 能匹配多个 target implementation family。
当前 `include/Intent/IntentOps.td` 仍含上述多类 target recipe；因此这是 B0 的实际清理项，而不是
已经完成的架构事实。extensional equation、tie/order/bounds、result type 与 region/yield contract
仍属于 op semantics，不应因清理 recipe 一并删除。

### 12.2 不恢复 per-op layer_b

B 仍由 canonical J enclosing boundary 单一拥有。MLIR op 上只允许由 J/B 确定性投影的 closed attrs，
不得重新接受任意字典或 consumer-private semantics。

### 12.3 reduction realization 不反向升级为 J permission

若 J/A 已用 structured loop、scan 或 recurrence 显式给出顺序，materializer 与 pass 必须保持该
结构；这不是 backend 可重调的 `reduction_order`。若 J 只是 extensional reduce，并由既有 closed
result relation 指定观察方式，那么 sequential/tree/reassociation 是 backend 候选 realization，实际
选择进入 backend receipt，并由声明的 `(D_exec,Q_exec)` 闭合。backend 不能因为希望搜索这些候选，
反向要求 J 新增 `reduce.assoc`、`reduction_order` 或 `allow_reassociation` permission；也不能把
Gate2 tolerance 解释成静态授权。

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

若 J/A 明确包含 online-softmax recurrence，默认 implementation family 可以直接 lower recurrence。
其他算法结构只有在 generic compiler library 中存在相应 contract-preserving transformation 时才进入
搜索。

不能为了扩大后端搜索空间先把 A 改写成模糊的 extensional attention，同时让 π 保存 online update
算术；这会破坏 J 的语义所有权和 source-blind Replay 边界。

## 14. Forge 内容的迁入方式

TianchenForge/历史 backend 中的内容可以按以下类别迁入当前工程：

1. 通用 implementation family；
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

历史数值可以用于选择优先迁入的 family 和搜索范围，但不替代当前系统的运行记录。

## 15. Target lowering 与性能选择

先要求静态准入：

\[
\Gamma_{J,t}\vdash M:\operatorname{Admissible}_t(J).
\]

对预先冻结的 backend execution domain/comparator \((D_{exec},Q_{exec})\)，定义：

\[
\operatorname{ExecClosed}_{D_{exec},Q_{exec}}(M,J)
\iff
\bigwedge_{x\in D_{exec}}
Q_{exec}\!\left(
\operatorname{Exec}(\operatorname{Lower}_t(M),x),
\operatorname{RefEval}(J,x)
\right).
\]

候选只有同时满足静态准入、compile/run 成功和上述闭合时才可参加性能选择。给定有限预算 b：

\[
\operatorname{Eligible}_{t,b,D,Q}(J;C)=
\left\{M\in\operatorname{Visited}_{t,b}(J;C)\mid
\Gamma_{J,t}\vdash M:\operatorname{Admissible}_t(J)
\land\operatorname{ExecClosed}_{D,Q}(M,J)
\right\},
\]

\[
M^*_{b,C}=\arg\min_{M\in\operatorname{Eligible}_{t,b,D,Q}(J;C)}
\operatorname{Cost}_t\bigl(\operatorname{Lower}_t(M)\bigr).
\]

Cost 可以来自静态模型、profiling、benchmark 或混合策略。correctness executions 与 performance
measurements 必须在 receipt 中分开记录；前者决定当前声明域内的 semantic acceptance，后者只在
accepted candidates 之间选择。

## 16. 作用域化闭合命题

假设 materializer、pass 与 target lowering 均通过静态安全准入，且候选 M 通过完整声明域
\((D_{exec},Q_{exec})\) 的执行闭合，则按定义：

\[
\forall x\in D_{exec},\quad
Q_{exec}\!\left(
\operatorname{Exec}(\operatorname{Lower}_t(M^*_{b,C}),x),
\operatorname{RefEval}(J,x)
\right).
\]

该命题只覆盖明确列出的 \(D_{exec}\) 与 \(Q_{exec}\)。若 D 是采样集，结论就是 sampled closure；若
D 是完整有限部署域，才是该域上的 exhaustive closure。它不外推到未知输入，也不把 pass contracts
夸写成静态语义证明。compiler TCB 仍至少包括 `Emit_v`、B projection、静态 safety admission、每个
启用 transformation、target lowerer 与 execution comparator implementation。

## 17. Backend conformance 的位置

backend search 对每个静态 admissible candidate 独立运行：

```text
Close_Q(Exec(Lower_t(M),x), RefEval(J,x))
```

它用于：

- compiler regression；
- CI；
- release preflight / smoke test；
- debug；
- 实验审计；
- 新 implementation family 的工程验证。
- 当前声明 execution domain 内的候选 semantic acceptance。

它不是：

- SCRC 的第四条 Gate2 边；
- Lift/J/C/π artifact 的 membership oracle；
- 把 failed candidate 反馈给 Lift 后补写 J 的机制；
- 未声明输入域上的静态证明。

所有候选必须来自已登记、source-neutral、kernel-name-independent 的 compiler transformations；execution
closure 不能为任意生成代码、Source 特化 patch 或绕过 safety admission 的 candidate 洗白。

## 18. 实验与消融设计

实验使用四个主要层级：

```text
E0  Source frontend implementation
E1  canonical generic lowering M0
E2  same-family schedule retuning
E3  contract-guided multi-lowering + schedule search
```

额外消融：

```text
E4  multi-lowering without C prior
```

对应问题：

- `E1 vs E0`：semantic interface 与 generic materialization 的基础质量；
- `E2 vs E1`：只调 schedule 的收益；
- `E3 vs E2`：实现结构/算法 lowering 多样性的额外收益；
- `E3 vs E4`：C 作为搜索先验的作用，而不是 correctness 作用。

报告指标至少包括：

- absolute latency/throughput；
- 相对 Source 与 vendor baseline 的性能；
- candidate 数、compile 数与测量数；
- time-to-best；
- implementation-family 命中与拒绝原因；
- schedule-only 与 multi-lowering 的增量；
- C prior 对搜索效率的影响；
- compile failure 与 backend conformance failure；
- pass/library 的跨 Kernel、跨 frontend、跨 target 复用。

“固定 implementation family”只作为实验控制变量，不建立系统冻结机制。

## 19. 版本与演进规则

后端可以快速演进，但每次演进必须保持对象边界清楚：

- 新 implementation family 只扩展 backend library，不改变 J；
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
- 保证 structured ops 不在 family selection 前过早消失。

### 阶段 B1：后端合同入口

- 固化 `Emit_v(J)`；
- 实现独立 `Attach_C(M0,C,t)` 与 `Erase_C` invariant；
- 建立 `ContractView(J,target)`；
- 让 materializer 对所有影响 target denotation、safety 与 ABI 的 backend-relevant B facts 完整消费
  或 fail closed；
- 定义 generic compiler pass/family registration API，只以 typed A pattern、B precondition 与
  target capability 为 key。

### 阶段 B2：第一批多 lowering

- elementwise producer/reduction fusion；
- reduction sequential/warp/block/multi-stage family；
- schedule candidate 与 C-prior ordering；
- lowering trace、rejection reason 与 performance receipt。

### 阶段 B3：算法结构与跨 target

- softmax-shaped graph；
- structured recurrence；
- contraction/conv 等通用 family；
- CUDA/RVV 或其他 target 的同一 J reuse；
- 迁入 Forge 通用 recipes。

### 阶段 B4：实验与消融

- E0--E4；
- search cost/time-to-best；
- family reuse；
- C prior；
- backend conformance 与性能结果分开报告。

## 21. 最终系统表述

> **IntentIR production backend consumes only J and optional C from an SCRC-accepted artifact. It deterministically materializes the immutable monomorphic semantic program into Intent MLIR, generates candidates only through registered source-neutral transformations under static safety contracts, and uses C solely to prioritize exploration. Candidate semantics are accepted only within a declared execution domain and comparator; performance selects among those accepted candidates. The claim is scoped to that domain and is not a static proof for unknown inputs.**

中文：

> **IntentIR production backend 只消费 SCRC 已接受 artifact 中的 J 与可选 C。它将固定的 monomorphic 语义程序确定性编码为 Intent MLIR，只通过登记的 source-neutral transformations 和静态 safety contracts 生成候选，并只用 C 提高搜索效率。候选语义由预先声明的 execution domain/comparator 闭合，性能只在已闭合候选间选择；结论严格限定于该执行域，不声称未知输入上的静态形式证明。**
