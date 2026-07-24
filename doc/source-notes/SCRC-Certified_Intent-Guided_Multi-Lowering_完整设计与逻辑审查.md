# SCRC-Certified Intent-Guided Multi-Lowering

## 从经认证的 IntentIR 到受合同约束的多实现 Lowering

> **文档状态**：后端方法与论文第四章的收敛设计草案。
> **目标**：在不修改已经冻结的 SCRC、C、π 信息边界的前提下，给出一套可实现、可解释、可实验验证的 production backend 方法；同时明确 A/B/C 需要做出的最小修订、数学边界、性能因果链与 novelty 边界。
> **核心结论**：**SCRC 负责把优化后的 Source 恢复并认证成不可变的语义合同 `J=(A,B)`；production backend 不读取 Source 或 π，也不重新猜测 J，而是从 J 确定性生成 Intent MLIR，在由 A 暴露、由 B 约束、由固定 compiler passes 实现的多个合法 lowering 中选择目标硬件上成本最低的实现。C 只提供非绑定的调度先验。**

---

# 0. 一页结论

完整系统分为两个彼此分离、但首尾衔接的阶段：

```text
优化后的 Source S
      │
      │  Lift + TypeCheck + source-blind Replay + Gate1 + Gate2
      ▼
SCRC-certified artifact (J, C, π)
      │
      │  production backend 只读取 J + C
      ▼
canonical Intent MLIR M0
      │
      │  J/B-conditioned contract-preserving lowerings
      ├────────► M1
      ├────────► M2
      └────────► ... Mn
                     │
                     │ deterministic target lowering
                     ▼
                 K1 ... Kn
                     │
                     │ 只比较性能
                     ▼
                    K*
```

其中：

\[
J=(\Sigma,A,B)
\]

- `A`：可执行的、结构化的 tensor-level computation；
- `B`：固定不变的 domain、access、effect、numeric 与 result contract；
- `C`：纯 implementation-selection / schedule hints；
- `π`：Source realization witness，只服务于 Replay 与 Gate1，不进入 production backend。

后端不做：

```text
任意生成候选
→ 跑 RefEval(J) 做有限测试
→ 测试接近就把候选当成合法实现
```

后端做的是：

```text
固定 J
→ 确定性 materialization
→ 只应用具有明确前置条件的 compiler lowering / transformation
→ 搜索发生在这些合法 derivation 之间
→ 只按目标性能选择
```

最简洁的学术表达是：

> **J supplies the invariant; Intent MLIR carries the implementation alternatives.**
>
> **J 给出不可改变的语义不变量；Intent MLIR 承载可搜索的实现选择。**

第四章的真正 novelty 不是“我们写了一个更好的 GPU backend”，而是：

> **IntentIR 将一个已经被 Source realization 和 schedule 固化的 Kernel，恢复成一个经 SCRC 认证、可以被普通确定性编译器重新多路 lowering 的优化接口。现有 high-level compiler 通常假设这个接口已经存在；IntentIR 从优化实现中把它重新恢复出来。**

---

# 1. 与冻结 SCRC/C/π 设计的硬兼容边界

本方法不得改变以下已经确定的边界。

## 1.1 Artifact 仍严格只有三件对象

\[
\operatorname{Lift}_H(S)=(J,C,\pi).
\]

不能新增：

- `OptimizedJ`；
- `J'`；
- `LoweringAgent` 输出；
- target-specific proof artifact；
- backend strategy artifact；
- 第四个 source-conditioned sidecar。

## 1.2 J 是唯一显式语义载体

\[
J=(\Sigma,A,B).
\]

production backend 运行期间，J 不可修改：

\[
J_0=J_1=\cdots=J.
\]

若修改 A 或 B，得到的是新 artifact：

\[
J'=(A',B')\neq J,
\]

旧 SCRC certificate 不再适用，必须按新 schema 重新 Lift、Replay、Gate1 和 Gate2。

## 1.3 Production backend 只读取 J 与 C

\[
K_t=\operatorname{Lower}_t(J,C).
\]

禁止读取：

- Source；
- Replay；
- π；
- Gate1 diff；
- Source compiler fingerprint；
- kernel name；
- RuleCard 或 per-kernel recipe。

## 1.4 π 的职责不变化

π 仍只表达 Source realization 的：

- regions；
- instances；
- routes；
- order；
- sharing / recomputation；
- placement；
- materialization；
- representation relation；
- Source 中的 C use-site。

π 不参与目标 backend 的实现选择。否则目标后端会退化为“模仿原 Source”，与跨硬件重新实现的目标冲突。

## 1.5 SCRC 不证明 production compiler 的每个 pass

SCRC 建立的是：

```text
Source S  →  certified J
```

第四章建立的是：

```text
certified J  →  target code Kt
```

两条可信链不同：

| 阶段 | 可信依据 |
|---|---|
| Source → J | SCRC 的 TypeCheck、Gate1、Gate2 |
| J → canonical Intent MLIR | deterministic materializer |
| Intent MLIR → alternative Intent/Lower IR | compiler pass contracts |
| Lower IR → target code | target lowerer |
| backend conformance test | CI / regression / evaluation，不是合法性 oracle |

---

# 2. 第四章的核心研究问题

第四章不应只问：

> IntentIR 能不能 lower 到目标代码？

因为任意一个可执行 IR 都可以有一个保守 lowering。

真正的问题是：

> **SCRC 恢复出的 A/B/C 分层，能否让目标 backend 获得 Source frontend 中已经被实现细节遮蔽的优化自由度，并且在不重新猜测语义的情况下，从多个满足同一 J 的实现中选择更快者？**

这对应三个子问题：

1. **A 是否暴露了普通低层 IR 难以可靠恢复的结构化计算机会？**
   例如 map、reduce、scan、contraction、select、gather/scatter、structured recurrence。

2. **B 是否提供了足够的固定语义事实，使 compiler 可以机械判断某种 lowering 是否适用？**
   例如 mask、domain、alias、effect、reduction order、accumulation dtype、duplicate update 和 result relation。

3. **C 是否保留了 Source 的调度知识，同时又不冻结目标实现？**
   C 可以帮助搜索，但不能决定 correctness。

因此三层的性能作用是：

\[
\boxed{
A\text{ exposes alternatives;}\qquad
B\text{ filters alternatives;}\qquad
C\text{ prioritizes alternatives.}
}
\]

中文：

\[
\boxed{
A\text{ 暴露候选实现；}\quad
B\text{ 筛除不合法实现；}\quad
C\text{ 提高搜索效率。}
}
\]

---

# 3. A、B、C 的最终学术身份

# 3.1 Layer A：结构化、可执行、不可变的 tensor program

A 应定义为：

\[
\boxed{
A=\text{a typed, executable, structured tensor program}.
}
\]

A 的职责是给出：

- logical operands/results；
- dtype 与 logical shape；
- tensor-level value relation；
- operator 与 dataflow；
- logical axes；
- structured region；
- reduction / scan / contraction / update / recurrence 的逻辑含义。

A 不能包含：

- block/thread/warp mapping；
- target intrinsic；
- shared-memory recipe；
- one-block-per-row；
- Hillis–Steele、bitonic、im2col 等 backend recipe 名称；
- pipeline stage 数；
- target-specific vector width。

## 3.1.1 A 必须保持可执行性

每个 A construct 必须同时具有：

1. 类型和 shape 规则；
2. `RefEval` 语义；
3. deterministic Intent MLIR materialization；
4. stable ID / axis / operand relation；
5. 可被 Replay 与 π 引用的 carrier。

不允许出现仅供分类、但没有 executable semantics 的 opaque label。

## 3.1.2 A 应保留结构，而不是立即 scalarize

例如：

```text
map → reduce → broadcast → map
```

应在 Intent MLIR 中继续以 structured region 存在一段时间，而不是在 materialization 后立刻退化成：

```text
load / scalar add / branch / store
```

否则 A 对 backend 的价值在优化真正发生前就丢失了。

## 3.1.3 A 的抽象原则

判断一个事实是否属于 A，可以使用：

> **改变它是否会改变逻辑 tensor value/state relation？**

- 会：进入 A；
- 只改变如何并行、如何存储、如何切块：不进入 A；
- 只描述 Source 实现拓扑：进入 π；
- 只描述目标 backend 的 recipe：只存在于 pass 实现中。

---

# 3.2 Layer B：固定的 portable semantic contract

B 不是一个可变的优化开关，也不是 transformation 白名单。

\[
\boxed{
B\text{ is immutable after SCRC certification.}
}
\]

B 给出所有合法实现必须满足的固定事实：

## 3.2.1 Logical domain 与 access contract

- logical domain；
- index relation；
- predicate；
- fill；
- output mapping；
- shape/layout interpretation；
- dynamic extent / output order。

## 3.2.2 State、alias 与 effect contract

- read / write / update；
- destination / inout；
- alias；
- conflict；
- observable effect；
- duplicate update policy；
- synchronization requirement；
- intermediate state 是否可观察。

## 3.2.3 Numeric 与 result contract

- operation order 是否固定；
- reduction tree 是否固定或允许变化；
- accumulation dtype；
- cast / signedness / overflow；
- NaN/Inf policy；
- tie-breaking；
- determinism；
- closed result-relation class。

## 3.2.4 Typed refinement relation

例如：

- signed widening；
- 已定义的 value-preserving representation conversion。

这些 relation 是 J 的固定组成部分，不是后端运行时修改的权限。

## 3.2.5 B 不应出现的内容

B 不应写：

```text
fusion_allowed = true
online_to_twopass_allowed = true
use_tensor_core = true
cpu_strategy = xxx
gpu_strategy = yyy
```

这些是 backend transformation 名称或 target policy。

正确方式是：

```text
B: no observable intermediate effect
B: no alias
B: reduction order = tree-permitted
```

然后 fusion / reduction pass 根据这些固定事实判断前置条件是否成立。

## 3.2.6 B 与外部 Q 的边界

B 可以声明闭合的语义类别，例如：

```text
exact
fp_elementwise
fp_reduction_fixed_order
fp_reduction_tree
atomic_update_relation
```

但具体 `atol/rtol/ULP` 由外部冻结的 Q 决定。Lift 不能在 B 中自由写一个任意宽松的 comparator。B 的 relation class 必须来自小型、版本化、封闭的全局词汇，并由 TypeCheck 检查它与对应 op、dtype、reduction/update 结构是否相容；否则一个过宽的 B 会把错误实现也纳入 `Impl(J)`。

---

# 3.3 Layer C：纯 implementation-selection / schedule prior

C 不进入 J 的 denotation。

C 可以包含：

- tile extent；
- vector width；
- warp/thread count；
- pipeline depth；
- unroll；
- prefetch；
- memory placement；
- buffer choice。

每个 C slot 至少需要：

```json
{
  "id": "c0",
  "kind": "constexpr",
  "role": "tile_extent",
  "target_ref": "n4",
  "axis_ref": "x1",
  "value_type": "int",
  "legal_domain": {
    "min": 16,
    "power_of_two": true
  },
  "strength": "hint",
  "retunable": true,
  "required": true
}
```

C 的作用是：

- 给 canonical/source-like lowering 一个起点；
- 为 `fixed` slot 实例化必须遵守的 implementation parameter；
- 为 `hint + retunable` slot 提供 schedule search prior；
- 缩小优先搜索区域；
- 记录 Source 中已有的性能知识。

C 可以限制或排序具体 implementation parameter，但无论取何种合法值，所有生成状态仍必须属于 `Impl(J)`；C 不定义新的语义合法性。

C 不负责：

- 判断某个 implementation form 是否语义合法；
- 表达 operator、predicate、fill 或 result relation；
- 保存 per-kernel backend recipe；
- 指定必须采用哪一种算法实现。

实现形式的选择属于 compiler 内部状态，不需要新增顶层 artifact。

---

# 3.4 π：保持原设计，不进入第四章 backend

第四章只需再次声明：

\[
\pi\notin\operatorname{BackendInput}.
\]

π 证明 Source realization 可以被 source-blind Replay 恢复，但目标 backend 的任务不是重现 Source realization，而是从 J/C 生成目标实现。

---

# 4. A/B/C 需要做出的最小修改

这些修改不改变 SCRC 的三对象结构，也不增加新的 Agent。

## 4.1 A 的修改

### 必须修改

1. **所有 Op 语义描述中移除 target-specific lowering recipe。**
   Op description 只保留 extensional semantics、shape/type 和 logical ordering。

2. **将 `intent.loop.tile_hint` 从 A op 中移到 C。**
   `tile_hint` 是明确的 Layer C 信息，不能继续作为 `Intent_LoopOp` 的语义 attr。

3. **保持 structured ops 到 multi-lowering 决策完成之后。**
   不要一进入 Intent MLIR 就立即全部 lower 到 generic loops。

4. **给每个 op/region 保留稳定 ID、axis role 和 operand/result carrier。**
   C 和 pass precondition 通过这些稳定引用工作，不读取 π 或 kernel name。

### 不建议修改

- 不把 A 退化成简单 operator-family 分类；
- 不把复杂计算隐藏在不可执行字符串中；
- 不为了后端方便把 target recipe 写回 A。

## 4.2 B 的修改

### 必须修改

1. **将模糊的 `result_relations: tolerance` 改成闭合、版本化的 relation class。**

2. **为 order-sensitive op 明确 numeric contract：**
   - reduction order；
   - accumulation dtype；
   - determinism；
   - NaN/Inf；
   - tie / duplicate semantics；
   - cast / overflow。

3. **将 `realization_authorizations` 解释为固定 typed refinement relation。**
   为避免误解，schema v2 可将其重命名为：

```text
refinement_relations
```

` signed_widen ` 可以保留；它表示一个固定、可机械检查的 value-preserving relation，不表示 B 在 backend 中可变。

4. **每个 B 字段必须有明确 consumer。**

| B 字段 | 必须至少被谁消费 |
|---|---|
| predicate/fill/access | RefEval 或 Intent MLIR dataflow |
| bounds/domain | verifier、guard 或 lowering condition |
| alias/effect | pass applicability 或 memory-effect interface |
| reduction order/result relation | reduction lowering precondition |
| signed widen/cast | TypeCheck 与 materializer/lowerer |
| output mapping | RefEval 与 target ABI/lowering |

没有 consumer 的 B 字段不能用于论文 correctness 或 performance claim。

### 禁止修改

- 后端运行时不得重写 B；
- 不新增 per-kernel transformation 名称；
- 不把 Q 的数值阈值放回 B；
- 不把 target-specific schedule 放进 B。

## 4.3 C 的修改

1. 补齐 `target_ref`、`axis_ref`、`legal_domain`、`strength`、`retunable`；
2. 从 A op attr 中移出 tile/vector/warp/pipeline；
3. production backend 可以忽略、覆盖或 retune hint，但不改变 J；
4. target 搜索产生的临时 schedule assignment 是 compiler internal state，不是新的 proof artifact；
5. 可以保存 lineage/receipt，但它不进入 SCRC Replay 或 Gate verdict。

## 4.4 π 不需要为第四章扩张

第四章禁止为了让 backend 更聪明而向 π 增加：

- strategy；
- target policy；
- backend recipe；
- performance hint；
- target-specific implementation name。

---

# 5. 当前 Op.td 的具体清理清单

当前 Op.td 的 op vocabulary 本身有价值，但很多 description 把语义与已有 GPU recipe 混在一起。建议只移动描述和字段，不需要推翻 op 集合。

| 当前内容 | 正确所有权 | 修改 |
|---|---|---|
| `map_unary` 中 “NVVM approx intrinsics” | backend | 从 A 描述移除；approx 是否可用由 B numeric contract + target pass 决定 |
| `select` 中 “lowered via arith.select” | backend | 移到 conversion pass 文档 |
| `iota` 中 block-cooperative 生成方式 | backend | 只保留逻辑 `0..N-1` 语义 |
| `cast` 的具体 arith 指令列表 | backend | A 只保留 cast 语义；lowering recipe 移出 |
| `scan` 的 Hillis–Steele/shared-memory/barrier | backend | A 只保留 inclusive scan 语义 |
| `arg_reduce` 的 pair tree | backend | A/B 保留 argmax/argmin 与 tie policy |
| `sort` 的 single-block bitonic | backend | A/B 保留排序、方向、top-k、tie/order contract |
| `gather/scatter` 的具体 GPU load/store/atomic | backend | A/B 保留 logical gather/update、duplicate/conflict/result contract |
| dynamic-output op 的 host prefix-sum sizing recipe | backend/runtime | A/B 保留 dynamic extent 与 output order |
| layout ops 的 one-block-per-row GPU recipe | backend | 只保留 logical index mapping |
| `polar` 中 approx cos/sin 和 `rel < 1e-3` | backend + Q | 从 A 描述删除 |
| `conv` 中 direct/no-im2col | backend | A 保留 convolution relation；多个 lowering 可选 |
| `reduce.assoc` | B | 保留，但改成更精确的 closed relation class |
| `intent.loop.tile_hint` | C | 从 op attr 移出，改为 C annotation/sidecar |

需要强调：

> **把 recipe 从 Op description 移走，不是删除后端能力，而是让同一个固定 A/B 可以对应多个 target implementation。**

---

# 6. Production backend 的完整方法

# 6.1 Deterministic canonical materialization

给定固定 J：

\[
M_0=\operatorname{Emit}_v(J),
\]

其中：

- `v` 是 materializer/schema 版本；
- 同一 J 与同一版本得到同一 canonical Intent MLIR；
- JSON → Python Intent objects → Intent MLIR 不调用 LLM；
- B 一部分成为显式 dataflow，一部分成为 op/region attrs、effects 与 verifier facts；
- C 作为独立 schedule annotation 或 sidecar 进入 lowering 边界。

“canonical”不表示 J 只有一个可能实现，只表示 backend 的解释起点唯一、稳定、可复现。

# 6.2 `Impl(J)` 判断

定义：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}(J).
\]

含义是：在目标 t 上，M 是 J 的一个合法实现状态。

抽象要求为：

\[
\forall x\in\operatorname{Dom}(J):
\operatorname{Safe}_B(M,x,t)
\land
\operatorname{ObsBeh}_t(M,x)
\subseteq
\llbracket J\rrbracket(x).
\]

其中 `Safe_B` 不只检查返回值，还覆盖：

- bounds；
- alias；
- effect；
- update/conflict；
- synchronization；
- numeric/result contract。

论文不必声称已经为所有 MLIR op 机器证明这个 judgement；它是 backend 的显式 contract 与 TCB 边界。

# 6.3 A-driven implementation families

backend 根据 A 中的 structured computation 匹配可用 lowering family：

```text
A pattern / op / region
    → applicable implementation families
```

例如：

```text
reduce region
    → sequential / vector tree / subgroup / block / hierarchical

scan
    → CPU sequential / SIMD prefix / subgroup scan / block scan

sort
    → bitonic / radix / merge / target library

conv
    → direct / implicit GEMM / im2col + GEMM

softmax-shaped graph
    → canonical multi-pass / CPU SIMD / GPU warp / GPU block / hierarchical
```

这些 family 是 compiler library 中的通用 lowering，不是 per-kernel 规则，也不进入 artifact。

# 6.4 B-conditioned legality

每个 lowering transformation `ρ` 都具有固定前置条件：

\[
\operatorname{Pre}_{\rho}(J,M,t).
\]

应用规则：

\[
\boxed{
\frac{
\Gamma_{J,t}\vdash M:\operatorname{Impl}(J)
\qquad
\operatorname{Pre}_{\rho}(J,M,t)
}{
\Gamma_{J,t}\vdash \rho(M):\operatorname{Impl}(J)
}.
}
\]

前置条件从以下事实机械读取：

- A 的 op/region/axis/value relation；
- B 的 domain、alias、effect、order、result relation；
- 当前 MLIR 状态；
- target capability。

它不读取：

- Source；
- π；
- kernel name；
- Gate1 diff；
- 运行时候选输出。

MLIR 的 verifier、interfaces 和 dynamic legality 可以执行这些条件，但不能把“MLIR legal”写成已经自动证明了 J-level semantic preservation。语义保持仍然是 materializer/pass/lowerer 的 compiler contract。

# 6.5 C-guided schedule and parameter selection

对每个已经合法的 implementation family，再搜索：

- tile；
- vector width；
- thread/block mapping；
- unroll；
- pipeline；
- memory placement。

C 只改变：

- 默认值；
- 初始候选；
- 搜索顺序；
- 优先邻域。

C 不改变 `Impl(J)` 的语义边界。它可以：

- 通过 `fixed` slot 约束某些 implementation parameter；
- 通过 `hint + retunable` slot 改变候选顺序和初始邻域。

# 6.6 Reachable implementation space

先定义由 J 与 target 决定的全部合同合法状态：

\[
\operatorname{Reach}_t(J)
=
\left\{
M\mid
M_0\xRightarrow[\Gamma_{J,t}]{\rho_1,\ldots,\rho_k}M
\right\}.
\]

给定 C 后，backend 实际访问的状态写成：

\[
\operatorname{Reach}_t(J;C)
\subseteq
\operatorname{Reach}_t(J),
\]

其中 `fixed` slot 可以限制参数，`hint + retunable` slot 主要定义排序：

\[
\operatorname{Order}_{C}
\left(
\operatorname{Reach}_t(J;C)
\right).
\]

关键不变量是：

\[
M\in\operatorname{Reach}_t(J;C)
\Longrightarrow
M:\operatorname{Impl}(J).
\]

因此 C 可以改变搜索路径和具体实现参数，但不能改变 J 的正确性边界。

# 6.7 Target lowering

若：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}(J),
\]

则 target lowerer 的合同是：

\[
\operatorname{Lower}_t(M)\models J.
\]

# 6.8 Performance-only selection

最终选择：

\[
\boxed{
M^*
=
\arg\min_{M\in\operatorname{Reach}_t(J;C)}
\operatorname{Cost}_t
\left(
\operatorname{Lower}_t(M)
\right).
}
\]

\[
K^*=\operatorname{Lower}_t(M^*).
\]

`Cost` 可以来自：

- 静态 cost model；
- profiling；
- benchmark；
- heuristic；
- 混合搜索。

搜索器本身可以是确定性的或启发式的。正确性不来自搜索顺序，而来自可达空间对 `Impl(J)` 的闭合。

---

# 7. 条件组合保持命题

## Proposition：Compositional Contract Preservation

假设：

1. `Emit_v` 正确 materialize J：

\[
\Gamma_{J,t}\vdash M_0:\operatorname{Impl}(J);
\]

2. 每个启用的 transformation `ρ` 满足：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}(J)
\land
\operatorname{Pre}_{\rho}(J,M,t)
\Rightarrow
\Gamma_{J,t}\vdash\rho(M):\operatorname{Impl}(J);
\]

3. target lowering 保持合同：

\[
\Gamma_{J,t}\vdash M:\operatorname{Impl}(J)
\Rightarrow
\operatorname{Lower}_t(M)\models J.
\]

则：

\[
\boxed{
M\in\operatorname{Reach}_t(J)
\Rightarrow
\operatorname{Lower}_t(M)\models J.
}
\]

从而：

\[
\boxed{
K^*\models J.
}
\]

这是一项**条件式、组合式 compiler correctness argument**。在没有对 materializer、passes 和 target lowerer 做 mechanized proof 前，不应称为 fully verified compiler theorem。

---

# 8. Softmax：完整示例

以下例子不依赖 opaque `softmax` 标签。A 使用现有的 structured tensor ops 表达稳定 softmax。

# 8.1 Layer A

逻辑计算：

\[
\tilde{x}_{ij}=\operatorname{select}(m_{ij},x_{ij},-\infty),
\]

\[
r_i=\max_j\tilde{x}_{ij},
\]

\[
e_{ij}=\exp(\tilde{x}_{ij}-r_i),
\]

\[
s_i=\sum_j e_{ij},
\]

\[
y_{ij}=e_{ij}/s_i.
\]

Intent MLIR 可以保持为：

```mlir
%masked = intent.select %mask, %x, %neg_inf
%max    = intent.reduce %masked {fn = "max", axes = [1]}
%shift  = intent.map_binary %masked, %max {fn = "sub"}
%exp    = intent.map_unary %shift {fn = "exp"}
%sum    = intent.reduce %exp {fn = "sum", axes = [1]}
%out    = intent.map_binary %exp, %sum {fn = "div"}
```

这里 `%max` 与 `%sum` 的 row-wise broadcast 关系由 J/B 中的 logical broadcast/access mapping 表达，不要求新增一个当前不存在的 `intent.broadcast` op。

# 8.2 Layer B

B 固定描述：

```text
Domain:
  0 <= row < M
  0 <= col < N

Mask:
  invalid elements are excluded
  masked fill = -inf

Effects:
  input/mask read-only
  output written exactly once
  no observable intermediate tensor
  no input/output alias

Numeric:
  max semantics and NaN policy
  sum accumulation dtype = fp32
  reduction order policy = fixed_order
                           or tree_permitted
  result relation class = fp_reduction_fixed
                          or fp_reduction_tree

Output:
  shape [M,N]
  same logical row/column mapping
```

B 不写：

```text
warp softmax
online softmax
two-pass
one-block-per-row
```

# 8.3 Layer C

例如：

```text
c0: tile_extent, target=row-reduction axis, preferred=128
c1: vector_width, target=column axis, preferred=4
c2: warp_count, target=kernel, preferred=4
c3: pipeline_depth, target=kernel, preferred=2
```

# 8.4 多个合法 lowering

## Form 0：Canonical generic lowering

- 独立 max reduction；
- 独立 exp + sum；
- normalize；
- 保守 tail handling；
- generic fallback。

## Form 1：Fused/recompute lowering

在 B 表明 intermediate `%exp` 不可观察且无 alias/effect 时：

- 不物化整个 `%exp` tensor；
- second pass 计算 exp 与 sum；
- normalize pass 可重新计算 exp；
- 以额外 compute 换取更少内存流量。

这不是修改 J，而是改变 intermediate materialization。

## Form 2：CPU SIMD row lowering

- row-level parallelism；
- SIMD max reduction；
- SIMD exp/sum；
- SIMD normalize；
- masked vector tail 或 scalar remainder。

它利用的是 A 的 reduce/map 结构与 B 的 domain/mask/numeric contract。

## Form 3：GPU warp-per-row lowering

当 N 与 target subgroup 能力满足前置条件时：

- 一个 warp 处理一行；
- register tile；
- subgroup max/sum；
- no/shared-minimal memory。

## Form 4：GPU block-level lowering

当 N 较大：

- thread-local partial；
- warp reduction；
- block-level merge；
- shared-memory staging；
- target-specific mapping。

## Form 5：Hierarchical/multi-stage lowering

当行极长、单 block 不合适，并且 B 表明中间状态不可观察、effect 与 reduction order contract 允许时：

- stage 1 生成 partial statistics；
- stage 2 合并；
- stage 3 normalize。

# 8.5 B 如何决定合法性

若：

```text
reduction_order = fixed_order
```

则改变 sum tree 的 warp/block/hierarchical lowering 不能自动应用；只能使用保持指定顺序的实现。

若：

```text
reduction_order = tree_permitted
accumulation_dtype = f32
result_relation = fp_reduction_tree
```

则相应并行 tree lowering 可以进入 `Reach_t(J)`。

B 没有改变。变化的是同一固定 B 下，哪些 pass 前置条件成立。

# 8.6 Online softmax 的边界

必须区分：

1. **A 是上述 extensional softmax graph**：online implementation 可以作为一个非平凡 lowering，但需要一个明确的 compiler transformation contract；
2. **A 自身显式表示 online recurrence**：recurrence 已经属于 J。此时 ordinary two-pass 不能被当成普通 schedule 变化；必须有一条单独、可信、前置条件明确的 recurrence-reformulation pass。

因此当前论文不能笼统声称：

> 任意 online softmax 都可以自动变成 two-pass。

可以声称：

> **对于 A 已经抽象出的 structured computation，backend 可以选择多个保 J 的实现；涉及非平凡算法恒等式的变化，只有在存在对应 compiler transformation contract 时才进入搜索空间。**

---

# 9. 当前 DyT JSON：可立即落地的 multi-lowering canary

当前 DyT J 的 A 是：

\[
t_5=
\operatorname{select}
\left(
 t_4,
 \tanh(t_1\cdot t_0)\cdot t_2+t_3,
 \tanh(t_1\cdot t_0)\cdot t_2
\right).
\]

B 已经提供：

- domain；
- masked loads/stores；
- broadcasts；
- read/write effects；
- output mapping；
- signed index widening。

在不修改 J 的前提下，当前后端即可支持：

1. **whole-chain fusion**：不物化 `v0..v3` 中间 tensor；
2. **broadcast hoisting**：scalar `t1` 与按轴广播的 `t2/t3` 复用；
3. **constexpr select specialization**：`t4` 固定时删除未使用分支和对应 load；
4. **vectorize k0**：masked vector tail、remainder loop 或 predication；
5. **parallelize k1**：CPU row parallel / GPU mapping；
6. **typed signed widening**：使用 B 中已声明的 index refinement；
7. **不同 target memory lowering**：register/cache/vectorized load/store。

DyT 适合作为：

- materialization 正确性 canary；
- B consumption canary；
- A/B/C 边界 canary；
- 第一个 multi-lowering 实现样例。

但它不足以单独证明“implementation-form search beyond schedule”，因为其主要机会是 fusion、specialization 与 vectorization。主性能 novelty 仍需 reduction/scan/sort/conv/softmax 等至少一个结构更丰富的 family。

---

# 10. 为什么性能可以归因于 IntentIR，而不只是 backend 写得好

外部 baseline 只能说明后端整体有竞争力。要证明 IntentIR 本身提供性能价值，必须在**同一 backend、同一 target、同一 codegen、同一 tuning budget**下做 representation ablation。

# 10.1 版本 A：Canonical lowering

```text
J → M0 → 单一 generic lowering → target
```

只证明 J 可执行。

# 10.2 版本 B：Schedule-only

固定 implementation form，仅搜索：

- tile；
- vector；
- mapping；
- pipeline。

它证明 Layer C 非绑定和 target retuning 的价值。

# 10.3 版本 C：A/B-guided multi-lowering

允许：

- 根据 A 选择 implementation family；
- 根据 B 筛选合法 family；
- 对每个 family 使用相同 schedule tuner。

最关键指标：

\[
\boxed{
\frac{
\operatorname{Perf}(\text{A/B-guided multi-lowering})
}{
\operatorname{Perf}(\text{schedule-only})
}.
}
\]

若该比值显著大于 1，才能证明收益来自 IntentIR 暴露出的实现空间，而不是普通 tile tuning。

# 10.4 版本 D：关闭 C prior

候选空间与版本 C 完全相同，只是不使用 Source C 排序。

报告：

- 达到最终最佳性能 95% 所需候选数；
- 编译次数；
- 调优时间；
- 固定预算下最佳性能。

这证明 C 是有价值但非绑定的 prior。

# 10.5 B-aware vs conservative B-blind

可增加一个很便宜的消融：

- `B-aware`：使用 domain、mask、no-alias、effect、order facts；
- `conservative`：仍保证正确，但不利用这些 facts。

例如 conservative 版本：

- 保留边界处理；
- 禁止改变 reduction tree；
- 不做跨节点 fusion/recompute；
- 假设可能 alias。

该比较说明：

> B 不是改变语义，而是消除后端不必要的保守性。

---

# 11. 建议的性能实验

## RQ1：SCRC 是否可靠恢复 backend 可消费的 J？

报告：

- TypeCheck pass；
- Gate1 closure@1 / closure@3；
- Gate2 pass；
- SCRC-certified rate；
- 按 A/B/C 字段的 materialization coverage；
- B 字段 consumer coverage。

## RQ2：A/B 是否打开 schedule-only 之外的实现空间？

在 1–2 个结构丰富 family 上比较：

```text
Canonical
Schedule-only
A/B-guided multi-lowering
```

至少报告：

- speedup over schedule-only；
- 选择了何种 implementation form；
- 有多少 shape/target 选择了非 canonical form；
- 每种 form 的 B precondition；
- 编译/搜索开销。

## RQ3：C 是否降低搜索成本？

比较：

```text
multi-lowering with C prior
multi-lowering without C prior
```

## RQ4：End-to-end backend competitiveness

可以保留现有 RVV/GPU 外部 baseline，作用是说明：

> 最终 backend 代码具有竞争力。

但不能用这些数字单独证明 IntentIR 因果收益。

## 最小可发表实现范围

时间受限时，建议：

- 当前 DyT 作为端到端 canary；
- 1 个 reduction/softmax family，至少 3 个 lowering form；
- 1 个 scan、sort 或 conv family，至少 2 个 lowering form；
- 至少一个 target 上完成完整三路消融；
- 第二 target 只做 strategy-selection case study 也有价值。

---

# 12. 与 SCRC 前文的逻辑一致性审查

# 12.1 不冲突的部分

## J 仍是唯一语义载体

multi-lowering 从不生成 J'，所以不破坏所有权模型。

## C 仍是 implementation selection

C 只指导 schedule，不参与 J denotation。

## π 仍然只服务 Replay/Gate1

production backend 完全不读取 π。

## Replay isolation 不受影响

第四章发生在 SCRC certification 之后，不给 Replay 增加任何输入。

## Gate1/Gate2 的语义不变

第四章不增加新的 SCRC Gate，也不把 backend conformance 塞进 Gate2。

## 新前端工程哲学不变

Source frontend 仍只参与 Lift/Replay/Compile/Run/Observe；production backend 只面对统一 J/C。

# 12.2 需要明确修订措辞的两处

## 冲突点一：C retune 后的 conformance

C 文档中可以保留 target-retuned schedule 的 conformance 记录，但必须明确：

> backend conformance 是 compiler CI、debug 和 evaluation audit；它不能作为“候选属于合法搜索空间”的数学 oracle。

正确性依据仍是 pass/lowering contract。

若论文写成：

```text
每个 retuned 候选必须先运行 RefEval(J) 才能被视为合法
```

会重新退化成经验验证搜索，应删除。

## 冲突点二：当前 B/result relation 的模糊度

当前 `tolerance` 过宽，不足以作为 pass precondition。需要替换为闭合 relation class，Q 保持外置。

# 12.3 A/B schema 改动如何不破坏旧证书

任何 A/B schema 变更都必须：

1. 增加 schema/materializer/evaluator 版本；
2. 更新 TypeCheck；
3. 更新 Replay prompt 对新字段的消费；
4. 重新跑 Gate1/Gate2；
5. 不将旧 receipt 静默升级成新版本证书。

这是系统版本升级，不是后端运行时修改 J。

# 12.4 防止 A 过度抽象导致 Gate1 失效

每次提高 A 的抽象层次时，必须满足四项准入：

1. A 仍能 RefEval；
2. A/B 仍包含所有会改变允许结果的语义；
3. π 只需表达 realization，不需要新增语义原子；
4. source-blind Replay 仍可在 Gate1 observer 下闭合。

若某个 Source 实现中的关键算术关系只能放进 π 才能闭合，说明 A 抽象过度或 J carrier 不完整。

---

# 13. Novelty 审查

# 13.1 单个组件本身并不新

下列单独概念已有大量工作：

- high-level tensor IR；
- algorithm/schedule separation；
- MLIR multi-stage lowering；
- tensor program superoptimization；
- program raising/lifting；
- LLM kernel generation；
- dynamic or probabilistic equivalence testing。

因此不能把 novelty 写成：

> 我们首次用 MLIR 做多 lowering。

也不能写成：

> 我们首次让 compiler 根据 tensor semantics 选择实现。

# 13.2 真正可辩护的空位

本方法的 novelty 是完整链条：

\[
\boxed{
\text{schedule-specialized heterogeneous Source}
\xrightarrow{\text{SCRC}}
\text{immutable certified }J=(A,B)
\xrightarrow{\text{source-independent multi-lowering}}
\text{target-specific implementation}.
}
\]

具体是四点的合取：

1. **输入端不同**：输入不是已有数学图，而是 Triton/TileLang/CUDA 等已经高度优化的 Kernel；
2. **恢复对象不同**：恢复的是 `J=(A,B)`、C、π 的分层 artifact，而不是单一等价程序；
3. **认证方式不同**：使用 source-blind Replay closure + Source/Replay/J test-domain closure；
4. **后端接口不同**：production backend 不读取 Source 或 π，只把固定 J 当成 multi-lowering interface。

# 13.3 与相邻工作的边界

| 工作方向 | 已有能力 | IntentIR/SCRC 的差异 |
|---|---|---|
| Halide / TVM / Ansor / Nautilus | 从已有高层算法/数学描述做 schedule 或 high-level optimization | IntentIR 解决高层 intent 已丢失、只剩优化 Source 的逆向入口问题 |
| Tenspiler / mlirSynth | 将顺序/低层程序 raise 到 tensor/high-level MLIR | 不处理相同的多前端 schedule-specialized GPU realization、J/C/π 因子化和 source-blind replay certificate |
| Mirage / Prism | 从已有 tensor program 做 algebraic/schedule superoptimization | IntentIR 不主张新的 superoptimizer search algorithm；它恢复并认证这些 optimizer 通常假设已经存在的输入接口 |
| CuBridge | 从专家 CUDA attention kernel lift–transfer–lower，并以用户 PyTorch spec 引导 | IntentIR 的目标是多前端、Source 自身恢复 J、π 不进入 production lowering，并在目标后端重新多路 lowering，而非保留专家执行 orchestration 进行 reference-guided reconstruction |
| 一般 LLM kernel agents | generate–test–profile–refine | 本方法的候选合法性不由运行测试决定；LLM 只在 Source→artifact 与 source-blind Replay 中使用 |

# 13.4 最稳妥的 novelty claim

英文：

> **IntentIR turns an already optimized, schedule-specialized kernel into an SCRC-certified multi-lowering interface. SCRC recovers an immutable semantic contract J=(A,B), non-binding implementation hints C, and a source-realization witness π. The production backend consumes only J and C: Layer A exposes structured implementation families, Layer B supplies immutable semantic side conditions for compiler legality, and Layer C prioritizes target schedules. The backend therefore selects among contract-preserving lowerings of a fixed J rather than translating the source realization or using execution tests to guess correctness.**

中文：

> **IntentIR 将一个已经优化并被 Source realization 固化的 Kernel，恢复成经 SCRC 认证的多实现 lowering 接口。SCRC 得到不可变语义合同 J=(A,B)、非绑定实现提示 C 和 Source realization witness π；production backend 只消费 J 与 C。Layer A 暴露结构化实现族，Layer B 提供决定 compiler legality 的固定语义侧条件，Layer C 为目标调度提供搜索先验。因此，后端是在固定 J 的保合同 lowering 中进行选择，而不是翻译 Source realization，也不是通过运行测试猜测正确性。**

# 13.5 Novelty 成立所需的实验条件

如果最终只有：

```text
J → 一个 lowering → 调 tile
```

novelty 仍主要是 SCRC 与 semantic lifting，第四章较弱。

第四章要具有独立说服力，至少需要证明：

1. 同一 J 对应两个以上结构明显不同的 lowering；
2. B 真实参与 applicability，而不是装饰 metadata；
3. multi-lowering 比 schedule-only 更快；
4. pass 是 family-level，而不是 benchmark name special case；
5. production backend 不读取 π/Source；
6. C 只影响搜索效率，不改变合法性。

满足这些条件后，第四章具有真实 novelty。它的新意不是某个具体 softmax/scan lowering 算法，而是：

> **首次将 open-world、SCRC-certified semantic recovery 直接转化为 source-independent multi-lowering optimization interface。**

“首次”字样仍需在正式投稿前做完整 related-work 核查；当前更稳妥的表述是：

> **We are not aware of prior work that jointly provides this recovered-and-certified optimization interface across schedule-specialized heterogeneous kernel frontends.**
