## 先把新的中心主张固定下来

**J 不是一个等待被机械 lower 的终点 IR，而是一个经过 SCRC 认证、可以重新展开优化空间的 MLIR 级编译接口。**

完整流程应当是：

[
\boxed{
\text{优化后的 Source}
\xrightarrow{\text{Lift + SCRC}}
J=(A,B),,C
\xrightarrow{\text{Intent-guided MLIR Search}}
P_t^\star
}
]

其中：

* (A) 描述计算语义；
* (B) 描述目标实现必须满足的合法性约束；
* (C) 保存 Source 的非绑定优化知识；
* SCRC 负责确认恢复出的 (J) 忠实于 Source；
* 后端负责在 (A/B) 允许的范围内搜索新的目标实现，而不是复刻 Source。

这延续了现稿最有价值的出发点：现有编译器通常从已经显式存在的高层 intent 开始，而你们面对的是已经被调度和硬件特化的 Triton、TileLang、CUDA Kernel，需要进行逆向语义恢复。

现稿也已经明确 (A/B) 是 binding，(C) 是 non-binding；这里不是改变三层定义，而是让后端真正消费这种区别。

---

# 一、最准确的 novelty 表达

## 中文版本

> **IntentIR 不只是从一个优化 Kernel 中恢复一个等价的高层程序，而是恢复一个经过认证、可供编译器重新搜索的受约束优化接口。Layer A 定义计算目标，Layer B 定义合法实现空间，Layer C 将 Source 中的优化知识保留为非绑定搜索先验。SCRC 认证这个接口忠实于 Source；随后，intent-guided backend 在 MLIR 中枚举满足 Layer B 的语义实现形式，再为目标硬件搜索调度、验证候选并选择最快实现。由此，一个原本已经固化为特定实现的 Kernel，被重新变成高级编译器可以优化的输入。**

## 英文版本

> **IntentIR does not merely recover an equivalent high-level program from an optimized kernel. It recovers a certified, constrained optimization interface. Layer A specifies the computation, Layer B defines the admissible implementation space, and Layer C retains source optimization knowledge as a non-binding search prior. SCRC certifies that this interface is faithful to the source. An intent-guided MLIR backend then enumerates semantic implementation forms admitted by Layer B, searches target-specific schedules, validates generated candidates against the recovered intent, and selects the best implementation. IntentIR thus turns a schedule-specialized implementation back into an optimizable compiler input.**

最简洁的一句话是：

[
\boxed{
\text{IntentIR reopens the semantic optimization space of an optimized kernel.}
}
]

即：

[
\boxed{
\text{IntentIR 重新打开已经被 Source 实现形式封闭的语义优化空间。}
}
]

---

# 二、这里的 novelty 具体不是什么、是什么

它**不是**：

* 发明 MLIR rewrite；
* 发明 tensor program search；
* 发明从程序提升到 tensor IR；
* 发明 schedule tuning；
* 发明某一种新的目标代码生成方式。

MLIR 已经提供了高层结构化运算、rewrite、Transform dialect、动态 legality 和目标信息描述等基础设施；Tenspiler 已经研究 verified lifting，Mirage 已经研究 tensor program superoptimization，CuBridge 已经研究从专家 CUDA attention Kernel 进行 lift–transfer–lower。([mlir.llvm.org][1]) ([arXiv][2])

你们可辩护的 novelty 是这条完整链：

[
\boxed{
\begin{aligned}
&\text{schedule-specialized heterogeneous Kernel}
\
&\xrightarrow{\text{open-world semantic lifting}}
\text{constrained optimization interface}
\
&\xrightarrow{\text{SCRC}}
\text{certified interface}
\
&\xrightarrow{\text{MLIR semantic implementation search}}
\text{target-specific reformulation}.
\end{aligned}
}
]

也就是：

> 已有 tensor optimizer 通常假设高层 tensor computation 已知；IntentIR 恢复并认证这样一个优化入口，使原本只有优化实现、没有高层程序的 Kernel 也能进入语义级搜索。

novelty 是**恢复并重新开放优化空间**，而不是单独的 lifting 或单独的 search。

---

# 三、J lower 到 MLIR 后，第一件事不是继续 lower

这是后端设计中最关键的原则：

[
\boxed{
\text{不要把 }J\text{ 一进入 MLIR 就立即降成 loops 或 }linalg.generic.
}
]

否则你们刚刚恢复出来的高级 operator、axis role、数值约束和合法实现边界，又会立即消失。

“J lower 到 MLIR”应理解为：

> **把 J materialize 成 semantic MLIR，而不是立刻变成低层 MLIR。**

建议保留两个层次。

## 1. Semantic MLIR

使用一个很薄的 `intent` dialect，或等价的高层 MLIR 表示：

```mlir
func.func @kernel(%input: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %result = intent.semantic_op %input
      {
        axis_roles = ...,
        numeric_policy = ...,
        layout_semantics = ...,
        valid_domain = ...,
        effects = ...,
        contract = #intent.contract<...>
      }
      : tensor<?x?xf16> -> tensor<?x?xf16>

  return %result : tensor<?x?xf16>
}
```

这里：

* op 和 dataflow 对应 Layer A；
* `contract` 对应 Layer B；
* Layer C 不应成为 payload IR 的合法性条件；
* (\pi) 不进入普通目标后端。

## 2. Standard MLIR

只有在选定某个实现策略后，才 lower 到：

[
\texttt{tensor/linalg/arith}
\rightarrow
\texttt{scf/vector/gpu}
\rightarrow
\texttt{target dialect}
\rightarrow
\texttt{LLVM/NVVM}.
]

MLIR 的 Linalg dialect本身能够显式表达 iterator 类型和 indexing maps，并支持 tiling、fusion、vectorization 以及向 loops、library call 或 target intrinsic 的 lowering；Transform dialect适合编排细粒度变换，PDL 可以表示结构化 rewrite，Dialect Conversion 可以根据具体 op 实例进行动态 legality 判断。([mlir.llvm.org][1])

---

# 四、J、B、C 在 MLIR 后端中的准确角色

## (A)：生成什么候选

(A) 决定：

* 这是怎样的 tensor computation；
* 输入输出关系；
* operator/dataflow；
* reduction axis；
* shape 与 dtype；
* layout interpretation；
* 哪些高层组合关系存在。

(A) 用来匹配可用的 semantic strategy。

---

## (B)：哪些候选合法

(B) 不是要求后端重演 Source 的执行过程，而是判断一个新实现是否仍然合法。

例如，(B) 应表达：

* 输入有效域；
* mask 与输出覆盖；
* shape/layout 约束；
* alias 和 effect；
* accumulation dtype；
* 是否允许 reassociation；
* 数值容差或精度政策；
* dependency 必须得到保持；
* 并发更新应满足何种结果合同。

它不应表达：

* 必须在 Source 的某一行放 barrier；
* 必须采用 Source 的 thread mapping；
* 必须以 Source 的 exact pointer arithmetic 实现；
* 必须使用 Source 的阶段数量。

例如，Layer B 可以说：

> 两个阶段之间存在必须保持的依赖。

但不应说：

> 必须使用 Source 中同样位置的一条 barrier 来保持依赖。

新策略可以通过：

* barrier；
* 分 kernel；
* 私有化；
* 重计算；
* 不同数据流；

来满足相同的 (B)。

因此：

[
\boxed{
B=\text{implementation legality},
\quad
B\neq\text{source execution script}.
}
]

这仍然保持：

[
J=(A,B).
]

只是避免把 Source realization 错误冻结进 (B)。

---

## (C)：只控制搜索顺序

(C) 可以影响：

* 首先尝试什么 tile；
* 首先尝试什么 vector width；
* 优先搜索哪些 mapping；
* 哪些 pipeline 参数更值得探索；
* 某类 strategy 的初始优先级。

但：

[
\boxed{
C\text{ 不决定候选是否合法。}
}
]

也就是说：

[
B \rightarrow \text{admissibility},
\qquad
C \rightarrow \text{priority}.
]

没有 (C) 时，后端仍应能搜索同一个合法空间，只是可能需要更多候选和更长时间。

---

# 五、J 到目标代码的完整后端流程

定义：

[
M_J=\operatorname{MaterializeMLIR}(J).
]

后端的目标不是：

[
\operatorname{Lower}(M_J)
]

得到唯一代码，而是构造一个候选集合：

[
\mathcal P_t(J).
]

完整流程如下。

---

## 第一步：生成 reference lowering

每个 semantic op 必须有一个确定性的 reference lowering：

[
\rho_{\mathrm{ref}}:
M_J
\rightarrow
M_{\mathrm{ref}}.
]

例如 lower 到：

[
\texttt{tensor/linalg/arith}.
]

它有三个用途：

1. 作为 (J) 的 executable reference；
2. 作为保守 fallback；
3. 作为 schedule-only baseline。

当前论文中 macro-operator 的 deterministic primitive lowering 可以保留，但它的身份应从“唯一 lowering”改成：

> **reference lowering。**

---

## 第二步：枚举 semantic strategies

为 semantic MLIR 定义一个 strategy registry：

```cpp
class IntentStrategy {
 public:
  virtual bool isApplicable(
      Operation *intentOp,
      const IntentContract &layerB,
      const TargetInfo &target) const = 0;

  virtual FailureOr<Operation *> materialize(
      Operation *intentOp,
      RewriterBase &rewriter) const = 0;

  virtual ScheduleSpace getScheduleSpace(
      Operation *materializedOp,
      const TargetInfo &target) const = 0;
};
```

每个 strategy 是一个**family-level MLIR rewrite**，而不是 benchmark-specific 代码。

它可以改变：

* 是否融合或拆分计算；
* 是否物化中间结果；
* 是否采用 streaming form；
* 是否采用单阶段或多阶段 reduction；
* reduction/dataflow 结构；
* layout transformation；
* 并行归约或私有化方式；
* target intrinsic 或 library implementation form。

这里真正新增的是：

[
\boxed{
\text{多个不同的 MLIR 计算图都可以实现同一个 }J.
}
]

---

## 第三步：用 B 过滤不合法策略

对于每个 strategy (\rho)，定义：

[
\operatorname{Pre}_{\rho}(B,t).
]

只有满足：

[
B,t\models\operatorname{Pre}_{\rho}
]

的策略才进入候选集合。

例如：

* strategy 需要浮点 reassociation，但 (B) 禁止，则拒绝；
* strategy 需要 contiguous layout，但 (B) 未保证，则拒绝或生成 guard；
* strategy 需要 shape 可整除，但只在部分 shape 成立，则生成 specialized path + fallback；
* strategy 改变更新顺序，但 (B) 要求 exact order，则拒绝；
* strategy 使用 target capability，但目标不支持，则拒绝。

这可以用普通 deterministic predicate 实现，不需要为 Source frontend 建立形式语义。

MLIR 的 Dialect Conversion 支持根据具体 operation、attribute 和 type 定义 dynamic legality，适合实现这一层；DLTI 可以携带目标硬件和 data-layout 信息。([mlir.llvm.org][3])

---

## 第四步：对每种 semantic strategy 做 schedule search

假设 semantic strategy 产生：

[
M_{\rho}.
]

再搜索：

[
\theta\in\Theta_t(M_{\rho}),
]

其中 (\theta) 可以包含：

* tiling；
* fusion；
* mapping；
* vectorization；
* memory placement；
* pipelining；
* unrolling；
* target-specific lowering choice。

MLIR Transform dialect可以作为 schedule 表达和执行层：payload IR 是候选 MLIR，transform IR 描述应用于它的变换序列。它的设计用途正是选择特定 IR 对象并编排 tiling、unrolling 等细粒度变换。([mlir.llvm.org][1])

Layer C 用于对 ((\rho,\theta)) 排序：

[
(\rho,\theta)\sim p_C,
]

但不会删除由 (B) 允许的其他候选。

---

## 第五步：lower 到目标 dialect

例如 NVIDIA 路径可以是：

[
\texttt{intent}
\rightarrow
\texttt{linalg/tensor}
\rightarrow
\texttt{vector/gpu/nvgpu}
\rightarrow
\texttt{nvvm}
\rightarrow
\text{binary}.
]

MLIR 的 `nvgpu` dialect 位于高层 GPU/Vector 与低层 NVVM 之间，而 NVVM dialect负责表示 NVIDIA 特定 intrinsic 和 ISA 能力。([mlir.llvm.org][4])

其他目标沿自己的 target dialect 或 Vector/LLVM 路径 lowering。

---

## 第六步：验证每个候选

对于生成的目标候选 (P)，执行：

[
O_P(x)
\approx_Q
O_J(x).
]

最终选中的候选最好在同一组隐藏输入上直接做：

[
O_S(x)\approx_Q O_P(x),
]

[
O_J(x)\approx_Q O_P(x),
]

并复用 SCRC 已记录的 Source 输出。

这样可信链是：

[
S
\xrightarrow{\mathrm{SCRC}}
J
\xrightarrow{\mathrm{candidate\ validation}}
P_t.
]

SCRC 本身已经把 TypeCheck、source-blind Replay、compiler-observer closure 和 Source–Intent 行为检查组合起来；目标后端只需要对统一的 (J) 接口验证候选，不需要为 Triton、TileLang、CUDA 各写一套目标验证器。

---

## 第七步：测量并选择

候选集合定义为：

[
\boxed{
\mathcal P_t(J)=
\left{
\operatorname{Codegen}*t
\left(
\operatorname{Schedule}*{\theta}
\left(
\rho(M_J)
\right)
\right)
;\middle|;
\begin{array}{l}
\rho\in\mathcal R_t,\
B,t\models\operatorname{Pre}_{\rho},\
\theta\in\Theta_t(\rho),\
\operatorname{Validate}_J(P)=1
\end{array}
\right}.
}
]

最后选择：

[
\boxed{
P_t^\star
=========

\arg\min_{P\in\mathcal P_t(J)}
\operatorname{Latency}_t(P).
}
]

或者采用吞吐、内存、能耗等复合 cost。

这里不需要发明复杂的新搜索算法。第一版可以是：

* strategy 枚举；
* cost model 剪枝；
* 固定预算 autotuning；
* 实测选择。

论文的创新点不是搜索算法本身，而是让以前无法进入这种搜索的优化 Source，重新拥有了一个可搜索的高层接口。

---

# 六、后端算法可以直接写成这样

```python
def compile_intent(J, C, target, tests):
    # J = (Layer A, Layer B)
    semantic_mlir = materialize_semantic_mlir(J)
    contract = J.layer_b

    target_info = query_target(target)
    candidates = []

    # Always available.
    semantic_forms = [reference_lowering(semantic_mlir)]

    # Enumerate alternative implementation forms.
    for strategy in strategy_registry.match(semantic_mlir):
        if strategy.is_applicable(contract, target_info):
            semantic_forms.append(
                strategy.materialize(semantic_mlir)
            )

    for form in semantic_forms:
        schedule_space = generate_schedule_space(
            form=form,
            target=target_info,
            prior=C,  # ordering/seed only
        )

        for schedule in schedule_space:
            candidate_mlir = apply_transform(form, schedule)

            if not check_layer_b_contract(
                candidate_mlir, contract, target_info
            ):
                continue

            binary = lower_and_compile(
                candidate_mlir, target_info
            )

            # Correctness before performance.
            if not validate_against_intent(
                binary, J, tests
            ):
                continue

            latency = benchmark(binary)
            candidates.append(
                (latency, binary, strategy, schedule)
            )

    if not candidates:
        return compile_reference_fallback(J, target_info)

    return min(candidates, key=lambda item: item[0])
```

这段算法已经把论文的核心思想全部体现出来：

* (J) 决定计算和合法性；
* strategy 改变实现形式；
* schedule 决定目标映射；
* (C) 只影响搜索效率；
* 候选先验证、再测速；
* reference lowering 保证始终存在 fallback。

---

# 七、MLIR 中建议实现的五个模块

后端团队不需要一开始建设大型通用系统。需要的是五个边界清楚的模块。

## 1. `Intent` dialect

负责表示：

* Layer A semantic ops；
* axis、shape、dtype；
* Layer B contract；
* reference semantics。

它必须保留 operator identity，不能过早退化成 generic loops。

---

## 2. `IntentStrategyInterface`

每种 strategy 实现：

```text
match(A)
check_precondition(B, target)
materialize_mlir()
schedule_space()
```

这里写一次的是 semantic family strategy，不是 Source frontend adapter，也不是 per-kernel special case。

---

## 3. `ContractChecker`

输入：

[
(M,B,t)
]

输出：

```text
LEGAL
ILLEGAL
REQUIRES_GUARD
```

它只检查 IntentIR/MLIR，不读取原始 Triton、TileLang 或 CUDA Source。

---

## 4. `TransformGenerator`

负责：

* 目标 schedule space；
* 将 C 变成搜索 prior；
* 生成 Transform dialect sequence；
* 保证不同策略使用公平的搜索预算。

---

## 5. `CandidateEvaluator`

负责：

* lower；
* compile；
* 用 (J) 和 Source cached outputs 验证；
* benchmark；
* 保存 strategy、schedule、target 和结果。

---

# 八、Layer B 必须做的一次语义清理

现稿中 Layer B 包含 index、mask、bounds、dependency、synchronization 和 controlled atomic，这一方向仍然成立。

但为了支持新的实现形式，建议逐字段检查：

> 这个字段描述的是“任何合法实现都必须满足的性质”，还是“Source 恰好使用的实现机制”？

例如：

| 信息                                   | 应放位置      |
| ------------------------------------ | --------- |
| 输出只定义在合法 index domain 上              | B         |
| 越界元素不参与 reduction                    | B         |
| accumulation dtype 为 fp32            | B         |
| 禁止/允许 reassociation                  | B         |
| 结果必须无数据竞争                            | B         |
| Source 使用了哪条 barrier                 | (\pi)     |
| Source 使用多少 stages                   | C         |
| Source 如何映射 thread/warp              | C 或 (\pi) |
| Source 的 exact shared-memory staging | C 或 (\pi) |
| 必须保持某种 logical dependency            | B         |
| 实现 dependency 的具体机制                  | 不应被冻结在 B  |

这是新 backend 能否真正离开 Source implementation form 的决定性条件。

---

# 九、这和普通“算子编译器”有什么不同

普通高层算子编译器的起点是：

[
\text{用户已经提供高层 operator/graph}.
]

你们的起点是：

[
\text{只有一个已经高度优化并混合了语义与 schedule 的 Kernel}.
]

因此你们不是声称：

> 我们比所有 tensor compiler 更会优化高层算子。

而是：

> 我们让只有优化实现、没有高层程序的 Kernel，重新获得可供 tensor compiler 或 MLIR optimizer 使用的高层入口。

后端仍然需要 strategy library。任何编译器要产生高性能代码都需要后端知识。

但其工程单位从：

[
\text{每个 Source frontend}
\times
\text{每个 target}
\times
\text{每个 Kernel}
]

变为：

[
\text{每个 semantic strategy}
+
\text{每个 target backend}.
]

一旦 Source 被 SCRC 认证成同一种 (J)，其来源是 Triton、TileLang 还是 CUDA，不再影响 semantic strategy。

你们项目原本的哲学也是让 LLM 吸收开放世界的前端复杂性，再用统一的小型验证核心限制错误。

---

# 十、性能实验应该如何设计

## 1. 当前绝对性能实验可以保留

当前 RVV、GPU 对外部 baseline 的性能结果仍可用于说明：

> 使用 IntentIR 的后端能够生成有竞争力的代码。

但它们不能单独证明新的 semantic-search novelty，因为还混入了：

* codegen；
* target intrinsic；
* cost model；
* dispatch；
* backend 工程。

所以把它们定位成：

# **End-to-end backend competitiveness**

而不是：

# **IntentIR representation causes all speedup**

---

## 2. 必须增加一组核心因果消融

从同一个 (J) 和同一个 MLIR backend 出发，比较四个版本。

### A. Reference

只使用 deterministic reference lowering：

[
J
\xrightarrow{\rho_{\mathrm{ref}}}
M_{\mathrm{ref}}
\xrightarrow{\text{default schedule}}
P_{\mathrm{ref}}.
]

### B. Schedule Search

固定 reference implementation form，只搜索 schedule：

[
J
\xrightarrow{\rho_{\mathrm{ref}}}
M_{\mathrm{ref}}
\xrightarrow{\theta\text{ search}}
P_{\mathrm{sched}}.
]

这基本对应现稿的 retune；当前实现主要重新搜索 vector width 和 tile size，因此它只能证明 schedule decoupling。

### C. Semantic Search

允许更换 semantic strategy，同时对每个 strategy 使用与 B 相同的 schedule tuner：

[
J
\xrightarrow{\rho\text{ search}}
M_{\rho}
\xrightarrow{\theta\text{ search}}
P_{\mathrm{semantic}}.
]

### D. Semantic Search + C Prior

与 C 完全相同的候选空间和预算，只使用 Layer C 改变搜索顺序：

[
P_{\mathrm{semantic+C}}.
]

---

## 3. 最重要的结果不是相对 Triton 的速度

最重要的主指标是：

[
\boxed{
\frac{\operatorname{Perf}
(P_{\mathrm{semantic}})}
{\operatorname{Perf}
(P_{\mathrm{sched}})}
}
]

它直接回答：

> 高级 intent 是否让后端获得了 schedule tuning 无法获得的新实现形式？

另外报告：

[
\frac{
\text{达到最终最佳性能 95% 所需候选数，无 C}
}{
\text{达到最终最佳性能 95% 所需候选数，有 C}
}
]

它回答：

> Layer C 是否真正作为有价值、但非绑定的 prior？

---

## 4. 每个性能结果还要报告“选了什么”

建议有一张非常直观的表：

| Kernel | Reference form | Target 1 selected form | Target 2 selected form | Speedup over schedule-only |
| ------ | -------------- | ---------------------- | ---------------------- | -------------------------: |
| K1     | form A         | form B                 | form C                 |                        ... |
| K2     | form A         | form A                 | form B                 |                        ... |

还应报告：

* 有多少 Kernel 最终选择了不同于 reference/Source 的 strategy；
* strategy 是否随 shape 改变；
* strategy 是否随 target 改变；
* 每种 strategy 的适用条件来自哪些 B 字段；
* 所有最终候选是否通过 J/Source validation。

这比只展示一组总速度更能证明“重新打开优化空间”。

---

# 十一、跨前端 J equality 不应是主实验

不需要要求：

[
J_{\mathrm{Triton}}
===================

# J_{\mathrm{TileLang}}

J_{\mathrm{CUDA}}.
]

不同 Lift 完全可以产生不同但等效的 MLIR 表示。

真正与新主张相关的可选实验是：

> 语义相同但来源不同的 Kernel，被分别 SCRC 认证以后，进入同一个 MLIR backend，是否能够进入相同的 strategy space，并得到接近的最终策略和性能？

也就是看：

[
\text{downstream optimization convergence},
]

而不是序列化文本或 graph hash 相等。

这项实验可以做，但不是新 novelty 成立的必要条件。

---

# 十二、SCRC 在新故事中的位置

SCRC 不再是论文的终点，而是后端搜索的可信入口。

完整故事为：

[
\boxed{
\begin{aligned}
\text{IntentIR} &: \text{恢复可执行、受约束的优化接口};\
\text{SCRC} &: \text{认证该接口忠实于 Source};\
\text{MLIR backend} &: \text{在该接口上搜索目标实现};\
\text{candidate validation} &: \text{检查最终实现仍满足 }J.
\end{aligned}
}
]

因此：

[
\boxed{
S
\xrightarrow{\mathrm{SCRC}}
J
\xrightarrow{\mathrm{semantic\ search}}
P_t^\star
}
]

是整篇论文最清楚的总图。

SCRC 不是替代后端正确性，而是把所有异构 Source 统一锚定到一个可执行的 (J)。后端只对 (J) 工作。

---

# 十三、论文结构应该怎样改

## Section 3：IntentIR

继续保留：

* Layer A；
* Layer B；
* Layer C。

但增加一段：

> J=(A,B) defines a constrained optimization interface rather than a unique lowering.

并明确：

[
\operatorname{Impl}_t(J)
]

是一个实现集合，不是一个程序。

---

## Section 4：Lifting and SCRC

讲：

* raw Source → (J,C,\pi)；
* source-blind Replay；
* Gate1；
* Gate2；
* SCRC-certified (J)。

---

## 新 Section 5：Intent-Guided MLIR Implementation Search

内容依次是：

1. Semantic MLIR materialization；
2. Reference lowering；
3. Strategy registry；
4. B-guided legality；
5. C-guided search；
6. target lowering；
7. candidate validation and selection。

---

## Evaluation

三组 RQ 即可：

### RQ1：Recovery and certification

SCRC 能否可靠地产生 (J)。

### RQ2：Reopened optimization space

Semantic search 是否优于 schedule-only search。

### RQ3：Portability and search efficiency

不同 target 是否选择不同实现，C 是否减少搜索成本，最终代码是否有竞争力。

---

# 十四、可以直接给后端团队的指令

你可以把后端工作概括为这七条：

1. **不要把 J 立即 lower 到 loops。**
2. **先实现一个保留 A/B 的 semantic MLIR dialect。**
3. **每个 semantic op 提供一个 deterministic reference lowering。**
4. **为可优化的 semantic region 注册多个 family-level strategy rewrite。**
5. **每个 strategy 必须用 B 和 target info 判断 applicability。**
6. **在 strategy 之后使用统一 Transform/schedule search；C 只负责排序和 seed。**
7. **所有候选先对 J/Source 输出验证，再测速并选择；无候选时使用 reference fallback。**

---

## 最终收敛

这篇论文最强的表达不再只是：

> 我们从 Triton、TileLang、CUDA 恢复一个 portable IR。

而是：

> **我们将一个已经被硬件和 schedule 固化的 Kernel，逆向恢复成一个经认证的 MLIR 优化问题。J=(A,B) 定义计算目标和合法实现空间，C 提供非绑定先验；后端据此重新选择语义实现形式和目标调度，而不是机械翻译原有实现。**

对应的核心公式就是：

[
\boxed{
P_t^\star
=========

\arg\min_{
\substack{
\rho,\theta\
B,t\models\operatorname{Pre}_{\rho,\theta}\
\operatorname{Validate}*J(P*{\rho,\theta})
}
}
\operatorname{Latency}*t
\left(
P*{\rho,\theta}
\right).
}
]

其中：

[
\rho=\text{semantic implementation form},
\qquad
\theta=\text{target schedule}.
]

这就是 J lower 到 MLIR 后，后端真正应该做的事情。
