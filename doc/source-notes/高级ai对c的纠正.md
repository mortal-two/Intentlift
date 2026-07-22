## 总判断

这份文档是目前关于 (C) 的设计里**最接近正确的一版**，但我不建议原样实现。

我的审查结论是：

[
\boxed{\text{所有权方向通过；Gate2 与 CSpec 需要架构级修订。}}
]

它最正确的地方，是把四类对象分清楚了：

[
J=\text{唯一语义对象},\qquad
C=\text{调度对象},\qquad
\pi=\text{源码 realization witness},\qquad
Harness=\text{运行事实}.
]

尤其是“生产 lowerer 不读 (\pi)”这一点必须保留，否则目标 backend 会退化成模仿原前端源码。

它对 Lift 与 Harness 的分工也基本正确：

> LLM 决定什么是 (C)、它绑定哪个源码事实、作用于哪个对象；Harness 只读取这次调用中的真实值。

这符合我们“理解权归 LLM，事实解析与裁决权归固定代码”的哲学，也没有引入新的前端语义分析器。

但是，它当前有一个**会削弱 (J/C) 语义分离证明的严重问题**，以及一个**可能重新制造通用 schedule IR 的工程风险**。

---

# 一、最严重的问题：不能让 Gate2 的语义腿消费 (C)

文档当前把 Gate2 写成：

[
O_{\mathrm{source}}
===================

\operatorname{Exec}(\operatorname{Compile}(S,C_0),H_0),
]

[
O_{\mathrm{replay}}
===================

\operatorname{Exec}(\operatorname{Compile}(R,C_0),H_0),
]

[
O_J
===

\operatorname{Exec}(\operatorname{Lower}(J,C_0),H_0).
]

并要求三条腿绑定同一个 (C_0)。

**Source 与 Replay 使用同一个 (C_0) 是完全正确的。**

因为 Gate1 要比较相同 specialization 下的编译结构，Gate2 也要执行相同实例。

但是：

[
\boxed{\text{(J) 的独立语义腿不能把 (C_0) 当作功能输入。}}
]

否则会出现一个致命反例。

## 反例：语义参数被错误放进 (C)

假设源码中的 `N` 决定真实逻辑 domain：

```text
for i in [0, N):
    Y[i] = X[i]
```

Lift 错误地没有把 (N) 写入 (J)，而是把它当成 (C)：

```text
C.c0 = N = 128
```

如果三条腿都使用 (C_0=128)：

* Source 按 128 执行；
* Replay 按 128 执行；
* `Lower(J,C0)` 也从 C 里取得 128 作为循环边界；

那么可能出现：

[
O_{\mathrm{source}}
===================

# O_{\mathrm{replay}}

O_J.
]

三角验证通过了，但事实上：

[
J
]

本身没有表达逻辑 domain。

也就是说，**同一个 C 被三条腿共同消费，反而可能掩盖 J/C 错误分类。**

这与引入独立 (J) 语义腿的原始目的相反。

---

## 正确改法：同 epoch，不同信息权限

应当改成：

[
O_{\mathrm{source}}
===================

\operatorname{Exec}
\bigl(
\operatorname{Compile}(S,\operatorname{Bind}*S(C*{\mathrm{source}})),
H_0
\bigr),
]

[
O_{\mathrm{replay}}
===================

\operatorname{Exec}
\bigl(
\operatorname{Compile}(R,\operatorname{Bind}*R(C*{\mathrm{source}})),
H_0
\bigr),
]

[
\boxed{
O_J
===

\operatorname{RefEval}(J,H_0).
}
]

正式 Gate2：

[
\boxed{
G_2=
[O_{\mathrm{source}}\equiv_B O_{\mathrm{replay}}]
\land
[O_{\mathrm{source}}\equiv_B O_J].
}
]

这里：

* Source 消费 (C_{\mathrm{source}})；
* Replay 消费同一个 (C_{\mathrm{source}})；
* `RefEval` 的接口里**根本没有 (C)**。

三条结果仍然属于同一个 artifact epoch：

[
e=
H(
J,\pi,C_{\mathrm{source}},S,R,H_0,
\text{tool versions}
),
]

`RefEval` 的 receipt 可以记录：

```text
artifact_epoch = e
associated_c_digest = ...
c_access = forbidden_by_interface
```

但不能把 (C) 对象交给 `RefEval`，再依赖它自报 `ignored`。

最强的信息隔离不是：

> 我把 C 给了它，它说自己没看。

而是：

[
\boxed{
\operatorname{RefEval}:J\times H\rightarrow O
}
]

这个函数类型里根本不存在 (C)。

---

# 二、必须区分 J reference 与 production lowering

文档里使用了模糊的：

[
O_J=\operatorname{Lower}(J,C_0).
]

这里必须明确是哪一种 `Lower`。

## 情况一：它是当前标量语义执行器

那么应改名为：

[
\operatorname{RefEval}(J)
]

或：

[
\operatorname{ScalarSemantics}(J).
]

它不应该读取 artifact (C)。它内部当然会有普通 CPU 循环，但这些循环只是 reference evaluator 的实现细节，不是 Intent 中的调度对象。

## 情况二：它是生产 target lowerer

例如：

[
\operatorname{Lower}*{\mathrm{CUDA}}(J,C),
\qquad
\operatorname{Lower}*{\mathrm{RVV}}(J,C),
]

那么它是**被验证对象**，不能同时充当 (J) 的独立语义 oracle。

正确结构是四个顶点：

[
\begin{array}{ccc}
&\operatorname{Source}&\
/&&\backslash\
\operatorname{Replay}&&\operatorname{RefEval}(J)\
&\backslash\quad/;&\
&\operatorname{Lower}_{t}(J,C_t)&
\end{array}
]

其中主 Gate2 三角仍是：

[
\operatorname{Source},
\quad
\operatorname{Replay},
\quad
\operatorname{RefEval}(J).
]

生产 backend 单独做：

[
\boxed{
G_{\mathrm{backend}}
====================

[
\operatorname{Exec}(\operatorname{Lower}_{t}(J,C_t),H_0)
\equiv_B
\operatorname{RefEval}(J,H_0)
].
}
]

文档后面已经意识到 target-only schedule 不能伪装成原三方 Gate2，应单列 backend conformance；这个方向是对的，但需要提前贯彻到第 7 节的正式定义中。

---

# 三、C 的数学定义不应从“谁消费它”开始

文档目前的核心原则是：

> C 的第一职责是指导生产 lowering。

这个工程定位合理，但不适合作为 (C) 的基础定义。

更稳固的数学定义应当先写：

[
\boxed{
C\text{ 是 denotation-preserving implementation-selection information}.
}
]

设：

[
\llbracket J\rrbracket(x)
]

是 (J) 对输入 (x) 允许的可观察结果集合，目标实现为：

[
I_t(J,c).
]

则 (C) 的纯调度性质定义为：

[
\boxed{
\forall c\in\operatorname{Adm}_t(J,H),\quad
\operatorname{Obs}
\left(
\operatorname{Exec}(I_t(J,c),x)
\right)
\in
\llbracket J\rrbracket(x).
}
]

也就是说：

* (C) 可以改变 loop；
* 可以改变 tile；
* 可以改变 vector width；
* 可以改变 warp/thread mapping；
* 可以改变 pipeline；
* 可以改变浮点 reduction order；

但不能改变：

[
\llbracket J\rrbracket.
]

对于浮点和并发程序，不应要求不同 (C) 的输出逐位相同，只要求它们都落在同一个 (J/B) 允许关系里。

然后再说：

> production lowerer 是 C 的主要消费者。

这样定义顺序不会反过来让 lowerer 的工程需要决定什么算 (C)。

---

# 四、当前 CSpec 太容易演化成第二套通用 schedule IR

文档要求每个 slot 至少包含：

```text
role
target_ref
axis_ref
legal_domain
strength
retunable
required
```

并希望由一个小型、版本化的全局词汇表达 tile、vector、warp、pipeline 等调度。

这些字段本身并非错误。

问题是：

[
\boxed{
\text{不应要求每个源码 C slot 都完整拥有这些字段。}
}
]

否则系统会逐渐变成：

```text
遇到一种新 schedule
    -> 新 role
    -> 新合法域
    -> 新 target/region 绑定
    -> 新 lowerer consumption
    -> 新 verifier
```

这与之前 `C.strategy + RuleCard + composer` 的膨胀路径高度相似，只是从 source realization 转移到了 schedule ontology。

文档最后也已经把 `target_ref`、`legal_domain` 和 C-purity profile 数量列为未决项；我认为这些不是实现细节，而是必须先处理的架构问题。

---

## 更合适的办法：一个 C，三个投影视图

不必增加三个顶层 IR，但应在逻辑上区分：

### 1. 事实绑定视图

服务于 source artifact 的可复现性：

```text
id
value_type
source binding sidecar
observed value
provenance
```

### 2. Replay 视图

服务于参数化重放：

```text
id
value_type
late-binding capability
```

无：

* 具体值；
* source name；
* source position。

C 的具体 use-site 主要由 (\pi) 和 Replay LLM 决定。

### 3. Lowering hint 视图

只有确实具有可迁移含义时才存在：

```text
role?
anchor_ref?
preferred_value?
retune_policy?
```

也就是：

[
C_{\mathrm{lower}}
==================

\operatorname{Project}*{\mathrm{lower}}(C*{\mathrm{source}})
]

可以是一个**部分投影**。

重要结论是：

[
\boxed{
\text{每个 source C 都应可绑定，但不必都能迁移到 target lowerer。}
}
]

例如一个非常 source-specific 的 meta-parameter：

* Gate1 为了复现源码必须记录；
* Replay 必须保留成空位；
* 生产 lowerer完全可以忽略；
* target autotuner重新选择自己的 schedule。

这不会损害正确性，只可能失去部分性能先验。

---

# 五、(C_{\mathrm{target}}) 不应只是给原 C slot 换一组值

文档当前把 target retune 示意为：

```json
{
  "parent_c_digest": "...",
  "values": {"c0": 256, "c1": 8}
}
```

这种形式适合简单情况，但不应成为一般定义。

现实中：

* Triton 的 `num_warps` 在 RVV 上没有直接对应物；
* source 的单一 tile 可能在目标上拆成多级 cache/register tile；
* target backend 可能引入源码里不存在的 knobs；
* 多个 source slots 可能合并成一个 target decision；
* 一个 source slot 也可能拆成多个 target decisions。

所以更准确的关系是：

[
\boxed{
C_t
===

\operatorname{Schedule}*t
\bigl(
J,
\operatorname{Hint}(C*{\mathrm{source}}),
\operatorname{Device}_t
\bigr).
}
]

并保存一个部分映射 receipt：

[
M_t:
C_{\mathrm{source}}^{\mathrm{hint}}
\rightharpoonup
C_t.
]

允许：

* accepted；
* transformed；
* split；
* merged；
* ignored；
* target-introduced。

而不是要求：

[
\operatorname{Ids}(C_t)
=======================

\operatorname{Ids}(C_{\mathrm{source}}).
]

这会避免为了跨目标复用而设计一套过强的“万能 schedule IR”。

---

# 六、`legal_domain` 不能直接由 Lift LLM 写成可信真值

调度值是否合法，通常不仅取决于常数类型或范围。

例如单 tile kernel 可能要求：

[
\texttt{BLOCK_N}\ge N.
]

合法性依赖：

* (J) 的逻辑 extent；
* 当前 specialization；
* target 资源；
* shared memory；
* register pressure；
* frontend 编译限制；
* source 是否真的对该值参数化。

因此：

[
\operatorname{Adm}_t
====================

\operatorname{Adm}_t(J,H,\mathrm{device},\mathrm{toolchain})
]

而不是一个简单的、完全由 Lift 声明的：

```json
{"min": 16, "power_of_two": true}
```

建议把：

```text
legal_domain
```

改成：

```text
proposed_domain
```

或：

```text
candidate_domain
```

它是 LLM 产生的候选，不是可信事实。

真正的合法性来自：

* frontend compile；
* target verifier；
* resource checker；
* Gate2/backend conformance。

同样，当前单个 `retunable` 也混淆了至少三个概念：

```text
source_rebindable
target_retunable
semantically_nonessential
```

三者不等价。

一个硬编码的 source tile：

* 可能不是 source-rebindable；
* 但语义上仍是纯调度；
* target backend 仍然完全可以 retune。

---

# 七、J/C 所有权应按“用途”分类，不应按“源码参数”二选一

文档正确地指出，`tl.constexpr` 只表示绑定时机，不决定它属于 J 还是 C。

还要再前进一步：

[
\boxed{
\text{同一个 source parameter 可以同时产生 J 事实和 C hint。}
}
]

例如：

```python
HEAD_DIM: tl.constexpr
```

它可能同时决定：

1. 逻辑 tensor 的最后一维大小；
2. reduction domain；
3. vectorization width 的 source preference。

那么：

* 逻辑 extent 必须进入 (J)；
* vectorization preference 可以进入 (C)。

不能因为一个参数有调度用途，就把整个参数归入 (C)；也不能因为它具有语义用途，就禁止产生 C hint。

所以 `CBindingMap` 最好允许：

[
\text{one source fact}
\rightarrow
\begin{cases}
\text{J specialization fact},\
\text{derived C hint}.
\end{cases}
]

所有权属于**事实和使用关系**，不是属于源码名字。

---

# 八、C-purity intervention 有价值，但不应进入基础在线 Gate2

文档建议为一个 slot 选择：

```text
tile=64
tile=128
tile=256
```

然后重新运行三角，以判断它是否真的是纯 (C)。

这个实验有价值，但更适合作为：

* schema/classifier CI；
* 论文消融；
* target retune conformance；
* 高风险 slot 的抽样审计。

不应成为每个 artifact 的基础在线 Gate2 条件，因为：

1. source 未必支持重新绑定；
2. 合法值难以机械生成；
3. 不同 C 可能合法改变浮点 reduction order；
4. 成本会随 slot 数量快速增长；
5. 主 Gate1+Gate2 已经可以认证当前 (C_{\mathrm{source}}) 下的 artifact。

正确实验判据不是：

[
O(C_0)=O(C_1)=O(C_2)
]

逐位相等，而是：

[
\boxed{
\forall c\in T_C,\quad
\operatorname{Obs}(I(J,c),x)
\in
\llbracket J\rrbracket(x).
}
]

并且只对：

```text
source_rebindable = true
```

或：

```text
target_retunable = true
```

且已验证 admissible 的值运行。

---

# 九、它与 LLM-first 哲学的最终关系

这份文档**没有天然违背** LLM-first。

它最好的部分正是：

* Lift LLM 决定 C 分类和 source binding；
* Harness 只解析事实；
* Replay 看到无值 C；
* Injector 只绑定，不生成源码结构。
* production lowerer 不读取 (\pi)。

这些都应保留。

真正的哲学防线应当是：

[
\boxed{
\text{C 的核心 schema 不需要完整理解所有 schedule；
未知 C 只降低性能利用率，不影响 J 的正确性。}
}
]

因此：

[
\operatorname{Lower}(J,\varnothing)
]

原则上应当仍可通过 generic fallback 工作。

一个 target 不理解某个 C hint：

* 可以记录 ignored；
* 可以重新 autotune；
* 可以使用默认 schedule；

不能因此说 J 不可执行。

换句话说：

> **J 必须完整；C 可以稀疏。**
> **C 的不完整影响性能与 source fidelity，不影响语义 soundness。**

这才真正避免新 frontend、新 schedule、新 target 一出现，就必须扩展核心固定代码。

---

# 十、我建议采用的最终模型

```text
Source + PrivateHarness
        │
        ▼
Lift LLM
  ├── J
  ├── π
  ├── C_source slots
  ├── optional portable C hints
  └── privileged source binding map
        │
        ▼
Generic Resolver
  只解析 observed values 和 provenance
```

## Gate1

```text
Replay LLM(J, π, value-free C projection)
    -> complete parameterized source

Compile(Source, C_source)
Compile(Replay, C_source)

frontend-local hard closure
```

即：

[
F_f(S,C_s)=F_f(R,C_s).
]

## Gate2 主三角

```text
Source(C_source)
Replay(C_source)
RefEval(J)             # 接口中没有 C
```

即：

[
O_S\equiv_B O_R,
\qquad
O_S\equiv_B O_J.
]

## Production lowering

```text
C_target =
    LoweringAgent/Autotuner(
        J,
        optional portable hints from C_source,
        target
    )

Target = Lower(J, C_target)
```

然后单独验证：

[
O_{\mathrm{target}}\equiv_B O_J.
]

---

# 给 agent 的直接结论

> 这份 C 设计的所有权边界总体正确，应保留 Lift LLM 分类、privileged CBindingMap、Harness 事实解析、Replay 隐藏具体值、Injector 只绑定、source/replay 同 C 编译、production lowerer 不读取 π，以及 source/target C lineage。
>
> 但不能按当前第 7 节原样实现。Source 和 Replay 必须消费同一个 (C_{\mathrm{source}})，而 (J) reference semantic leg 必须在接口层面禁止访问 C；同一个 C digest 只能作为 artifact epoch 元数据，不能作为 `RefEval` 的功能输入。否则一个错误归入 C 的语义参数可能被三条腿共同使用，导致三角错误通过。Production lowering 若消费 C，应作为独立 backend conformance 腿，而不是 J 的语义 oracle。
>
> 同时，不应要求每个 C slot 都完整具有 role、target、axis、legal domain、strength 与 retunable 合同。应把 source binding、Replay late binding 和可迁移 lowering hint 作为同一 C 的不同投影；只有确实可迁移的 slot 才产生 lowering hint。`legal_domain` 应是待验证的候选，不是 Lift LLM 声明的可信真值；`source_rebindable` 与 `target_retunable` 必须分开。(C_{\mathrm{target}}) 可以重参数化、扩展或忽略 (C_{\mathrm{source}})，不要求两者拥有相同 slot 集。
>
> C-purity 多值干预保留为 CI、论文消融和 target-retune conformance，不进入基础在线 Gate2。基础数学定义应是：对任意 admissible C，lowered implementation 的结果仍属于同一个 (\llbracket J\rrbracket)，而不是不同 C 的输出必须逐位相同。

最终应把文档的核心原则改成：

[
\boxed{
\text{(C) 是可选的、可审计的、保持 (J) denotation 不变的实现选择信息；}
}
]

[
\boxed{
\text{Source/Replay 用同一个 (C_{\mathrm{source}}) 闭合，}
\quad
\text{(J) reference 不读取 C，}
\quad
\text{production backend 独立消费或重调 (C_{\mathrm{target}})。}
}
]
