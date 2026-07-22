你抓到的是那 **1% 里最关键的 1%**。我前一版确实引入了两个不属于你们系统的东西：

1. 把 `CBindingMap / portable hints / resolver` 写成了 Lift 除 (J,C,\pi) 之外的额外产物；
2. 凭空加入了 `LoweringAgent/Autotuner`。

这两点都应该撤回。

---

# 一、数学上，Lift 的证明相关输出必须严格只有三件东西

正确的 codomain 应当是：

[
\boxed{
\operatorname{Lift}_H(S)\rightarrow (J,C,\pi)
}
]

而不是：

[
\operatorname{Lift}(S,H)
\rightarrow
(J,C,\pi,B,\text{hint},\text{binding map},\ldots).
]

这里的 (H) 是已有 Harness 环境，是函数的外部环境或参数，不是 Lift 生成的第四份 artifact。

特别是 Replay 的输入必须严格满足：

[
\boxed{
R
=

W_f
\bigl(
J,\pi,\operatorname{Mask}_C(C);
U_f
\bigr)
}
]

其中：

* (U_f) 是固定、公共、与当前源码无关的 frontend 文档和工具上下文；
* (\operatorname{Mask}_C(C)) 是隐藏具体数值和源码绑定后的 C 视图；
* 不允许存在任何额外的、由当前源码条件化产生的 (Z(S))。

更正式地说，Replay 必须满足信息隔离：

[
\boxed{
R
\perp
(S,H,C_{\mathrm{value}},H_{\mathrm{binding}})
\mid
\left(
J,\pi,\operatorname{Mask}_C(C),U_f
\right).
}
]

也就是：给定 (J,\pi,\operatorname{Mask}_C(C)) 和公共 frontend context 后，Replay 的输出分布不能再受 source、Harness、完整 C 值或 source binding 影响。

如果再给 Replay 一个额外对象：

[
Z=\phi(S,H),
]

那么 Replay 实际变成：

[
R=W(J,\pi,\operatorname{Mask}_C(C),Z).
]

除非能证明：

[
Z=g(J,\pi,\operatorname{Mask}_C(C)),
]

否则 (Z) 就可以携带任意源码语义，原来的三对象因子化：

[
S\rightarrow(J,C,\pi)\rightarrow R
]

立即失效。

所以你说“从数学层面无法接受”是准确的。

---

# 二、文档里的 `CBindingMap` 应该删除为独立 Lift 产物

当前草案把 `CBindingMap` 定义为 Lift LLM 额外产生的 privileged sidecar，然后再交给 resolver。即使草案规定它不直接给 Replay，这仍然把 Lift 的输出空间从：

[
J\times C\times\Pi
]

扩张成了：

[
J\times C\times\Pi\times B.
]

而且 (B) 是一个从源码提取的、有表达能力的结构，之后很容易被其他流程误用。

更正确的做法是：

[
\boxed{
\text{绑定关系属于 Harness 的求值环境，不是 Lift artifact。}
}
]

或者，如果实现上必须在 C 内记录某个匿名来源坐标，那么它也是 **C 的内部私有字段或 provenance**，而不是一个第四对象；并且在 Replay 投影中必须被擦掉。

---

# 三、Harness 的正确数学身份：环境，不是 IR，也不是模型产物

Harness 更像语义学中的环境：

[
\rho_H:
{\text{logical input},\text{C slot}}
\rightarrow
{\text{concrete tensor/scalar/value}}.
]

例如：

[
\rho_H(c_0)=128,
\qquad
\rho_H(c_1)=4.
]

它还可以持有两个机械调用关系：

[
\rho_H^{S}(c_0)
===============

\texttt{Source.BLOCK_N}=128,
]

[
\rho_H^{R}(c_0)
===============

\texttt{Replay.c0}=128.
]

这些关系的作用只是：

* 把值送到正确的调用位置；
* 用同一个 C commitment 编译 Source 和 Replay；
* 记录 invocation、grid、device、compiler option 和 receipt。

Harness 不负责：

* 判断一个值是 J 还是 C；
* 给 C 生成语义 role；
* 生成 Replay use-site；
* 修改 Replay AST；
* 增加 loop、mask、reduction；
* 向 Replay 暴露源码参数名、位置或具体值。

当前草案中“Injector 只绑定，不修源码”的约束是正确的，应当保留。

---

# 四、C 应该是一个对象，不是 `C_source + hints + binding map`

可以直接定义：

[
\boxed{
C=(\Sigma_C,\nu_C)
}
]

其中：

* (\Sigma_C)：匿名参数化 slot，例如 `c0`、类型、binding kind；
* (\nu_C)：Harness 持有或提交的具体值。

例如：

[
\Sigma_C=
{
c_0:\texttt{constexpr<int>},
c_1:\texttt{compiler_option<int>}
},
]

[
\nu_C=
{
c_0\mapsto128,
c_1\mapsto4
}.
]

Replay 只接收投影：

[
\boxed{
\operatorname{Mask}_C(C)
========================

(\Sigma_C,\bot).
}
]

即：

```text
c0: constexpr<int>, value hidden
c1: compiler_option<int>, value hidden
```

Replay LLM 输出参数化源码：

```python
def kernel(..., c0: tl.constexpr):
    ...
```

完成生成并冻结后，Harness 才执行：

[
\operatorname{Bind}_H(R,C).
]

Source 与 Replay 分别使用：

[
\operatorname{Compile}
\left(
S,\rho_H^S(C)
\right),
]

[
\operatorname{Compile}
\left(
R,\rho_H^R(C)
\right).
]

这正是当前实现里已经存在、而且基本正确的机制：

* canonical `ScheduleSpec` 保存 `slots + values`；
* Replay view 只返回 slots；
* 具体 values 被视为 secret；
* Replay 生成后，Harness 再用同一 C 编译 Source 与 Replay。

所以没有必要创造：

* `C_source slots`；
* `optional portable C hints`；
* `privileged CBindingMap`；
* `Generic Resolver`；

作为新的顶层对象。

最多存在一个 C 的若干**确定性视图**：

[
V_{\mathrm{replay}}(C)=\operatorname{Mask}_C(C),
]

[
V_{\mathrm{gate}}(C)=C,
]

[
V_{\mathrm{lower}}(C)=C.
]

这些是一个对象的投影，不是额外 artifact。

---

# 五、谁决定某个 source 参数属于 C？

这里需要分清“坐标枚举”和“语义分类”。

Harness 可以机械地枚举：

```text
q0 = 第 0 个参数
q1 = 第 1 个参数
q2 = 第 2 个 constexpr
q3 = 第 0 个 compiler option
```

Harness 不能预先声明：

```text
q2 是 schedule
q1 是 logical domain
```

这种分类仍由 Lift 完成。

但是 Lift 的分类结果直接体现在 (J) 和 (C) 中：

* 若 `q1` 决定 logical domain，则对应事实进入 (J)；
* 若 `q2` 是纯 tile，则对应匿名 slot 进入 (C)。

不再额外生成：

```text
CBindingMap = q2 -> c0
```

作为第四对象。

实现上有两种干净方法：

### 方法 A：预先匿名化坐标

Lift 看到匿名 bindable coordinate，并直接在 C 中采用该 handle。

### 方法 B：关系是 C 的私有 provenance

若必须记录 `q2 -> c0`，它是 C 内部的私有 provenance，Replay view 擦除它。

两者都满足：

[
\operatorname{Lift}_H(S)=(J,C,\pi).
]

---

# 六、Harness 不应被整体送进 Lift 或 Replay

更准确的图不是：

```text
Source + PrivateHarness
        ↓
Lift LLM
```

而应该画成：

```text
                    Source S
                       │
                       ▼
                   Lift LLM
                       │
                  (J, C, π)
                       │
          ReplayView(J, MaskC(C), π)
                       │
                       ▼
                  Replay LLM
                       │
             Parameterized Replay R
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
Harness binds full C            Harness binds full C
to Source invocation            to Replay invocation
        │                             │
        ▼                             ▼
   Compile/Run S                 Compile/Run R
```

Harness 在旁边，作为隐藏执行环境：

```text
Harness H:
  concrete inputs
  complete C values
  source invocation coordinates
  replay invocation coordinates
  grid / device / compiler options
  receipts
```

但：

[
\boxed{
H\notin\operatorname{ReplayInput}.
}
]

Lift 是否需要看一个极小的、匿名化的 compile-time interface，是实现问题；无论如何，它不能生成一个额外 source-conditioned object 再送进 Replay。

---

# 七、Gate2 里的 Harness 关系

Gate2 应写成：

[
O_S
===

\operatorname{Exec}
\left(
S,\rho_H^S(C),x
\right),
]

[
O_R
===

\operatorname{Exec}
\left(
R,\rho_H^R(C),x
\right),
]

[
\boxed{
O_J
===

\operatorname{RefEval}(J,x).
}
]

即：

* Source 用完整 C；
* Replay 用同一完整 C；
* 标量 J 语义腿不消费 C。

当前实际主线中，Source 和 Replay 执行时使用 schedule values，而 scalar J lowering 忽略 C，这一点本来就是合理的。

Harness 可以让三者共享：

* 同一组逻辑输入；
* 同一个 artifact epoch；
* 同一 comparator；
* 同一 shape/runtime condition。

但不能让 J reference 从 C 中恢复逻辑 domain、predicate 或其他语义。

---

# 八、`LoweringAgent/Autotuner` 应该完全删除

这个确实是我凭空引入的，不属于你们当前架构。

当前核心只需要：

[
\boxed{
K_t=\operatorname{Lower}_t(J,C).
}
]

也就是：

```text
J + C
  -> Intent MLIR
  -> target lowering
  -> executable
```

草案本身描述的生产路径也是直接的 `J + C -> Intent MLIR + schedule annotations -> target IR`，并没有一个必要的 Lowering Agent。

因此删除：

```text
C_target =
    LoweringAgent/Autotuner(...)
```

当前正式架构里不需要：

* Lowering Agent；
* C 生成 Agent；
* 额外的 target C artifact；
* 新的 LLM 调度阶段。

如果未来真的实现 target autotuning，那只是 production backend 内部的可选优化：

[
C'\in\operatorname{Adm}_t(J)
]

然后验证：

[
\operatorname{Exec}
\left(
\operatorname{Lower}_t(J,C')
\right)
\in
\llbracket J\rrbracket.
]

但这不应进入当前核心定义，也不能被写成一个已经存在的系统角色。

---

# 九、修正后的唯一主架构

[
\boxed{
\operatorname{Lift}_H(S)=(J,C,\pi)
}
]

[
\boxed{
R=
\operatorname{ReplayLLM}
\left(
J,\pi,\operatorname{Mask}_C(C),U_f
\right)
}
]

[
\boxed{
G_1=
\left[
F_f
\left(
\operatorname{Compile}_H(S,C)
\right)
=

F_f
\left(
\operatorname{Compile}_H(R,C)
\right)
\right]
}
]

[
\boxed{
G_2=
\bigwedge_{x\in T}
\left[
\operatorname{Obs}_H(S,C,x)
\approx
\operatorname{Obs}_H(R,C,x)
\right]
\land
\left[
\operatorname{Obs}_H(S,C,x)
\approx
\operatorname{RefEval}(J,x)
\right].
}
]

生产 lowering 单独为：

[
\boxed{
K_t=\operatorname{Lower}_t(J,C).
}
]

整个证明相关 artifact 始终只有：

[
\boxed{
A=(J,C,\pi).
}
]

---

# 可以直接发给 agent 的修正

> 删除当前设计中作为独立 Lift 产物的 `CBindingMap`、`optional portable C hints` 和 `Generic Resolver`。证明相关的 Lift codomain 必须严格是 `(J, C, π)`；任何额外的 source-conditioned 对象，只要进入 Replay，就会成为未受约束的第四信息通道，使 `Source -> (J,C,π) -> Replay` 的因子化命题失效。
>
> Harness 不是 LLM 产物，也不是 IR。它是预先存在的隐藏求值环境，持有 concrete inputs、完整 C values、source/replay invocation coordinates、grid、compiler options 和 receipts。C 可形式化为 `(slot schema, concrete valuation)`；Replay 只接收 `MaskC(C)`，即匿名 slots 而无 concrete values、source names、positions 或 binding。Replay 生成并冻结参数化源码后，Harness 才把完整 C 分别绑定到 Source 与 Replay。Injector 只能做 argument/option binding，不能修改 Replay AST 或增加 use-site。
>
> 若实现需要保存 source coordinate 到 canonical C slot 的对应关系，它必须是 Harness 已有的机械调用关系，或 C 内部在 ReplayView 中被擦除的 private provenance；不能成为第四个 Lift artifact，也不能进入 Replay packet。
>
> 删除 `LoweringAgent/Autotuner`。当前 production 接口只写成 `Lower_target(J, C)`；是否忽略、消费或未来重调某些 C slot，是 backend 的实现与 conformance 问题，不新增一个 LLM 角色。当前 Gate2 仍是 Source(C)、Replay(C) 和 `RefEval(J)` 三角，其中 J reference 接口不消费 C。
>
> 最终不变量：
> [
> \operatorname{Lift}*H(S)=(J,C,\pi),
> \qquad
> R=W(J,\pi,\operatorname{Mask}*C(C)),
> ]
> [
> R\perp(S,H,C*{\mathrm{value}},H*{\mathrm{binding}})
> \mid
> (J,\pi,\operatorname{Mask}_C(C)).
> ]
> 任何 audit/location/diagnostic sidecar 即使保留，也必须是 proof-irrelevant：不进入 Replay、不进入 Gate verdict、不改变 lowering。

最简洁的一句话是：

[
\boxed{
\text{Harness 负责给符号赋值，不负责产生新的符号系统；}
\quad
\text{Replay 只看 }J+\pi+\operatorname{Mask}(C).
}
]
