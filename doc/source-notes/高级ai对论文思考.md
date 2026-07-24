下面这个版本，我建议直接作为 **PPoPP 10 页稿的骨架锁定**。整篇只讲一个科学故事：

[
\boxed{
\text{优化 Kernel 缺少可复用的高层语义接口，造成 }N\times M\text{ 的语义恢复与目标移植耦合。}
}
]

[
\boxed{
\text{IntentIR 用 SCRC 恢复并认证该接口，再由多个后端独立消费它。}
}
]

旧稿的 (N\times M) 出发点是对的：没有共享语义表示时，每一种 Source frontend 与每一种 target backend 之间都可能重复发生语义理解、正确性建立和目标优化；IntentIR 希望把接口改造成 (N) 侧恢复加 (M) 侧消费。

---

# 建议标题

## **IntentIR: Certified Semantic Interfaces for (N\times M) Kernel Portability**

也可以稍微强调后端：

## **IntentIR: SCRC-Certified Lifting and Multi-Lowering for Kernel Portability**

第一版更简洁，更像 PPoPP。

---

# 页面预算

按总共约 10 页控制：

| 部分                                |  目标页数 |
| --------------------------------- | ----: |
| Abstract                          |  0.25 |
| 1 Introduction                    |   1.2 |
| 2 Related Work                    |   0.8 |
| 3 SCRC-Certified Semantic Lifting |   1.8 |
| 4 IntentIR-Guided Multi-Lowering  |   1.8 |
| 5 Evaluation                      |   2.7 |
| 6 Limitations and Conclusion      |   0.3 |
| References                        | 约 1.2 |

图表已经包含在各节预算内。主文最多保留 **4 张图和 1 张双面板表格**。

---

# Abstract

摘要只做四件事：

1. 提出 (N\times M) 问题：优化 Kernel 的语义与 Source schedule 缠绕，无法直接作为多后端的公共输入。
2. 提出 IntentIR：从 Source 恢复 (J=(A,B))、(C) 和 (\pi)。
3. 提出两项核心方法：

   * SCRC 认证 Source 到 IntentIR；
   * J-guided multi-lowering 从固定 J 生成多个目标实现。
4. 给出三类结果：

   * 多前端 SCRC 覆盖；
   * 前端/后端工程解耦；
   * RVV/GPU 性能与 multi-lowering 消融。

不要在摘要中出现：

* O1–O7；
* FULL/PARTIAL/OOS；
* FIC/ESC；
* masked-J 细节；
* SMT；
* 大量证书术语。

---

# 1 Introduction

Introduction 不分小节，按六段写。

## 第一段：问题场景

高性能 Kernel 来自 Triton、TileLang、CUDA 等前端，并针对不同 GPU 和硬件进行深度特化。语义、索引、线程映射、存储层次和调度混合在 Source 中。

因此，一个 target backend 面对的不是干净的 tensor program，而是一个已经固化的实现。

## 第二段：(N\times M) 问题

给定：

[
F={f_1,\dots,f_N}
]

个 Source frontends，以及：

[
T={t_1,\dots,t_M}
]

个 target backends。

在没有公共高层语义接口时，每条 source-target 路径都可能重复：

* 理解 Source 算什么；
* 区分语义与 schedule；
* 建立正确性；
* 选择目标实现。

这里不要绝对声称所有系统真的实现了 (NM) 个完整编译器。更准确地说：

> 缺少高层接口会让 source-specific semantic recovery 与 target-specific implementation 在 source-target integration edges 上重复发生。

## 第三段：为什么现有系统不能直接解决

高层 tensor compiler 很擅长：

[
\text{已知高层计算}\rightarrow\text{目标实现}.
]

但你们的问题是反向的：

[
\text{优化实现}\rightarrow\text{可信高层计算}.
]

普通 LLM lifting 覆盖广但不可信；为每个前端建立完整形式语义和 verifier，又恢复了 N 侧工程负担。

## 第四段：核心方案

放全文最重要的总图：

```text
Triton ─┐
TileLang├── LLM Lift ── SCRC ── (J=(A,B), C, π)
CUDA ───┘                           │
                                    │ π stops here
                                    ▼
                                  J + C
                                    │
                         J-guided Multi-Lowering
                              ┌─────┴─────┐
                              ▼           ▼
                            GPU          RVV
```

SCRC 最终只认证三件事：

[
\operatorname{TypeCheck}
\land
\operatorname{Gate1}
\land
\operatorname{Gate2}.
]

当前最终设计也已将 artifact 严格限定为 ((J,C,\pi))，并将证书定义为所有权检查、source-blind Replay closure 和测试域行为闭合的合取。

## 第五段：为什么后端能够加速

一句话讲清楚：

[
\boxed{
A\text{ exposes alternatives;}
\quad
B\text{ filters alternatives;}
\quad
C\text{ prioritizes alternatives.}
}
]

后端不修改 J，不读取 Source 或 (\pi)。它根据 A 识别实现族，根据固定 B 判断哪些 lowering 合法，再使用 C 选择和调优参数。

## 第六段：三项贡献

只写三项：

### Contribution 1：SCRC

一个无需为每个 Source frontend 建立完整形式语义和专用 verifier 的逐 artifact 认证协议。

### Contribution 2：Certified Multi-Lowering Interface

IntentIR 不只是可执行 IR，而是一份经过认证的优化接口。后端可以在固定 J 下选择多个保合同实现。

### Contribution 3：Measured Decoupling and Performance

量化：

* 多前端覆盖；
* 新 frontend/target 的增量工程；
* shared lowering 的复用；
* multi-lowering 相比 schedule-only 的收益；
* RVV/GPU 绝对性能。

---

# 2 Related Work

第二章仍然叫：

# **2 Related Work**

控制在约 0.8 页，最多三个小节。

## 2.1 Tensor Compilers and Performance Portability

讨论：

* Halide；
* TVM / Ansor；
* MLIR / Transform dialect；
* tensor compiler 与 schedule search。

核心区别：

> 它们通常从已有高层 computation 开始；IntentIR 从已经调度特化的 Kernel 中恢复这个入口。

## 2.2 Semantic Lifting and LLM-Assisted Compilation

讨论：

* program lifting / decompilation；
* tensor-level lifting；
* LLM code translation、repair 和 compiler agents。

核心区别：

> IntentIR 不只生成一个高层候选，而是生成 ((J,C,\pi))，并用 source-blind Replay 检查该表示是否足以重新构造 Source realization。

## 2.3 Translation Validation and Implementation Search

讨论：

* translation validation；
* formal kernel equivalence；
* tensor superoptimization；
* equality saturation 或 implementation search。

核心区别：

> 验证系统通常需要受控语言语义；superoptimizer 通常假定 specification 已经存在。IntentIR 的工作是从开放的优化 Kernel 中恢复并认证 specification/optimization interface，再将其交给后端搜索。

Related Work 末尾只用一句话总结空位：

> IntentIR connects certified recovery from schedule-specialized kernels with source-independent target multi-lowering.

---

# 3 SCRC-Certified Semantic Lifting

这一章只回答：

[
\boxed{
\text{恢复出的 IntentIR 为什么值得相信？}
}
]

最多三个小节。

---

## 3.1 Factorized Artifact and Typed Ownership

定义：

[
\operatorname{Lift}_H(S)=(J,C,\pi),
\qquad
J=(\Sigma,A,B).
]

这里只简要说明角色：

* (J)：唯一显式语义载体；
* (C)：implementation selection；
* (\pi)：source-realization witness。

A/B/C 的完整设计留到第四章。

定义：

[
\mathsf{TC}(J,C,\pi).
]

正文只保留两个核心所有权性质：

[
\operatorname{SemDecl}(C)
=========================

# \operatorname{SemDecl}(\pi)

\varnothing,
]

[
\operatorname{SemRef}(C,\pi)
\subseteq
\operatorname{Ids}(J).
]

放一个小型表格说明 J、C、(\pi) 各自能放什么。

不要展开：

* FIC；
* erasure theorem；
* information theory；
* covert channel 的所有细节。

---

## 3.2 Source-Blind Replay Closure

Replay 只接收：

[
R=
W_f
\left(
J,\pi,\operatorname{MaskC}(C);U_f
\right).
]

它看不到：

* Source；
* C values；
* Harness；
* Gate1 diff；
* 旧 Replay。

定义 observer：

[
\Phi_f(P,c)
===========

\operatorname{Canon}_f
\left(
\operatorname{Observe}_f
\left(
\operatorname{Compile}_f(P,c)
\right)
\right).
]

Gate1：

[
\boxed{
G_1=
\left[
\Phi_f(S,c)
===========

\Phi_f(R,c)
\right].
}
]

准确结论：

> ((J,C,\pi)) 足以在 Source 隔离条件下，构造一个与 Source 落在同一 compiler-observer class 中的 Replay。

Relift 只用一段说明：

* Gate1 diff 返回 source-aware Lift；
* 每轮重新输出完整 ((J,C,\pi))；
* 每轮生成新的 source-blind Replay；
* 最多三轮。

---

## 3.3 Behavioral Closure and Certificate

对测试输入 (x)：

[
O_S(x),\quad O_R(x),\quad O_J(x).
]

比较政策 (Q) 外部冻结。

Gate2：

[
\boxed{
\begin{aligned}
G_2=
\bigwedge_{x\in T}
\Big[
&
\operatorname{Close}_Q(O_S,O_R)
\
{}\land{}&
\operatorname{Close}_Q(O_S,O_J)
\
{}\land{}&
\operatorname{Close}_Q(O_R,O_J)
\Big].
\end{aligned}
}
]

最终：

[
\boxed{
\mathsf{SCRC_Cert}
==================

\mathsf{TC}
\land
\mathsf{ReplayIsolation}
\land
G_1
\land
G_2.
}
]

只声明三种保证：

1. artifact 语义所有权受约束；
2. Source-blind Replay 在声明 observer 下闭合；
3. Source、Replay 和 J 在测试域内两两一致。

随机错误接受上界只用一句话陈述，证明放附录。

不要在主文单独介绍：

* FIC/ESC；
* observer monotonicity；
* masked-J 数学状态机；
* J-authority challenge。

---

# 4 IntentIR as a Certified Multi-Lowering Interface

这一章只回答：

[
\boxed{
\text{可信的 IntentIR 为什么对后端有用？}
}
]

最多三个小节。

---

## 4.1 The A/B/C Optimization Interface

完整介绍三层。

### A：Structured Executable Computation

A 是结构化 tensor program：

* map；
* select；
* reduce；
* scan；
* contraction；
* gather/scatter；
* structured loops 和 state relation。

当前 Intent dialect 本身就是以 Layer A tensor semantics、Layer B constraints 为基本组织原则。

### B：Immutable Semantic Boundary

B 固定不变，描述：

* domain；
* predicate/fill；
* access；
* alias/effect；
* update/conflict；
* order/numeric/result relation。

B 不包含：

* pass 名；
* target strategy；
* `fusion_allowed`；
* `use_tensor_core`；
* backend recipe。

### C：Non-Binding Implementation Selection

C 包含：

* tile；
* vector width；
* warp/thread count；
* pipeline；
* memory placement。

C 需要稳定引用 J node/axis，声明合法域和是否可 retune。当前 C 规范也明确规定：J 是唯一语义对象，C 是 source-neutral、可替换的 lowering hint，production backend 只消费 J+C，不读取 (\pi)。

### Deterministic Materialization

[
M_0=\operatorname{Emit}_v(J).
]

JSON 到 Intent MLIR 完全确定，不使用 LLM。

---

## 4.2 J-Guided Contract-Preserving Multi-Lowering

定义：

[
\Gamma_{J,t}
\vdash
M:\operatorname{Impl}(J).
]

每个 transformation (\rho)：

[
\frac{
\Gamma_{J,t}\vdash M:\operatorname{Impl}(J)
\qquad
\operatorname{Pre}*{\rho}(J,M,t)
}{
\Gamma*{J,t}\vdash
\rho(M):\operatorname{Impl}(J)
}.
]

搜索空间：

[
\operatorname{Reach}_t(J)
=========================

\left{
M
\mid
M_0\xRightarrow{\rho_1,\ldots,\rho_k}M
\right}.
]

核心表达：

[
A\rightarrow\text{可选 lowering family},
]

[
B\rightarrow\text{lowering applicability},
]

[
C\rightarrow\text{schedule values / priority}.
]

### 只放一个 softmax-shaped 示例

A：

```text
select(mask)
→ reduce(max)
→ sub
→ exp
→ reduce(sum)
→ div
```

同一个固定 J 可对应：

* generic reduction lowering；
* RVV vector-row lowering；
* GPU warp lowering；
* GPU block lowering；
* B 允许时的 hierarchical lowering。

这里不要声称任意 online recurrence 都能自动转为 two-pass。涉及非平凡数学恒等式的改变，必须有单独的保合同 compiler transformation。

---

## 4.3 Target Selection and Correctness Boundary

对合法状态：

[
K_M=\operatorname{Lower}_t(M).
]

最终：

[
M^\star
=

\arg\min_{M\in\operatorname{Reach}_t(J)}
\operatorname{Cost}_t(K_M).
]

C 可以决定：

* 第一批 tile；
* 搜索顺序；
* 初始 schedule。

但不能决定候选是否合法。

正确性链：

| 阶段                       | 依据                         |
| ------------------------ | -------------------------- |
| Source (\rightarrow J)   | SCRC                       |
| (J\rightarrow M_0)       | deterministic materializer |
| (M_i\rightarrow M_{i+1}) | compiler pass contract     |
| (M\rightarrow K_t)       | target lowerer             |
| 候选选择                     | 只比较性能                      |

Backend conformance 只作为 CI 和实验审计，不作为 production search 的合法性 oracle。

---

# 5 Evaluation

Evaluation 只回答 **两个问题**。

开头用约三分之一页介绍：

* 数据集；
* LLM；
* frontends；
* GPU/RVV 平台；
* correctness policy；
* compiler versions。

不要单独设 5.1 Experimental Setup。

---

## 5.1 Can a Fixed SCRC Core Scale Across Frontends?

这个小节同时回答：

* SCRC 覆盖；
* N 侧工程成本；
* SCRC 的核心组件是否必要。

### 实验一：SCRC 漏斗

将现有 100-kernel 和 FlagGems 数据去重，形成约 200 个 Kernel 的多前端 corpus。

只报告：

| 指标              | 含义                |
| --------------- | ----------------- |
| TypeCheck@1     | 首轮合法 artifact     |
| Gate1 closure@1 | 首轮 Replay closure |
| Gate1 closure@3 | 三轮内 closure       |
| Gate2 pass      | 三方行为闭合            |
| SCRC-certified  | 最终通过              |

按 Triton、TileLang、CUDA 分组。

### 实验二：工程解耦表

一个双面板表格即可。

#### Panel A：新增 frontend

| Frontend | Adapter ΔLoC | 人工时间 | SCRC core changes | GPU backend changes | RVV backend changes | Per-kernel rules |
| -------- | -----------: | ---: | ----------------: | ------------------: | ------------------: | ---------------: |
| Triton   |              |      |                   |                     |                     |                  |
| TileLang |              |      |                 0 |                   0 |                   0 |                0 |
| CUDA     |              |      |                 0 |                   0 |                   0 |                0 |

你们的主张不是“新增 frontend 零代码”，而是：

> 只增加 compile/run/observe glue，不增加 frontend formal semantics、专用 verifier 或 target-backend 分支。

最终 SCRC 设计本身也将新 frontend 的最低接口限制为 `Compile/Run/Observe`，并明确不要求 J→frontend lowerer、完整形式语义或 per-kernel RuleCard。

#### Panel B：backend reuse

| Lowering family | Artifacts | Source frontends | Targets | Source-specific branches |
| --------------- | --------: | ---------------: | ------: | -----------------------: |
| Family 1        |           |                  |         |                        0 |
| Family 2        |           |                  |         |                        0 |

核心结果是：

> 一条 J-level lowering 能服务来自多个 Source frontends 的 artifact。

### 实验三：一个紧凑的 SCRC intervention 表

不要再做 3,200 个旧式 mutants。

只做 30 个代表 Kernel 上的四种 intervention：

| Intervention            | 应由谁发现            |
| ----------------------- | ---------------- |
| 删除 (\pi)                | Gate1 closure 下降 |
| masked-J                | J 的操作性必要性        |
| 修改 J operator/predicate | Gate2            |
| 修改 (\pi) route/order    | Gate1            |

报告通过/拒绝率即可。

这张表替代：

* 旧 O1–O7 mutation；
* FULL/PARTIAL/OOS；
* semantic-class accuracy；
* 大规模 masked-J 状态图。

---

## 5.2 Does IntentIR Improve Target Implementations?

这个小节同时回答：

* M 侧复用；
* IntentIR 本身是否带来性能；
* 最终后端是否有竞争力。

---

### 核心因果消融：Freeze / Retune / Multi-Lowering

三组使用：

* 同一个 J；
* 同一个 backend；
* 同一个 target；
* 同一个 code generator；
* 同一个 tuning budget。

#### Freeze

保留 Source C 和 canonical implementation form。

#### Retune

固定 implementation form，只重新选择：

* tile；
* vector；
* thread；
* pipeline。

这基本复用当前 RVV `freeze vs retune` 实验。旧稿已经显示 retune 相比 freeze 在 RVV 上有 1.13× 几何平均收益。

#### Multi-Lowering

依据 A/B 选择多个合法 implementation forms，并对各 form 使用相同 schedule tuner。

最重要的指标：

[
\boxed{
\frac{
\operatorname{Perf}(\text{Multi-Lowering})
}{
\operatorname{Perf}(\text{Retune})
}.
}
]

这直接证明：

> 收益不仅来自调 tile，而来自 IntentIR 恢复出的结构化语义扩大了后端可安全选择的实现空间。

只需在真正具有多个 lowering 的代表性 subset 上做，不要求覆盖全部 200 个 Kernel。

同时报告：

* 多少 Kernel 选择了非 canonical form；
* form 是否随 target 或 shape 改变；
* 每个 form 由哪些 B 条件允许或拒绝。

---

### RVV 绝对性能

保留两个 baseline：

1. Triton-CPU LLVM：portability baseline；
2. ggml/llama.cpp RVV：手工优化现实 baseline。

比较时必须保持：

* 相同 RVV 机器；
* 相同线程数；
* 相同 dtype；
* 相同 shape；
* 相同 operator semantics。

不要把量化 ggml kernel 与非量化 IntentIR kernel 混成一个 geomean。

主文只放：

* single-thread geomean；
* 16-thread geomean；
* 少量代表 operator。

详细 per-kernel 数据放 supplementary material。

---

### GPU 绝对性能

保留：

* H100 vs Triton；
* RTX 5090 D vs Triton；
* 复杂 operator suite 的 geomean。

旧稿已有 H100 与 RTX 5090 D 上的 Triton 对比，可以作为 end-to-end backend competitiveness 保留。

但明确区分：

* 外部 baseline 说明后端有竞争力；
* `Multi-Lowering / Retune` 才证明 IntentIR 本身带来额外优化能力。

---

# 6 Limitations and Conclusion

只保留约三段。

## Limitations

诚实说明：

* SCRC 是 observer-relative 和 test-domain bounded；
* backend materializer/pass/lowerer 属于 TCB；
* 新 semantic construct 仍需共享 J/RefEval 和 target lowering 支持；
* IntentIR 消除的是 frontend-target 乘积耦合，不是消除所有 compiler engineering。

## Conclusion

最后一句：

> IntentIR restores the common-IR (N+M) architecture for optimized kernels that no longer have a high-level program: SCRC recovers and certifies a source-independent semantic interface, and J-guided multi-lowering reuses that interface across target backends.

---

# 主文图表清单

## Figure 1：全局 (N\times M) 架构

放 Introduction。

同时显示：

* N frontends；
* SCRC；
* (J,C,\pi)；
* (\pi) 停止；
* J+C；
* M backends。

## Figure 2：SCRC

放第三章：

```text
Lift → TypeCheck → Source-blind Replay → Gate1 → Gate2
```

## Figure 3：SCRC 规模结果

两个 panel：

* 认证漏斗；
* 四种 intervention。

## Figure 4：性能

三个 panel：

* Freeze / Retune / Multi-Lowering；
* RVV vs Triton-CPU / ggml；
* GPU vs Triton。

## Table 1：工程解耦

两个 panel：

* frontend onboarding；
* lowering/backend reuse。

不要再增加主文图表。

---

# 必须从旧稿删除的内容

以下全部从主文删除：

* CanonicalEvidence 主路径；
* O1–O7；
* FULL/PARTIAL/OOS；
* semantic-class accuracy；
* 30 对跨前端 J exact/structural equality；
* 旧 3,200 mutant 总结果；
* evidence-on/off；
* contract-off；
* host-dispatch-off 主实验；
* FIC/ESC 独立章节；
* J-authority challenge 细节；
* 多 C purity；
* 30-kernel SMT audit。

其中 masked-J、完整数学证明、per-kernel 性能、更多 mutation 可以放 supplementary material，但不要与主故事竞争。

---

# 最终读者应该记住的三句话

[
\boxed{
\text{优化 Kernel 缺少公共语义接口，因此 Source 与 Target 发生 }N\times M\text{ 耦合。}
}
]

[
\boxed{
\text{SCRC 从开放 Source 中恢复并认证固定的 }J=(A,B)\text{，不需要每 frontend 的完整形式语义。}
}
]

[
\boxed{
\text{后端只消费 }J+C\text{，根据 A 构造实现族、根据 B 判断合法性、根据 C 选择调度。}
}
]

这个版本足够覆盖科学问题、方法创新、工程扩展性和性能，同时能压进 PPoPP 的 10 页限制。
