下面这段可以直接发给 agent。它延续你们已经确定的“**裁决权归代码，理解权归 LLM**”，同时明确反对系统继续退化成 deterministic composer、RuleCard 和 source-pattern 枚举器。

---

# 系统总哲学：LLM 驱动、验证器闭合的开放前端系统

我们要构建的，不是一个“用 LLM 辅助传统编译器开发”的系统，也不是一个逐渐积累前端 lowering、API 映射和 kernel-specific 规则的框架。

我们的目标是：

> **让 LLM 吸收碎片化生态中的理解与生成复杂度；让一个极小、通用、可审计的验证核心约束 LLM 的不确定性。**

可以概括为：

[
\boxed{
\text{LLM absorbs ecosystem entropy;}
\qquad
\text{the verifier bounds model uncertainty.}
}
]

中文即：

[
\boxed{
\text{LLM 吸收前端碎片性，Gate 约束模型不确定性。}
}
]

---

## 一、LLM 不是辅助模块，而是开放世界的主执行者

GPU Kernel 生态具有以下特点：

* 前端数量多；
* API 持续变化；
* 源码 realization 极度多样；
* 同一语义在不同项目中可能采用完全不同的写法；
* 新 DSL、新 intrinsic、新编译器版本不断出现。

在这种生态中，任何试图由固定代码持续维护：

[
J\rightarrow\text{frontend API}
]

或者：

[
\text{source pattern}\rightarrow\pi
]

的系统，最终都会产生类似：

[
|\text{frontend}|
\times
|\text{operator}|
\times
|\text{realization pattern}|
]

的工程复杂度。

也就是：

[
\Theta
\left(
|F|\cdot|O|\cdot|R|
\right).
]

这在真正碎片化的生态中不可持续。

因此，我们明确把下面这些开放世界任务交给 LLM：

[
\operatorname{Lift}_{\mathrm{LLM}}:
S
\rightarrow
(J,\pi,C)
]

以及：

[
\operatorname{Replay}_{\mathrm{LLM}}:
(J,\pi,C,\operatorname{Context}_f)
\rightarrow
R_f.
]

其中：

* (S) 是任意前端的源 Kernel；
* (J) 是语义；
* (\pi) 是源实现 realization 的见证；
* (C) 是纯调度；
* (R_f) 是 LLM 直接生成的、完整可编译的目标前端源码；
* (\operatorname{Context}_f) 可以是动态检索的文档、仓库示例和工具说明。

LLM 自己负责决定：

* Triton 中该使用什么 API；
* TileLang 中该如何表达 reduction；
* CUDA 中该采用什么线程级实现；
* 一个从未见过的新 DSL 中如何写出合法代码；
* 某个 source realization 如何在同一前端中被重建。

这些知识不应被固化成永久的、具有语义权威的 RuleCard 或 deterministic lowering。

---

## 二、系统中只有 (J) 是语义，(\pi) 是见证，(C) 是调度

顶层定义应保持：

[
\boxed{
\operatorname{Intent}=J(A+B)+C
}
]

并且：

[
\boxed{
\pi\notin\operatorname{Intent}.
}
]

(\pi) 是 Gate1 使用的 source-realization witness：它保存那些不能仅由 (J) 推出的源码实现选择，但本身不是第二份语义。这个职责划分已经在现有讨论中明确。

三者分别回答：

[
J:\quad\text{程序是什么意思？}
]

[
\pi:\quad\text{源码选择了哪一种实现这个意思的 realization？}
]

[
C:\quad\text{这份 realization 使用了什么调度参数？}
]

我们不要求：

[
I(\pi;\operatorname{Sem}(S))=0.
]

因为实现拓扑天然可能与语义相关。

我们只要求：

* (\pi) 不被赋予独立 denotation；
* (\pi) 不成为最终语义裁决路径；
* 一个错误的 (J) 不能依靠 (\pi) 通过 Gate2 的 Source–(J) 比较。

因此，masked-(J) 审计的是 **J/(\pi) 信息分工**，而 Gate1+Gate2 认证的是 **源码与 (J) 的约语义等价**。两者不能混为同一个命题。

---

## 三、这不是 LLM 自洽，而是外部锚定的双闭环

系统不能被描述为“让两个 LLM 互相同意”。

两个 LLM 互相同意没有证明力，因为它们可能共享相同先验和相同错误。

我们真正建立的是一个由外部对象闭合的系统。

### 第一条：构造闭环

[
S
\xrightarrow{\operatorname{Lift}*{\mathrm{LLM}}}
(J,\pi,C)
\xrightarrow{\operatorname{Replay}*{\mathrm{LLM}}}
R_f
]

随后由同一个前端编译器观察：

[
\boxed{
F_f(S)=F_f(R_f).
}
]

Replay LLM 看不到源码，因此这条闭环证明：

> (J+\pi+C) 足以构造性地重建一个与源码处于同一 frontend-local compiler observation class 的实现。

这是 Gate1 的职责。

---

### 第二条：语义闭环

对同一组输入执行：

[
\operatorname{Obs}(S,x),
]

[
\operatorname{Obs}(R_f,x),
]

[
\operatorname{Obs}(E_J,x),
]

其中 (E_J) 是只依赖 (J) 的标量语义执行路径。

正式裁决要求：

[
\boxed{
\operatorname{Obs}(S,x)
\approx_\tau
\operatorname{Obs}(R_f,x)
}
]

以及：

[
\boxed{
\operatorname{Obs}(S,x)
\approx_\tau
\operatorname{Obs}(E_J,x).
}
]

这是 Gate2 的职责。你们目前 Source、Replay、(J) 标量 lowering 的三角执行，本身已经实现了这一闭环。

所以系统不是：

[
\text{LLM output}\leftrightarrow\text{LLM judgment}.
]

而是：

[
\boxed{
\text{LLM 生成 witness}
;\longrightarrow;
\text{编译器与独立执行路径闭合 witness}.
}
]

更准确的名称应是：

# **LLM-driven, verifier-closed system**

而不是单独称为 “LLM closed system”，后者容易被误解为模型自洽或 self-consistency。

---

## 四、前端开放性：新前端不应要求新的语义工程

一个新前端接入系统时，最低要求不应是：

* 写一个 (J\rightarrow) frontend lowerer；
* 写一套 frontend API 映射；
* 写一个 frontend (\rightarrow) 公共语义 IR 的反编译器；
* 为每一种 kernel realization 增加 RuleCard；
* 修改 Intent schema；
* 修改 composer。

最低要求只应是一个黑盒执行合同：

```text
Build(source, specialization, C)
Run(artifact, logical_inputs)
Observe(outputs, inout state, effects)
OptionalDump(artifact)
```

也就是：

[
\boxed{
\text{零新增语义 lowering；
仅保留不可避免的 build/run/observe harness。}
}
]

完全零前端工作并不现实，因为系统至少必须知道如何构建和运行程序。

但我们要求新增工作的复杂度近似为：

[
O(c_{\mathrm{harness}})
]

而不是：

[
O(
|\text{J primitives}|
\times
|\text{frontend APIs}|
\times
|\text{source patterns}|
).
]

---

## 五、Gate1 应是观察器参数化的，而不是所有前端统一强度

不同前端能够暴露的编译器信息天然不同，因此不应强迫所有前端 lower 到同一个公共摘要 (Q)。

可以把 frontend-local observer 写成分级关系：

[
F_f^{(0)}
\preceq
F_f^{(1)}
\preceq
F_f^{(2)}.
]

其中高层观察器比低层观察器区分能力更强。

例如：

### Level 0：通用闭合

* 源和 Replay 均可编译；
* 接口、参数角色和 specialization 一致；
* source-blind Replay 能形成合法候选。

### Level 1：必要条件闭合

* AST 结构；
* intrinsic/instruction multiset；
* mask、atomic、barrier 等必要条件；
* 编译资源信息。

### Level 2：强结构闭合

* TTIR/TIR 高层数据流；
* reduction function 与 axis；
* predicate/fill；
* effect；
* 规范化后的 realization topology。

Gate1 的结论始终显式参数化为：

[
F_f^{(k)}(S)=F_f^{(k)}(R_f).
]

新前端可以立即以 Level 0 或 Level 1 接入。

以后增加一个更强的 compiler observer，只会增强 Gate1 的构造闭合结论，而不是改变 Intent、Replay 或 Gate2，也不会阻塞前端首次接纳。

因此：

> **强 Gate1 adapter 是可插拔的证明增强器，而不是新增前端的准入门槛。**

Gate2 的 Source–(J) 语义声度不依赖 Gate1 使用的是哪个观察等级。

---

## 六、固定代码只负责验证，不负责理解和生成

固定系统允许做的事情应严格限制为：

* schema 与基本类型检查；
* opaque handle 与引用合法性；
* source/Replay 信息隔离；
* artifact hash 与 commitment；
* 编译和执行；
* 读取编译器已有 artifact；
* 确定性比较；
* Gate2 输入生成；
* tolerance 与 effect comparator；
* receipt、provenance 和审计日志。

这些代码回答的是：

> 候选是否合法？
> 两个已存在对象是否一致？
> 某个差异是否存在？

固定系统不应回答：

> 这个语义在 Triton 应该写成什么？
> 这个 LayerNorm 应该先 mask 还是先 square？
> 这个新 DSL 应该调用哪个 reduction API？
> 这个 kernel 属于哪个 StrategyFamily？

后面这些问题必须交给 LLM。

最重要的设计边界是：

[
\boxed{
\text{固定代码可以拒绝错误，但不应替 LLM 构造正确实现。}
}
]

换一句更容易执行的原则：

> **我们不把前端知识编码进核心系统；我们只把“怎样发现和拒绝错误”编码进核心系统。**

---

## 七、LLM 的不确定性不会直接成为最终声度

Lift 和 Replay LLM 可以是随机的。

不同 seed 可能生成不同的：

[
(J,\pi,C,R_f).
]

这影响的是：

> 系统能否找到一个可接受 witness。

也就是 completeness 或成功率。

但最终 artifact 一旦被冻结：

[
A=(S,J,\pi,C,R_f),
]

它就成为一个固定候选。

随后才生成隐藏 Gate2 输入：

[
x_1,\ldots,x_m
\overset{\mathrm{iid}}{\sim}\mathcal D.
]

定义：

[
d_{\mathcal D,\tau}(S,J)
========================

\Pr_{x\sim\mathcal D}
\left[
\operatorname{Obs}(S,x)
\not\approx_\tau
\operatorname{Obs}(E_J,x)
\right].
]

若：

[
d_{\mathcal D,\tau}(S,J)>\varepsilon,
]

则无论：

* Lift LLM 是什么模型；
* Replay LLM 是否真正读取了 (J)；
* (\pi) 泄漏了多少信息；
* 模型先验有多强；
* Replay 是否碰巧正确猜出了源码；

该错误 artifact 通过 (m) 个隐藏 Gate2 输入的概率都至多为：

[
\boxed{
(1-\varepsilon)^m.
}
]

因此：

[
\boxed{
\sup_{\pi,\operatorname{Lift},\operatorname{Replay}}
\Pr[
\operatorname{Accept}
\land
d_{\mathcal D,\tau}(S,J)>\varepsilon
]
\le
(1-\varepsilon)^m.
}
]

这就是系统如何把 LLM 的“大概率正确”转化为整个 artifact 的概率声度：

> LLM 的随机性只影响找到 witness 的机会；
> verifier 的随机 challenge 决定错误 witness 的通过上界。

---

## 八、修复循环也必须服从证明，而不是追求过测试

Replay 失败后，不能把源码、旧 Replay 和 source/replay 差分直接交给 Replay LLM 做局部补丁。

否则系统证明的将变成：

> 一个看过答案和差异的 Agent 最终能够把代码修到通过。

而不是：

> (J+\pi) 本身足以指导一次 source-blind 重放。

因此失败只能：

[
\text{failure evidence}
\rightarrow
\text{source-aware Lift}
\rightarrow
(J',\pi',C')
\rightarrow
\text{fresh blind Replay}.
]

旧 Replay 只保留为审计记录，不成为新 Replay 的输入。

这保证每一次 Gate1 closure 都对应一个新的、完整的 source-blind construction witness，而不是 patch accumulation。

---

## 九、任何新设计都应通过三个审查问题

### 问题一

这个组件是在：

* **理解或生成前端代码**，还是
* **检查一个已经生成的候选**？

若是前者，优先交给 LLM。

---

### 问题二

它的维护成本是否随以下乘积增长：

[
|\text{frontend}|
\times
|\text{operator}|
\times
|\text{realization pattern}|?
]

若是，应当拒绝。

---

### 问题三

它是在增强外部闭合证据，还是在复制一份新的语义真值？

若它需要重新定义一份与 (J) 同样完整的公共语义摘要、frontend linker 或 API 映射，那么它很可能正在创建第二个 IR 或第二个 lowering 系统。

---

# 最终设计宣言

> **IntentIR 不是要把碎片化 GPU 生态重新编码成一套人工维护的规则系统。我们让 LLM 负责理解任意源码、抽取 (J/\pi/C)，并在看不到源码的情况下直接生成任意前端的完整 Replay；固定代码不理解 kernel family，也不负责把 (J) 翻译成前端 API，只负责隔离、编译、执行、比较和审计。Gate1 用同前端编译器验证 source-blind 构造闭合，Gate2 用 Source、Replay 和 (J)-only 标量语义执行验证约语义等价。新增前端只需要最小 build/run/observe 合同，强编译器观察器可以增量加入，但不成为接纳前提。由此，生态复杂性由 LLM 吸收，模型不确定性由 Gate1+Gate2 的外部闭合和概率声度约束。**

最适合作为项目口号的一句话是：

[
\boxed{
\text{不把碎片化生态编进系统，只把可验证性编进系统。}
}
]
