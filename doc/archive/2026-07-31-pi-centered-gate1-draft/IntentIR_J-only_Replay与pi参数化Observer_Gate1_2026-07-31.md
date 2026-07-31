# IntentIR：J-only Replay 与 typed π 参数化的 Gate1 Observer

> 日期：2026-07-31
> 状态：Task 20 活动方法设计；取代 Task 19 的 Target-Normal IR / 确定性 Composer 路线。

## 1. 唯一裁决

外部 SCRC 流程与 Figure 3 的拓扑保持不变：

```text
Source + Harness
  -> Lift(J,C,π)
  -> TypeCheck
  -> Source-Blind Replay
  -> Bind same C
  -> Gate1
  -> Gate2
  -> SCRC_CERTIFIED
```

只修改两个内部定义：

```text
R = Replay_f(J, MaskC(C); U_f)              # π 不进入 Replay

Gate1 = equality after typed π-relative
        compiler-observer canonicalization  # π 只进入固定 observer
```

不新增 Composer、Target-Normal IR、第四 artifact、额外 Gate 或 Source-conditioned sidecar。

## 2. 对象与信息边界

Lift 仍完整作者化：

\[
\operatorname{Lift}_H(S)=(J,C,\pi).
\]

Replay 改为：

\[
\boxed{
R=\operatorname{Replay}_{\theta,f}
\left(J,\operatorname{MaskC}(C);U_f\right).
}
\]

Replay request、workspace、工具和 continuation 均不得包含：

- π 或 π 的摘要、查询接口、消费 ledger；
- Source、Harness、Gate、compiler observation 或 diff；
- C value、Source binding、旧 Replay 或跨 branch 上下文。

因此 π 不能帮助 Replay 重建 Source。Replay 只负责把 J 编译成 frontend `f` 中的一份完整实现。

π 保持为依赖于 `(J,C)` 的 typed source-realization witness；在 Gate1 前它只是 Lift 提出的 candidate
witness，Gate1 后也只对 receipt 中实际 used clauses 给出 observer-relative 的局部使用证据，不声称
整份 π 是 Source realization 的完备真值。π 不进入 production backend，也不定义新的 J denotation；
它的新认证消费者仅是版本化 Gate1 canonicalizer。

## 3. Gate1 的正式定义

先定义未规范化的 frontend observation：

\[
O_{f,p}(P,c)
=
\operatorname{Observe}_{f,p}
\left(\operatorname{Compile}_f(P,c)\right).
\]

在固定 valuation `c_0` 上得到 `O_S=O_{f,p}(S,c_0)` 与
`O_R=O_{f,p}(R,c_0)`。对冻结的 π schema `Σπ`、frontend observer profile `(f,p)` 与
π-normalizer version `ν`，定义二侧归一化：

\[
(\Phi_S^\pi,\Phi_R^\pi,\rho)
=
\operatorname{CanonPair}^{\Sigma_\pi}_{f,p,\nu}
\left(O_S,O_R;J,C,\pi\right).
\]

Gate1 继续是相等判定：

\[
\boxed{
G_1^{f,p,\nu}(S,R,J,C,\pi;c_0)
=
\mathbf 1\left[
\Phi_S^\pi=\Phi_R^\pi
\right].
}
\]

因此 Figure 3 中 Gate1 的“两边 Compile -> Observe -> equality”保持不变；正文只需把 `F_f` 明确为
typed π-relative canonical observation。

## 4. `Diff ⊆ π` 的准确身份

设基础 observer 的确定性结构化差异为：

\[
D=\Delta_{f,p}(O_S,O_R).
\]

关键约束是：π 不是“看到 diff 以后就可以写一个例外”的白名单。每个 relation
constructor `q` 在冻结的 `Σπ` 与 `(f,p,ν)` 中都有一个固定的、可能未定义的局部
二侧匹配和归一化：

\[
A^{q}_{f,p,\nu}(O_S,O_R;J,C)\;\rightharpoonup\;
(\bar O_S,\bar O_R,\rho_q).
\]

`A^q` 只有在下列前提全部成立时才适用：

1. `q` 的 kind、字段、引用和权限属于 `Σπ`，并唯一锚定一个已有的 J/C typed
   occurrence；
2. Source 与 Replay 两侧在该 exact scope 上都满足 `q` 的固定 profile-local
   recognizer。recognizer 可以识别一个预先定义的允许二侧形式对，但不能读取 Kernel
   名称、Source 文本、历史 diff 或候选 prompt；
3. 作用域不重叠，或按 `Σπ` 规定的固定顺序可合流；重复应用是幂等的；
4. hard projection 不变：完整 observation 默认全 hard；`q` 只能改写其 profile rule
   白名单列出的 exact atom kinds。对两侧 `i∈{S,R}`，
   `H_{J,C}^{q}(\bar O_i)=H_{J,C}^{q}(O_i)`。

任一前提不成立，`A^q` 返回 `undefined`，该 relation 不产生任何放宽；相关 residual
继续参加 equality。也就是说，候选写出一个 typed relation 并不等于该 relation 被使用。

normalizer 必须产生：

```text
base_diff
used_pi_relations
covered_diff_atoms
uncovered_after_normalization
hard_preserved_atoms
```

定义二侧 normalizer（所有 relation 的适用性由同一个固定过程决定）：

\[
(\bar O_S,\bar O_R,\rho)
=
\operatorname{CanonPair}^{\Sigma_\pi}_{f,p,\nu}
(O_S,O_R;J,C,\pi).
\]

归一化后的 residual 为：

\[
D^{\pi}_{\mathrm{res}}
=\Delta_{f,p}(\bar O_S,\bar O_R).
\]

Gate1 的实际判定是 `D^π_res = ∅`，而不是候选自己提供的 coverage 数字。
当 profile 对 diff atoms 与 canonical rewrite 的对应是精确的，可以把 Gate1 诊断简写为：

\[
D_{\mathrm{realization}}
\subseteq
\llbracket\pi\rrbracket_{f,p,\nu}
\quad\land\quad
D_{J\text{-owned}}=\varnothing.
\]

正式 verdict 仍以 `D^π_res=∅`（等价于两侧 normalized observation equality）为准；集合包含是
coverage receipt，不是允许候选定义任意 exception 的接口。若某个 profile 尚未实现
`D` 与 relation denotation 的双向精确 matcher，则不得把该 shorthand 当作独立的通过条件。

不要求 `D = [[π]]`。π 可以描述 Source 已有、但本次 Replay 恰好采用相同形式而未产生 diff 的
realization；这些 clause 记为 unused，不获得额外 acceptance 权限。

## 5. 为什么不是循环认证

候选只提交 `π ∈ Π_{Σπ}`；它不能修改：

- π relation 的种类、字段和引用类型；
- 每种 relation 的 observer denotation；
- J-owned hard projection；
- frontend/profile 的 canonicalization rules；
- Gate1 comparator。

即使删除 LLM，所有可接受 observation pairs 的集合也已经由
`(Σπ,f,p,ν)` 的固定 partial recognizer、hard projection 与 comparator 预先确定。LLM 只搜索一个
typed witness，不能扩张接受关系；relation 是否真正生效由实际两侧 observation 的固定
applicability check 决定。

循环只会在下列情况下出现，协议必须禁止：

- arbitrary `allowed_diff`、wildcard、free-text equivalence；
- 按 Kernel、失败样本、Source 路径或 Gate1 diff 动态注册 normalization rule；
- π relation 擦除 operator、reducer、predicate/fill、logical domain、observable effect 或 output；
- Replay 读取 π、coverage receipt 或 Gate1 diagnostics。

对固定 `(Σπ,f,p,ν,c0)`，即使允许 relift 搜索任意合法 π，接受集合也可在模型运行前定义：

\[
\mathcal A_{\Sigma_\pi,f,p,\nu,c_0}
=
\left\{
(O_S,O_R,\pi)
\mid
\operatorname{TypeCheck}(J,C,\pi)
\land
D^{\pi}_{\mathrm{res}}=\varnothing
\right\}.
\]

若把 π 隐去，则接受一对 observation 的条件只是对这个冻结集合存在一个 typed witness；LLM 可以
提高找到 witness 的机会，不能改变集合本身。这里与“LLM 写一个 Python comparator 决定自己通过”
不同：后者会改变接受关系，前者只选择固定 relation algebra 的参数。

## 6. π schema 与 TypeCheck

现有 π 已经是 closed schema，无自由文本或 wildcard，并具有 region、instance、route、order、sharing、
refinement、program axis/linearization、iteration/carrier、terminal-control 等 typed relations。Task 20
不把所有 relation 自动升级为 Gate1 quotient 权限。

候选 wire 不新增 `allowed_diff`、`normalize_to`、regex、目标 API 或 rule id。Gate1 的可信
profile metadata 另有一个版本化 `π-normalization capability table`；它不是 Lift artifact，也不进入
Replay。所有 relation kind 在一个新 profile 中默认都是 `witness_only`，只有该 profile 已发布完整
rule spec 的 exact kind 才能升级。状态只有：

1. `normalizable`：具有固定、局部、profile 已实现的 canonical rewrite；
2. `conditional`：只有 J/B 的 closed authorization 与 observer 前提均满足时可 normalize；
3. `witness_only`：保留在 π，但不能放宽 Gate1；
4. `unsupported`：当前 profile 无法安全消费，相关 residual 保持 hard。

每条可信 rule spec 至少固定：

```text
relation_kind
accepted_anchor_sorts
required_J_B_C_premises
left/right local recognizer
erasable_atom_kinds          # closed allowlist
canonical replacement
scope/overlap key
composition order
receipt fields
```

候选不能选择或修改这些字段。`required_J_B_C_premises` 可以读取通用 typed rank/dtype、J-authored
dimension relation、C attachment 与本次 `c0`，但不能读取 Kernel identity、benchmark ordinal、Source
路径或“当前 concrete extents 恰好相等”来发明关系；J/B authorization 只是必要前提，绝不自动赋予
normalization 权限。

第一版候选边界如下；表中“可候选”不表示已经获得权限，未发布 rule spec 时一律 hard：

| π relation | profile-wide rule 的最高候选权限 |
|---|---|
| program axis/linearization、纯 coordinate spelling | 可候选 conditional rule；logical domain/access/predicate、axis multiplicity 保持 hard |
| address base/lane、route、route extent spelling | 可候选 conditional rule；memory origin、canonical coordinate、mask、alias/effect 保持 hard |
| placement、identity materialization | 可候选 conditional rule；intermediate 必须不可观察、无 alias/effect，且 multiplicity/provenance hard |
| physical iteration wrapper | 可候选 conditional rule；reducer、domain、value recurrence、order 与 result contract 保持 hard |
| sharing、order、loop carrier、terminal control | 默认 witness_only；没有另行窄升就不能 normalize |
| reassociation、accumulation dtype、numeric cast、atomic/update refinement | 始终 witness_only；π/J-B authorization 都不能让 Gate1 擦除数值或状态差异 |

TypeCheck 继续证明：

- 所有 π refs 解析到 J/C 已有 typed handles；
- region/order/route/carrier 拓扑闭合、唯一且无环；
- 无未知字段、重复 relation、自由公式、predicate、constant、Source/API 或 target recipe；
- sensitive refinement 具有 J/B 已作者化 authorization。

TypeCheck 不读取 Source observation，也不声称 π relation 对 Source 为真。其职责是保证 π 只能实例化
协议允许的 realization algebra；具体 relation 是否在本次 Gate1 canonicalization 中被使用，由固定
normalizer 的二侧 applicability receipt 决定。这不是重新 Lift Source，也不是一个 Source→π
完备验证器：它只检查某个已经提交的、窄类型 relation 是否满足该 profile 的局部闭合前提；无法
局部匹配就保持 residual hard。

### 6.1 当前不改 π wire

Task 20 的第一阶段不新增 π 顶层对象、target recipe、`allowed_diff`、`normalizable` 标记或
observer-specific 字段。现有 `regions/instances/routes/order/sharing/iterations/carriers/
terminal_controls` 等关系继续由 Lift 完整作者化；它们从“Replay construction obligation”改为
“Gate1 candidate witness”，其中绝大多数先保持 `witness_only`。

`normalizable/conditional` 状态、二侧 recognizer、erasable atom allowlist 与 canonical replacement
属于 profile/version 的固定 metadata，不写进 π。`used/unused/covered/residual` 属于 Gate1
proof-irrelevant receipt，也不是第四 artifact。只有在真实 vertical slice 证明现有 relation 没有
稳定 typed anchor/scope 时，才另行提出最小、版本化的 generic anchor 扩展；不得为了预想的 kernel
或某个 frontend 预先加字段。这样可以改变 π 的消费位置而不改变 Lift 的语义表达或信息边界。

## 7. π-relative Canonicalizer 的硬性质

每个 `(f,p,ν)` 实现必须满足：

1. deterministic：相同输入得到相同 observation 与 receipt；
2. symmetric：交换 Source/Replay 只交换两侧输出和 receipt 角色，适用与否及通过/失败不变；
3. idempotent：重复 canonicalization 不继续擦除信息；
4. source-neutral：不读取 Kernel 名、Source 文本、历史 winner 或失败样本规则；
5. hard-by-default：完整 observation 默认参与 equality。对 rule `q`，只从其 closed
   `erasable_atom_kinds(q)` 中改写 exact matched atoms；其补集
   `H_{J,C}^{q}=O\setminus E_q` 全部保持。interface、J operator/operand、logical domain、
   predicate/fill、reducer/value recurrence、observable effect/control/post-state、output binding，
   以及 C identity/type/valuation/attachment 必须总在 hard 集；alias、memory ordering、numeric
   metadata、地址 provenance 等未知或未列举 atom 也因默认规则保持 hard；
6. scoped：每条 rewrite 只作用于 π relation 的 exact typed scope，不能通过多个 relation
   叠加扩大 scope；
7. fail-closed：匹配歧义、缺少 anchor、profile 未支持或前置条件不足时不 normalize；
8. atomic：一个 relation 只有在二侧匹配和 hard-preserving 检查都通过时才被记为 `used`，
   否则两侧都保持原子并进入 residual。

当前 observer 没有统一 J-anchor diff vocabulary。因此第一版不能假装支持所有 π：只允许能够从现有
typed occurrence/topology 唯一匹配的窄 relation；其余 residual 保持 hard。后续扩展必须按 frontend
profile/version 全局加入，连同 capability table、anchor vocabulary 与 rule spec 一起发布；不能按
Kernel 加规则。

### 7.1 TCB 与“sound”的准确含义

`Σπ`、compiler、raw observer、profile rule table、`CanonPair` 与 comparator 都属于 Gate1 TCB。
Task 20 只把下列性质作为每条 rule 的准入合同：closed、deterministic、symmetric、idempotent、
scoped、atomic、hard-by-default、fail-closed。当前没有把每条 realization quotient 写成“对所有程序
保持行为”的形式定理，也不把 `D^π_res=∅` 解释为语义等价。

因此一个错误或过宽的 profile rule 是 observer/TCB 缺陷，必须升 profile version、使旧证书保持
原解释并重新运行；它不是候选 π 获得了修改 rule 的权限。主证书用 Gate2 的 `Q,𝒯` 三边执行来提供
声明执行域内的行为证据。若论文需要声称某条 quotient 全域语义保持，就必须为那条 rule 另给证明，
不能从本协议自动推出。

## 8. 多前端

π 的 wire schema、kind 名称和 ownership 标签是共同的 closed syntax；relation 的 denotation
与 applicability 必然是 frontend/profile-relative：

```text
PiNormalizer_f,p,ν:
  (source observation, replay observation, J, C, π)
  -> (canonical source, canonical replay, consumption receipt)
```

不强制 Triton、CuTe、TileLang、CUDA 共享同一 raw diff 格式。每个 profile 只需：

- 输出自身稳定的 compiler observation；
- 声明支持的 π relation 子集；
- 实现相应固定 canonical rewrites；
- 保留统一的 hard ownership边界；
- 在证书中记录 frontend、compiler stage、observer/profile、normalizer 与 π schema version。

无法实现某种 π relation 的 frontend 报 `observer_unsupported` 或保留 residual hard；不能借用另一个
frontend 的弱 observation 自动通过。同一个 π instance 在不同 profile 下可能是
`normalizable`、`witness_only` 或 `unsupported`；不同 profile 的证书强度分别报告，不宣称天然等价。
跨 profile 共同的只是 wire syntax、opaque ownership 与引用类型；同名 relation kind 并不是同一个
observer 语义函数，不能把一个 profile 的 PASS 转移到另一个 profile。

工程量按 frontend observer profile 增长，而不是按 Kernel 增长。共享 PTX/TIR/MLIR 分析可以复用，
但禁止为单个 Kernel 增加 relation matcher。

## 9. Relift 与反馈

Gate1 失败反馈仅包含固定 structured diff 与 coverage receipt：

```text
uncovered diff atoms
used / unused π clauses
ambiguous / unsupported relation diagnostics
hard J-owned residuals
```

Relift 仍看到 Source 与该 closed feedback，但看不到 Replay source、Replay trajectory 或私有上下文。

- 若 J 或 C 改变：旧 Replay 作废，执行 fresh J-only Replay 与 fresh Gate1；
- 若只修改 π，且 J、MaskC、Replay request policy 与 exact Replay bytes 均不变：可复用同一 R，重新
  TypeCheck 与 Gate1；这是 witness-only recheck，不读取 Gate2；
- Gate2 hidden inputs/results 不进入同一次 relift；若实验协议允许跨实验修复，最终候选必须使用新的
  holdout。

反馈引导 witness synthesis 不改变接受谓词；`Σπ`、observer/profile 与 normalizer version 在一次正式
实验内冻结。这里不需要统计意义上的 precommit：Gate1 不是 p-value，而是在固定 relation algebra
中搜索一个由确定性 checker 验证的 witness。自适应轮数只影响搜索成本和
first-pass/closure@k，必须报告；它不改变任一 candidate 的 verdict。Gate1 证书只认证 receipt 中
实际 `used` 的 π clauses；proposed 但 unused 的 clauses 不获得“Source 中为真”或“跨 profile
有效”的结论。

## 10. Gate2 与最终 claim

Gate2 保持外部政策 `Q` 与声明测试域 `𝒯` 上的三条直接比较：

\[
G_2^{Q,\mathcal T}
=
\bigwedge_{x\in\mathcal T}
\left[
\operatorname{Close}_Q(O_S(x),O_R(x))
\land
\operatorname{Close}_Q(O_S(x),O_J(x))
\land
\operatorname{Close}_Q(O_R(x),O_J(x))
\right].
\]

最终认证谓词为：

\[
\boxed{
\operatorname{SCRC\_CERTIFIED}
\iff
\operatorname{TypeCheck}(J,C,\pi)
\land G_1^{f,p,\nu}
\land G_2^{Q,\mathcal T}.
}
\]

可以声称（均限定到记录的 `f,p,ν,c_0` 与 Gate2 的 `Q,𝒯`）：

- Replay 在不读取 Source 与 π 的条件下，从 J/MaskC 构造完整 frontend program；
- Source/Replay 的 compiler-visible residual 在 frozen typed π-relative observer quotient 下闭合；
- Source、Replay、RefEval(J) 在声明 test domain/Q 下三边闭合。

不能声称：

- π relation 对所有输入都是形式化语义保持定理；
- Gate1 提供完整 Source→J 或 Source→π 验证；
- observer 未观察部分相等；
- test domain 外全称等价；
- 不同 frontend profile 具有相同证明强度。

## 11. Figure 3 与论文最小改动

Figure 3 的节点和箭头全部保留。只修改正文与必要标签：

1. Replay equation 删除 π：`Replay(J,MaskC(C);U_f)`；
2. Gate1 inset 的 `Observe F_f` 在正文定义为 `π-relative Canon/Observe`，并在 Gate1 标签或图注明确
   “consumes frozen π from Lift”；不向 Replay 画 π 输入；
3. Gate1 pass 文案改为 `structural closure under typed π-relative observer`；
4. relift 仍接收 structured diff，流程箭头不变；
5. Gate2 与最终 certificate 公式不变，仅 Gate1 的 `Φ` 增加 `(π,ν)` 参数。

最终原则：

> Replay只从J构造；π不生成Replay，也不定义语义。π只能请求实例化预先冻结的、hard-by-default
> realization quotient；固定二侧 recognizer 决定 relation 是否实际使用。Gate1比较归一化后的空
> residual，Gate2再闭合声明执行域中的行为。
