# IntentIR：J-only Replay 与 J/C Structural Gate1

> 日期：2026-07-31
> 状态：Task 20 活动实现规范；取代 π-centered `CanonPair/coverage` 草案。
> 范围：只窄化 Replay 与 Gate1 的内部定义，不改变 Figure 3 的阶段、三对象或 Gate2。

## 1. 核心定位

论文真正提升、复用并交付 production backend 的对象是 `J+C`。π 是依赖于 `(J,C)` 的
certification-only source-realization witness：它只帮助 Gate1 排除允许的 realization 表达噪声，
不进入 Replay，不进入 production backend，也不是最终结论要证明正确或完备的研究输出。

```text
Lift_H(Source) -> (J,C,pi)
TypeCheck(J,C,pi)
Replay_f(J,MaskC(C);U_f) -> R                 # 不读取 pi
Gate1_pi(Source,R,c0;f,p)                    # J/C structural closure
Gate2(Source,R,RefEval(J);Q,T)                # 三边行为闭合
```

不新增 Composer、Target-Normal IR、第四 artifact 或额外 Gate。

## 2. Replay 的唯一输入边界

\[
R_J=\operatorname{Replay}_{\theta,f}
\bigl(J,\operatorname{MaskC}(C);U_f\bigr).
\]

Replay request、workspace、工具、continuation 与 cache key 均不得包含 π、Source、Harness、C value、
Gate1 observation/diff、旧 Replay 或 sibling 状态。`S_J=M(J)` 可以作为 J-only、只读、
proof-irrelevant 的编译视图；它不能读取 π 或生成 target skeleton。

## 3. Gate1 的最小数学

固定 frontend `f`、observer profile `p` 和 Harness valuation `c₀`。raw observer 完全不读取 π：

\[
\Phi_{f,p}(P,c)
=
\operatorname{Canon}_{f,p}
\bigl(\operatorname{Observe}_{f,p}(\operatorname{Compile}_f(P,c))\bigr).
\]

π 只参数化一个由 profile 预先冻结的、可能未定义的单臂 normalizer：

\[
\Phi^{\pi}_{f,p}(P,c)
=
\mathcal N^{\pi}_{f,p}\bigl(\Phi_{f,p}(P,c)\bigr).
\]

`N^π` 对 Source 和 Replay 独立调用；一次调用不得读取另一臂、两臂 diff、Kernel identity、Source
文本或历史失败。若某种 realization 不能由该固定单臂规则局部识别，就返回 `undefined` 或保留为
hard observation，不能临时退化成 pairwise exception matcher。

Gate1 只做 definedness 与 equality：

\[
G_1=
\mathbf 1\left[
\Phi^{\pi}_{f,p}(S,c_0)\downarrow
\land
\Phi^{\pi}_{f,p}(R_J,c_0)\downarrow
\land
\Phi^{\pi}_{f,p}(S,c_0)=\Phi^{\pi}_{f,p}(R_J,c_0)
\right].
\]

令 `H^{J,C}_{f,p}` 是 profile 预先固定的 J/C-owned observation 投影。它从两臂共同使用的匿名
observer vocabulary 与 `(J,C)` typed contracts构造，不读取 Source 文本、peer observation 或 diff；
候选 π 无权选择投影、scope、normalization rule 或可擦除字段。`H` 不能是空投影，必须至少覆盖该
profile 能观察的 interface、operator/operand/dataflow、logical domain/access/index、predicate/fill、
reducer/recurrence、result relation、observable effect/control/post-state/output，以及 C 的
identity/type/valuation/attachment；未知 observation atom 默认 hard。无法建立唯一 typed anchor 时，
profile 报 `observer_unsupported`，不能删掉该 atom。

定义候选外的 `ProfileOK(f,p,ν,J,C)`：normalizer version、relation-kind allowlist、局部 recognizer、
scope 与 canonical replacement 全部冻结；每条启用规则都已由 profile implementation 的机械
hard-preservation check 或独立 rule audit确认满足：

\[
H^{J,C}_{f,p}(\mathcal N^{\pi}_{f,p}(O))
=
H^{J,C}_{f,p}(O).
\]

`ProfileOK` 是 observer/normalizer TCB 的发布条件，不由 TypeCheck 或候选 π 证明。TypeCheck 只保证
π 是可供固定规则读取的 well-typed instance。若 profile 不能建立 `ProfileOK`，该 relation 必须
unsupported，不能进入正式 Gate1。`ProfileOK` 只是对上述既有 TCB 合同的简写，不是新 artifact、
运行时 Gate、receipt 或 LLM 输出；论文正文也可直接写成“under the declared valid observer profile”。

因此 Gate1 的结论只谈 J/C：

\[
\boxed{
\mathsf{ProfileOK}_{f,p,\nu}(J,C)
\land \mathsf{TC}(J,C,\pi)\land G_1
\Longrightarrow
H^{J,C}_{f,p}(\Phi_{f,p}(S,c_0))
=
H^{J,C}_{f,p}(\Phi_{f,p}(R_J,c_0)).
}
\]

π 从结论中消失。该结论是固定 `(f,p,c₀)` 下 deterministic、runtime-input-independent、
observer-relative 的 J/C structural closure；它不是全输入域语义等价，也不是完整 Source→J verifier。

## 4. π 与 TypeCheck 的边界

新 Lift 只作者化 versioned compact π：`schema` 与 `program_axes`。TypeCheck 验证 relation 只引用
已声明的 J domain，并且不证明 π 对
Source 为真、完整或最小。旧 artifact 的 regions/instances/routes/carriers/refinements 等 realization
graph 仍可只读解析和审计，但不再进入活动 Lift schema，也不再形成新候选的 totality obligation。

π 不得携带：

- comparator、`allowed_diff`、wildcard、free-text equivalence 或 tolerance；
- Kernel/frontend API/Source identity 或 target recipe；
- J-owned operator、operand、logical domain/access/index、predicate/fill、reducer/recurrence、result relation、
  observable effect/control/post-state/output；
- C 的 identity、type、valuation 或 attachment 的替代定义。

active π schema 不暴露尚无固定单臂 normalization 的 relation；它们不能放宽 Gate1。
调试日志可以记录 applied/unsupported relation，但这不是第四 artifact、主公式或 π correctness claim。

## 5. 最小 normalizer，而不是 π 证明系统

Task 20 先只实现 corpus 真实需要且能保守定义的纵切：

1. π 无关的 alpha/SSA 名称与纯打印顺序规范化继续由 raw `Canon` 负责；
2. `program_axes` 继续作为 typed witness 保留，但当前 profile 将其保持 hard。旧的
   `x/y/z -> x` 标签折叠无法证明 logical-domain/physical-axis 绑定不变，因而撤销；
   只有当 observer 能将每个 physical occurrence 唯一回指到 J domain，并通过
   hard-preservation audit 后，新 profile 才可启用对应一元 canonical form；
3. simple placement/materialization 只有在 observer 能独立证明 J/C hard projection 不变时才加入；
4. carrier、terminal control、sharing/order、numeric reassociation、atomic/update 与一般 loop quotient
   默认不 normalize。

因此 compact π 不是被压缩编码的旧 realization graph，而是更窄的 relation language。新增 relation
必须先有 Kernel 无关的一元 normalizer 与 hard-preservation argument，再进入下一版 schema；不能先让
Lift 填字段、以后再决定它表示什么。

任何规则必须按 `(f,p,version)` 全局定义并满足上述 `ProfileOK`；禁止按 Kernel、benchmark ordinal
或某次 diff 加规则。
false negative 可以通过后续通用 profile 升级解决，false positive 不接受。

当前 `jc-normalizer.v2` 因此是 identity/fail-closed 基线：π 非空仍被记录，但不
改变 observation。这不表示 π 永久无效，而是禁止在没有 sound consumer 前先声称
已有消融能力。后续 π ablation 只在启用已审计 rule 的 profile 上报告，不用当前
identity 基线伪造收益。

## 6. Gate1 与 Gate2 的分工

Gate1 提供三项证据：

- J/C 因子化的 compiler-visible 结构闭合；
- 在 profile 确实观察这些类别时，于 Gate2 前提前拒绝 wrong
  op/reducer/mask/fill/access/recurrence/effect 等明显错误；
- 向 source-aware Lift owner 提供结构化 counterexample。

Gate1 不执行数值语义，也不证明 π。Gate2 保持三条直接比较：

\[
G_2^{Q,\mathcal T}
=
\bigwedge_{x\in\mathcal T}
\left[
S(x)\approx_Q R_J(x)
\land S(x)\approx_Q \operatorname{RefEval}(J,x)
\land R_J(x)\approx_Q \operatorname{RefEval}(J,x)
\right].
\]

Gate2 是声明执行域内的最终行为锚点；Gate1 不是其替代品，二者都不声称测试域外全称等价。

## 7. 失败分流与 freshness

| 失败 | 处理 |
|---|---|
| Replay compile/API failure | 只向 Replay Agent 返回 J-visible compiler diagnostics，继续同一 source-blind episode |
| 已支持 realization 未能规范化 | 同一实验只可修 π 实例并复用 R；若需改 normalizer，结束本实验、升 profile 后新开实验 |
| J/C-owned hard mismatch | source-aware owner 判断 J/C 缺失或 Replay 未忠实；J/MaskC 改变后 fresh Replay |
| observer/profile 不支持或歧义 | fail closed / `observer_unsupported`，不得临时放宽 |

完整 Gate1 comparison 保存在外部审计 evidence；同一 branch 的 source-aware Lift owner 只收到它的
确定性、有限 typed frontier。上一轮 Replay source、Replay TTIR、agent context 与私有 realization
metadata 均不返回 Lift，也绝不进入 Replay。这样 Lift 不能为了某次 Replay 的偶然实现改写 J；若
Source 与当前 J/C 已完整，而 frontier 只表明 Replay 未忠实，则保持 artifact 不变并启动 fresh Replay。

- π-only 改变：复用 exact R，重跑 TypeCheck 与 Gate1；
- J 或 `MaskC(C)` 的结构/attachment 改变：fresh Replay；
- 仅 `c₀` 改变：可复用 Replay source，但必须重新 Bind/Compile/Observe/Gate；
- observer/normalizer profile 改变：属于新实验；可复用 Replay source bytes 作为输入，但旧 Gate1
  evidence 不升级，必须重新 Bind/Compile/Observe/Gate。

Agent 执行层可以在不改变上述谓词的情况下提前拒绝明显欠完整的草稿。当前 SourceReconcile 只检查
三种 one-bit 必要条件：Source-visible masked access、含非线性算子的 multi-state recurrence、terminal
effect control 是否被 J 整体遗漏。它不生成 predicate、trip、carry、update 或 branch；完整正确性仍由
TypeCheck、fresh Replay 和 Gate1 裁决。closed submit 一旦接受 exact bytes，host 可停止无证明意义的
结尾 prose，恢复该 checkpoint 并独立重跑同一检查；这只缩短搜索时间，不改变 artifact 或 acceptance。
实现可对 Agent 正在写的唯一 output 做稳定字节监视；只当同一份 LLM-authored
exact bytes 已通过与显式 submit 完全相同的 CandidateCheck+SourceReconcile（Replay 为
ingress+construction compile）后，才能中止当前 provider turn。Host 随后仍恢复并重检
这份 exact bytes。监视器不修改文件、不产生字段、不读取 Gate，也不把旧 formal-round
candidate 当作当前新交付。

## 8. 多前端与实验

Triton、CuTe、TileLang、CUDA 可以共享 π wire syntax 与 J/C ownership，却分别定义
`Phi_{f,p}`、`H^{J,C}_{f,p}` 和窄 `N^π_{f,p}`。同名 relation 不保证跨 frontend 同一 denotation，
一个 profile 的 PASS 不能转移到另一个 profile。设计目标是把工程量限制为按 profile、而非按
Kernel 增长；是否达成必须由实现与实验报告。

实验主要报告 Gate1 对论文主张的实际价值，而不是 π clause 统计：

- Gate1 对最终 Gate2 失败候选的提前拒绝率与成本；
- 有/无 Gate1 结构反馈的 relift 轮数与 `closure@k`；
- wrong op/reducer/mask/fill/index/effect 的 mutation sensitivity；
- 合法 realization difference 的误拒率；
- Gate1 提前拦截的 compile/runtime failure。

## 9. Figure 3 与最终 claim

Figure 3 的拓扑不改，只更新内部标签：

- Replay 输入删除 π，只保留 `J + MaskC(C)`；
- π 只连到 Gate1 内部 normalizer；
- Gate1 标签为 `J/C Structural Closure`；
- Gate2 三边与最终 `SCRC_CERTIFIED` 不变。

最终可以防守的表述是：

> `J/C` 是可复用的 lifting result。Replay 以 J 为唯一显式语义权威，并只保留匿名 MaskC 接口；
> Gate1 在固定 frontend/profile/valuation 下，使用 π 仅消除预先定义的 realization 噪声，确定性检查
> Source 与 J-conditioned Replay 的 J/C-owned compiler structure 是否闭合；Gate2 再在声明执行域上
> 独立锚定 Source、Replay 与 RefEval(J) 的行为。
