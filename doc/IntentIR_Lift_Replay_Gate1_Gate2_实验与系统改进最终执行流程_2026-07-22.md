# IntentIR：Lift、Replay、Gate1、Gate2 实验与系统改进最终执行流程

> 状态：最终执行设计（2026-07-22）
>
> 前置设计：[IntentIR C：所有权、绑定、消费与三方验证设计](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
>
> 前置设计：[IntentIR π：依赖类型实现见证与 Replay 安全设计](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)
>
> 数学与 claim 总述：[IntentIR / SCRC 最终数学、执行流程与论文表述](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)
>
> Claim 收紧复审：[最终数学与三份设计的局部收紧建议](source-notes/高级ai对最终数学与三份设计的局部收紧建议_2026-07-22.md)
>
> 纠错与复审：[旧执行流程纠错](source-notes/高级ai对于之前流程的纠错和判断.md)、[三份核心设计复审与四项修正](source-notes/高级ai对三份核心设计的复审与四项修正_2026-07-22.md)

## 1. 这份文档解决什么问题

C 和 π 已经确定了对象所有权、Replay 可见范围以及 Gate1/Gate2 的验证关系。本文件只确定运行方式：

1. 一次实验怎样从 Lift 开始并走到 Gate2；
2. Gate1 失败后怎样进行 relift；
3. 什么算一次正式实验轮次；
4. 什么时候停止实验；
5. 实验结束后怎样修改系统并重新开始。

本文件不引入新的证明对象、额外 Gate、额外 Agent 或第四个 Replay 输入。

最核心的原则是：

> **一次实验运行期间，系统规则保持不变；实验结束后，才根据失败证据修改系统。**

这里的“系统规则”包括 J/C/π schema、Lift prompt、Replay prompt、Gate1 observer profile、比较
政策 Q、编译器和本次实验使用的工具能力。它只表示一次实验中的运行纪律。

## 2. 系统中的对象和职责

### 2.1 Lift

Lift 是 source-aware 的抽取过程：

```text
Lift_H(Source) -> (J, C, π)
```

它可以看到 Source 以及完成 J/C 分类所需的 Harness 机械信息，但不能产生 `(J,C,π)` 之外的证明相关对象。

### 2.2 J

J 是唯一的语义对象，负责表达：

- value 和 state relation；
- logical domain、predicate、fill；
- logical index 和 typed index relation；
- logical signature（inputs、outputs、runtime scalar、dtype、rank、shape、alias/inout）；
- operator、reducer、effect、alias、result relation；
- 必要的数值、宽度、signedness、overflow 或地址语义。

如果一个事实可能改变 J 的允许结果关系，它必须先由 J 或 B 表达，不能只放进 π 或 C。

如果工程实现需要函数接口视图，只能从 J 确定性得到 `InterfaceView(J)`。它不是第四个 Lift
artifact，也不作为 Replay 的额外输入；C slots 仍由 `MaskC(C)` 提供，调用位置由 Harness
机械绑定。

实现中的模型 response 在成为 Lift artifact 前，必须经过唯一的固定代码标准化：

```text
response JSON -> Decode/Elaborate -> (J,C,pi) | deterministic diagnostic
```

Decode/Elaborate 成功后只有一个 typed J，不存在 `J_wire` 与 `CanonicalJ` 两个对象。该步骤将
`k0`/`[k0]`、`k1,k0`/`[k1,k0]` 等 closed surface sugar 归一为 structured axes，将
domain/predicate/index 解析为 ContractExpr AST，并物化 closed op signature 唯一确定的 result type、
消去可机械证明的 identity output mapping。它不读 Source、C/π、Harness concrete value 或 Gate
结果；需要猜 operator、predicate、effect、authorization 或类型选择时必须 fail closed。

### 2.3 C

C 是保持 J denotation 的 implementation-selection 对象：

```text
C = (Σ_C, ν_C)
```

其中：

- `Σ_C` 是匿名 slot schema；
- `ν_C` 是 Harness 持有的 concrete valuation。

C 可以表达 tile、vector width、warp、pipeline、prefetch、buffer 等纯调度和实现选择，但不能保存 J 的语义副本。

### 2.4 π

π 是 source-realization witness：

```text
TypeCheck(J,C,π)
```

π 可以表达 instance、region、route、operand slot、order、sharing、placement、materialization 和受约束 representation relation，但不能拥有独立语义。

如果 π 表达整数宽度、numeric cast、reassociation、accumulation dtype 或其他可能改变结果关系的事实，TypeCheck 必须确认该事实已经由 J/B 授权，或者属于系统已经规定的合法 representation refinement。

### 2.5 Replay

Replay 是全新的 source-blind 生成：

```text
Replay = W_f(J, π, MaskC(C); U_f)
```

它只能看到：

- J；
- π；
- 去掉具体值后的 `MaskC(C)`；
- 与当前 Source 无关的公共前端知识 `U_f`。

它不能看到 Source、Harness、C 的具体值、source parameter name/position、BindingMap、独立
interface sidecar、Gate1 diff、source/compiler fingerprint、旧 Replay、旧候选代码、旧模型
conversation context 或 per-kernel RuleCard。

### 2.6 Harness

Harness 只负责：

- 枚举匿名参数和 compiler slot；
- 保存 concrete inputs 和 C values；
- 将同一份 C 绑定到 Source 和 Replay；
- 建立 compile/run invocation；
- 执行 Gate1/Gate2 所需的编译和运行。

Harness 不负责：

- 判断事实属于 J 还是 C；
- 生成 loop、mask、address、reduction 或 sharing；
- 修改 Replay AST；
- 创建新的 source-conditioned artifact。

## 3. 一次实验和系统改进必须分开

### 3.1 一次实验

一次实验使用当时系统中的同一套：

- J/C/π schema；
- TypeCheck 规则；
- Lift 和 Replay prompt；
- Gate1 observer profile 和比较政策 Q；
- 编译器、前端和工具能力；
- relift 次数上限。

实验运行期间不能针对某一个 Kernel：

- 临时增加 schema 字段；
- 临时增加 `extsi_mode`、`splat_policy` 等专用字段；
- 临时修改 Gate1 observer；
- 临时删除某个 compiler feature；
- 临时让 Replay 看到 diff 或旧代码；
- 临时增加新的 Agent 或 RuleCard。

### 3.2 系统改进

一次实验结束后，才分析失败属于哪一层：

1. Lift 没有使用已有表达能力；
2. Replay 没有正确消费已有 J/C/π；
3. J 的语义表达能力不足；
4. π 的 realization 表达能力不足；
5. C 的所有权或绑定规则不足；
6. Gate1 observer 比较了不应要求恢复的全局 compiler 差异；
7. 前端、编译器或工具链能力不足；
8. Gate2 的 reference 或 comparator 不完整。

系统改进必须是通用修改，不能只为单个 Kernel 添加分支。修改完成后，重新开始一次新的实验。

## 4. 一次实验的完整流程

```text
Source + Harness 机械信息
          │
          ▼
       Lift
          │
          ▼
  response JSON
          │
          ▼
 Decode/Elaborate -> (J,C,π)
          │
          ├─ 失败：Lift 修正；不进入 Replay
          │
          ▼
 TypeCheck(J,C,π)
          │
          ├─ 失败：Lift 修正；达到修正上限则实验失败
          │
          ▼
  source-blind Replay
          │
          ▼
  Harness 绑定 C
          │
          ▼
        Gate1
          │
          ├─ 失败：结构化 diff 只返回 Lift，重新生成完整 (J,C,π)
          │
          └─ 通过：进入 Gate2
                         │
                         ▼
                       Gate2
                         │
                         ├─ 失败：实验失败，进入系统改进
                       └─ 通过：SCRC_CERTIFIED
```

实际候选搜索采用同一 Source、同一 Harness 与同一 `c0` 下的有界并行 fan-out，而不是把不同
Kernel 并发混在一次认证中。设 branch 数 `B=5`；每个 branch 拥有独立 Lift 上下文、完整
`(J,C,pi)`、TypeCheck、全新盲 Replay、Gate1 evidence 与 branch-local relift history。branch 之间
不得读取、拼接或投票组合 artifact。Harness 仍是机械绑定环境，不变成新的 LLM Agent。

## 5. 第一步：Lift 生成完整三元 artifact

Lift 的每一次正式输出都必须是完整的：

```text
(J,C,π)
```

不能只输出：

- J 的补丁；
- π 的局部片段；
- 一个额外 BindingMap；
- 一个 Resolver 输出；
- 一个自由文本诊断结论。

如果是第 r 次 relift，Lift 可以看到：

```text
Source
上一轮完整 (J,C,π)
上一轮 Gate1 的结构化 diff
```

但它仍然必须重新输出完整的 `(J,C,π)`，而不是输出对旧 artifact 的 patch。

## 6. 第二步：Decode/Elaborate 与 TypeCheck(J,C,π)

Decode/Elaborate 与 TypeCheck 都不是新的 Gate或证明对象，而是 Replay 之前的确定性闭合 IR 前端。
前者从 response 构造唯一 J并消除合法 surface form 的偶然差异；后者再检查所有权、引用和授权。
形式上：

```text
Decode/Elaborate(response) = (J,C,pi)
and TypeCheck(J,C,pi)
```

Replay packet、RefEval 与 production lowering 只消费这一个 J，不得各自重新解析 response string。
RefEval 与正式 backend 的 MLIR 路径共享同一个 `J -> Intent MLIR` 入口，随后才在 reference
execution 与 target/C lowering 处分叉。解析 AST/axes 可作为 J 的私有缓存，但不序列化、不传递、
不具有独立身份。

### 6.1 J 检查

检查 J 的：

- node、value、tensor、access、effect 引用；
- 类型和 operand arity；
- domain、predicate、fill、index relation；
- result relation 和 update relation；
- typed index、numeric conversion 等语义关系。

### 6.2 C 检查

检查 C 的：

- slot 是否匿名且唯一；
- binding kind 和 value type 是否匹配；
- concrete value 是否来自 Harness 已有坐标；
- 是否把 predicate、formula、logical shape 或 result semantics 错误放进 C。

### 6.3 π 检查

检查 π 的：

- 每个 semantic reference 是否存在于 J/C 的声明环境；
- instance、region、route、order、sharing 是否闭合；
- route slot 是否与被引用 J carrier 相容；
- 是否出现未由 J/B 授权的 observable stage；
- 是否新增 operator、predicate、fill、formula、effect 或 semantic constant；
- representation refinement 是否属于系统封闭 schema，且参数与类型是否合法；
- sensitive refinement 是否引用 J/B 中已有的 authorization，或属于系统安全白名单；
- 是否存在 source text、frontend API、C value 或可执行自由文本。

artifact schema 必须 closed-world：未知字段、未知 relation 和自由文本扩展均 fail closed。若
`SemDecl` 只计直接创建的显式逻辑语义声明，则 TypeCheck 支撑的是：

```text
FreshLogicalSemDecl(C,pi | J) = empty
```

或在合法引用解析回 J 后写成 `SemDecl(J,C,pi)=SemDecl(J)`。它不证明 π 没有统计语义线索，
不证明 J 对 Source 正确，也不证明 realization 保持 J 行为。

TypeCheck 不证明某种 representation 在一般情况下与 J 语义等价。对于 cast、integer width、
reassociation、atomic form 等 sensitive refinement，它只检查两件事：J/B 是否已有明确的 typed/effect
authorization，或者该 refinement 是否属于系统统一维护的安全白名单；两者都不满足就直接失败。

### 6.4 格式修正和正式实验轮次

以下错误仍属于 Lift 输出修正：

- JSON 格式错误；
- 必填字段遗漏；
- enum 拼错；
- route 引用不存在的 instance；
- schema 层级错误；
- TypeCheck 能直接指出的引用和类型错误。

这些错误在 Replay 之前处理，不算一次 Gate1 正式轮次。但每次实际 LLM 调用仍应计入普通调用记录，并设置修正次数上限，不能无限重试。

只有 TypeCheck 通过、能够进入全新 Replay 的候选，才算一次正式实验轮次。

## 7. 第三步：source-blind Replay

TypeCheck 通过后，Replay 只接收：

```text
J + π + MaskC(C) + U_f
```

Replay 必须：

1. 从 J 读取 operator、predicate、fill、index、effect 和 result semantics；
2. 从 π 读取 instance multiplicity、route、slot、order、sharing、placement 和 representation；
3. 从 `MaskC(C)` 读取匿名 slot schema，不读取 concrete values；
4. 根据 `U_f` 选择合法的 frontend API spelling；
5. 输出完整、参数化、source-blind 的 frontend source。

Replay 不得：

- 根据自己的猜测覆盖 J 语义；
- 丢弃 π 中的显式 stage；
- 折叠 π 要求保留的 materialization 或 sharing；
- 从旧 Replay 或 Gate1 diff 推断修补方案；
- 等待 Harness 在生成后补出新的 use-site。

`U_f` 必须是版本化、source-neutral 的公共前端知识，不能按当前 Kernel、operator 或失败样本选择。
runner 可以保存实际 request/output 和 compiler metadata 供追溯，但这些内容不能回流 Replay packet；
不需要为此增加 hash、commitment 或签名门。

## 8. 第四步：Harness 绑定 C

Replay source 生成并完成后，Harness 执行：

```text
Bind_H(Source,C)
Bind_H(Replay,C)
```

绑定必须使用同一个 `ν_C`。

记本次实际绑定为 `c0 = Bind_H(C)`。当前证书只覆盖这个 observed c0；即使 C 的设计合同要求
所有 admissible choice 保持 J，单次 Gate1/Gate2 也不能外推成全配置证明。多 C 测试只能按实际
覆盖的有限配置集合另行报告。

Harness 可以把同一个 C slot 绑定到 Source 和 Replay 的不同参数名或不同调用位置，但这只是机械调用关系，不得修改 Replay AST。

如果 C slot 无法绑定，实验在 Gate1 之前失败，原因归入 C/接口绑定问题；不能靠 Injector 新增使用点解决。

## 9. 第五步：Gate1

设当前 frontend 与 observer profile 为 `(f,p)`，其中 p 固定 observer/version、compiler/dump stage
与 canonicalization version：

```text
F_source = F_(f,p)(Compile_H(Source,c0))
F_replay = F_(f,p)(Compile_H(Replay,c0))
```

Gate1 为：

```text
G1 = [F_source == F_replay]
```

Gate1 的职责是检查 source-blind Replay 是否恢复了当前 observer 要求的 frontend realization。

Gate1 不负责：

- 运行语义等价证明；
- 在当前 Kernel 上临时判断某个 op 能否忽略；
- 用范围分析或 SMT 临时放宽比较；
- 判断 J 是否正确；
- 替 Lift 做语义分类。

### 9.1 canonicalization 的规则

Gate1 可以使用协议中已经确定的全局 canonicalization，例如：

- SSA 名归一；
- 参数名归一；
- source location 和路径归一；
- 无意义的声明顺序归一；
- 其他已经明确、全局适用的 frontend presentation noise。

如果未来要把某类 compiler transformation 全局 NormalizeAway，必须在实验结束后的系统改进阶段统一修改 observer。不能在某个 Kernel 失败时临时忽略。

因此，Gate1 的准确含义是：

```text
在当前 observer 和当前 canonicalization 规则下精确比较。
```

每份证书必须记录 `frontend`、observer profile/version、compiler/dump stage 和 canonicalization
version。mutation test 可以在 CI 或消融中评估 observer 能抓住哪些错误，但不进入主 Gate，也
不能被表述成 observer 完备性证明。

## 10. 第六步：Gate1 diff 与 relift

Gate1 失败后，Comparator 只输出确定性的结构化事实：

```text
Diff = Diff(F_source,F_replay)
```

例如：

```json
{
  "missing_ops": [
    {"name": "arith.extsi", "count": 2}
  ],
  "unexpected_ops": [],
  "dataflow_differences": [],
  "predicate_differences": [],
  "effect_differences": []
}
```

Comparator 不直接下结论：

- “这是 J 的错”；
- “这是 π 的错”；
- “这个差异语义无关”；
- “应该修改 prompt”。

这些判断由 source-aware Lift 在下一轮处理，或者在实验结束后由系统改进阶段处理。

### 10.1 relift 的输入

第 r+1 轮 Lift 可以使用：

```text
Source
Artifact_r = (J_r,C_r,π_r)
D_r
```

它必须输出：

```text
Artifact_(r+1) = (J_(r+1),C_(r+1),π_(r+1))
```

接着重新执行：

```text
TypeCheck
→ source-blind Replay
→ Harness 绑定 C
→ Gate1
```

旧 Replay 只作实验记录，不进入下一轮 Replay 输入。

### 10.2 branch 数与 relift 预算

当前搜索预算固定为 `B=5` 个并行 branch。每个 branch 最多进行 `K=3` 次能够进入 Replay 的
TypeCheck-valid 正式 Gate1 round，`K` 包含首次候选；schema/TypeCheck 的 pre-Replay rejection
另按既定上限计数，并必须报告 logical 与 physical provider calls。

五个 branch 都在各自预算内未通过 Gate1，且没有 infrastructure-incomplete branch 时：

```text
experiment_status = NO_CLOSURE_(B=5,K=3)
```

`NO_CLOSURE_(B,K)` 只表示：该次有界并行搜索在所使用的 J/C/π 表达能力、prompt、Replay 能力和
Gate1 observer 下没有找到闭合候选。只要存在尚未完成的 provider/infrastructure branch，就只能
报告搜索不完整，不能伪装成 `NO_CLOSURE`。

它不自动说明：

- J 一定错误；
- π 一定泄漏；
- Replay 模型一定无能力；
- Gate1 observer 一定设计错误。

这些要到实验结束后的系统改进阶段分类。

## 11. 第七步：Gate2

只有 Gate1 通过的候选才进入 Gate2。

对测试输入 `x`：

```text
O_S = Exec(Source, c0, x)
O_R = Exec(Replay, c0, x)
O_J = RefEval(J, x)
```

Gate2 为：

```text
G2 = Close_Q(O_S,O_R)
     and Close_Q(O_S,O_J)
     and Close_Q(O_R,O_J)
```

其中：

- Source 和 Replay 使用同一 C；
- `RefEval(J,x)` 不接收 C；
- `RefEval` 不读取 π；
- B 只声明封闭 comparison class 与必要语义合同，外部比较政策 Q 据此构造 `Close_Q`。

### 11.1 比较关系的来源和范围

B 只能声明系统支持的 comparison class 和必要语义合同，例如 exact 或受限浮点 tolerance。Q 是
实验协议或 benchmark correctness policy 预先确定的可信政策，根据经验证的 class、dtype、
reduction extent、reassociation 权限等构造 comparator，并检查 tolerance 上限、NaN/Inf 合同等
全部参数；不支持的类别或非法参数直接 fail closed。Q 不是 Lift artifact，也不进入 Replay。

Lift 不能自由写入任意 Python comparator，也不能通过任意大的 `rtol/atol` 或“所有输出均可接受”
来放宽自己的 Gate2。B 保留对语义比较类别与必要合同的所有权，但候选不能自己决定及格线。

`Close_Q` 必须是确定性、对称的比较谓词；若底层 `isclose` 有方向性，Q 必须显式对称化或执行
双向检查。第三条 Replay–J 边不增加一次执行，只比较三个已经产生的输出；它防止近似关系不传递
时，两条 Source-centered 边都通过但 Replay 与 J 彼此超出范围。

当前主线只覆盖 deterministic / contract-deterministic Kernel：内部执行顺序可以变化，但 J/B
在协议可观察层面必须给出唯一合同结果。真正集合值的语义不因“非确定性”一词自动降低 claim，
而是留给未来 `Accept_{J,Q}(x,o)` 的逐观察 conformance；有限次 membership 不能宣称完整
`Beh(P,x) subseteq Spec_J(x)`。

### 11.2 Gate2 输入的生成顺序

令 `tau_pre` 表示五个 branch 的 Lift、TypeCheck、Replay 与 Gate1 记录，但不含任何 Gate2 输入或
结果。当前实现等待这一 search wave 完成，然后使用预先确定的选择函数：

```text
a* = sigma(tau_pre)
sigma = 按 branch_id、formal_round 的固定顺序选择第一个 Gate1-closed 候选
```

只有这个完整候选 `a*=(J,C,pi,Replay)` 被选定；随后才生成 Gate2 输入，本次 Gate2
期间不再修改该候选。如果采用隐藏随机输入模式，必须从预先声明的分布 D 独立同分布地抽取输入，
并对 Lift/Replay 隐藏这些输入；Gate2 结果也不回流本次实验。令 `g2_Q(a*,x)` 表示三条边在 x
上全部通过，只有当固定候选的 `Pr_D[not g2_Q(a*,x)] >= epsilon > 0` 时，才能报告 m 个
challenge 全部通过的上界 `(1-epsilon)^m`。Source–J 错误质量是其中的充分特例；“错误 J”本身
不自动保证在 D、Q 下有可检测 disagreement。

如果只使用预先给定的有限测试集合 `T`，则只报告：

```text
for every x in T:
    Close_Q(Source(x),Replay(x))
    and Close_Q(Source(x),RefEval(J,x))
    and Close_Q(Replay(x),RefEval(J,x))
```

不能在只使用固定 `T` 时同时引用隐藏随机测试的概率结论。两种输入模式都不需要新的证明对象
或新的运行阶段。

未选 branch 即使 Gate1 闭合也不执行 Gate2。Gate2 失败后，不 fallback 到 sibling candidate，
不把结果返回任何 Lift，也不在同一次运行中继续局部修补。运行结束，进入通用系统改进，再开始
新的完整 fan-out。

## 12. masked-J 的位置

masked-J 不是主认证条件，也不是每次实验必须增加的额外 Gate。

如果研究需要，可以在 Gate1 通过后单独记录：

```text
J_NECESSARY = masked-J hard mismatch
```

它只说明：在当前 J、π、C 和 Replay 模型下，删除 J 后是否仍能完成相同的 Gate1 closure。
即使状态记录为 `PROVED`，结论也只相对于记录的模型、解码配置、Replay packet 和 masked-J
干预成立，不是对所有生成器的全局不可替代性证明。

masked-J hard-close 不直接等同于语义泄漏，也不自动推翻：

```text
TypeCheck(J,C,π) + Gate1 + Gate2
```

如果需要进一步检查 J 的语义权威，可以另行运行离线 J-authority challenge；它不自动触发，
结果独立记录，也不改变主流程的对象边界。

## 13. `arith.extsi` 的处理方式

在一次实验中，若 Gate1 报告 Source 存在 Replay 缺少的 `arith.extsi`：

1. 不在 Gate1 中临时删除该差异；
2. 不新增 `extsi_count` 之类的 Kernel 专用字段；
3. 将结构化 diff 返回 source-aware Lift；
4. Lift 检查 J 是否已经表达对应的 typed conversion relation；
5. 如果 J 已经授权该 relation，π 表达显式 materialization、placement、order 和 sharing；
6. Replay 根据 J、π 和公共 frontend 知识生成相应实现；
7. 再次执行 TypeCheck、Replay 和 Gate1。

如果整数宽度、signedness 或 overflow 可能改变 J 的允许结果，那么这些内容必须由 J/B 拥有，π 只能表达 realization 关系。

如果当前 J/π schema 无法表达该 relation，各 branch 达到 `K=3` 且完整 `B=5` fan-out 仍未闭合后
结束本次运行。之后在系统改进阶段决定：

- 扩充 J 的 typed index/conversion 能力；
- 扩充 π 的 generic representation relation；
- 修改 Replay 对 representation relation 的消费方式；
- 或通过跨案例证据决定该 compiler difference 是否应由 Gate1 全局归一。

不能在同一实验里临时改变其中任何一项。

## 14. 系统改进阶段的修改方法

### 14.1 Lift 能表达但没有抽取出来

修改：

- Lift prompt；
- Lift 输入说明；
- Lift 的公共示例；
- 结构化 diff 的表达方式。

不修改：

- 单个 Kernel 专用字段；
- Replay 的额外输入；
- Harness AST 修补逻辑。

### 14.2 π 能表达但 Replay 没有消费

修改：

- Replay prompt；
- π 消费顺序说明；
- instance/route/order/sharing 的公共解释。

不把旧 Replay 代码提供给下一次 Replay。

### 14.3 J 能力不足

修改：

- J schema；
- J 的类型和语义关系；
- Gate2 的 J-only reference。

任何可能影响结果关系的表示属性，都先判断是否属于 J/B。

### 14.4 π 能力不足

修改：

- generic representation relation；
- placement/materialization/order/sharing 的表达；
- `TypeCheck(J,C,π)` 的依赖检查。

不按 kernel family 增加 `softmax_stage`、`row_address`、`extsi_mode` 等专用 ontology。

### 14.5 Gate1 observer 不合适

只有在多个实验结果和人工/自动变异证据都说明某类差异不是应该恢复的 realization 时，才修改 observer 的全局规则。

修改后重新开始实验，不能对旧实验结果静默重判。

### 14.6 Gate2 或前端工具不足

如果 Source、Replay 和 J 的行为比较不完整，或者前端编译/执行工具失败，应修复对应通用工具，再重新实验。

不能用 Gate1 通过代替 Gate2，也不能用 production lowering 代替 `RefEval(J)`。

## 15. 实验结果记录

每个实验至少记录：

- Source、J、C、π、Replay 的文本或结构结果；
- TypeCheck 是否通过；
- 每次正式轮次的 Gate1 结果；
- 每轮结构化 diff；
- Gate2 结果及失败关系；
- LLM 调用次数和失败原因；
- 最终状态。

这些是实验记录，不是新的证明对象，也不是 Replay 输入。

建议分别报告：

```text
first-attempt TypeCheck pass rate
Lift 格式修正次数
closure@1 / closure@2 / closure@3
Lift call count / Replay call count / total LLM call count
NO_CLOSURE failure categories
Gate2 pairwise Source/Replay/J pass rate among Gate1-closed candidates
SCRC_CERTIFIED rate
```

`closure@k` 定义为截至第 k 个正式 Gate1 轮次的累计闭合任务比例，不是第 k 轮的独立成功率。
证书绑定实际生成的 Replay，而不是不报告搜索成本的抽象 `exists R`。

这样能够区分：

- 一次 Lift 就能抽取完整关系；
- 经过 Gate1 反馈后才闭合；
- 语义上能够闭合但 Gate2 失败；
- schema 或模型能力确实不足。

## 16. 主结论

一次实验的完整逻辑是：

```text
B=5 个独立 branch 并行：
  Lift response → Decode/Elaborate → (J,C,π) → TypeCheck(J,C,π)
       → source-blind Replay(J,π,MaskC(C);U_f)
       → Harness 绑定同一 c0 → Gate1
       → Gate1 diff 只返回本 branch 的 Lift 并 relift
→ sigma 仅根据 pre-Gate2 evidence 选择一个 Gate1-closed winner
→ 只为 winner 生成隐藏输入并执行 Gate2
```

运行期间不改变系统规则；达到 branch/relift 预算仍失败就结束本次搜索。系统改进只能在完整运行
结束后进行，并且必须以通用能力修改的方式解决：

```text
Lift 能力问题       → 改 Lift
Replay 消费问题     → 改 Replay
J 语义能力问题      → 改 J
π realization 问题  → 改 π
C 所有权问题        → 改 C/绑定规则
观察器问题          → 改 Gate1 observer
语义验证问题        → 改 Gate2/reference
工具问题            → 改通用 frontend/toolchain
```

最终认证条件只有：

```text
SCRC_CERTIFIED_(f,p,Q,T,c0)(Source,J,C,pi,Replay)
```

其中 p 是 observer profile，c0 是本次实际绑定。为便于实现可继续使用下面的简写，但论文不得
把它外推到其他 observer、比较政策、C 配置或全输入域：

```text
SCRC_CERTIFIED
    iff
TypeCheck(J,C,π) = PASS
and Gate1 = PASS
and Gate2 = PASS
```

这套流程保持 C 和 π 的对象边界，不引入第四通道，也不把一次实验中的失败修补误认为系统已经获得新的表达能力。
