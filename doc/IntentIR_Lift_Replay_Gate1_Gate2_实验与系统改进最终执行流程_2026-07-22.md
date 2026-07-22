# IntentIR：Lift、Replay、Gate1、Gate2 实验与系统改进最终执行流程

> 状态：最终执行设计（2026-07-22）  
> 前置设计：[IntentIR C：所有权、绑定、消费与三方验证设计](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)  
> 前置设计：[IntentIR π：依赖类型实现见证与 Replay 安全设计](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)

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

这里的“系统规则”包括 J/C/π schema、Lift prompt、Replay prompt、Gate1 observer、Comparator、编译器和本次实验使用的工具能力。它只表示一次实验中的运行纪律。

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
- operator、reducer、effect、alias、result relation；
- 必要的数值、宽度、signedness、overflow 或地址语义。

如果一个事实可能改变 J 的允许结果关系，它必须先由 J 或 B 表达，不能只放进 π 或 C。

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

它不能看到 Source、Harness、C 的具体值、Gate1 diff、旧 Replay 或旧候选代码。

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
- Gate1 observer 和 Comparator；
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
      (J,C,π)
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

## 6. 第二步：TypeCheck(J,C,π)

TypeCheck 不是新的 Gate，也不是新的证明对象。它只是 Replay 之前的确定性合法性检查。

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
- representation refinement 是否被 J/B 允许；
- 是否存在 source text、frontend API、C value 或可执行自由文本。

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

## 8. 第四步：Harness 绑定 C

Replay source 生成并完成后，Harness 执行：

```text
Bind_H(Source,C)
Bind_H(Replay,C)
```

绑定必须使用同一个 `ν_C`。

Harness 可以把同一个 C slot 绑定到 Source 和 Replay 的不同参数名或不同调用位置，但这只是机械调用关系，不得修改 Replay AST。

如果 C slot 无法绑定，实验在 Gate1 之前失败，原因归入 C/接口绑定问题；不能靠 Injector 新增使用点解决。

## 9. 第五步：Gate1

设当前 frontend observer 为 `F_f`：

```text
F_source = F_f(Compile_H(Source,C))
F_replay = F_f(Compile_H(Replay,C))
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
A_r = (J_r,C_r,π_r)
D_r
```

它必须输出：

```text
A_(r+1) = (J_(r+1),C_(r+1),π_(r+1))
```

接着重新执行：

```text
TypeCheck
→ source-blind Replay
→ Harness 绑定 C
→ Gate1
```

旧 Replay 只作实验记录，不进入下一轮 Replay 输入。

### 10.2 relift 次数

一次实验最多进行三次能够进入 Replay 的正式轮次。三次均未通过 Gate1：

```text
experiment_status = NO_CLOSURE
```

`NO_CLOSURE` 只表示：在本次实验所使用的 J/C/π 表达能力、prompt、Replay 能力和 Gate1 observer 下，没有找到闭合候选。

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
O_S = Exec(Source, ρ_H^S(C), x)
O_R = Exec(Replay, ρ_H^R(C), x)
O_J = RefEval(J, x)
```

Gate2 为：

```text
G2 = (O_S equivalent_B O_R)
     and
     (O_S equivalent_B O_J)
```

其中：

- Source 和 Replay 使用同一 C；
- `RefEval(J,x)` 不接收 C；
- `RefEval` 不读取 π；
- comparator 和允许结果关系来自 J/B。

Gate2 失败后，不在同一次实验中继续使用该结果做局部修补。实验结束，进入系统改进，再开始下一次实验。

## 12. masked-J 的位置

masked-J 不是主认证条件，也不是每次实验必须增加的额外 Gate。

如果研究需要，可以在 Gate1 通过后单独记录：

```text
J_NECESSARY = masked-J hard mismatch
```

它只说明：在当前 J、π、C 和 Replay 模型下，删除 J 后是否仍能完成相同的 Gate1 closure。

masked-J hard-close 不直接等同于语义泄漏，也不自动推翻：

```text
TypeCheck(J,C,π) + Gate1 + Gate2
```

如果需要进一步检查 J 的语义权威，可以使用单独的 J-authority challenge；该结果独立记录，不改变主流程的对象边界。

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

如果当前 J/π schema 无法表达该 relation，最多三轮后结束本实验。之后在系统改进阶段决定：

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
Lift 首轮 TypeCheck 通过率
Lift 格式修正次数
第一轮 Gate1 closure rate
三轮内 Gate1 closure rate
Gate2 Source/Replay/J closure rate
SCRC_CERTIFIED rate
```

这样能够区分：

- 一次 Lift 就能抽取完整关系；
- 经过 Gate1 反馈后才闭合；
- 语义上能够闭合但 Gate2 失败；
- schema 或模型能力确实不足。

## 16. 主结论

一次实验的完整逻辑是：

```text
Lift
→ TypeCheck(J,C,π)
→ source-blind Replay(J,π,MaskC(C);U_f)
→ Harness 绑定 C
→ Gate1
→ Gate1 diff 返回 Lift 并 relift
→ Gate1 通过后执行 Gate2
```

实验期间不改变系统规则；达到 relift 次数仍失败就结束实验。系统改进只能在实验结束后进行，并且必须以通用能力修改的方式解决：

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
SCRC_CERTIFIED
    iff
TypeCheck(J,C,π) = PASS
and Gate1 = PASS
and Gate2 = PASS
```

这套流程保持 C 和 π 的对象边界，不引入第四通道，也不把一次实验中的失败修补误认为系统已经获得新的表达能力。
