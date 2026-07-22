# IntentIR C：所有权、绑定、消费与验证设计

> 状态：新架构核心设计（2026-07-22）
>
> 修订依据：[高级 AI 对 C 设计的纠正](source-notes/高级ai对c的纠正.md)、[高级 AI 对 C 设计的自省](source-notes/高级ai对c的自省.md)
>
> 联合复审：[三份核心设计复审与四项修正](source-notes/高级ai对三份核心设计的复审与四项修正_2026-07-22.md)
>
> 范围：定义 C 在 Lift、Replay、Gate1、Gate2、reference semantics 与直接 production lowering 中的最小合同
>
> 核心不变量：**证明相关 artifact 严格只有 `(J,C,pi)`；Replay 只看 `J + pi + MaskC(C)`。**

## 1. 最终判断

本版撤回此前草案中两个不属于当前系统的扩张：

1. 不再把 `CBindingMap`、portable hints、Generic Resolver 写成 Lift 的额外产物；
2. 不再引入 LoweringAgent、C 生成 Agent、retune agent 或额外 `C_target` artifact。

正确主架构是：

```text
Lift_H(Source) -> (J, C, pi)

TypeCheck(J,C,pi)

Replay = ReplayLLM(
    J,
    pi,
    MaskC(C),
    public frontend context
)

Gate1:
    Observe(Compile_H(Source, C))
    ==
    Observe(Compile_H(Replay, C))

Gate2:
    Source(C)
    Replay(C)
    RefEval(J)       # function interface contains no C

Production:
    Lower_t(J, C)
```

其中：

- Harness 是预先存在的隐藏求值环境，不是 IR，也不是 Lift 的第四输出；
- C 是一个对象，不拆成多个顶层 artifact；
- logical signature 属于 J；任何工程 `InterfaceView` 都只能是 `J` 的确定性投影；
- Replay view、Gate view 和 lowering view 都是 C 的确定性投影；
- `RefEval` 在接口层面禁止访问 C；
- production lowerer 可以消费或忽略 C，但不读取 pi；
- backend cost model 与未来 retune 是后话，不进入当前核心设计或当前 28-kernel 验收。

## 2. 严格三对象因子化

### 2.1 Lift codomain

证明相关的 Lift 输出严格定义为：

```text
Lift_H(S) -> (J, C, pi)
```

不能扩张为：

```text
Lift(S,H) -> (J,C,pi,binding_map,hints,resolver_output,...)
```

这里下标 `H` 表示 Lift 在既有 Harness/调用环境下工作；H 是外部环境，不是 Lift 新生成的对象。

### 2.2 Replay 因子化

对 frontend `f`，Replay 定义为：

```text
R = W_f(J, pi, MaskC(C); U_f)
```

其中 `U_f` 是固定、公共、与当前源码无关的 frontend 文档和工具上下文。

禁止存在额外的当前源码条件化输入：

```text
Z = phi(Source, Harness)
R = W(J,pi,MaskC(C),Z)
```

除非可以证明 `Z` 是 `J/pi/MaskC(C)` 的确定性函数，否则它就是未受约束的第四信息通道。

### 2.3 条件信息隔离

Replay 必须满足：

```text
R independent of
    Source,
    Harness,
    concrete C values,
    source/replay binding
given
    J,
    pi,
    MaskC(C),
    fixed public frontend context U_f.
```

也就是说：

```text
Source -> (J,C,pi) -> Replay
```

是当前研究必须保住的因子化命题。

### 2.4 Logical signature 属于 J

输入、输出、runtime scalar、destination、dtype、rank、shape、alias、inout 与 observable effect
都属于 J 的 logical signature 或 J/B 边界。它们不能形成独立的第四个 Lift artifact。

如果前端适配器或实现代码需要一个接口视图，只允许定义：

```text
InterfaceView(J)
```

`InterfaceView(J)` 是 J 的确定性投影，不具有独立语义，不单独序列化为证明对象，也不作为
Replay 的额外输入。C slots 仍通过 `MaskC(C)` 单独提供；Harness 只保存 source/replay 的机械
调用坐标。

Lift 可以从 H 获得匿名、未分类的 bindable coordinates，以便形成 J signature 和 C；该机械输入
不是 `InterfaceView` artifact，也不能替 Lift 预先完成 J/C 分类。

### 2.5 Proof-irrelevant sidecars

实现可以保存 audit、location、diagnostic、provider receipt、compiler receipt 等 sidecar，但它们
必须是 proof-irrelevant：

- 不进入 Replay；
- 不进入 Gate verdict 的功能输入；
- 不改变 production lowering；
- 不成为从源码到 Replay 的额外通道。

## 3. J、C、pi 与 Harness

| 对象 | 数学身份 | 所有内容 | 禁止内容 |
|---|---|---|---|
| `J=A+B` | 唯一 denotational semantic object | logical signature、value/state relation、domain、predicate、fill、effect、alias、result relation | schedule、frontend API、source topology |
| `C` | denotation-preserving implementation-selection object | anonymous slots、binding kind、value type、concrete valuation，以及已有的最小 source-neutral schedule metadata | J 语义副本、predicate/formula、kernel-specific executable policy |
| `pi` | source realization witness | instance、region、route、slot、order、sharing、placement、representation topology | C value、J 语义副本、production target policy |
| `H` | 隐藏求值环境 | inputs、C values、source/replay invocation coordinates、grid、device、compiler options、receipts | 新 IR、语义分类、Replay construction |

最重要的关系是：

```text
J：算什么；
C：选择哪个保持 J denotation 的实现参数；
pi：原源码怎样 realization J 与 C；
Harness：给匿名坐标赋本次真实值并执行。
```

## 4. C 的数学定义

### 4.1 Denotation-preserving implementation selection

设：

```text
[[J]](x)
```

是 J 对输入 `x` 允许的可观察结果集合。设 target `t` 在 implementation selection `c` 下生成：

```text
I_t(J,c)
```

合法选择集合为：

```text
Adm_t(J,H,device_t,toolchain_t)
```

C 的纯调度性质是：

```text
for every c in Adm_t(J,H,device_t,toolchain_t):
    Obs(Exec(I_t(J,c),x)) belongs to [[J]](x)
```

C 可以改变 loop、tile、vector width、thread/warp mapping、pipeline、buffer、memory placement 与
允许范围内的浮点 association，但不能改变 `[[J]]`。

不同 C 的结果不要求逐位相等，只要求满足同一个 J/B relation。

### 4.2 J 完整，C 可稀疏

必须满足：

```text
J 必须完整；C 的缺失或 target 不理解只能影响性能或 source fidelity，不能补充 J 语义。
```

Gate2 的 J reference 不读取 C，正是为了检测“把语义事实错误塞进 C”这一类所有权错误。

## 5. C 是一个对象及其确定性视图

### 5.1 Canonical C

定义：

```text
C = (Sigma_C, nu_C)
```

其中：

- `Sigma_C`：匿名参数化 slot schema；
- `nu_C`：Harness 持有、解析并提交的 concrete valuation。

示意：

```json
{
  "slots": [
    {
      "id": "c0",
      "value_type": "int",
      "binding_kind": "constexpr"
    },
    {
      "id": "c1",
      "value_type": "int",
      "binding_kind": "compiler_option"
    }
  ],
  "values": {
    "c0": 128,
    "c1": 4
  }
}
```

实现可以保留当前已有、经过审查的最小 `role` 等字段，但它们属于同一个 C，不能再扩张成
portable-hint artifact、schedule ontology 或 LoweringAgent 输入协议。

### 5.2 Replay view

Replay 只接收：

```text
MaskC(C) = (Sigma_C, bottom)
```

示意：

```json
{
  "slots": [
    {
      "id": "c0",
      "value_type": "int",
      "binding_kind": "constexpr"
    },
    {
      "id": "c1",
      "value_type": "int",
      "binding_kind": "compiler_option"
    }
  ]
}
```

Replay 看不到：

- `values`；
- `BLOCK_N`、`num_warps` 等 source spelling；
- source parameter position；
- source/replay invocation binding；
- Harness；
- prior Replay 或 compiler diff。

### 5.3 Gate view

Gate1 的 compile/run harness 使用完整 C：

```text
V_gate(C) = C
```

同一个 `nu_C` 被机械绑定到 Source 和 Replay 的各自调用坐标。

### 5.4 Lowering view

当前 production 接口直接写成：

```text
V_lower(C) = C
K_t = Lower_t(J,C)
```

backend 可以消费、部分消费或忽略 C；当前设计不新增 `C_target`、LoweringAgent 或 cost model。
如果未来 backend 内部 retune，它是 production 实现细节与独立 conformance 问题，不改变核心
artifact `(J,C,pi)`。

### 5.5 不存在额外顶层对象

本设计删除以下额外 Lift 产物：

- 独立 `CBindingMap`；
- `optional portable C hints`；
- `Generic Resolver` 输出；
- `C_source -> C_target` artifact；
- target schedule lineage 作为当前核心证明对象。

若实现必须记录 source coordinate 到 C slot 的关系，它只能是：

1. Harness 已有的机械调用关系；或
2. C 内部的 private provenance，且在 `MaskC(C)` 中被确定性擦除。

它不能成为第四个 Lift artifact，也不能进入 Replay packet。

## 6. Harness 是隐藏求值环境

### 6.1 环境语义

Harness 可以形式化为：

```text
rho_H : {logical input, C slot} -> {concrete tensor/scalar/value}
```

例如：

```text
rho_H(c0) = 128
rho_H(c1) = 4
```

它还持有两个机械调用关系：

```text
rho_H^S(c0) = Source.BLOCK_N := 128
rho_H^R(c0) = Replay.c0       := 128
```

这些关系只负责把同一个 C value 送到两个程序的正确调用位置。

### 6.2 Harness 可以做什么

- 枚举参数、constexpr 与 compiler option 的机械坐标；
- 保存 concrete inputs 与 C values；
- 将相同 C valuation 绑定给 Source 和 Replay；
- 保存 grid、device、toolchain、compiler options 与 receipts；
- 执行 compile/run/measure。

### 6.3 Harness 禁止做什么

- 判断一个事实属于 J 还是 C；
- 创建 C 的语义 role；
- 生成 Replay use-site；
- 修改 Replay AST；
- 增加 loop、mask、address、reduction 或 sharing；
- 向 Replay 暴露 source name、position 或具体 value；
- 生成第四份 source-conditioned artifact。

### 6.4 Harness 不整体进入模型输入

正确关系是：

```text
                    Source
                       |
                       v
                    LiftLLM
                       |
                    (J,C,pi)
                       |
           ReplayView(J,MaskC(C),pi)
                       |
                       v
                  ReplayLLM
                       |
              Parameterized Replay
                       |
       Harness privately binds full C
       to Source and Replay invocations
```

Lift 可以读取 H 提供的极小、匿名化机械坐标；最终 logical signature 必须进入 J。Replay 若需要
接口信息，只能读取 `InterfaceView(J)` 这一确定性投影。H 不进入 Replay，也不能产生额外
source-conditioned object 再送给 Replay。

## 7. 谁决定一个 source coordinate 属于 C

### 7.1 枚举与分类分离

Harness 可以机械枚举：

```text
q0 = 第 0 个参数
q1 = 第 1 个参数
q2 = 第 2 个 constexpr
q3 = 第 0 个 compiler option
```

但不能预先声明：

```text
q2 是 schedule
q1 是 logical domain
```

语义分类由 Lift 完成，并直接体现在 J 和 C 中：

- logical domain、predicate、shape、result consequence 进入 J；
- 纯 tile/warp/pipeline 等 implementation selection 进入 C。

### 7.2 所有权按事实用途，不按源码名字

同一个 source parameter 可能有多种用途。例如 `HEAD_DIM` 同时决定逻辑 extent 与 source
vectorization preference：

- logical extent 必须进入 J；
- 可分离的纯 schedule use 可以在 C 中体现。

这仍然不需要第四个 binding artifact。分类结果已经由 J 和 C 的内容表达。

### 7.3 两种干净实现

方法 A：Harness 在 Lift 前机械匿名化 bindable coordinates，Lift 直接在 C 中采用该 handle。

方法 B：若必须保存 `q2 -> c0`，将其作为 C private provenance，Replay view 确定性擦除。

两种方法都必须保持：

```text
Lift_H(S) = (J,C,pi)
```

## 8. Replay

### 8.1 唯一输入

Replay 的当前 kernel 条件化输入严格是：

```text
J + pi + MaskC(C)
```

另有固定、公共、与当前源码无关的 frontend context `U_f`。

不允许增加：

- binding map；
- source span/location；
- complete C values；
- Harness view；
- selected RuleCard；
- prior candidate；
- source/compiler diff。

### 8.2 参数化生成

Replay 生成：

```python
@triton.jit
def scrc_k0(..., c0: tl.constexpr):
    lane = tl.arange(0, c0)
    ...
```

它知道存在 `c0`，但不知道 source 名、source position 和具体 value。

pi 负责表达 C slot 在 source realization topology 中的 use；Injector 不得在生成后补 use-site。

### 8.3 生成后绑定

Replay source 生成完成后，Harness 执行：

```text
Bind_H(Source,C)
Bind_H(Replay,C)
```

Injector 只允许 argument/option binding，禁止修改 Replay AST。

## 9. Gate1

设 frontend observer 为 `F_f`：

```text
F_source = F_f(Compile_H(Source,C))
F_replay = F_f(Compile_H(Replay,C))
```

Gate1：

```text
G1 = [F_source == F_replay]
```

Source 与 Replay 使用同一个完整 C，因为 constexpr 和 compiler option 会改变 IR shape、address
topology、reduction tree 与浮点 realization。

Gate1 不能从额外 binding artifact、RuleCard 或 source-conditioned sidecar 获得 construction 信息。

## 10. Gate2

### 10.1 J reference 接口中没有 C

若语义参数 `N` 被错误放进 C，而 Source、Replay 和 `Lower(J,C)` 三腿共同读取 `C.N`，三角可能
错误通过，即使 J 没有表达 logical domain。

因此：

```text
RefEval : J x runtime inputs -> observable outputs
```

函数类型中不存在 C。最强隔离不是“把 C 给它后声明 ignored”，而是它没有访问 C 的接口。

### 10.2 正式 Gate2

对测试输入 `x`：

```text
O_S = Exec(Source, rho_H^S(C), x)
O_R = Exec(Replay, rho_H^R(C), x)
O_J = RefEval(J, x)
```

Gate2：

```text
G2 = (O_S equivalent_B O_R)
     and
     (O_S equivalent_B O_J)
```

其中 `equivalent_B` 是预先确定的系统比较协议依据 B 所声明的封闭 comparison class 构造出的
关系；下标 B 不表示候选可以提供任意 comparator。系统根据 dtype、reduction extent、
reassociation 权限等构造 comparator，并检查全部参数，包括 tolerance 上限、NaN/Inf 合同和
atomic/allowed-result 合同；不支持的类别或非法参数直接 fail closed。Lift 不得在 J/B 中写入
任意可执行 comparator，也不得用任意大的 `rtol/atol` 或“所有输出均可接受”来定义自己的及格线。

因此这里的所有权分工是：

```text
B declares a closed comparison class and its required semantic contract
the pre-established system protocol validates all parameters and builds the comparator
```

而不是候选 J 可以自由定义比较程序。

### 10.3 同一次实验，不同权限

三腿可以共享：

- 同一 logical inputs；
- 同一组 J/C/pi 与 Replay；
- 同一 comparator；
- 同一 runtime shape/condition；
- 同一实验记录。

但 `RefEval(J,x)` 不得从 C 恢复 logical domain、predicate、fill 或 result relation。

RefEval 的实验记录可以关联当前 J/C/pi，但其函数接口必须保证：

```text
c_access = forbidden_by_interface
```

该记录只说明接口权限，不向 `RefEval` 提供 C。

## 11. Production lowering

### 11.1 当前唯一接口

当前 production 接口只定义为：

```text
K_t = Lower_t(J,C)
```

不引入：

- LoweringAgent；
- C-generation agent；
- retune agent；
- 新的 LLM schedule phase；
- `C_target` 顶层 artifact；
- 当前任务必须完成的 per-backend cost model。

### 11.2 Lowering 与 RefEval 分离

production lowerer 是被验证对象，不能同时充当 J 的独立 semantic oracle。

若当前阶段验证 backend：

```text
O_target = Exec(Lower_t(J,C), x)
G_backend = O_target equivalent_B RefEval(J,x)
```

该 conformance 与 Gate2 主三角分开记录。

### 11.3 C 在 lower IR 中的生命周期

直接 production 路径可以是：

```text
J + C
  -> Intent MLIR + schedule annotations/sidecar
  -> target lowering
  -> tile/vector/thread/pipeline/memory decisions
  -> lower target IR
  -> erase consumed schedule annotations
  -> executable
```

production lowerer可以消费、部分消费或忽略 C；但不能读取 pi 或 source-specific hidden binding。
若它忽略未知 C，必须仍能从完整 J 走 generic correctness fallback。

### 11.4 未来 retune 与 cost model

未来若实现 target autotuning，可以在 backend 内部选择：

```text
C' in Adm_t(J)
```

并验证：

```text
Exec(Lower_t(J,C')) belongs to [[J]]
```

这属于 future work，不改变当前核心 artifact `(J,C,pi)`，也不产生新的系统角色。

面对“每个 backend 是否都要重建 cost model”的 reviewer 问题，当前回答是：

- correctness 架构不依赖 cost model；
- 当前只要求 backend 能消费或忽略 C 并通过 J/B conformance；
- target-specific cost model 是未来的性能优化层，而不是 Lift/Replay/Gate 正确性的前置条件；
- 本阶段不实现、不声称已经实现，也不把它加入 28-kernel canary 验收。

## 12. C-purity 与多值实验

C 的数学定义要求合法 C 不改变 `[[J]]`，但多值 intervention 不进入每个 artifact 的基础在线
Gate2。

原因：

- source 未必支持重新绑定；
- admissible value 难以统一生成；
- 不同 C 可以合法改变浮点 association；
- 成本随 slot 数量增长；
- 当前 Gate 已认证 observed C 下的 artifact。

它适合作为：

- classifier/schema CI；
- 高风险 slot 抽样；
- 论文消融；
- future backend retune conformance。

正确判据是：

```text
for every verified admissible c in test set T_C:
    Obs(I(J,c),x) belongs to [[J]](x)
```

不是不同 C 的输出逐位相同。

## 13. 反泄漏边界

### 13.1 Replay 可见

- J；
- pi；
- anonymous C slot schema；
- value type 与 binding kind；
- fixed public frontend context。

### 13.2 Replay 不可见

- C concrete valuation；
- source/replay invocation mapping；
- Harness；
- source name/path/span/position；
- audit/location/diagnostic sidecar；
- prior Replay/compiler diff；
- RuleCard/Registry；
- future retune/cost-model state。

### 13.3 Injector 可做与禁止

Injector 可以：

- 将 C values 绑定到已存在的 Replay parameters/options；
- 建立 compile/run invocation；
- 保存 receipt。

Injector 禁止：

- 修改 Replay AST；
- 新增 C use-site；
- 生成 loop、mask、address、reduction；
- 根据 Source 或 diff 修补 Replay。

## 14. 当前实现映射

### 14.1 已经符合的部分

当前 `ScheduleSpec` 已保存：

```text
slots + values
```

当前 `ScheduleSpec.rewriter_view()` 只返回 slots，并明确将 concrete values 视为 secret。Gate1 在
Replay 生成后，由 Harness 使用同一 C 编译 Source 与 Replay。这正是 `C=(Sigma_C,nu_C)` 与
`MaskC(C)` 的基本实现，应保留。

### 14.2 需要修正或审计的部分

1. formal manifest 是否预先替 Lift 完成了 J/C 语义分类；
2. Lift 是否被要求复述 Harness 已分类的 schedule values；
3. 当前 anonymous coordinate 是否足以让 Lift 直接在 C 中采用 handle，而不生成 BindingMap；
4. 任意 binding/private provenance 是否在 Replay view 中被确定性擦除；
5. Replay packet 是否严格只有 `J + pi + MaskC(C) + fixed U_f`；
6. Gate2 RefEval 类型是否机械禁止 C；
7. production lowerer是否仍保持直接 `Lower_t(J,C)`，且不读取 pi。

## 15. 当前阶段明确不做

- 不新增 `CBindingMap` artifact；
- 不新增 portable-hint artifact；
- 不新增 Generic Resolver 作为证明对象；
- 不新增 LoweringAgent、retune agent 或 C-generation agent；
- 不设计每个 backend 的 cost model；
- 不新增 `C_target` 顶层对象；
- 不把 future retune 写进当前 Gate1/Gate2；
- 不把多 C intervention 变成当前在线基础 Gate；
- 不为 target schedule 建立新的 LLM 调用阶段。

当前任务只需闭合：

```text
Lift_H(S) = (J,C,pi)
Replay(J,pi,MaskC(C))
Gate1(Source(C),Replay(C))
Gate2(Source(C),Replay(C),RefEval(J))
```

## 16. 禁止的退化方案

1. Lift 输出 `(J,C,pi)` 之外的 source-conditioned proof object。
2. 独立 interface/BindingMap/portable hints/Resolver output 进入 Replay 或 Gate verdict。
3. Harness 根据 `tl.constexpr` 或参数名做 J/C 语义分类。
4. Harness 整体进入 Replay。
5. Replay 看到 C values 或 source binding。
6. Injector 修改 Replay AST 来新增 C use-site。
7. RefEval 接收 C 后自报 ignored。
8. `Lower_t(J,C)` 同时充当 production implementation 与 J semantic oracle。
9. production lowerer读取 pi、source path、RuleCard 或 compiler diff。
10. 凭空引入 LoweringAgent/Autotuner/C_target 并作为当前系统组成。
11. 将未来 backend cost model 设为当前正确性验证前提。
12. audit/location/diagnostic sidecar 改变 Replay、Gate verdict 或 lowering。

## 17. 建议验证标准

### 17.1 Artifact factorization

- proof-relevant Lift codomain 只有 J、C、pi；
- logical signature 属于 J，`InterfaceView(J)` 只能是确定性投影；
- 额外 sidecars 不进入 Replay/Gate/lowering；
- Replay request schema 不存在第四个 kernel-conditioned payload；
- historical source/binding names在 Replay packet 中不可搜索到。

### 17.2 C 与 Harness

- canonical C 只有一个对象，保存 slots 与 values；
- Replay view 确定性删除 values 和 private provenance；
- Harness 只枚举/绑定坐标，不做 J/C 分类；
- Source/Replay 使用同一个 C 对象和同一份 valuation；
- Injector 只绑定，不修改 AST。

### 17.3 Gate1

- Replay 在 value-free C 下生成；
- compile 前才注入完整 C；
- source/replay observer 在同一 specialization 下比较；
- Gate1 不读取额外 BindingMap、RuleCard 或 source-conditioned sidecar。

### 17.4 Gate2

- Source 与 Replay 使用相同 C；
- RefEval API 没有 C 参数；
- RefEval 实验记录可注明 `c_access=forbidden_by_interface`；
- J/B 只选择系统允许的 comparison class；tolerance 和允许结果参数通过系统上限检查；
- production backend conformance 与主 Gate2 分开。

### 17.5 Production boundary

- 当前接口是直接 `Lower_t(J,C)`；
- lowerer 不读取 pi；
- 未理解的 C 可被忽略而不破坏 generic correctness；
- future retune/cost model 不进入当前 acceptance。

## 18. 后续实现中需要确定的通用接口

1. `Sigma_C` 的最小字段是否保留当前 `role`，还是只保留 handle/type/binding kind？
2. Harness 的匿名 bindable coordinate 如何机械形成，避免预先做 J/C 分类？
3. source coordinate 到 C slot 的关系优先采用同 handle，还是 C private provenance？
4. 一个 source parameter 的多用途如何同时体现在 J 与 C，而不新增第四对象？
5. Lift 最小可见的 H 机械坐标包含哪些结构事实，以及如何确定性形成 `InterfaceView(J)`？
6. 如何从类型/API 层机械证明 RefEval 无法访问 artifact C？
7. 当前 production lowerer对 C 的 consume/ignore receipt 需要做到什么粒度？
8. future backend cost model 在 reviewer response 中如何表述而不形成当前实现承诺？

这些开放项不能通过新增 proof-relevant artifact 或 LLM agent 静默解决。

## 19. 最终规范

```text
Lift_H(S) = (J,C,pi)

TypeCheck(J,C,pi)

Replay = W_f(J,pi,MaskC(C);U_f)

Gate1:
    F_f(Compile_H(S,C)) == F_f(Compile_H(R,C))

Gate2:
    Obs_H(S,C,x) equivalent_B Obs_H(R,C,x)
    and
    Obs_H(S,C,x) equivalent_B RefEval(J,x)

Production:
    K_t = Lower_t(J,C)
```

> **Harness 负责给既有符号赋值，不负责产生新的符号系统；Replay 只看 `J + pi + MaskC(C)`。当前证明相关 artifact 始终只有 `(J,C,pi)`，LoweringAgent、retune agent 与 backend cost model 均属于未来工作。**
