# Lift 与 Replay 合同

> 状态：当前派生规范。上位设计见 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。

## Lift

```text
Lift_H(Source) -> (J,C,pi)
```

Lift 直接完成 whole-tensor semantic lifting、C 分离与 source realization π 的提出。proof-relevant
输出只有 J、C、π。diagnostic、location、provider receipt 等可以作为 proof-irrelevant sidecar
保存，但不得进入 Replay、Gate verdict 或 production lowering。

J 必须包含 logical signature；不允许 root sibling `interface` 成为独立对象。实现若需要接口，
只能按需计算 `InterfaceView(J)`。H 可以向 Lift 提供匿名、未分类的 bindable coordinates，但
不得预先替 Lift 做 J/C/π 语义分类。

匿名 handle 是机械 wire contract：J tensor 必须按 `intent.inputs` 后接 `intent.outputs` 独立、
无间隙地编号为 `t0,t1,...`，不得把 `p*` 参数位置直接复制为 `t*` 后缀。例如 input pointer `p0`、
output pointer `p1`、runtime extent scalar `p2` 对应 input `t0`、output `t1` 与 dimension `x0`。
`harness_compiler_view` 中 runtime scalar 的当前 value 只是本次 Harness binding；若 Source 用它控制
logical extent、index、predicate 或 result，J 必须保留 symbolic `x*` 并在相关 tensor shape 中引用，
不得把当前 value 特化成 J 常量。`x*` 是 signature/type handle，不是 SSA value，不能进入 portable
scope definitions/captures 或 storage。

provider 已返回、但因 schema/匿名合同不合法而被拒绝的响应不进入 TypeCheck，也不计正式 Gate1
round；Runner 对这类 pre-Replay rejection 使用有限调用上限并如实报告。拒绝响应、receipt、耗时
和 token 作为 proof-irrelevant sidecar 保存，不能只留下一个错误字符串，也不得进入 Replay 或
认证谓词。上限耗尽可以结束为 `NO_CLOSURE`，但报告必须显示正式 Gate1 round 为 0，而不是伪造
一次 Gate1 失败。

## Replay 前检查

```text
TypeCheck(J,C,pi)
```

最小 TypeCheck 必须机械检查：

1. 所有 J/C 引用存在；
2. route operand slot 与类型兼容；
3. observable instance 锚定 J-owned node/access/effect；
4. π 不含语义原子、Source/API、C value 或可执行自由文本；
5. sensitive refinement 引用 J/B authorization，或属于系统级安全白名单。
6. 未知字段、未知 relation 与自由文本扩展 fail closed。

TypeCheck 不做 SMT、range proof 或一般语义等价证明。检查失败时当前 artifact fail closed，
不进入 Replay。它只支撑 `FreshLogicalSemDecl(C,pi | J)=empty` 的显式所有权结论，不证明 π
没有统计语义线索，也不证明 realization 行为正确。

## Replay

```text
ReplayLLM(J,pi,MaskC(C);U_f) -> complete source R_f
```

`U_f` 是版本化、source-neutral 的公共前端文档与工具上下文，不按当前 Kernel/op/pattern/失败
样本选择。Replay 看不到 Source、Harness、C values、source parameter name/position、真实
BindingMap、独立 interface sidecar、source/compiler fingerprint、旧候选、旧 Replay、Gate1 diff、
旧模型 conversation context 或 per-kernel RuleCard。`MaskC(C)` 只暴露 C 的匿名 schema/slot
视图，不暴露 valuation 与私有 provenance。

Replay 必须生成完整、可独立构建的当前前端源码；固定代码只做绑定和检查，不根据 π 或 compiler
diff 施工。

π region placement 不是隐式代码生成指令：`placement=program` 不等于存在 `program_id`。只有 π
中另有一个表示 compiler-visible program/grid coordinate 的独立 `realize` instance 与 routes 时，
Replay 才能生成对应 stage；不得按常见 kernel 模板从 placement、grid 惯例或 C 猜出额外 grid
offset、整数乘加或 affine address。

现有关系的机械解释是：锚定 J domain、位于 program region 且 scalar dependency path 进入
`address_base` 的 `realize` stage 是 program coordinate；进入 `address_lane` 的 domain/C path 是
lane coordinate。Triton `U_f` 按声明顺序把前者映射为 `tl.program_id(axis)`，后者才使用
`tl.arange` 等 lane realization；placement 单独出现时两者都不能推断。Triton 的 `tl.arange` 等
compile-time-only API 必须直接读取仍带 `tl.constexpr` 注解的匿名 `c*` 参数，不得先复制到普通
Python local 再传入。单个锚定 `c*` 的 `realize` instance 由参数 binding 和直接 routed use 即可
满足，不要求恒等 alias。`IntentFunction.reduce.axes` 是 logical axes；匹配的
`J.boundary.reductions.tile_axis` 才是 Triton reduction call 使用的物理 tile axis。这里都是固定、
source-neutral 的前端合同，不是 per-kernel hint。

## Gate1 失败与 relift

Gate1 的结构化 diff 只返回 source-aware Lift。下一轮 Lift 可以读取 Source、上一轮完整
`(J,C,π)` 与累计 Gate1 diff，但必须重新输出完整新三元组。随后使用新三元组发起全新的盲
Replay。

禁止把 diff 交给 Replay、patch 旧 Replay、只输出 artifact 增量，或跨候选复用 Replay。
最多三轮正式候选；仍不闭合则 `NO_CLOSURE`。
