# IntentIR 系统数据流与状态

> 状态：当前派生规范。上位设计为 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、[π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。
> 数学与 claim 边界见 [SCRC 最终数学总述](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。

## 主数据流

```text
Source + Harness coordinates
        |
        v
Lift_H(Source) -> (J,C,pi)
        |
        v
TypeCheck(J,C,pi)
        |
        v
Replay W_f(J,pi,MaskC(C);U_f) -> R
        |
        v
Gate1(Source,R,C)
   | fail                         | pass
   v                              v
structured diff -> relift      Gate2(Source,R,RefEval(J))
   |                              | fail: end experiment
   +-- complete new artifact      | pass
       + new blind Replay         v
                              SCRC_CERTIFIED
```

TypeCheck 是 Replay 前的确定性合法性检查，不是新的证明对象或额外 Gate。只有 TypeCheck 通过并
进入一次全新 Replay 的候选才计为正式实验轮次；最多三轮，之后报告 `NO_CLOSURE`。

## 可见性矩阵

| 组件 | 可以读取 | 不得读取或产生 |
|---|---|---|
| Lift | Source、H 的最小匿名机械坐标、上一轮完整 artifact 与 Gate1 diff | 第四个 proof artifact |
| Replay | J、π、`MaskC(C)`、版本化公共 `U_f` | Source、H、C values、source name/position、BindingMap、interface sidecar、compiler diff/fingerprint、旧 Replay/conversation、per-kernel RuleCard |
| Harness | 调用坐标、inputs、C valuation、device/grid/toolchain | 语义分类、实现构造 |
| Gate1 | Source/Replay 的 compiler observation、同一 c0 与 observer profile | 数值语义裁决、Replay 修补 |
| RefEval | J 与 logical inputs | C、π、Source、Replay |
| Gate2 | 三腿输出、外部 Q 及三条直接 `Close_Q` 比较 | 候选自定义及格线、真正集合值全行为证明 |
| Production lowerer | J、可选 C、target | π、Gate verdict、源码身份 |

## 状态分解

运行记录必须区分：

```text
main_status          = SCRC_CERTIFIED | NO_CLOSURE | GATE2_FAILED | ...
typecheck_status     = PASS | FAIL
j_necessity_status   = PROVED | NOT_PROVED | INCONCLUSIVE
```

`TypeCheck=FAIL` 使当前 artifact 失败。masked-J hard close 且 TypeCheck 通过，只表示
`J_NECESSARY` 未证明，不自动改变 `main_status`。

## 系统改进边界

同一次实验中不修改 TypeCheck、observer profile、比较政策 Q、prompt 或模型配置。Gate2 结果
不回流本次实验。只有实验结束后，才能聚合失败、修改系统并重新开始；新实验不得冒充旧实验的
续轮。

证书与报告必须索引到 frontend、observer profile/version、compiler/dump stage、canonicalization
version、Q、实际 c0 和测试域。搜索预算另行报告首轮 TypeCheck、`closure@1/2/3`、Lift/Replay/
总 LLM 调用、`NO_CLOSURE` 分类及 Gate2 条件通过率；这些记录不是新的 proof artifact。
