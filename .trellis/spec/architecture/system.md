# IntentIR 系统数据流与状态

> 状态：当前派生规范。上位设计为 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、[π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。
> 数学与 claim 边界见 [SCRC 最终数学总述](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。
> SCRC 之后的 production 数据流见
> [后端最终设计](../../../doc/IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)。

## 主数据流

```text
Source + Harness coordinates
        |
        v
LLM_H(Source) -> complete candidate bytes
        |  agent 可显式调用 bounded CandidateCheck
        |  submission: exact bytes 若无本 session PASS，则自动 preflight
        |      FAIL -> closed diagnostics -> LLM 重写完整 candidate
        |      PASS -> 仅允许提交，不产生 artifact 或证书
        v
formal Runner wrapper: privacy/provenance; rerun source-free core
        |
        v
strict parser + Decode/Elaborate(response) -> (J,C,pi)
        |
        v
anonymous-view consistency + TypeCheck(J,C,pi)
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
                                  |
                                  | orchestration 只交付 J+C
                                  v
                           M0 = Emit_v(J)
                                  |
                                  v
                         S_C = Attach_C(M0,C,t)
                                  |
                                  v
                       contract-preserving Intent MLIR search
                                  |
                                  v
                         performance-only target selection
```

Decode/Elaborate 与 TypeCheck 是 Replay 前的确定性 closed-IR 前端，不是新的证明对象或额外 Gate。
接受后的 typed/canonical archive 就是唯一 J；响应 JSON 不叫另一份 J。只有二者通过并
进入一次全新 Replay 的候选才计为正式实验轮次；最多三轮，之后报告 `NO_CLOSURE`。
`CandidateCheck` 共享正式 Runner 的同一 source-free
parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行 source-identifier privacy、
receipt/provenance，并在提交后重新运行该 core。CandidateCheck 只返回 diagnostics，不修改候选、
不签发证书，也不进入 Replay；禁止维护第二套近似 checker。激活的 vNext SCRC adapter 必须安装
该 checker；缺失或不可用时属于提交前 infrastructure failure。generic environment 可以不安装 checker
用于非认证测试，但这种执行不属于 vNext SCRC 认证路径。
production search 在 `SCRC_CERTIFIED` 之后独立发生，不改变该状态。`Emit_v` 严格 J-only；只有
后置 `Attach_C` 可以产生可擦除 C-derived search state，且 `BaseIR(S_C)=M0`。

## 可见性矩阵

| 组件 | 可以读取 | 不得读取或产生 |
|---|---|---|
| Lift | Source、H 的最小匿名机械坐标、上一轮完整 artifact 与 Gate1 diff；对当前完整 candidate 的只读 CandidateCheck | 第四个 proof artifact；由工具补写语义或 patch |
| Decode/Elaborate | Lift response 的 closed J/C/π grammar | H/C values、Gate output、语义猜测或修复 |
| Replay | J、π、`MaskC(C)`、版本化公共 `U_f` | Source、H、C values、source name/position、BindingMap、interface sidecar、compiler diff/fingerprint、旧 Replay/conversation、per-kernel RuleCard |
| Harness | 调用坐标、inputs、C valuation、device/grid/toolchain | 语义分类、实现构造 |
| Gate1 | Source/Replay 的 compiler observation、同一 c0 与 observer profile | 数值语义裁决、Replay 修补 |
| RefEval | J 与 logical inputs | 未接受的响应文本、C、π、Source、Replay |
| Gate2 | 三腿输出、外部 Q 及三条直接 `Close_Q` 比较 | 候选自定义及格线、真正集合值全行为证明 |
| Production lowerer | J（由 orchestration 仅在 SCRC acceptance 后交付）、可选 C、target、generic pass/cost library | 未接受的响应文本、π、Gate verdict、源码身份、per-kernel RuleCard |

## 状态分解

运行记录必须区分：

```text
main_status          = SCRC_CERTIFIED | NO_CLOSURE | GATE2_FAILED
typecheck_status     = NOT_RUN | PASS | FAIL
j_necessity_status   = PROVED | NOT_PROVED | INCONCLUSIVE
```

`TypeCheck=FAIL` 使当前 artifact 失败。masked-J hard close 且 TypeCheck 通过，只表示
`J_NECESSARY` 未证明，不自动改变 `main_status`。provider、compiler、device 等基础设施异常单独
抛出并保存 partial evidence，不能伪装成上述任一主认证状态。

## 系统改进边界

同一次实验中不修改 TypeCheck、observer profile、比较政策 Q、prompt 或模型配置。Gate2 结果
不回流本次实验。只有实验结束后，才能聚合失败、修改系统并重新开始；新实验不得冒充旧实验的
续轮。

证书与报告必须索引到 frontend、observer profile/version、compiler/dump stage、schema/decode/
typecheck/canonicalization version、Q、实际 c0 和测试域。搜索预算另行报告首轮 TypeCheck、
`closure@1/2/3`、Lift/Replay/
总 LLM 调用、CandidateCheck 次数、`NO_CLOSURE` 分类及 Gate2 条件通过率；这些记录不是新的 proof
artifact。backend 另行报告 implementation family、pass trace、candidate/compile/measure 数、
time-to-best 与 performance，不把这些字段写回 SCRC 证书。
