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
branch-local persistent LLM_H(Source) + isolated workspace
        |  first packet may carry <=64 KiB exact allowlisted input bytes
        |  generic exact-text patch only applies LLM-authored old/new bytes
        |  direct-submit may atomically write LLM-authored complete bytes + preflight
        |  old tool payloads may become deterministic transport receipts
        v
complete candidate bytes
        |  agent 可在 active wall time 内调用 CandidateCheck
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
deterministic J-only M(J) -> read-only parameterized S_J
        |
        v
fresh Replay Compiler Agent reads S_J+MaskC(C)+U_f -> complete R
        |
        v
Gate1_pi(Source,R,J,C,pi)
   | fail                         | pass
   v                              v
bounded feedback frontier + exact diff -> same owner  Gate2(Source,R,RefEval(J))
   |                              | fail: end experiment
   +-- complete new artifact      | pass
       + fresh blind Replay       v
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

Decode/Elaborate、TypeCheck 与 `S_J=M(J)` 是 Replay 前的确定性 closed-IR 前端，不是新的证明对象或
额外 Gate。`S_J` 只在当前 Replay episode 内作为 J 的 proof-irrelevant 编译视图；外部 Replay 接口为
`W_f(J,MaskC(C);U_f)`。π 不进入 Replay，而由 Gate1 的 frontend/profile-specific typed
canonicalizer 对 Source/Replay observations 作对称规范化；证书和 production backend 都不新增
`S_J` 输入。
接受后的 typed/canonical archive 就是唯一 J；响应 JSON 不叫另一份 J。只有二者通过并
进入一次全新 Replay 的候选才计为正式实验轮次；每次运行显式声明正有限的 `K`，每 branch 最多
执行 `K` 轮，之后才能报告 branch-local `NO_CLOSURE`。当前 CLI限制 `1<=K<=12`，Task 14默认
上限为5；历史 `K=3` evidence仍按原预算解释。
`CandidateCheck` 共享正式 Runner 的同一 source-free
parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行 source-identifier privacy、
receipt/provenance，并在提交后重新运行该 core。CandidateCheck 只返回 diagnostics，不修改候选、
不签发证书，也不进入 Replay；禁止维护第二套近似 checker。激活的 vNext SCRC adapter 必须安装
该 checker；缺失或不可用时属于提交前 infrastructure failure。Gate1 完整 diff 始终 exact 落盘；
runtime 只可按版本化 closed category priority 投影有限原始 witness 形成 feedback frontier，便于 owner
先处理 interface/reduction/effect 再按需读取完整 dataflow。frontier 不解释 mismatch、不产生 patch、
不改变 Gate1 verdict。generic environment 可以不安装 checker
用于非认证测试，但这种执行不属于 vNext SCRC 认证路径。
production search 在 `SCRC_CERTIFIED` 之后独立发生，不改变该状态。`Emit_v` 严格 J-only；只有
后置 `Attach_C` 可以产生可擦除 C-derived search state，且 `BaseIR(S_C)=M0`。

## 可见性矩阵

| 组件 | 可以读取 | 不得读取或产生 |
|---|---|---|
| Lift | 同一 branch 持久 conversation/workspace 中的 Source、H 的最小匿名机械坐标、当前完整 artifact、bounded feedback frontier 与可按需读取的 exact Gate1 diff；对当前 candidate 的只读 CandidateCheck；LLM 明示 old/new 的 exact-text patch | sibling/Replay/Gate2 状态；第四个 proof artifact；由 checker/runtime 推断、推荐或补写语义/corrected patch |
| Decode/Elaborate | Lift response 的 closed J/C/π grammar | H/C values、Gate output、语义猜测或修复 |
| Replay | exact J、J-only 只读 `S_J=M(J)`、`MaskC(C)`、版本化公共 `U_f`；Agent 全局编译完整 `S_J` | π、Source、H、C values、source name/position、BindingMap、target skeleton/hole、interface sidecar、compiler diff/fingerprint、旧 Replay/conversation、per-kernel RuleCard |
| Harness | 调用坐标、inputs、C valuation、device/grid/toolchain | 语义分类、实现构造 |
| Gate1 | Source/Replay 的 compiler observation、同一 c0、J/C/π、冻结的 observer/π-normalizer profile | 数值语义裁决、Replay 修补、动态扩张 π relation algebra |
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
`closure@1/2/3`、Lift owner session/active wall time、Lift/Replay 总 LLM 调用、CandidateCheck
次数、`NO_CLOSURE` 分类及 Gate2 条件通过率；调用次数是 telemetry 而非 capability cutoff。这些记录不是新的 proof
artifact。backend 另行报告 implementation family、pass trace、candidate/compile/measure 数、
time-to-best 与 performance，不把这些字段写回 SCRC 证书。
