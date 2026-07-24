# 设计来源与审计记录

本目录保存设计推理、外部回答、纠错记录和实现审计。材料按形成时间保留原始判断，其中可能
包含后来被否定的 hash、epoch、commitment、单次 Replay、旧 masked-J 状态机或其他历史概念。
这些内容不是当前规范，不能直接约束实现。

当前规范性结论只见五份权威设计：

1. [C 设计](../IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
2. [π 设计](../IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)
3. [最终执行流程](../IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)
4. [IntentIR / SCRC 最终数学、执行流程与论文表述](../IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)
5. [Contract-Preserving Multi-Lowering 后端最终设计](../IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)

前四份定义 SCRC，第五份只定义 SCRC 之后的 production backend。

## 2026-07-24：后端设计来源

| 材料 | 性质 | 主要吸收位置 |
|---|---|---|
| [Intent 搜索建议](高级ai建议intent搜索.md) | 早期搜索方向、曾包含在 J/MLIR 搜索位置上的反复判断 | 第五份后端设计 |
| [SCRC-Certified Multi-Lowering 设计草案](SCRC-Certified_Intent-Guided_Multi-Lowering_完整设计与逻辑审查.md) | A/B/C、`Impl(J)`、multi-lowering 与论文第四章建议；部分 C/authorization 表述未采纳 | 第五份后端设计与派生 spec |
| [高级 AI 对论文思考](高级ai对论文思考.md) | N+M 论文主线、后端章节与实验消融设想 | 第五份后端设计的动机与实验结构 |

## 2026-07-22：三份核心设计的形成与复核

| 材料 | 性质 | 主要吸收位置 |
|---|---|---|
| [高级 AI 对 C 设计的自省](高级ai对c的自省.md) | 三对象因子化、Harness 与 C 边界分析 | C 设计 |
| [高级 AI 对 C 设计的纠正](高级ai对c的纠正.md) | RefEval、Gate2 与 C 权限纠错 | C 设计 |
| [高级 AI 对 π 的思想](高级ai对于π的思想.md) | realization witness、泄漏分类与 masked-J 分离 | π 设计 |
| [高级 AI 对旧执行流程的纠错](高级ai对于之前流程的纠错和判断.md) | Gate1 diff、relift 与实验/系统改进边界 | 最终执行流程 |
| [三份核心设计复审与四项修正](高级ai对三份核心设计的复审与四项修正_2026-07-22.md) | interface、TypeCheck、comparator 与随机 Gate2 前提 | 三份核心设计 |
| [最终数学与三份设计的局部收紧建议](高级ai对最终数学与三份设计的局部收紧建议_2026-07-22.md) | TypeCheck claim、Q、Gate2 三边、c0、observer 与搜索预算 | 三份核心设计、数学总述与活动 spec |
| [IntentOp 健壮性审计](IntentOp_健壮性审计_2026-07-21.md) | 旧实现能力与语义旁路审计 | 后续实现重构依据 |

## 2026-07-21：LLM-first 收敛过程

| 材料 | 历史作用 |
|---|---|
| [高级 AI 哲学](高级ai哲学.md) | LLM-first 与 verifier-closed 的总体方向 |
| [高级 AI 思考](高级ai思考.md) | 当时的 Gate1/Gate2 与概率声度分析 |
| [回答 3](721回答3.md) | 无 composer 的直接 π、在线成本与最小 J |
| [回答 2](721回答2.md) | 严格 Gate1、π 在 Intent 外、匿名参数与轻 Harness |
| [回答 1](721回答1.md) | 对旧系统膨胀和历史漂移的审计 |

这些来源材料解释了设计演进，但冲突时不再按材料之间的“优先级”裁决，一律以当前五份权威
设计及与之对齐的活动 spec 为准。
