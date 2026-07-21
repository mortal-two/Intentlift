# SCRC 规划与历史报告归档

这里保存已经由 v8 实现吸收、已被取代或明确未采用的 SCRC 规划/报告。归档只整理
文档位置，不改变历史内容所记录的实验事实。

当前权威入口：

- `doc/IntentIR_SCRC_方案_重写版.md`：SCRC 总主线；
- `.trellis/spec/verification/scrc-gate1.md`：当前 Gate1 v8 规范；
- `.trellis/spec/verification/scrc-gate2.md`：当前 Gate2 规范；
- `doc/SCRC_PI_优化汇总.md`：π 当前定义与演进依据；
- `doc/SCRC_PI_v8效果专项测试_2026-07-17.md`：最新五核 v8 基线。
- `doc/SCRC_v8后续实施与规模验证流程.md`：未完成项、优化项与后续执行流程。
- `doc/SCRC_Gate1_规则库与数百Kernel扩展计划.md`：规则库与规模扩展路线图。

归档分类：

| 文件 | 状态 | 当前替代 |
|---|---|---|
| `SCRC_J语义完备与确定性PI重构_候选方案.md` | 核心实现已完成；复杂特性真实验证未完成 | Gate1 spec、π 优化汇总 |
| `SCRC_PI_拓扑去泄漏_计划外候选报告.md` | 被确定性 π v8 取代 | Gate1 spec |
| `SCRC_多分面PI_计划外候选报告_2026-07-15.md` | 未采用；strict-v5 口径失效 | 确定性 flat π + RuleRegistry |
| `SCRC_Gate1_MaskedJ_低熵Kernel阻塞报告_2026-07-15.md` | 旧 masked-writer/fuzzy-ratio 口径已废止 | actual-π masked-J hard non-closure |
| `SCRC_Gate1_Gate2_本轮成果与边界.md` | v7 阶段快照 | v8 效果专项测试 |
| `SCRC_首批24个真实Kernel执行报告_2026-07-15.md` | 旧 v7 held-out 快照 | schema-23 held-out batch |
| `SCRC_Gate1_多Specialization门禁_计划外候选报告_2026-07-15.md` | fixed-C 决策已实现，剩余集成已并入统一路线图 | v8 后续实施流程 |
| `SCRC_真实验证架构与IntentIR_ABC衔接计划.md` | 两门架构与 A/B/C 归因原则已吸收 | Gate1/Gate2 spec、规模扩展计划 |
| `TIANCHENIR_CLAUDE_READONLY_AUDIT_2026-07-10.md` | SCRC 重构前审计快照 | 当前代码与 v8 spec |

禁止把归档中的 strict-v5/v6/v7 artifact、cache、masked score 或成功率当作 v8 正式
结果。尚未完成的 branch、复杂 loop carry、buffer lifetime、frontend scope、atomic/
scan/sort 和规模 held-out 验证仍属于当前主线，不因相关候选文档归档而视为完成。
