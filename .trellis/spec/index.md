# IntentIR 当前派生规范

> 状态：2026-07-24 当前派生规范。

`.trellis/spec` 只把四份 SCRC 设计与第五份 production backend 设计投影为 schema、API、runner、
lowering 和测试要求，不形成另一套设计。上位文档按作用域组合，不构成一条让旧 SCRC 文档覆盖
新 backend 设计的全局线性优先级：

- [C：所有权、绑定、消费与三方验证](../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
  [π：依赖类型实现见证与 Replay 安全](../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与
  [Lift、Replay、Gate1、Gate2 最终执行流程](../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)
  详细定义 SCRC 对象和运行协议；
- [IntentIR / SCRC 最终数学、执行流程与论文表述](../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)
  统一 SCRC 数学与 claim；
- [Contract-Preserving Multi-Lowering 后端最终设计](../../doc/IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)
  在服从前四份对象边界的条件下，权威定义 post-SCRC production backend；
- 本目录只做上述五份文档的派生投影。

若派生规范与上位设计冲突，应修订派生规范；不得通过增加 interface artifact、hash gate、
epoch、commitment、额外认证 Gate 或兼容旁路来调和冲突。

标注为“下一 protocol version”的 CandidateCheck、C `subject_ref/axis_ref` attachment、numeric-cast
dependent typing 与 result-relation compatibility 是规范性目标合同；只有 schema、实现、测试、
receipt version 与 fresh 实验同步升级后才成为活动认证语义，不能用于静默重判历史 artifact。

## 导航

- [项目边界](project/constitution.md)
- [系统数据流与状态](architecture/system.md)
- [J、C、π 与 TypeCheck](ir/j-c-pi.md)
- [J/IntentOp 语义核心](ir/intentop-core.md)
- [Lift 与 Replay](extraction/lift-replay.md)
- [Frontend Harness](integration/frontend-harness.md)
- [Gate1](verification/gate1.md)
- [Gate2](verification/gate2.md)
- [RefEval 与 Contract-Preserving Multi-Lowering](lowering/contract.md)
- [旧实现退役与首步重构](migration/legacy-removal.md)

历史 spec 已迁至 [`.trellis/archive/spec`](../archive/spec/2026-07-21-pre-llm-first/README.md)，
不属于活动规范树。
