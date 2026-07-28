# IntentIR 当前派生规范

> 状态：2026-07-27 当前派生规范。

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

当前 fresh-run 栈为 `artifact.v9 / decode.v13 / typecheck.v23 / candidate-check.v1 / replay-check.v9`。它继承 rank-0
closed signature、Layer-A total typing、显式 broadcast、canonical reduce/contraction、absent carrier
拒绝、functional-domain closure、rank-0 liveness、padded-mean necessary condition 与 closed
`pi.iterations`，并验证 Lift 已作者化的 value/dimension、consumer-extension、access-guard、typed
control/inout/output-binding/terminal-control 关系。fixed code只诊断，不翻译别名、补齐 attrs、生成 handle/relation、
改图或修复 artifact；历史 wire/receipt 继续按自身版本读取，不静默升级或重判。当前 Agent execution
receipt 为 `intentir.lift-knowledge.v59 / minisweagent.intentir-replay.v18 / openai.codex-cli.intentir-replay.v14`；
artifact.v9 的 `physical_iterations/physical_loop_carriers` 只表达 J reduce/contraction 外的物理时间，
feedback frontier 与 Replay local-axis 合同只改 agent 消费，不改变接受集合。Harness adapter v5 另把 concrete
shape valuation 与 Harness-only `h*` shape identity 分开，并可显式声明 valuation-matched semantic
constexpr ↔ `h*`；最终 `x*`/equality 仍由 Lift 完整作者化。
Codex Replay v14 通过隔离 file mailbox提供匿名 C、J boundary与完整 π 的 lossless workspace view；该 view
不生成 realization，不遗漏 physical-time/terminal/address relation，也没有Gate或认证权限。
`replay-check.v9` 只在 J.signature carrier kind 与 π address routes 能完全决定时，核对 compiled
pointer-address operation occurrence；独立的 integer base/lane contribution 不被误当 pointer stage，
但每个 transfer 必须有同 ref 的 pointer-producing stage，且不能存在未绑定 transfer 的额外 pointer
stage。无法闭合的 topology 不猜，不读取 Source，也不生成地址公式。

当前 Gate1 observer profile 是 `triton-ttir-3.6-gate1-v11`；它保留既有对已建模 op 与 public-entry
reachable、static、structured、acyclic `tt.call` 的 interprocedural comparison rules，并包含 typed
single-block、encodingless `tt.scan`、closed structured `scf.if` 与 exact terminal binary fork observation。
operation/effect classification 是全局 closed profile；unknown op、malformed generic/body line、超出
closed profile 的 multi-block CFG/structured region 与未支持 control/region fail closed。历史
v5–v10 evidence 不由 v11 重判；该 profile 只是当前 compiler
contract，不是形式完备性证明。同 case fan-out 协议是
`intentir.scrc.candidate-fanout.v4`：首个被观察到的完整 Gate1 closure 成为 winner，sibling
cooperative stop并保存 receipt；hidden Gate2 尚未生成，因此调度次序不进入认证谓词。
outer batch protocol 是 `intentir.scrc.batch.v3`；其 `observer_unsupported` 只表示 Source observer
能力缺口，不是候选非法或 `NO_CLOSURE`，历史 batch v2 report 不重判。

当前 scalar RefEval 是 `j_scalar_cpu_mlir_v19`，shared functional-boundary receipt 是
`intentir.j-functional-boundary.v18`。主 SCRC runner 选定 Gate2 winner 后，才为当前 invocation 调用
RefEval-only 入口；projection library 不选择 winner。该入口用 J 自身
rank-0 logical `i1`/integer values 和完整 closed dimensions 逐 access 证明 predicate 对完整 logical
domain 恒真；不改 J、不读 Source/C/π、不进入 Lift feedback。invocation-local
proof subreceipt 不保存 raw value/value digest；这是 audit-data minimization，不是
confidentiality/privacy proof。
production projection 仍是 value-free `ProjectFunctionalBoundary(J)`。CPU-only
`expand-rsqrt-for-libm` 位于共享 Intent-to-Linalg 之后，不改变共享/GPU lowering。
v18/v19 还完整消费同一 inout handle 上多个 closed store→reload stage：每一对必须由唯一 producer
SSA、相同坐标/predicate、同 scope 的 store-before-load effect order 且无中间覆盖共同决定；只有最终
canonical output binding 对应的 store 是 observable output。J 中未被 logical access/predicate 使用的
矩形 domain 作为 `context_domain` 保留在 projection receipt 中，供 π 锚定 compiler-visible program
context；它不参与 Layer-A 数值求值，也不生成 access/program 公式。TypeCheck v23 另要求每个
canonical output binding 的完整 dtype/shape 与声明输出精确相等。Gate2 Triton 两条执行 leg 共享
`triton_static_dot_transpose_v1`，只把已移除的静态 `trans_a/trans_b` 等价写法归一成 `tl.trans`。

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
