# IntentIR 当前派生规范

> 状态：2026-07-29 当前派生规范。

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

当前 fresh-run 栈为 `artifact.v10 / decode.v15 / typecheck.v27 / candidate-check.v1 / replay-check.v11`。它继承 rank-0
closed signature、Layer-A total typing、显式 broadcast、canonical reduce/contraction、absent carrier
拒绝、functional-domain closure、rank-0 liveness、padded-mean necessary condition 与 closed
`pi.iterations`，并验证 Lift 已作者化的 value/dimension、consumer-extension、access-guard、typed
control/inout/output-binding/terminal-control 关系，并检查 v10 authored route-local extent transform 的
existing-route、C/J attachment 与 closed divisor/rounding。fixed code只诊断，不翻译别名、补齐 attrs、生成 handle/relation、
改图或修复 artifact；历史 wire/receipt 继续按自身版本读取，不静默升级或重判。当前 Agent execution
receipt 为 `intentir.lift-knowledge.v69 / intentir.lift-knowledge.v65 / minisweagent.intentir-replay.v30 / openai.codex-cli.intentir-replay.v26`；
artifact.v10 的 `physical_iterations/physical_loop_carriers` 表达 structured J loop 外、由已作者化
reduce/contraction 或 boundary access/domain 锚定的物理时间，
feedback frontier/category、candidate-control 与 Replay local-axis 合同只改 agent 消费，不改变接受
集合。持久 Lift owner 固定使用创建时的私有 Source/Harness/schema/knowledge/tool 快照；续轮只刷新
closed feedback，host 仓库并发升级不得改变既有 owner。Harness adapter v5 另把 concrete
shape valuation 与 Harness-only `h*` shape identity 分开，并可显式声明 valuation-matched semantic
constexpr ↔ `h*`；最终 `x*`/equality 仍由 Lift 完整作者化。
Replay v28/v24 明确同一 region 中 `physical_loop_carriers` declaration order 是跨
init/state/next/result 共享的 carrier-tuple ordinal；Lift knowledge v66/v62 必须从 Source frontend
loop 的 argument/yield/result tuple 作者化该顺序，Replay 不按 dtype/ref/mechanism 重排。该 tuple slot
顺序不是 event execution order，后者仍只由 `pi.order` 拥有。v28/v24 继承 v27/v23；v27/v23 对不超过
64 KiB 的完整匿名 J/C/π/U_f 只在首次上下文 exact embedding 一次，并额外
并置一个只重索引 semantic dimension carrier、IEEE special typed constant、physical iteration、
`reduction_linearization + consumer route`、loop/branch lexical region membership、cross-region
order 与 route-local extent transform，以及 authored broadcast 的 operand type/access/named consumer
的 exact `critical_facts.v4`；超限
packet 才接收一个自带 authored op/boundary、π instance mechanism 与 time/control/terminal 清单的
compact manifest，并只查询明确遗漏页。Codex Lift v11 / knowledge v69 同样直接嵌入 exact Harness ABI
与 relift frontier，并一次并列展开非递归 direct hard categories；原有只读工具退为可选局部自读。
两条路径都不生成 artifact/realization，没有 Gate
或认证权限。

含 broadcast/lexical/cross-region/route-extent critical facts 的 Replay 在 construction compile 前还必须对当前 exact bytes
完成 `replay-obligation-audit.v3`；该工具只并置 J/π ledger 与 Replay 自己的编号源码，不映射 relation/instance、
不诊断、不修复，也不参与 Gate 接受。
`replay-check.v11` 还要求 synthetic TTIR 至少包含 π 已作者化 `reduction_linearization` 数量的
distinct `tt.reshape -> tt.reduce(axis=0)` dataflow，避免无关 reshape 冒充，但不从 reduction/shape
生成 relation；地址检查只在 J.signature carrier kind 与 π
address routes 能完全决定时，沿作者化 route
传播 pointer reachability，并核对 compiled pointer-address operation occurrence；独立的 integer
base/lane contribution 不被误当 pointer stage，address instance 的 `semantic_ref` spelling 不必等于
transfer ref，但每个 transfer 必须从匹配 boundary tensor 的 pointer/tensor signature carrier 可达，
并按位于 program region、沿作者化 route 到达 `address_base` 的 distinct J domain 数量核对
compiled `tt.get_program_id` occurrence；同一 `p*` 的多个 π instance 始终共享一个既有 ABI carrier，
placement 本身不产生 program coordinate。
且每个 pointer-producing address stage 必须流向至少一个 transfer。无法闭合的 topology 不猜，不读取
Source，也不生成地址公式。`typecheck.v27` 只新增 route extent transform 的 closed dependent check；
`typecheck.v26` 只新增 direct `x* == 1`/`1 == x*` 对显式 broadcast
shape alignment 的 singleton dependent check；不做 SMT/transitive inference，也不从 concrete
valuation 猜测。它继承 v25；v25 继承 v24 的 pointer route 与 floating typed constant检查，并把
`reduction_linearization` 限定为一个 `realize` stage 经唯一 slot-0 route 进入一个 J
`reduce` realize stage；fixed code只检查 Lift 已作者化的事实。

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

当前 scalar RefEval 是 `j_scalar_cpu_mlir_v22`，shared functional-boundary receipt 是
`intentir.j-functional-boundary.v20`；relation-indexed pointwise 的 J-only 执行 receipt 是
`intentir.j-contract-indexed-cpu-reference.v1`。主 SCRC runner 选定 Gate2 winner 后，才为当前 invocation 调用
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
context；它不参与 Layer-A 数值求值，也不生成 access/program 公式。TypeCheck v24 继承 v23，要求每个
canonical output binding 的完整 dtype/shape 与声明输出精确相等。Gate2 Triton 两条执行 leg 共享
`triton_static_dot_transpose_v1`，只把已移除的静态 `trans_a/trans_b` 等价写法归一成 `tl.trans`。
v21/v20 还只消费 J direct singleton constraint 与同一 `effect_if` 两个 sibling arm 的完整 store；
v22 的 ContractIR reference 则逐节点执行 J 已作者化的 typed index/predicate/fill，并只接受穷尽、
无重叠的 output stores。它不把一个 node 的 access 传播到 sibling，也不读取 Source/Gate 来补 access；
因此关系缺失会形成真实 Source--J 数值分歧而不是被 consumer 猜成正确语义。历史 v20/v19/v8
另把 predicated-load padding provenance 限定在 J 已绑定的 operand edge；同 tensor 的
显式 bounded gather 不再被错误并入 padded stream。J 已作者化的 `effect_if` branch-local full
store 可由其 access/effect destination 投影并按实际 arm 执行；fixed code不补 branch value、select
或 output pre-state，路径上没有已定义 output 时仍拒绝。
动态 gather 的 index tensor shape 只在每一维都能唯一对应 closed logical domain axis时形成
`index_bounded_gather` receipt；integer range 对 dynamic iota 和非零 signed div 做保守、overflow-closed
传播。node-less padded load 与显式 reload 共存还要求已有 consumer extension，且 provenance 按
operand edge 隔离。

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
