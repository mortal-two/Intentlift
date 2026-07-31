# IntentIR 当前派生规范

> 状态：2026-07-31 Task 20 设计迁移中。本文入口与系统数据流已切换到 J-only Replay / J/C
> Structural Gate1；详细 Replay、Gate1、multi-frontend runtime 条款在 Task 20 实现时同步，当前旧
> runtime 文字不得被解释为新路径已经完成。

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
- [J-only Replay 与 J/C Structural Gate1](../../doc/method/IntentIR_J-only_Replay与JC结构Gate1_2026-07-31.md)
  窄化 Replay/Gate1 接口：Replay 只读 `J+MaskC(C)`；π 只进入 frontend/profile 预定义的窄
  normalizer，Gate1 结论只谈 J/C，不把 π 作为认证目标；
- 本目录只做上述文档的派生投影。

若派生规范与上位设计冲突，应修订派生规范；不得通过增加 interface artifact、hash gate、
epoch、commitment、额外认证 Gate 或兼容旁路来调和冲突。

当前 fresh-run 栈为 `artifact.v12 + pi.normalization-witness.v1 / decode.v18 / typecheck.v31 / candidate-check.v1 / replay-check.v15`。
J/C 继续验证 rank-0 closed signature、Layer-A total typing、显式 broadcast、canonical
reduce/contraction、functional-domain closure、value/dimension、consumer-extension、access-guard、typed
control/inout/output-binding 等全部语义事实。active π 只作者化 closed
program-axis relation；当前 profile 在无 sound anchor matcher 时将它保持 hard。旧
`regions/instances/routes/iterations/carriers/refinements`
仅供历史 artifact 按原版本读取。fixed code只诊断，不翻译别名、补齐 attrs、生成语义
handle/relation、改图或修复 artifact；历史 wire/receipt 不静默升级或重判。当前 Agent execution
receipt 为 `openai.codex-cli.intentir-lift.v39 / intentir.lift-knowledge.v99 /
minisweagent.intentir-replay.v33 / openai.codex-cli.intentir-replay.v62`；Codex runtime v5 在
closed submit 工具接受 exact bytes 后恢复并复核 checkpoint；稳定但被同一 preflight 拒绝的 exact
bytes 也可立即中断当前 turn、保留同一 thread 并用 closed diagnostics 续修，不等待 turn deadline；
Replay 内部 J-only 视图为 `intentir.replay-semantic-program.v3`，另有只并置现有 J/MaskC refs 的
`intentir.replay-compiler-index.v1`；二者都不是 artifact、新语义或 target schedule；
artifact.v11 的 `dimension_relations`只接受identity、常数加/乘、常数ceil-div与round-up的acyclic closed grammar；
relation由Lift从Source作者化，所有J消费者只能验证/求值，不能从Harness valuation、grid、C、π或TTIR生成。
迁移前 artifact.v10 的 `physical_iterations/physical_loop_carriers` 曾表达 structured J loop 外的物理时间；
这些旧 relation 不属于 fresh compact π。当前
feedback frontier/category、candidate-control 与 Replay local-axis 合同只改 agent 消费，不改变接受
集合。持久 Lift owner 固定使用创建时的私有 Source/Harness/schema/knowledge/tool 快照；续轮只刷新
closed feedback，host 仓库并发升级不得改变既有 owner。Harness adapter v6 另把 concrete
shape valuation 与 Harness-only `h*` shape identity 分开，并可显式声明 valuation-matched semantic
constexpr ↔ `h*`；最终 `x*`/equality 仍由 Lift 完整作者化。v6的`input_profile`仅记录Harness
valuation provenance，不进入semantic artifact或Gate。
Replay v28/v24 明确同一 region 中 `physical_loop_carriers` declaration order 是跨
init/state/next/result 共享的 carrier-tuple ordinal；Lift knowledge v66/v62 必须从 Source frontend
loop 的 argument/yield/result tuple 作者化该顺序，Replay 不按 dtype/ref/mechanism 重排。该 tuple slot
的四个 role 必须为 distinct occurrences，普通 route 闭合 `init -> state -> next`；跨迭代和 exit
仍由 carrier relation 自身表示。slot 顺序不是 event execution order，后者仍只由 `pi.order` 拥有。
`replay-check.v14` 仅在 packet route
和 Replay 自身 TTIR 都提供唯一对应时检查 ordered transfer 的词法先后，不从 Source 或 Gate 构造
对应。当前 Replay 不再向首次模型上下文嵌入完整 packet、critical-facts ledger、π 或 compact manifest；
启动消息只给 immutable J/S_J、MaskC schema、frontend knowledge 的文件路径和字节数。Agent 读取
typed-SSA `semantic-program.intent`，但不能读取 `pi.intent` 或查询 π 局部事实。normal 与 MaskJ
各自从本轮可见 J fresh materialize，禁止 full-J
`S_J` 泄漏到 masked episode。Codex Lift v12 / knowledge v70 同样直接嵌入 exact Harness ABI
与 relift frontier，并一次并列展开非递归 direct hard categories；此外以
`intentir.lift-source-ttir.v1`在同一匿名specialization上编译一次Source，TTIR与observer fingerprint只在
source-aware owner workspace中按需查询，不进入Replay或artifact。原有只读工具退为可选局部自读。
Codex Lift v18 还用同一公共 obligation 投影对账候选已写出的 realization 与 Source TTIR；它只拒绝和
诊断，不生成语义、不修改artifact，也不替代 fresh Replay/Gate1。
Codex Lift v19 将 relift 的 bulk feedback 改成 workspace 内按既有 Gate1 label 分区的 exact 文件，
不再重复输入上一 artifact；这只缩小 agent 上下文，不改变任何接受集合。
Codex Lift v20 删除重复的 feedback host tool，特殊回调只保留两个 reject-only checker。
两条路径都不生成 artifact/realization，没有 Gate
或认证权限。

当前 `S_J=M(J)` 按 declaration order 保存每个 J node 的 typed
`id/op/operands/attrs/result`、structured control、boundary 与 output binding，并保留 J handle；它不读取
π，也不生成 target code。当前 compact π 只作者化 closed `program_axes`；
当前 profile 在无唯一 J-domain anchor 时将 axis identity 保持 hard；
mixed-radix decode 在有 sound normalizer 与新 schema 版本前保持 hard。logical block interval/mask
必须显式属于 J，π 不复制 partition formula。

旧 `replay-obligation-audit.v3` 与 π-consuming `replay-check.v14` 只用于解释历史 evidence；Task 20
新 Replay 不查询或消费 π，必要的 realization noise 只由 Gate1 的固定 normalizer处理。历史
`replay-check.v14` 曾要求 synthetic TTIR 至少包含 π 已作者化 `reduction_linearization` 数量的
distinct `tt.reshape -> tt.reduce(axis=0)` dataflow，避免无关 reshape 冒充，但不从 reduction/shape
生成 relation；地址检查只在 J.signature carrier kind 与 π
address routes 能完全决定时，沿作者化 route
传播 pointer reachability，并核对 compiled pointer-address operation occurrence；独立的 integer
base/lane contribution 不被误当 pointer stage，address instance 的 `semantic_ref` spelling 不必等于
transfer ref，但每个 transfer 必须从匹配 boundary tensor 的 pointer/tensor signature carrier 可达，
并按位于 program region、沿作者化 route 到达 `address_base` 的 distinct J domain 数量核对
compiled `tt.get_program_id` occurrence；artifact v12还按π作者化的`program_axes{domain_ref,axis}`核对
精确launch axis，不从声明顺序/grid/C猜测；同一 `p*` 的多个 π instance 始终共享一个既有 ABI carrier，
placement 本身不产生 program coordinate。domain-owned branch 至少要求 compiled structured branch，
`terminal_controls` 至少要求 compiled terminal conditional；若同一 `(domain,region)` 被声明为域外
终止，它只计 terminal、不能再重复计普通 branch。上述检查都只拒绝最低结构，不选择条件、arm 或
目标写法。
且每个 pointer-producing address stage 必须流向至少一个 transfer。无法闭合的 topology 不猜，不读取
Source，也不生成地址公式。当前 `typecheck.v31` 只对 compact π 检查 direct J
program-axis/dimension anchors；迁移前 `typecheck.v30` 新增双臂完整store的J-only conditional post-state
output binding，v29新增program-axis completeness dependent check，v28新增closed
dimension relation check；`typecheck.v27` 只新增 route extent transform 的 closed dependent check；
`typecheck.v26` 只新增 direct `x* == 1`/`1 == x*` 对显式 broadcast
shape alignment 的 singleton dependent check；不做 SMT/transitive inference，也不从 concrete
valuation 猜测。它继承 v25；v25 继承 v24 的 pointer route 与 floating typed constant检查，并把
`reduction_linearization` 限定为一个 `realize` stage 经唯一 slot-0 route 进入一个 J
`reduce` realize stage；fixed code只检查 Lift 已作者化的事实。

当前 Gate1 observer profile 是 `triton-ttir-3.6-gate1-v14`。v14 保留 v13 的 raw
fingerprint，但撤销不 sound 的 program-axis label quotient；`program_axes` 在能唯一
回指 J domain 之前保持 hard。v13 只把 direct typed FMA 与相同
typed operands 的 direct mul+add（或 direct negated addend 对应的 mul-sub）规范为同一 value tree；
不移动 cast、不重关联更大表达式，也不是 production FMA permission。v13 继承 v12；v12 对 compiler 图先作通用 ownership
切片：所有进入返回值、store/update value 的计算，完整 predicate/boolean 树及其直接常量，以及
reduction、control、effect、region/loop/carrier、program-coordinate 与 pointer/transfer anchor 继续作为
hard evidence；仅把不进入这些切片、只拼写 program scheduling、address offset、rank lift 或物理 loop
bound 的 known-pure 子图收缩为带类型和原始 ABI/region carrier 的不透明 representation edge。具体
`div/rem/min/add/mul/splat/broadcast` 树不由 J/C/π 拥有，不能成为 source-blind hidden answer；unknown
op、错误 predicate/fill/reducer/effect/transfer 数量或 carrier provenance 不在该 quotient 内。
v13 保留既有对已建模 op 与 public-entry
reachable、static、structured、acyclic `tt.call` 的 interprocedural comparison rules，并包含 typed
single-block、encodingless `tt.scan`、closed structured `scf.if` 与 exact terminal binary fork observation。
operation/effect classification 是全局 closed profile；unknown op、malformed generic/body line、超出
closed profile 的 multi-block CFG/structured region 与未支持 control/region fail closed。历史
v5–v13 evidence 不由 v14 静默重判；该 profile 只是当前 compiler
contract，不是形式完备性证明。同 case fan-out 协议是
`intentir.scrc.candidate-fanout.v4`：首个被观察到的完整 Gate1 closure 成为 winner，sibling
cooperative stop并保存 receipt；hidden Gate2 尚未生成，因此调度次序不进入认证谓词。
outer batch protocol 是 `intentir.scrc.batch.v3`；其 `observer_unsupported` 只表示 Source observer
能力缺口，不是候选非法或 `NO_CLOSURE`，历史 batch v2 report 不重判。

当前 scalar RefEval 是 `j_scalar_cpu_mlir_v24`，shared functional-boundary receipt 是
`intentir.j-functional-boundary.v22`；structured与relation-indexed J-only执行receipt分别是
`intentir.j-structured-cpu-reference.v10 / intentir.j-contract-indexed-cpu-reference.v2`。主 SCRC runner 选定 Gate2 winner 后，才为当前 invocation 调用
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
