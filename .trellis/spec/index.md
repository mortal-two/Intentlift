# IntentIR 当前派生规范

> 状态：2026-07-25 当前派生规范。

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

自 2026-07-24 的 fresh run 起，CandidateCheck、C `subject_ref/axis_ref` attachment、numeric-cast
dependent typing 与 result-relation compatibility 已由 `artifact.v2 / decode.v2 / typecheck.v3 /
candidate-check.v1` 同步激活；随后 total graph typing 与显式 non-trailing broadcast 合同将
TypeCheck 升为 `typecheck.v4`。v5 继承 v4 的 rank-0 closed signature、total typing 与显式
broadcast 合同，并新增已知 `reduce`/binary `contraction` 的 canonical fail-early 以及 absent
`opaque(mode=value,dtype=none)` 的语义引用拒绝。v6 再加入 J-only
functional logical-domain closure：只对可识别的矩形 direct access/padding profile 检查
index/predicate/domain/typed extent 一致性，不从 Source/C/π/hidden input 补造，也不把 general logical
predicate 因 consumer 未实现而判为非法。当前 fresh-run 的 `typecheck.v7` 又对 rank-0 logical input
执行 output/boundary/shape liveness，并允许显式 `logical_ref=null` 的 rank-0 representation scalar
只作为 π carrier；该 null form 仅限 runtime scalar，bound constexpr 必须属于 J logical semantics，
纯 schedule constexpr 归 C，否则 Harness invalid。v7 还只在无 guard、完整零起点矩形 direct
access/domain、bound 与 typed extent 同轴匹配，且 masked axis 可证明被 `reduce(mean)` 消去而 J 无
valid-count contract 时返回 `padded_mean_count_unproven`；guard/cross-axis/unknown-op 是 consumer
unsupported，不是 J illegal。上述规则都不按 Source 名称/位置判断、不自动解绑/改图，也不推广成
一般 tensor liveness 或 mean blanket ban。fixed code 不翻译别名、补齐 attrs、生成 dummy carrier
或修复 artifact。当前 wire/decode 已因 closed `pi.iterations` 物理 induction-basis relation 升为
`artifact.v3 / decode.v3`；`tile_extent/reduction_tile_extent` 保留 role 只触发 relation 缺失的机械
totality，显式 relation 的合法性不依赖 role，loop axis 可写 `0` 或对应 `x*`。Source 中被 Lift 整体
遗漏的 physical region，以及 custom/misclassified role 下完全遗漏的 relation，仍由 Gate1 检出。
`artifact.v1/v2` 只按
原 wire 保留历史读取能力并禁止该新字段；新 Lift response 必须使用 v3，新规则不得用于静默重判
既有 receipt 或历史证书，DYT/SwiGLU 也不因协议升级被重跑或冒充 fresh v3 evidence。

当前 Gate1 observer profile 是 `triton-ttir-3.6-gate1-v5`；它继承 v4 只 NormalizeAway closed
load/store cache-policy hint，并展开 public entry 可达、静态、structured、acyclic 的 `tt.call`
graph。未知/递归 call、reachable arbitrary multi-block CFG 与未支持 control/region fail closed；这
只是当前 compiler contract，不是形式完备性证明。同 case fan-out 协议是
`intentir.scrc.candidate-fanout.v3`，在 pre-Gate2 wave barrier 后按固定
`(branch_id, formal_round)` 选择 winner，wall-clock 到达次序不参与选择。
outer batch protocol 是 `intentir.scrc.batch.v3`；其 `observer_unsupported` 只表示 Source observer
能力缺口，不是候选非法或 `NO_CLOSURE`，历史 batch v2 report 不重判。

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
