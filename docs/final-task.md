# 最终任务：真实 Kernel 的语义闭包与生产 Lower

这是仓库唯一的活动执行记录，只保留当前合同、权威闭包事实、未解决阻断和下一步；不保存候选、Source、TTIR、provider 响应、拒绝草稿、轨迹或临时运行日志。

## 当前合同

- 工作负载全集由 `spec/lowering/high-performance-kernels.json` 定义，共 28 个真实 Kernel。
- 只有 Gate1 为 `replay_closed`，且 Gate2 的 Source–Replay、Source–J、Replay–J 三边均通过，才能计为闭包。CandidateCheck、TypeCheck 和任何局部投影通过均只是诊断，不能计作进度。
- 真实运行严格串行：一次仅运行一个 Kernel，使用一个全新的独立 external root，至多三次 fresh Lift attempt；每次 attempt 都须来自新的合法生命周期。
- 不得跨 attempt 或 root 复用 rejected candidate、draft、trajectory、session 或 repair base；不得补写 Source-owned 语义、放宽任何检查器，或将失败草稿作为修复基础。
- 当前真实验证范围仅为 `high_performance:liger_poly_norm`。真实运行只可使用 `openai` transport；`cislc` 已永久停用。出现 provider rate/usage、transport、step、cost 或 time limit 时立即停止。
- 不创建或恢复 `tests/`、测试目录、mock、synthetic、smoke 或 canary；验证只能走真实 Kernel lifecycle。临时 external root 仅在提取安全终态后移入系统回收站，不提交新 evidence。
- `docs/project-guide/real-evidence` 顶层只保留 `liger_swiglu` 的既有闭包证据。后续运行不添加 evidence。provider 凭据、Kernel Source、TTIR、candidate、replay Source、provider response 与拒绝产物不得写入仓库、文档或报告。

## 权威真实闭包事实

活动 Kernel 总数：`28`

当前可审计闭包总数：`5/28`

- `high_performance:liger_softmax`
- `high_performance:liger_rms_norm`
- `high_performance:liger_swiglu`
- `high_performance:liger_fused_add_rms_norm`
- `high_performance:liger_geglu`

`docs/project-guide/real-evidence/liger_swiglu` 是仓库中唯一保留的 closure evidence。它不能替代其他 Kernel 的独立 fresh 运行。

## 当前状态与未解决阻断

主机 GPU 环境已确认可用；此前 GPU 不可见仅因 Codex 沙箱未透传 `/dev/nvidia*`，不是 driver 故障。涉及 GPU/Triton 的真实执行必须在主机环境进行，且不得重装、重载 NVIDIA 模块或重启机器。

`liger_poly_norm` 当前为 `NO_CLOSURE`。其 Source memory observation 已严格闭合：真实 Source-only 观察得到 9 条 occurrence、9 条 binding；kind-specific compiler anchor 仅使用 load 的 pointer 与 store/update 的 value，未使用 Source/AST/candidate 顺序、计数或语义猜测。原始 Source compiler 坐标、ledger、Direct Responses、Codex/MCP 路径和 Gate1 Replay 前授权复核已使用同一 host-only Source observation；这些 host-only 数据不进入 provider prompt、artifact、Replay payload 或 Gate1 digest。

Source transfer reorder 曾遗漏已签发的 `semantic_authorizations` receipt，导致 `source_transfer_reorder_execution_failed`。该投影缺陷已修复：仅按唯一连续 ordinal 原样传递 ledger receipt，不生成、重签、修改或提升 owner/guard 授权。修复后的新鲜真实运行已越过该阶段。

此前三次 fresh provider attempt 的 FunctionalBoundary 拒绝已定位为一个更早的 Source authorization 缺口，而不是 provider 语义、transport、并行、GPU、Source ledger 或 replay 问题。最新真实 host-side Source-only 观察已确认：2 条 masked access 都有 parser-validated `i1` 直接 predicate edge、唯一 Source 直线 predicate def-use 与 family 一致性，因此 strict ledger 已为这 2 条 access 签发 computed-guard receipt；producer 仍无 standalone raw compiler debug location，但不再以错误的 local-signature 或文本 token 边界推断其类型。store owner 现在仅在“唯一 Source 直线 value definition、已精确匹配的 memory access、direct compiler value edge、family 一致”同时成立时签发，结果为 4 条 store（含 masked store）通过。4 条 load 缺少可接受的 direct compiler semantic consumer，另 1 条 masked load 有多个 Source consumer，均继续保持 unavailable；普通 access 不得因此提升为 provider-local owner。

authoring contract 已同步为 fail-closed：masked predicate 只有在已有闭合逻辑表达式、或 ledger 已签发 exact computed-guard receipt 时才可进入 provider authoring；无 receipt 的 provider-local predicate 不再被描述为合法义务。该 preflight 同时接入 Responses 和 MCP。此前唯一的 masked-store representation 缺口已经以严格的 `output-only, domain-total, realization-guarded store` profile 补齐，且保留原有 `inout + select` 分支：access/effect 必须共同锚定 Source-attested、provider-authored stored-value producer；该 producer 的唯一结果必须精确等于 output-only logical output binding 并严格同型；J 中 access predicate 必须是字面量 `true`；实际 `i1` 只由唯一 `access_guard_relation` 以唯一 typed broadcast 轴携带，既不能成为 logical producer operand，也不能在 `boundary.broadcasts` 重复出现。profile 不引入 input/inout pre-state、select merge、effect_if 或 host-authored operation。TypeCheck、FunctionalBoundary 和 contract materialization 都独立检查并消费同一关系；CPU logical RefEval 继续只执行 load-side guard，因为该 store guard 只约束物理 tail realization，不改变完整 logical output。

最新的真实 Source-only 复核（未调用 provider）为：Source observation `pass`、transfer ledger `pass`、9 条 occurrence/9 条 binding；2 条 masked access（1 load、1 store）都有 computed-guard receipt；5 条 load 的 owner receipt 均严格 absent，4 条 store 均有 exact owner receipt；5/5 load access-view receipt 已签发且是 wire 必填项；`source_authoring_blocker_count=0`。此项只说明可以开始新的合法 provider lifecycle，不是 Candidate、TypeCheck、Gate1 或 Gate2 的通过，也不增加闭包数。

`semantic_access_authorization` 只证明 `node_ref=null` 的 load boundary row 是 exact Source/compiler transfer ledger 中的一条独立 physical access occurrence；它不替代 owner 或 guard receipt，也不授予 consumer operation、predicate、index、domain、effect 或 J 语义。该 contract 已消除先前的 `functional_boundary_access_view_ambiguous` 阻断，且 Gate1 Replay 前会重新验证相同 receipt。

最新独立真实 `liger_poly_norm` lifecycle 均使用 `openai`、Responses Lift/Replay、fresh external root、严格单 worker 与 masked-J，且所有 root 已在仅提取安全终态后回收。Replay generation 的嵌套循环缺陷已由两轮真实生命周期确认修复：每个 TypeCheck-passed 候选只执行初次 Replay 和至多一次 fresh generation retry；两次都在正式比较前被拒绝后，控制流会离开 Replay 内层循环并启动新的 fresh Lift，不再对同一候选无界重复调用。被拒绝 Replay 不消费 formal K、不产生 repair base，也不继承 candidate、draft 或 trajectory。最近两轮分别只执行 4 次 Replay writer invocation，不再出现异常的 32 次累计；两轮 `formal_gate1_rounds=0`，Gate2 均未运行。

当前 provider 基础设施已进一步 fail-closed：HTTP 429、408、认证错误和一般 HTTP 错误分别投影为安全 failure class；完整响应严格等于已知服务过载状态提示时归类为 `provider_overloaded`，不再作为 candidate 送入 CandidateCheck。Lift 的任何 `provider_*` preparation failure 现在直接成为 `SCRCInfrastructureError`，保存的 `lift_provider_generation_failed` receipt 仅含规则码、路径、failure class、物理调用数与 `candidate_present=false`。该过载分类修复已通过静态检查，但由于最新真实运行已观察到 provider 高负载，尚未再次发起长 Kernel 请求验证。最新真实运行的前两个 fresh Lift 均产生候选并通过 TypeCheck，但各自两次 Replay 都在正式比较前被拒绝；第三次 fresh Lift 遇到服务过载。公共 Source-vs-J semantic probe 在这些候选上持续为 `unavailable`，属于另一个诊断性基础设施阻断，不具 Gate 权限。当前无可接受 Gate1 candidate，`liger_poly_norm` 仍为 `NO_CLOSURE`。

不得把普通 access 提升为 provider-local owner，不得制造 owner/guard token，不得根据被拒绝候选推断谓词、index、domain、effect 或 operation 语义，也不得放宽 FunctionalBoundary、SemanticArchive、CandidateCheck、TypeCheck、Gate1 或 Gate2。

## 下一步

1. 保持 load owner 的 unavailable 结果；除非存在同样严格的唯一 Source def-use 与 exact compiler dataflow counterpart，否则不补发 receipt。access-view receipt 不可替代 owner receipt。
2. 保持 Source-only 聚合检查、`compileall`、关键导入、`git diff --check` 与本地 provider 配置 `{}`；确认新 receipt 和运行归属不让 Source、TTIR、candidate 或 provider payload 进入活动树。
3. 当前因 `provider_overloaded` 停止真实测试；不得立即创建新 root、切换 cislc、并行或用探针/canary 追加请求。等待 `openai` transport 恢复后，只依据新的安全 failure class 判断是否可继续。
4. provider 恢复后，使用一个新的 external root 重跑 `high_performance:liger_poly_norm`：显式 `openai`、Responses Lift/Replay、masked-J、严格串行，最多三次彼此独立的 fresh Lift；不复用任何旧 submission。该轮还须确认服务过载只形成 `lift_provider_generation_failed` 基础设施终态，绝不形成 rejected candidate。
5. 保持 TypeCheck、CandidateCheck、SemanticArchive、FunctionalBoundary、authorization revalidation、Gate1 与 Gate2 的严格条件。host 不得生成 operation、predicate、owner、index、domain、effect、select 或 repair candidate。
6. 只有 Gate1=`replay_closed` 且 Gate2 三边全部通过，才把该 Kernel 计为闭包；否则只报告安全规则码、结构路径、调用计数与 Gate 状态，并停止该轮。不得把 candidate、Source、TTIR、provider response、replay Source、rejected draft 或 trajectory 写入仓库或文档。

## 完成条件

任务完成仅当每个活动 Kernel 都具备独立真实 lineage、Gate1=`replay_closed`、Gate2 三边通过，并且 Source、J、C、π、Boundary、authorization 与 Gate 路径的职责边界保持完整、严格且可审计。
