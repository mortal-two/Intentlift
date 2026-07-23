# Task 04：mini-swe-agent 有界 Lift Worker 接入

> 状态：执行中
>
> 类型：SCRC 系统生成能力新增/重构；不是论文实验、性能实验或新的认证流程。

## 直接目标

把当前一次性 Responses Lift 替换为 mini-swe-agent 驱动的有界 Lift worker，使每个 worker 只负责：

```text
读取本轮允许的 Lift 信息
    -> 按需理解 Source、schema 与通用 frontend 知识
    -> 输出唯一完整 candidate.json = (J,C,pi)
    -> 结束本轮
```

现有 `SCRCGateRunner` 继续唯一维护 TypeCheck、source-blind Replay、Gate1、relift、winner 选择与
Gate2。mini-swe-agent 不接管流程，不成为 HarnessAgent、ReplayAgent、Gate Agent 或第四 artifact。

## 第一性原理边界

对 branch `b`、Lift round `k`，worker 只实现：

```text
MiniLift[b,k](Source, HarnessView, Schema, LiftKnowledge, Feedback[b,k-1])
    -> one complete (J[b,k], C[b,k], pi[b,k])
```

其中：

- 第一轮 feedback 为空；
- relift 只接收外部 runner 明确允许的本分支上一轮 TypeCheck 或 Gate1 feedback；
- worker 内部可以有多次模型/工具步骤和草稿修改，但不能调用正式 dependent TypeCheck、Replay、
  Gate1、masked-J、RefEval、Gate2 或 Q；
- worker 一旦提交完整 candidate，本轮结束；外部检查失败后的任何修改都属于下一轮；
- malformed/missing candidate 仍消耗本轮候选预算；provider/worker 基础设施失败单独分类；
- 每个分支/轮次使用独立模型 history、独立工作目录和独立 trajectory，不读取 sibling candidate。

正式证书谓词完全不变：

```text
SCRC_CERTIFIED
iff TypeCheck(J,C,pi) and Gate1 and Gate2.
```

## 最小实现形态

只增加一个生产接入面：

```text
SCRCGateRunner
    -> MiniSWELiftAdapter     # 实现既有 LiftAdapter protocol
        -> 仓库内 minisweagent.DefaultAgent / existing model configuration
        -> candidate.json
```

明确不增加：

- `CPAResponsesModel` 或第二套 provider abstraction；
- 新的 Lift/Gate 状态机；
- 新 Replay writer、Replay session 复用或端到端 mini-swe orchestrator；
- vector database、per-kernel RuleCard 或独立 RAG artifact；
- hash/commitment/epoch/freeze 门；
- 自动修补 parser、TypeCheck、observer 或 comparator 的旁路。

若 upstream 现有 model configuration 能连接当前 CPA，就直接配置；只有出现具体、可复现的不兼容时
才修改最小现有点，不预先建设 transport wrapper。mini-swe trajectory 中已有的模型调用/时间信息
直接作为 runner evidence 保存，不另造证明对象。

## Worker 输入与工具边界

每轮建立独立工作目录，逻辑视图为：

```text
input/source.py                 # 本轮 Source slice，只读
input/harness_view.json        # 允许的机械 compiler/Harness view，只读
input/schema/                  # 当前 J/C/pi wire/schema 与字段合同，只读
input/lift_knowledge/          # 版本化、source-neutral 通用 frontend 文档/示例，只读
input/feedback.json            # 仅本 branch 上一轮允许反馈，只读；首轮为空
output/candidate.json          # 唯一候选输出
```

首版使用普通文件与 `rg`/`jq` 按需检索，不建设向量 RAG。worker 不挂载仓库根目录、历史 `runs`、
prior Replay、Gate2 evidence、其他候选、provider secret 或正式 Gate executable。优先使用 upstream
已有的容器/受限 environment 配置；若它不能表达上述文件边界，只补一个窄 environment 适配，不
把隔离责任交给 prompt。

允许 worker：

- 读取、搜索上述输入；
- 编辑本轮草稿和 `output/candidate.json`；
- 检查 JSON 语法与静态 JSON Schema；
- 查看 source-neutral 的合法/非法通用示例。

禁止 worker：

- 运行项目的 dependent `type_check`、compiler observer、Replay 或 Gate2；
- 修改 input/schema/runner/system code；
- 访问网络检索未记录知识、宿主环境变量或工作目录外文件；
- 从自然语言 final response 代替 `candidate.json`；
- 用自己的完成声明代替外部 Parse/TypeCheck。

## 模块与依赖纪律

- 完整 `mini-swe-agent v2.4.5` 源码直接进入 `python/minisweagent`，成为仓库一级模块；
- 在该模块内部直接按 IntentIR 要求修改，不维护 upstream + patch overlay，不依赖 PyPI
  `mini-swe-agent` 包，也不建立 `third_party`；
- 复用其中的 `DefaultAgent`、Responses model 与 Bubblewrap environment，只在真正影响 Lift
  边界、隔离和预算处直接修改；
- 依赖不可用时给出明确 infrastructure/unavailable 结果，不静默回退 one-shot Lift；
- provider 配置只读取仓库既有、gitignored 的 `llm_providers.local.json`；key 仅用于宿主侧模型
  调用，不进入 agent shell、prompt、trajectory 或 evidence。

## 实施阶段

### A. 接口与离线 worker

1. 定义 `MiniSWELiftAdapter`，实现现有 `LiftAdapter.lift(request, feedback)`；
2. 将现有 source anonymization、Harness view 与 feedback 投影成独立 worker 输入；
3. 配置 mini-swe-agent 输出唯一 `candidate.json`，worker exit/submission 只表示候选交付；
4. 从 candidate 复用现有 `Gate1Artifact.from_json` 与 anonymous/compiler-view 检查；
5. 把 worker invocation、trajectory、模型调用数、wall time、termination 和 rejected candidate 放入
   现有 Lift metadata/evidence。

### B. 隔离、预算与轮次

1. 每个 branch/round 新 worker、空 history、独立目录；
2. 外部 TypeCheck/Gate1 feedback 只进入下一轮同 branch；
3. 五 branch 不共享可写目录、trajectory、candidate 或 feedback；
4. worker step/wall-time/model-call 上限配置为有限值，并在报告中如实记录；
5. infrastructure failure 与候选拒绝/`NO_CLOSURE` 分开；取消分支不伪装成失败。

### C. 现有 runner/CLI 接入

1. 保持 `SCRCGateRunner.search_gate1/complete_gate2` 不改证书语义；
2. CLI 显式选择 Lift implementation，首版不静默切换默认路径；
3. Replay 继续使用当前独立、source-blind direct Responses writer；
4. Task 03 的 `B=5,K=3` fan-out 直接复用新 adapter；
5. evidence 能恢复每轮 worker 输入清单、trajectory、candidate 与外部验证结果。

### D. 验证与真实恢复

1. 单元测试覆盖输入 allowlist、唯一输出、无候选、非法候选、超时与 worker failure；
2. 集成测试证明 TypeCheck/Gate1 不在 worker 内执行，反馈只创建下一轮；
3. 并发测试证明五 worker 目录/history/feedback 隔离；
4. 原 one-shot Lift 测试与现有 Gate/Replay/Gate2 全量回归不退化；
5. 完成离线验证后，用 `high_performance:liger_dyt` 新的 `fanout-002` 做真实 canary；只有外部
   TypeCheck、Gate1、Gate2 的正式结果可以形成 Task 03 新结论。

## 验收条件

1. mini-swe-agent 只实现既有 LiftAdapter，现有 runner 仍是唯一认证状态机；
2. 每个 worker 最多交付一个完整 `(J,C,pi)`，外部反馈后明确开始下一轮；
3. worker 无法读取 Source slice/Harness view/schema/knowledge/本轮 feedback 之外的信息；
4. worker 无法调用 TypeCheck、Replay、Gate1、Gate2 或修改可信规则；
5. Replay packet/source isolation、同一 c0、external Q 三边与 Gate2 不反馈纪律不变；
6. 五路候选继续并发且互不读取，winner/Gate2 规则不变；
7. 不存在新 CPA model layer、第二 provider truth、第四 artifact 或隐式 one-shot fallback；
8. worker trajectory/调用数/耗时作为 evidence 保存，但不进入 certificate predicate；
9. 定向测试、全量 Python 测试、compileall 与 diff check 通过；
10. dyt 新 run 产生可审计的正式候选搜索记录，随后 Task 03 恢复执行。

## 当前基线

- 进入本任务前代码 checkpoint：`f0e96a0`；全量回归 `801 passed, 4 skipped`。
- `liger_dyt/fanout-001`：五分支真实并发，12 次 Lift、0 次 TypeCheck/Replay；四分支正常耗尽，
  一分支 HTTP 408，未产生 winner。该记录属于 Task 03，不作为本任务实现结果。
- 当前任务从 upstream 依赖/接口可行性与离线 adapter 测试开始，不在未完成隔离与轮次测试前调用
  真实 provider。

## 2026-07-23 实现 checkpoint

- `mini-swe-agent v2.4.5` 的完整 `src/minisweagent` 已直接成为 `python/minisweagent`，来源 commit
  为 `e187bcb2ff5825d85761a6f9c1f98c9fa6cfbc79`；模块在仓库内直接修改，不存在 patch overlay；
- `MiniSWELiftAdapter` 已实现既有 `LiftAdapter`，继续复用唯一的 Source anonymization、
  `Gate1Artifact` canonical decoder、compiler-view 检查和 runner feedback；
- IntentIR 模式直接复用 `DefaultAgent + LitellmResponseModel`，从既有
  `llm_providers.local.json` 读取 CPA，未新增 `CPAResponsesModel`；本地假 CPA 已机械确认请求走
  `/v1/responses`、model 正确、`reasoning.effort=xhigh`、`parallel_tool_calls=false`；
- 一个 invocation 使用有限串行 model/action turn，每个 response 恰好一个 bash action；模型层、
  LiteLLM 与 OpenAI transport 的隐式 retry 均关闭，IntentIR 模式不参加 upstream process-global
  model budget；
- sandbox 已机械验证 `/input` 只读，宿主 repo、`intentir` package、provider secret 与网络不可见；
  根、`/dev` 及 system-package mask 只读，只有受总额/文件数/深度/输出/进程资源约束的
  `/work`、`/output`、`/tmp` 可写，异常退出使用非递归清理；
- 每轮只接受唯一 `candidate.json`；trajectory、knowledge retrieval、草稿和 provider envelope
  只作为 proof-irrelevant evidence，不进入 artifact 或 Replay；
- 依赖以 `lift-agent` extra 声明，wheel 已确认包含完整模块、IntentIR config、preflight、validator、
  license 与 upstream provenance；无 extra 的 core wheel 仍可 `import intentir` 和
  `import minisweagent`；
- 定向 agent/CPA/隔离/并发/预算测试通过；最终全量 Python 回归为 `818 passed`，`compileall` 与
  staged diff check 通过。

## 2026-07-23 DYT implementation canary：run-001

本次运行是 MiniSWE Lift backend 的实现 canary，不是 SCRC 认证实验：

- 命令显式选择 `--lift-backend miniswe --effort xhigh`；
- fresh Lift invocations 为 3，每个 invocation 恰好 8 个 model turn；
- physical provider calls 为 24；
- 三个 worker 耗时分别为 49.073、36.893、45.317 秒；
- submitted `(J,C,pi)` candidate 为 0；
- canonical decode、TypeCheck、Replay、Gate1、RefEval 与 Gate2 执行次数均为 0；
- workload 的阶段分类是 `lift_response_rejected`；aggregate `main_status=NO_CLOSURE` 只表示本次
  有限搜索没有 winner，不能解释成 Gate1 已执行后不闭合。

三个 invocation 均为独立的新 session/workspace。它们读取 Source、Harness view 与合同后，把剩余
turn 全部用于逐段浏览 1420 行静态 schema，最终以 `LimitsExceeded` 结束，没有 artifact 越过 Lift
输出边界。因此这份记录只定位 MiniSWE 的 schema discovery、任务引导和 action-budget 分配问题；
它不构成 TypeCheck failure、Gate1 failure 或 SCRC certificate evidence，也不能说明 `(J,C,pi)`
设计过严。完整运行记录保存在 `runs/liger_dyt/run-001/`。

针对该失败，knowledge profile 升级为 `intentir.lift-knowledge.v2`：加入固定、source-neutral 的合法
wire 示例与全局 IntentIR operation reference，提供 `top/catalog/op/section` 的确定性叶子查询，明确
禁止翻页 raw schema，要求最迟第 5 turn 写出首个完整 candidate，并在每次 observation 中显示剩余
turn。预算小幅调整为 10 turn，新增内容不进入 artifact、Replay 或 certificate predicate。定向验证
为 `21 passed`，全量 Python 回归为 `822 passed`，`compileall` 与 diff check 通过。下一步使用同一
case 运行 run-002；只有单 worker 能稳定提交 candidate 后才恢复五分支 fan-out。

## 2026-07-23 DYT implementation canary：run-002

run-002 使用 `intentir.lift-knowledge.v2` 和 10-turn budget。与 run-001 相比：

- 三个 fresh worker 全部在第 9 turn `Submitted`，submitted candidate 从 0/3 提升为 3/3；
- physical provider calls 为 27；三个 worker 分别耗时 309.628、303.912、394.474 秒；
- 三份 candidate 均通过 Agent 内静态 schema validator，并实际到达外部 canonical decoder；
- attempt 1/2 被拒为 `p5.logical_ref=v0`，但 `v0` 是 body SSA value，不是 bindable signature handle；
- attempt 3 把 `p5.logical_ref` 改为 `t4`，但 `t4` 是 f32 output，而 `p5` 是 input-side i1 semantic
  constexpr，因此 direction/dtype 不相容；
- canonical artifact 仍为 0，TypeCheck、Replay、Gate1、RefEval、Gate2 执行次数仍均为 0。

这次结果证明 run-001 的 schema 翻页/无提交问题已解决，同时暴露一个真实合同冲突：静态 schema
description 曾写 `logical_ref` 可指向 “J tensor, value, or dimension”，canonical parser 实际只允许
声明的 J input/output tensor 或 symbolic dimension。候选第 5 turn 生成完整 15--17 KB JSON 时输出
8.6k--10.8k tokens，单次 provider 生成耗时 155--196 秒；其余大多数请求为 2--29 秒。run-002
约 16.8 分钟的主因不是 Lift+Replay 混入同一请求，也不是本地 shell，而是一个分支串行生成三份
大候选；第二/第三次没有获得上一 rejected candidate，只能重新生成完整 artifact。

knowledge-v3 因此做两项系统修正：

1. schema description、Lift contract、preflight 与 canonical diagnostic 统一规定：logical data scalar /
   semantic constexpr 使用 dtype 相容、shape `[]` 的 rank-0 J input 并进入 dataflow；extent-only integral
   handle 才绑定 `x*`，`logical_ref` 不得指向 `v*` 或不相容 output；
2. canonical parser rejection 后，仅把同 branch、紧邻、与 diagnostic 精确对应的有限 JSON object
   送入下一次全新 worker。新 session/history/workspace 只做聚焦修正并重新提交完整三元组；不恢复
   trajectory、不在同轮运行 parser/TypeCheck，也不把 rejected candidate 送入 Replay/证书。

v3 定向回归为 `120 passed`，全量 Python 回归为 `827 passed`，`compileall` 与 diff check 通过。
下一步 run-003 必须首先验证 canonical decode/TypeCheck 是否被真正到达；未达到前不扩大到五分支
正式 fan-out。
