# Task 13：Lightweight Single-Owner MiniSWE / Codex Agent Runtime

> 状态：已完成（2026-07-27）；fresh 28-Kernel 闭合已移交 Task 14
>
> 类型：Task 11 的通用 Agent 执行引擎支撑；继承已归档 Task 12 的反例与可复用代码，但不继承其
> 性能验收结论，不终止、替代或归档 Task 11。
>
> 当前闭合基线：**18/28**；本任务只提升 Lift/Replay agent 能力，已有证书不重跑、不重标。

## 目标

实现两套遵守同一 SCRC 数学与信息边界的 agent runtime：

1. **MiniSWE 轻量路径**：用于日常闭合和能力迭代，优先做到低延迟、低累计上下文、少量高价值工具
   调用，并保持一个 branch-local source-aware Lift owner 贯穿完整 Gate1 闭合过程；
2. **Codex 主力/加速路径**：当前可用于 28-Kernel 主闭合、较大批量并发或独立交叉验证，复用相同 workspace、checker、隐私与
   evidence 合同，补齐真正的 CandidateCheck / ReplayCheck / public synthetic compile 工具循环；
3. 两条路径都必须让 LLM 完整 author `(J,C,pi)` 或 Replay source。固定代码只保存/搬运 exact bytes、
   执行 closed check、返回诊断和管理隔离，不生成、分类、补全或修复语义。

Task 13 完成后把两套通过工程验收的 runtime交回 Task 11/Task 14。Codex 成为 fresh 28-Kernel
闭合的主力并发 backend，MiniSWE 保持轻量可用路径；闭合率、Gate1/Gate2 与 corpus freshness 属于
Task 14，不再阻塞本任务的代码提交。该调度选择不改变 artifact、Gate 或数学合同。

## 不可改变的系统合同

### 1. Lift 生命周期

每个 branch 只有一个 source-aware Lift owner：

```text
Source + anonymous Harness/compiler view
  -> complete (J_0,C_0,pi_0)
  -> CandidateCheck / external Decode / TypeCheck
  -> fresh source-blind Replay_0
  -> Gate1
  -> if FAIL: same Lift owner receives
       Source + complete (J_r,C_r,pi_r) + closed Gate1 diff
  -> owner authors complete (J_{r+1},C_{r+1},pi_{r+1})
  -> fresh Replay_{r+1}
```

- 同一 owner 的 workspace 与逻辑 session 跨 pre-Replay rejection 和 formal Gate1 round 保留；
- relift 必须输出完整 artifact，不能只交 patch；exact-text patch 只是 owner 编辑自己的当前文件；
- `B>1` 时每个 branch 是独立完整 owner，彼此不共享 candidate、Replay、diff、notes 或历史；不是多
  agent 合作构造一个 artifact；
- staged planner/builder/reviewer 不再是活动执行路径，也不能由 CLI/runtime 参数重新启用。

### 2. Replay 生命周期

每个 TypeCheck-valid formal round 创建一个全新的 source-blind Replay episode：

```text
J + pi + MaskC(C) + fixed U_f
  -> one Replay owner reads/inspects the exact packet
  -> authors complete parameterized Replay source
  -> ReplayCheck
  -> advisory public synthetic compile
  -> owner repairs its own source when needed
  -> exact submit
```

- 一个 Replay episode 可以有多个模型/工具步骤；“fresh”不等于 one-shot response；
- 下一 formal round 不继承旧 Replay、旧 conversation、Gate1 diff 或 Source；
- public synthetic compile 使用公开代表值，不使用实际 Harness/`nu_C`，不返回 Source/Gate 信息，且
  没有 acceptance authority；
- exact Replay 提交后，Harness 才私下用同一 `nu_C` 绑定 Source 与 Replay 并执行正式 compile/Gate1。

### 3. C 的时序和权限

Replay 生成时看到 `MaskC(C)=(Sigma_C,bottom)`：匿名 `c*` slot 的 kind、role、type、required 与
source-neutral `subject_ref/axis_ref`；看不到 concrete values、Source 名称/位置或 invocation mapping。
Replay 必须自己 author 已存在的参数和 use-site。生成后 Injector 只做 argument/option binding，禁止
修改 Replay AST、增加 C use-site、loop、mask、address 或 reduction。

### 4. Gate1 / Gate2 分流

- Gate1 FAIL：closed diff 只回到同一 Lift owner，随后完整 relift与 fresh Replay；
- Gate1 PASS 后候选固定，才进入三边 Gate2；Gate2 隐藏输入/结果不回流本次 Lift；
- 若 Gate2 失败且 exact J/Replay 已正确表达，问题只是通用 RefEval/Q/frontend consumer 缺口，则修复
  consumer 后零模型复用 exact Gate1-closed artifact/Replay 做 Gate2-only recheck；
- 只有证据表明 J/B 的语义、类型或 relation 本身错误时才开始新的完整 Lift；不能让 Gate2 tolerance
  或 fixed consumer偷偷修改错误 J。

## 第一性原理诊断

Task 12 证明主要成本不在 Source 大小、JSON 写盘或 checker，而在执行面：

- B4 的单个 formal round仍发生 41–128 次 provider request；无状态 relay 每个工具边界重发增长历史；
- GroupNorm Source 约 2.3 KiB、compact artifact 约 19.6 KiB，却因重复 view、分页 query 和历史回放
  形成百万级累计 token；
- MiniSWE Replay 曾执行 409 次 packet query，其中约 70% 为重复请求，最终没有运行一次有效 Replay；
- Codex C3 Lift 虽保持单 thread/workspace，但读取 README、contract、preflight、ops、schema 和 example
  等重叠输入，3 个 CLI turn耗时约 678 秒；
- Codex Replay recheck 已降到一个 CLI turn、5 个 model segment、10 个 tool action，但当前 adapter只
  host-run ingress，没有给 Codex 同 episode 的 public synthetic compile 工具；
- Codex C1/C2 的只读 output/sandbox错误没有首次 fail-fast，而被错误续调 10–16 次。

因此本任务首先优化 **当前状态、工具面与调度**。fresh canary另暴露并促成两项经权威边界审查的
通用输入能力：adapter v5 的 source-grounded shape identity，以及 artifact.v8 的 π-owned
physical-time relations；二者均由 LLM作者化/消费且 fixed code只检查。除此之外，本任务不增加
J semantic schema、Kernel recipe、token预算或固定 semantic authorizer。

## 设计与实施

### P0：收回错误执行路径

- 移除/拒绝 MiniSWE `staged` 活动入口；历史 v124 evidence 保持不可变，但运行时不再启动 planner、
  builder、reviewer 三个作者；
- 删除默认 prompt 中 staged、reviewer、mandatory ledger/plan 等无效注意力负担；notes 仅由同一 owner
  自愿写入 `/work`；
- 所有 sandbox、provider、workspace 初始化失败立即归为 terminal infrastructure failure；相同 blocked
  状态不得再次恢复模型；
- configured `xhigh` 在 CandidateCheck/ReplayCheck repair 或 Gate1 continuation 中保持 `xhigh`；不得
  无依据自动降为 `high`。

### P1：MiniSWE 当前状态 transport

- MiniSWE 保持同一 branch owner id、workspace 和可审计 trajectory，但 provider packet 不再携带从
  第一轮开始的完整 conversation；
- 每个 Lift continuation 的 kernel-conditioned 当前状态严格由以下 exact authority 组成：短系统合同、
  exact Source、匿名 Harness/compiler view、当前完整 compact candidate、最新 closed CandidateCheck/
  TypeCheck/Gate1 feedback，以及可选的 model-authored notes；
- 已落盘且可按 digest 重读的旧 tool arguments/outputs 和旧 candidate 只保留 deterministic
  bytes/hash/path receipt，不做 fixed semantic summary；完整原始 trajectory 继续落盘审计；
- Replay 每轮首包提供一份完整、无损的 current packet view，不再同时强制 overview/ABI/graph/
  realization/obligation 多套重叠分页；细节查询仅在 agent 主动需要时使用；
- 允许一个 response 原子执行多个互不依赖的只读请求、一个完整 direct submit 或一个 exact patch+
  submit。模型/工具/check 次数只作为 telemetry，不设低次数型 capability cutoff；owner仍由 active
  wall time、sandbox资源、transport safety与 formal `K` 治理；
- 增加 deterministic stagnation detection：只有连续相同工具请求、相同 bytes/digest 和相同 closed
  diagnostic 均无状态进展时才终止为 orchestration stall；它不判断语义正确性。
- 删除重复 contract/example/schema 话术并保持完整闭合约束；prompt 大小本身是执行成本，不能通过
  反复加载多份等价说明来换取“看似更充分”的上下文。

### P2：MiniSWE 工具面

Lift owner至少具有：

```text
read exact current inputs / batch schema sections
write or exact-patch current candidate
CandidateCheck on exact current bytes
commit / rollback exact checked bytes (optional fallback)
direct submit complete candidate
```

Replay owner至少具有：

```text
read complete exact packet / focused lossless view
write or exact-patch current Replay source
ReplayCheck on exact current bytes
public synthetic compile check
commit / rollback exact checked bytes (optional fallback)
direct submit complete Replay source
```

所有工具只操作 owner 已获授权的 bytes；checker PASS 不是证书，host在正式路径独立重跑。

### P3：Codex 同合同执行面

- 保留 `CodexCLISession` 的单 thread resume、持久 Lift workspace、fresh Replay workspace 与双层
  bubblewrap；保持 Codex `multi_agent=false`；
- 向隔离 Codex workspace 暴露与 MiniSWE 等价的受控工具入口，而不是在 prompt 中描述“另一个
  runtime 的不可用命令”；工具通过窄 host RPC/allowlisted dispatcher 执行，不挂载 repo、不暴露
  provider secret、Source 到 Replay 或实际 C values；
- Codex Lift 在自己的 agent turn 内可调用 CandidateCheck/direct submit；Codex Replay 可调用
  ReplayCheck/public synthetic compile/direct submit；host结束后仍独立复核 exact bytes；
- Codex 当前可作为主闭合、批量并发或交叉验证 backend；它不引入不同 artifact、不同 Gate、不同
  privacy 或不同 acceptance 规则。

### P4：观测与失败分类

分别记录：

```text
time_to_first_complete_candidate
time_to_first_candidate_check_pass
model segments / physical provider calls
tool actions and duplicate-query ratio
cumulative input / cached input / output
CandidateCheck / ReplayCheck / compile-check calls
formal Gate1 rounds and Gate1 frontier
same-owner continuation identity
fresh-Replay identity
infrastructure / orchestration / semantic failure class
```

进度阶段分离为 Lift、TypeCheck、Replay generation、Replay public compile、Gate1 compare、Gate2；不能在
Replay 尚未生成时显示 `Gate1 started`。

同 case fan-out 使用 first-observed Gate1 closure：首个完整 closure 原子成为 winner，sibling 在
provider retry 之外 cooperative stop并保存 superseded receipt。该规则不能读取 hidden Gate2，且不把
sibling candidate/feedback交给 winner；它只避免已有 winner 后继续烧完整 K 轮。

## 验证与量化验收

### 实现 checkpoint（2026-07-27）

- 活动 MiniSWE staged 路径已移除；Lift 使用唯一持久 owner，Replay 每个 formal round 使用全新 owner；
- MiniSWE current-state transport 只保留 exact Source/current candidate/latest feedback 与旧 history
  receipts；Replay 首包只提供一个 exact packet；repair/continuation 保持 configured `xhigh`；
- MiniSWE direct submit、exact patch、CandidateCheck、ReplayCheck、public synthetic compile 与可选
  exact commit/rollback 已接通；三次完全相同 action/bytes/closed-output 才触发 deterministic
  `OrchestrationStall`；
- Codex CLI 已接通隔离 allowlisted host bridge，Lift/Replay 在同一各自 episode 内可调用相同 closed
  checker；host仍对最终 exact bytes 独立复核；
- current Lift knowledge v56 在 v55 compact Gate1 frontier/local-axis 基础上，增加 closed
  `bound_constexpr_shape_coordinates`：只有 adapter 显式声明且 valuation 相等的 semantic constexpr
  才关联 `h*`；Lift 必须直接绑定 `x*` 或自行作者化 rank-0 `t*=x*`，fixed code不生成 relation；
- artifact.v8/decode.v12/typecheck.v20 已加入 J reduce/contraction 外的
  `physical_iterations/physical_loop_carriers`，Replay v12/Codex v6 按 active physical tile 使用 local
  reduction axis；这些 π relations 不进入 production backend；
- live progress 已将 Replay 与 Gate1 分开；Replay 未形成前不再显示 `gate1 started`；
- candidate fan-out v4 已把首个 Gate1 closure 接到 MiniSWE/Codex Lift 与 Replay 的在途取消；取消
  保留 winner id、已消费调用与 bytes/hash audit，不能被 transport retry 吞掉；
- Triton workload adapter v5 已把 concrete tensor valuation 与 source-grounded parametric shape identity
  分开：Harness-only `h*` 可跨 tensor/scalar/grid 绑定，Lift 仍必须显式作者化双射 `x*` 与必要的
  value-dimension equality；v5 还闭合 explicit bound-constexpr ↔ `h*` 与真实 scratch inout role；fixed
  code只拒绝 specialization/遗漏，不生成 J relation；
- Lift instruction 已从 20,054 字符压到 17,899 字符，删除重复说明而未删除 closed contract；相关
  prompt/shape/cancellation tests 通过。最终工程验收的 runtime/shape/physical-time/batch 定向矩阵为
  `457 passed`；全量 Python 为 `1822 passed, 4 skipped`；C++/MLIR 完整构建后以可用 lit入口运行
  `122 passed`。生成的 `check-intent` wrapper仍引用环境中不存在的 `/llvm-lit`，改用同一 build/test
  suite的本地 lit可执行文件后全部通过；这是 build配置路径问题，不是代码失败。上述结果只完成
  Task 13 工程验收，不冒充Task14的闭合率或性能结论。

### fresh runtime canary checkpoint（2026-07-27；不是新证书）

Codex Terra high 的 v55 DYT shape canary 使用 `B=3,K=3`，evidence root 为
`/tmp/intentir-task13-v55-codex-terra-high-dyt-parametric-b3k3-r2-20260727`。branch 2/formal round 1
Gate1 closure 后三边 Gate2 全部通过，Source–Replay 误差 0、两个 J leg 最大误差
`2.9802322387695312e-08`；first-observed winner 后两个 sibling cooperative stop。winner J 已把旧
`[2,x0]` 提升为 `[x0,x1]`，但 p6 semantic constexpr `N` 被写成 rank-0 `t4` 时没有 `t4=x1`；旧 v55
Harness 尚未声明/强制该 constexpr-coordinate equality。因此该 immutable run 只维持其固定 valuation
证书，不能作为 v56 parametric acceptance；v56 必须 fresh rerun。

Codex Terra high 的 full Flash forward v55 canary 使用 `B=3,K=3`，evidence root 为
`/tmp/intentir-task13-v55-codex-terra-high-flash-fwd-b3k3-r2-20260727`。9/9 complete artifact
TypeCheck PASS、9 次 fresh Replay/Gate1，终态 `NO_CLOSURE_(3,3)`；Lift/Replay physical calls 分别为
55/70，总墙钟约 2241 秒，无 provider/infrastructure failure。branch 3 最终闭合 interface 与 reduction，
并作者化一条 physical iteration；剩余为 online-softmax 的 physical loop carriers、tile shape/effect 与
dataflow realization。另两个 branch 的 persistent continuation 未消除 TMP output/inout mismatch；审计
确认 adapter 将 Source store-then-load scratch TMP 错标 output，现已改为 inout。该 factual ABI 修正和
v56 shape contract 都要求后续 fresh run，不能回写 v55 verdict。

Codex Terra high 对 Linear 使用 `B=3,K=3` 完整终止，evidence root 为
`/tmp/intentir-task13-v52h-codex-terra-high-linear-b3k3-20260727`：9 次 Lift 均形成
TypeCheck-valid artifact，9 次 fresh Replay/Gate1 均为 `NO_CLOSURE`；Lift/Replay physical calls 分别为
73/38，总墙钟约 1642 秒，无 provider/infrastructure failure。最好 branch 的 op/dataflow score 从
`0.590/0.055` 进展到 `0.817/0.263`，但最终仍缺 `scf.for`、loop carry/pointer advance 与对应
predicate/effect topology。

该结果不能归因于 single-owner runtime 或把 physical loop 硬塞进 J：winner J 正确以 contraction
表达 logical GEMM，C 已声明 reduction tile，π 也有 placement=loop region，但 `pi.iterations=[]`；当前
typed iteration relation 只锚定 structured J loop，无法为普通 contraction/reduction 的物理 tile loop
声明 trip/direction/basis 与 realization carriers。Replay 因此只生成一次 direct dot。这个 fresh
反例把阻塞点收敛为 Task11 后续需要共同审查的 source-neutral π-time expressivity，而不是允许本 Task
用 kernel prompt、旧 Replay patch 或 semantic composer修复。Sol high 与 MiniSWE canary 未终止前只记
live telemetry，不写最终 verdict。

Codex Replay v6 的固定 DYT artifact diagnostic recheck 位于
`/tmp/intentir-task13-v58-codex-full-realization-dyt-fixed-recheck-20260727`。v6 的隔离 mailbox 已成功
返回 anonymous C、J boundary 与完整 π（包括 physical-time fields）的 lossless realization view；同一
fresh Replay episode 首次 public synthetic compile 失败后自行修复并通过。正式 Gate1 仍因额外
`arith.addi/tt.addptr` 与 address dataflow 不一致而 `NO_CLOSURE`；该 run 是
`diagnostic_only=true`，不进入 Gate2、不改变历史 verdict。它证明 Task 13 的 agent/tool loop 可用，
剩余 Replay realization fidelity 属于 Task 14 的通用闭合能力迭代，不能继续用 TypeCheck或 prompt
特化冒充本任务工程完成。

### 单元与隔离测试

1. MiniSWE 单 branch 跨 CandidateCheck、TypeCheck 与 Gate1 continuation 保持同一 owner/workspace；
2. staged workflow 从活动配置与 CLI 拒绝，不能创建 `lift_plan.json` 或启动第二 writer；
3. continuation provider packet大小有确定上界，并且 exact Source/current candidate/latest feedback仍可
   逐字恢复；旧 history receipt不改变本地 trajectory；
4. Replay 两次 `generate` 使用不同 session/workspace，且都能在同 episode 执行
   ReplayCheck→synthetic compile→repair；
5. MiniSWE 与 Codex 受控工具的 PASS/FAIL、exact-byte mismatch、额外 output、privacy、sandbox、secret
   和 infrastructure fail-fast 均有正反例；
6. configured `xhigh` 在所有 repair/continuation 中保持不变；
7. C values/Source/Gate/prior Replay 无法在 Replay input、tool output或 Codex event stream 中检索到。

### 工程完成门槛与 Task 14 交接

Task 13 不再把 fresh Kernel closure 作为代码提交前置条件。闭合本身会继续暴露 J/π/Replay能力缺口，
必须在独立 Task 14 中按 immutable commit 与 fresh evidence迭代，不能让一个长期脏工作区同时承载
runtime实现和实验 verdict。Task 13 的完成门槛是：

- MiniSWE 的同 owner continuation、fresh Replay、closed checker、direct submit/patch、exact
  stagnation、privacy 与 configured effort 正反例通过；
- Codex 的持久 Lift thread、fresh Replay、isolated file-mailbox tool bridge、完整 source-blind
  realization view、host exact-byte复核、same-episode repair与 first-winner cancellation正反例通过；
- Harness adapter v5 对可唯一确定的 shape identity显式提供 `h*`，同时保持 J `x*`/equality完全由
  Lift作者化；RoPE/MRoPE安全直接映射落地，SplitK/product/derived extent不猜测；
- artifact.v8 physical-time relations能 round-trip并 dependent TypeCheck，Replay packet与工具不会
  遗漏这些作者化字段；production backend仍不消费π；
- staged writer与 stdout marker不可恢复，TypeCheck/CandidateCheck/ReplayCheck不被写成 capability
  或认证指标；
- 定向与全量本地测试通过，Task12归档、Task13/14/Task11/spec/权威文档一致，提交后工作区干净。

Task 12 的 40–128 次 provider request、上百次 packet query、累计 input 与 time-to-first-valid 继续
作为历史 telemetry 对照；只有同模型、同 effort、同并发的归一化 A/B 才可声称百分比改进。Task 14
默认由 Codex Terra high 以 `B=3,K<=5` fresh重闭合全部28项；调用数不是 agent内部硬停止上限。

## 非目标

- 除已列明的 adapter v5 shape contract 与 artifact.v8 π physical-time relations外，不继续修改
  J/C/π语义、A/B/C、TypeCheck/Gate1/Gate2数学或 production backend；
- 不增加 typed skeleton、IRBuilder、composer、semantic patcher、自动 handle/type/reference 修复；
- 不增加 Kernel 专用 prompt、模板 artifact、RuleCard、跨 case semantic cache 或 sibling 信息共享；
- 不让 Replay 看到 Source、Gate1 diff、实际 C value、旧 Replay 或 Lift conversation；
- 不用更多 branch、更多 formal Gate1 round 或无限 token代替 agent execution能力。
