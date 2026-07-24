# Task 03：28 项真实 Kernel Manifest、顺序 Batch 与代表性纵切

> 状态：已归档（原任务已结束；后续工作不恢复本任务）
>
> 类型：SCRC 系统运行能力扩展；不是性能实验、论文统计或 28 项一次性 provider 全跑。

## 2026-07-24 执行范围收敛

六个 representative case 只是下一协议的 bring-up 顺序，不是 Task 03 的最终覆盖终点。当前项目
顺序调整为：

1. 完成 Task 09 的非语义重复解释器清理；
2. 实现同一权威 CandidateCheck、C attachment、numeric-cast dependent typing 与
   result-relation compatibility；
3. 用 DYT、SwiGLU、Attention reduction 三项完成新协议纵切；
4. 继续顺序闭合 registry 的全部 28 个 Kernel，使每项都留下 accepted J/C/π、Replay、Gate1、
   Gate2 与预算证据；
5. 稳定 J corpus 完成后，再独立启动后端 B0/B1；不要求每闭合一个 Kernel 就紧邻开发其 backend。

这里的“不是 28 项一次性 provider 全跑”只禁止在通用协议尚未纵切闭合时发起一个不可诊断的
单体批次，不排除最终逐项完成 28/28 SCRC。某项失败时先结束该 fresh run，依据跨字段、跨 Kernel
可复用的 evidence 修正通用系统并重新运行；不得用 per-kernel prompt、手写 J 或后端特例换取通过。

## 上游基线

Task 01 已完成 canonical `(J,C,pi)`、dependent TypeCheck、source-isolated Replay、Gate1 observer、
external-Q 三边 Gate2 与统一状态机。Task 02 已用三个全新真实流程分别认证 exact、logical-domain
mask 与 masked single-axis sum，最终回归为 `736 passed, 4 skipped`。

本任务继续服从四份核心设计和活动 spec，不新增 Agent、Gate、IR、RuleCard、LoweringAgent、hash
门或第四 artifact。实验运行中的候选只由既定 Lift/Replay 流程改变；通用系统能力只在一次 run
结束后修改并从新 run 重走。

## 唯一 28 项来源

28 项集合只来自现行 lowering capability registry：

```text
spec/lowering/high-performance-kernels.json
python/intentir/backend/cuda/capabilities.py
```

registry 已机械要求 ordinal 恰为 `1..28`、ID 唯一，并另记不计作 direct kernel 的 scenario 29。
Task 03 不复制或手写第二份 28-ID truth；SCRC manifest 必须是该 registry 的确定性投影。lowering
registry 继续只陈述 backend execution，不加入 Gate verdict 或认证字段。

当前 28 项实际组成：17 个 Liger、6 个 FlashAttention、5 个 xFormers direct kernel；具体计数与
ID 一律由 loader 计算，不在 runner 中另建常量列表。当前工作区已只读验证 28/28 source
module/file 与 entry symbol 均可解析。

## 目标架构

```text
lowering capability registry (28, existing truth)
                 |
                 v
SCRC workload projection (source/Harness readiness, no verdict)
                 |
                 v
sequential batch runner -> per-case existing SCRCGateRunner
                 |
                 v
per-case evidence + aggregate facts
```

Manifest/runner 配置属于 Harness side，不进入 `(J,C,pi)`、Replay packet、TypeCheck 或 Gate predicate。

## 阶段 A：Manifest 与离线批量骨架

新增一个轻量、typed 的 SCRC workload projection，至少包含：

```text
ordinal
case_id                    # 直接引用 capability.kernel_id
suite / symbol / mechanisms
frontend
source_module / resolved source file
source_alignment
observer profile/version
Q policy/version
lowering evidence refs
SCRC adapter readiness
representative tier
```

要求：

1. projection 每次从 lowering registry 构造，28 个 ID/ordinal 不得漂移；
2. Liger source 优先从安装模块解析，必要时回退同级 `Dataset/Liger-Kernel/src`；
3. FlashAttention/xFormers source 从可导入模块或同级 `Dataset` tree 解析；不提交用户绝对路径；
4. source file 必须可解析，目标 function definition 必须恰好出现一次；
5. optional dependency 缺失归类为 `CASE_UNAVAILABLE`，不能伪装成 Gate 失败；
6. `--list`、`--validate-manifest`、dry-run 不调用 provider；
7. 单 case infrastructure failure 不终止后续 case；默认顺序执行，不建设异步队列。

CLI 第一阶段：

```text
python tools/scrc_batch.py --list
python tools/scrc_batch.py --validate-manifest
python tools/scrc_batch.py --dry-run --tier representative
python tools/scrc_batch.py --case <registry-id> --dry-run
```

## 阶段 B：六个真实代表纵切

以下 selector 绑定到现行 registry ID，不复用 Task 02 内嵌 canary：

| tier | registry ID | 目的 |
|---|---|---|
| core.elementwise | `high_performance:liger_dyt` | 真实 elementwise/source ABI |
| core.masked_pointwise | `high_performance:liger_swiglu` | program row、tail mask、cast/sigmoid/mul |
| core.single_axis_sum | `high_performance:flash_attn.flash_attn_triton._bwd_preprocess_do_o_dot` | 两输入 product-sum 与真实 stride ABI |
| frontier.softmax | `high_performance:liger_softmax` | max/sum/broadcast/div 组合 |
| frontier.normalization | `high_performance:liger_rms_norm` | square/mean/epsilon/rsqrt/affine |
| frontier.contraction | `high_performance:flash_attn.ops.triton.linear.kernel_fwd` | dot/tile/accumulator/epilogue |

六项都必须使用真实 source file/symbol 和显式 Harness factory。前三项是最接近 Task 02 当前能力的
compatibility candidates，不预先宣称必然通过；后三项是能力边界。Task 03 的正确结果可以是
`SCRC_CERTIFIED`、`NO_CLOSURE`、`GATE2_FAILED/reference_unsupported` 或明确 infrastructure
unavailable，但不能用 per-kernel prompt patch、手写 J、Task 02 source 或 Python oracle 伪造通过。

六项完成只表示 bring-up 结束；随后进入全部 28 项的顺序 closure。最终目标不是用一份 aggregate
状态代替 individual evidence，而是每个 registry item 都有独立 fresh lifecycle 与 accepted artifact。

## 失败分类与汇总

批量层只规范化既有事实，不增加 Gate：

```text
MANIFEST_INVALID
CASE_UNAVAILABLE
HARNESS_INVALID
LIFT_RESPONSE_REJECTED
TYPECHECK_FAILED
REPLAY_RESPONSE_REJECTED
REPLAY_COMPILE_FAILED
NO_CLOSURE
REFERENCE_UNSUPPORTED
GATE2_FAILED
INFRASTRUCTURE_FAILED
SCRC_CERTIFIED
```

每个 case 记录 source location、observer/Q 版本、main/j-necessity 状态、TypeCheck、formal Gate1 rounds、
Gate1 primary diff、Gate2 三边、RefEval profile/unsupported reason、真实 Lift/Replay/compile/execution
预算和 evidence 路径。aggregate 只计数，不重新定义 certification。

## 验收条件

### Manifest/runner

1. 28 项投影与 lowering registry exact ID/ordinal equality；scenario 29 不混入 direct cases；
2. 当前工作区 28/28 source file 与 symbol 静态解析；
3. list/validate/dry-run 零 provider 调用，并有单测；
4. sequential runner 对单项失败隔离，summary 稳定、机器可读；
5. manifest/source/Harness/C concrete values 不进入 Replay packet；
6. Task 01/02 生命周期与测试不回退。

### 真实纵切

7. 六项各有真实 source 与 Harness adapter，不从性能测试 stdout 反解析；
8. 不同 Kernel 仍按六项顺序运行；同一 Kernel 使用 `B=5` 个候选 branch 并行搜索，保存每 branch
   的完整 pre-Gate2 evidence、固定 winner 坐标、唯一 Gate2 结果与调用预算；
9. 当前支持域若暴露通用缺口，先结束 run，再增加通用 schema/consumer/prompt 回归并新跑；
10. frontier unsupported 必须准确落在具体阶段，不能归成未知异常；
11. Task 03 完成时运行全量测试，工作区无 secret 或无主 sidecar。
12. 六项 bring-up 后继续顺序运行全部 registry item；Task 03 只有在 28/28 各自形成
    `SCRC_CERTIFIED` artifact 与独立 evidence 时完成，aggregate summary 不能代替单项闭合。

## 本任务明确不做

- 不一次性调用 provider 跑完 28 项；
- 不做性能计时、排名、beat/parity 结论或 retune；
- 不为 softmax/RMSNorm/matmul 写 per-kernel RefEval 特例；
- 不增加一般集合值并发、atomic、scatter/update、alias/inout 支持；
- 不建设跨 Kernel 异步队列、分布式调度、复杂 resume 或 hash/commitment 系统；同 case 的固定
  5-worker candidate fan-out 是本任务明确建设的搜索能力；
- 不让旧 Replay、旧 conversation 或其他 Kernel 的 Gate1 diff 跨 case 传播。

## 后续

Task 03 的首个五候选真实运行表明，当前一次性 Responses Lift 才是继续纵切前的系统瓶颈；因此
Task 04 先把 mini-swe-agent 接成现有 Lift 的有界候选生成实现，外部 TypeCheck、Replay、Gate1、
Gate2 与本任务的五分支控制面保持不变。Task 04 完成后从全新 dyt fan-out 恢复本任务；六项纵切
全部完成后，继续按同一协议顺序推进其余 registry item。每次失败仍先依据结构化结果做可泛化
系统改进再 fresh rerun，直到形成 28/28 accepted corpus。达到 SCRC 门禁覆盖与达到后端性能目标
继续作为不同任务处理，后端 B0/B1 不在本 Task 内穿插实现。

## 执行记录

### Phase A：离线 Manifest 与顺序 Batch 底座（已完成）

- 新增 `intentir.verify.workload_manifest`，每次从现行 lowering capability registry 构造 28 项
  typed 投影，没有复制第二份 Kernel 清单；六个 representative tier 只是一层小型选择视图。
- source resolver 优先解析已安装 module，并以仓库同级 `Dataset` 为 fallback；没有提交任何用户
  绝对路径。当前工作区静态校验结果为 `28/28 resolved`、零 unavailable、零 invalid。
- 新增 `intentir.verify.scrc_batch` 与 `tools/scrc_batch.py`：支持 list、validate、代表项/单 case
  dry-run、稳定 JSON 结果和顺序故障隔离；这些离线命令不导入或调用 Lift/Replay provider。
- 六个代表项 dry-run 均为 `ready`；此状态只表示真实 source/symbol 已解析，不能解释为 Harness
  adapter 完成、Gate 通过或 `SCRC_CERTIFIED`。
- manifest/batch/CLI 及原有 high-performance registry 的相关回归共 `22 passed`。独立 CLI 已验证
  可从仓库根目录直接运行；机器可读输出只记录 `module://` / `dataset://` portable location，不写入
  用户绝对路径。

Phase B 仍在执行：需要为六项建立真实 Harness adapter，并通过统一 SCRC 生命周期保存各自的
terminal evidence 与真实调用预算。在六项纵切完成以前，本 Task 保持“执行中”。

### Phase B：真实 Source/Harness 与统一生命周期（进行中）

- 已新增声明式 Triton workload adapter 与 fixed-C AST source projection。六个 representative
  case 均从现行 registry 解析真实 file/symbol，机械检查 runtime/constexpr ABI；需要的 imported
  `@triton.jit` helper 由 source closure 解析，不通过导入带 native extension 的外部 Python package
  获取。六项真实 CUDA/Triton frontend compile 检查均已通过。
- compiler view 现记录机械 argument mode；Gate1 对 interleaved semantic/C constexpr 使用匿名
  handle 分类而非位置猜测。Replay 的公共 Triton 合同统一使用 `libdevice` module access，没有
  per-kernel prompt patch。
- 统一 `SCRCGateRunner` 已接入真实 workload：Gate1 首次闭合后才构造隐藏有限输入；Gate2 绑定
  Source/Replay/J 三腿与 external-Q 三边；rank-0 logical scalar、boolean guard 和受限 pure
  pointwise logical-domain tail mask 已作为通用能力接入，masked reduction profile保持独立。
- evidence 路径现在保存被 schema/validator 拒绝的 Lift/Replay provider response、receipt、实际
  physical call 数、partial artifact、TypeCheck 与 Gate1 证据；批量分类可以区分 Lift/TypeCheck/
  Replay response/compile/Harness/reference/Gate2/infrastructure 终态。

### 首个真实 case 暴露的问题与处理

`high_performance:liger_dyt` 初始产生七个独立 run。`run-001`/`run-005` 是候选未闭合，`run-002`/
`run-003` 暴露并复现了 Gate2 rank-0 logical scalar binding 缺口，`run-004`/`run-006`/`run-007`
是外部 provider 502/408。上述 run 的原始 terminal verdict 不回写；历史记录的计数/证据限制与
离线诊断分别写在各 run 的 `AUDIT.md`。

对 `run-003` accepted Gate1 candidate 的离线只读诊断显示：Source/Replay exact，RefEval 三腿在
通用 scalar/mask 支持补齐后均可执行，但 candidate 把浮点 tanh/乘加结果声明为 `exact`，两条
J 比较边最大绝对差约 `2.98e-08`，因此仍不能认证。这个事实已用于收紧通用 Lift comparison-class
提示；没有修改历史 artifact、Q 或 verdict。

后续 `liger_dyt` 正式记录如下：

- `run-008` (`high`)：第一次 Lift HTTP 请求达到 600 秒客户端上限，自动重试开始后在总耗时超过
  12 分钟时人工中断；未产生 candidate，归为 `INFRASTRUCTURE_FAILED`。
- `run-009` (`medium`)：三次 Lift response 均在完整 artifact 解析阶段失败，分别为两次
  `tolerance` 错带空 `update_node_ref`，以及一次 access/effect `node_ref` 不一致；正式结果为
  `NO_CLOSURE / LIFT_RESPONSE_REJECTED`，0 次 TypeCheck/Gate1。
- 结束 run-009 后只收紧通用 wire instruction，没有放宽 parser：`exact/tolerance` 必须完全省略
  `update_node_ref`；access/effect 的 kind/target/node_ref/access_ref 必须 exact match。
- `run-010` (`medium`)：4 个 logical Lift call 中 3 个完整 artifact 进入 TypeCheck 与正式 Gate1；
  两个 Replay 编译成功但在 `op_multiset/dataflow/predicate_effect` hard mismatch，另一次 Replay
  provider response rejected，最终为 `NO_CLOSURE`。具体差异是 Source 的 3 个 signed widening
  在 π 中全部遗漏，且整数 coordinate 的 base/lane 组成被误写成额外 pointer address stages，
  Replay 因而生成 0 个 `arith.extsi` 与 10 个 `tt.addptr`，Source 分别为 3 与 6。

上述差异推动两项 source-neutral 系统改进：Lift 必须为每个 compiler-visible sensitive
representation stage 建立 J authorization 与 π refinement，Replay 必须在对应 instance 消费；
integer coordinate arithmetic 不因随后用作 offset 就自动成为 pointer address stage。Gate1
observer、TypeCheck 与 strict parser 均未放宽。

同时完成 prompt attention 收敛：Responses structured-output schema 继续承担完整字段/枚举/结构
合同，Lift natural-language instructions 从 26,180 字符降为 6,294 字符；完整 Gate1 comparison
继续落盘，relift 只接收每类至多 8 个代表差异的确定性 actionable view，不再重复发送低层节点
fingerprint、components 与 score/rank。相关回归 `92 passed`。

### Phase C：同 case 多候选并发（进行中）

单链 `B=1` 容易让一次 schema/π 提取偏差串行占满 wall time，也不足以满足 dyt 必须实际闭合的
目标。现将搜索控制面改为：

```text
B=5 个 branch 并行
每 branch: 最多 K=3 个 TypeCheck-valid Gate1 round + branch-local relift
wave barrier 后按 (branch_id, formal_round) 固定顺序选择一个 Gate1-closed winner
winner 选定后才生成隐藏输入，且只执行这一次 Gate2
```

已把统一 runner 拆为 `search_gate1` 与 `complete_gate2`，原单候选 `run()` 继续组合两者且保持兼容；
新增 fixed-size thread fan-out、独立 evidence 目录、稳定 branch 汇总、异常隔离与 aggregate physical
call budget。测试用五线程 barrier 证明 Lift branch 实际重叠执行，并验证即使五个 branch 都闭合，
也只有固定 branch 1 进入 Gate2。并发不同 Kernel、候选拼接、Gate2 fallback 和 HarnessAgent 均禁止。
并发重构后的全量回归为 `801 passed, 4 skipped`。

此前启动的 `liger_swiglu/run-001` 是旧单链 xhigh 运行，在尚未产生 Lift response 时因本次控制面
重构被人工停止；它不计 `NO_CLOSURE` 或 Gate 失败。重构完成后先回到 `liger_dyt`，使用
`--candidates 5 --effort xhigh` 获得完整 Gate1 closure 与 winner Gate2 结果，再进入 swiglu。

首次真实五候选运行 `liger_dyt/fanout-001` 已完成并留证：五个 branch 同时启动；branch 3 的首个
provider 请求在约 301 秒后以 HTTP 408/stream disconnected 结束，归为
`INFRASTRUCTURE_FAILED`；branch 1/2/4/5 各产生三份完整 provider response，但均在正式
TypeCheck 前被严格 artifact parser 拒绝，终末诊断为 π refinement `auth_ref` 不是 J 的 `u*`
authorization handle。总预算为 12 次 logical/physical Lift、0 次 TypeCheck、0 次 Replay、0 次
Gate1/Gate2，winner 为空。四个正常耗尽分支是 branch-local `NO_CLOSURE`，但整个 fan-out 含一个
基础设施失败，不能把它概括成五分支全部正常耗尽。

该 run 从五分支目录初始化到汇总约 25 分 46 秒。当前 rejected-response evidence 保存了完整响应，
但没有逐请求 usage/elapsed receipt，因此不能从仓库证据反推出每个并发调用的精确排队与推理时间；
CPA 侧观测应与早期单链 receipt 分开解释。更重要的是，12 份高成本响应仍未交付可进入 TypeCheck
的候选，说明继续堆叠 one-shot prompt 或放宽 parser 不是合适的系统改进。Task 04 据此把 Lift
替换为一次一候选、可按需读取 schema/通用文档的有界 mini-swe worker；任何外部 TypeCheck/Gate1
反馈仍严格开启下一轮 worker，不能藏在同一轮内部修正。

本阶段 checkpoint 前已再次验证：manifest `28/28 resolved`、六个 representative dry-run 均为
`ready`；Phase B checkpoint 的全量回归为 `793 passed, 4 skipped`，Phase C fan-out checkpoint
更新为 `801 passed, 4 skipped`。接下来用可用 provider 配置从全新 run
继续六项外层顺序纵切。Task 03 在 dyt 获得完整 closure、且六项各自收敛为结构化 evidence 前
仍保持“执行中”；当前先执行 Task 04，完成后从新的 `fanout-002` 恢复。
