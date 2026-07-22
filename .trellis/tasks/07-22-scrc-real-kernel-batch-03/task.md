# Task 03：28 项真实 Kernel Manifest、顺序 Batch 与代表性纵切

> 状态：执行中
>
> 类型：SCRC 系统运行能力扩展；不是性能实验、论文统计或 28 项一次性 provider 全跑。

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
8. 六项顺序运行，保存每项完整 terminal/failure evidence 与调用预算；
9. 当前支持域若暴露通用缺口，先结束 run，再增加通用 schema/consumer/prompt 回归并新跑；
10. frontier unsupported 必须准确落在具体阶段，不能归成未知异常；
11. Task 03 完成时运行全量测试，工作区无 secret 或无主 sidecar。

## 本任务明确不做

- 不一次性调用 provider 跑完 28 项；
- 不做性能计时、排名、beat/parity 结论或 retune；
- 不为 softmax/RMSNorm/matmul 写 per-kernel RefEval 特例；
- 不增加一般集合值并发、atomic、scatter/update、alias/inout 支持；
- 不建设异步、分布式、复杂 resume 或 hash/commitment 系统；
- 不让旧 Replay、旧 conversation 或其他 Kernel 的 Gate1 diff 跨 case 传播。

## 后续

Task 03 完成后，Task 04 依据六项结构化结果只选择一个可泛化语义能力扩展，例如 compositional
row reduction、softmax 组合或 contraction。达到门禁覆盖与达到性能目标继续作为不同任务处理。

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
