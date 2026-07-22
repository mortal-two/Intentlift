# Task 02：真实 SCRC Exact / Masked / Reduction Canary

> 状态：执行中
>
> 类型：新门禁系统的真实 source-aware Lift / source-blind Replay canary；不是 28-kernel 批量或
> 性能优化任务。

## 上游条件

Task 01 已完成 canonical `(J,C,pi)`、dependent TypeCheck、typed ReplayPacket、Gate1 observer、
外部 Q 三边 Gate2 与唯一 `SCRCGateRunner`，最终回归为 `701 passed, 4 skipped`。本任务不再重写
这些设计，而是用真实 provider 和真实 Triton compiler 检查它们能否承载最小代表性 Source。

上位依据仍是四份核心文档与活动 spec：

1. [C：所有权、绑定、消费与三方验证](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
2. [π：依赖类型实现见证与 Replay 安全](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)
3. [Lift、Replay、Gate1、Gate2 执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)
4. [SCRC 最终数学与 claim](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)

## 目标与顺序

按能力从小到大依次运行三个真实 Triton canary：

1. `exact_vector_add`：固定 extent、无边界 mask 的 exact elementwise；
2. `masked_copy`：runtime extent、masked load/store 与 fill；
3. `masked_row_square_sum`：program row、masked access、乘法与 reduction。

先完成并解释 exact，才能进入 masked；先完成并解释 masked，才能进入 reduction。任何失败都先按
当前协议结束该次 canary，区分：

```text
LIFT_RESPONSE_REJECTED
TYPECHECK_FAILED
NO_CLOSURE
GATE2_FAILED
INFRASTRUCTURE_FAILED
```

若是通用 schema、TypeCheck、observer 或 adapter 能力不足，结束本次运行后修改系统并重新开始；
不把 Gate2 结果回流，不给 Replay diff，不 patch 旧 Replay。

## 固定边界

- provider 使用本地已配置的 `gpt-5.6-sol` Responses API；三类 canary 使用同一公开 prompt、
  TypeCheck、observer profile、Q 和最多三次正式 Gate1 轮次。
- proof artifact 只有 `(J,C,pi)`；真实 Source、Harness、provider receipt 与报告只是运行 sidecar。
- Replay 每轮由新 writer/attempt adapter 生成，输入只有 `J + pi + MaskC(C) + U_f`。
- Source/Replay 使用同一 `HarnessBindingInstance(c0)`；Gate2 suite 必须显式绑定同一 c0。
- Gate2 输入只在 Gate1 通过后构造，并保存 Source–Replay、Source–J、Replay–J 三边。
- masked-J 默认作为独立诊断；其结果不改变 `main_status`。
- 运行目录使用普通顺序编号 `run-001`、`run-002`；不增加 hash/commitment/epoch/freeze 有效性门。

## 执行记录

### `exact_vector_add / run-001`

- 真实 `gpt-5.6-sol/xhigh` Lift 完成一次物理调用，并已进入 TypeCheck 后的 Gate1 Source compile；
- terminal：`INFRASTRUCTURE_FAILED`，没有形成 `NO_CLOSURE` 或 Gate2 结论；
- 根因：canary Harness 把 canonical dtype `f32/i32` 直接交给 Triton compile worker；该 worker 的
  runtime argument spelling 要求 `float32/int32`，因此 Source 在 TTIR 之前失败；
- 同时发现基础设施异常路径只保存 failure 摘要，未保存内存中已有的 Lift artifact/response。

已在结束 run-001 后做两项通用修正：

1. canary Harness 在机械边界显式映射 canonical dtype 到 runtime dtype，并新增真实 CUDA/TTIR 回归；
2. `SCRCInfrastructureError` 携带普通 partial evidence，CLI 在异常时保存已有 Lift artifact、receipt、
   response 与 Gate1 diagnostic；这些记录不改变认证规则。

修正后的 canary/runner 定向测试：`15 passed`。下一次运行从新的 `run-002` 开始，不覆盖或重判
run-001。

### `exact_vector_add / run-002`

- 三轮 Lift 均通过 dependent TypeCheck；前两轮 Gate1 分别因 Replay 编译失败、结构不闭合而结束，
  第三轮 `replay_closed`；真实预算为 3 次 Lift、3 次 normal Replay、1 次 masked-J Replay；
- Gate2 Source 与 Replay 均成功执行，Source–Replay 在 external Q 的 exact relation 下误差为 0；
- terminal：`GATE2_FAILED`，case 子状态为 `reference_unsupported`，不是数值
  `pairwise_diverged`；具体原因是 scalar RefEval 一律拒绝 J 中合法的 load/store access relations，
  因而没有产生 `O_J`，Source–J 与 Replay–J 未被伪造为比较失败；
- 按协议，本次运行到此结束，Gate2 结果没有返回 Lift，也没有 patch 第三轮 Replay。

结束 run-002 后新增通用 `dense_functional_v1` RefEval consumer：只接受 direct dense
`[axis0, ...]`、与 J shape 一致的可选 full-domain、完整 logical input/output 覆盖，以及一一对应的
load/read、store/write、J node operand/result 与 output mapping。predicate/fill、affine/gather、
update、alias 和 mutable storage 继续 fail closed。正例、domain mismatch 与 predicated access
负例均已加入测试。

用 run-002 已选定的第三轮 `(J,C,pi,Replay)` 离线复验新 consumer（不重写历史 verdict）：
Source–Replay、Source–J、Replay–J 三条 exact 边均通过、最大误差均为 0。下一步必须从全新的
`run-003` 重新走完整主流程，才能形成新的认证结论。

### `exact_vector_add / run-003`

- terminal：`SCRC_CERTIFIED`；这是修复后的全新主流程，不复用 run-002 候选或离线 verdict；
- 两轮 Lift 均通过 TypeCheck；第一轮 Gate1 不闭合，第二轮 `replay_closed`，因此该样本为
  `closure@2`，accepted attempt 为 2；
- Gate2 在实际 `c0={c0:128}` 上由 `dense_functional_v1` RefEval 产生 `O_J`，Source–Replay、
  Source–J、Replay–J 三条 external-Q exact 边全部通过，三个最大绝对误差均为 0；
- 调用预算：2 次 Lift、2 次 normal Replay、1 次 masked-J Replay、1 次 Source/Replay/RefEval
  Gate2 execution；Gate2 case 数为 1；
- `j_necessity_status=not_proved`：masked-J 也 hard-close。该结果只说明本次 topology 下没有证明
  J 的操作性必要，不改变主状态 `SCRC_CERTIFIED`。

至此 exact canary 已明确完成。下一步按既定顺序进入全新的 `masked_copy` 运行；不从 exact 结果
外推 masked predicate/fill 已被 RefEval 支持。

### `masked_copy / run-001`

- terminal：`NO_CLOSURE`，但不是三轮 Gate1 结构不闭合；三次 provider 响应均在 TypeCheck 前被
  anonymous Lift contract 拒绝，正式 Gate1 round 为 0；
- 三次拒绝的诊断完全相同：`J tensors must use declaration-order handles t0, t1, ...`。该 source 的
  参数顺序为 input pointer `p0`、output pointer `p1`、runtime extent scalar `p2`，模型把 `t*`
  后缀与 `p*` 位置混用，没有按 intent inputs 后接 outputs 形成独立连续 tensor sequence；
- 真实预算为 3 次 Lift/3 次物理模型调用，TypeCheck、Replay、Gate1 Source compile、Gate2 与
  masked-J 均为 0；因此不能从本次运行推断 predicate/fill 的 RefEval 状态；
- 同一次运行没有修改 prompt，也没有把错误响应送入 Replay；pre-Replay rejection 上限耗尽后结束。

本次还暴露了证据保存缺口：`SCRCLiftError` 内存中含拒绝 response/receipt，但
`ResponsesSCRCLiftAdapter` 只把 diagnostic 写入 record，所以 run-001 只能保存三条错误字符串，
无法事后检查具体错误 JSON。结束 run-001 后已做两项通用修正：

1. prompt 明确 J tensor 是独立于 `p*` 位置、按 inputs 后接 outputs 的 gap-free sequence，并给出
   pointer/scalar 混排的机械示例；校验诊断同时返回 expected/actual handles；
2. rejected Lift record 保留 provider response、receipt、usage、耗时与普通 trace，并由 CLI 落盘；
   这些 sidecar 不进入 TypeCheck、Replay 或 verdict。

相应适配器与落盘回归已通过。下一次从新的 `masked_copy/run-002` 开始，不覆盖 run-001，也不把
run-001 的三个拒绝当作正式 closure round。

### `masked_copy / run-002`

- terminal 仍为 `NO_CLOSURE`，预算仍是 3 次 Lift、0 次 TypeCheck/Replay/正式 Gate1/Gate2；
- 新 sidecar 已完整保存三份匿名响应和 receipt，确认 run-001 的 tensor 编号问题已经解决：三份均为
  input `t0`、output `t1`，signature 也正确使用 `p0→t0`、`p1→t1`、`p2→x0`；
- 第 1/3 份把 tensor shape 特化为当前 runtime value `[1000]`，同时让 scalar `p2` 引用不存在的
  symbolic `x0`，parser 因 unknown logical handle 拒绝；
- 第 2 份正确保留 shape `[x0]`，但把 dimension `x0` 错当成 SSA definition 写入 function scope，
  portable scope 的 derived definitions 检查因此拒绝；
- 三次拒绝均发生在 TypeCheck 前，没有进入 Replay。receipt 显示每个逻辑 Lift 各对应一次物理调用，
  response/usage/耗时均已保存，证明拒绝证据链修复有效。

结束 run-002 后，prompt 与活动 spec 进一步明确：runtime scalar 的当前 Harness value 不是 J 常量；
只要它控制 logical extent/index/predicate/result，就必须以 symbolic dimension 出现在相关 tensor
shape 中，且 dimension handle 不得进入 SSA scope definitions/captures 或 storage。对应 prompt 回归
已加入。下一次从新的 `masked_copy/run-003` 开始。

## 实施内容

### 1. 可复现真实入口

新增一个薄的 canary CLI/catalog，复用生产 `ResponsesSCRCLiftAdapter`、每轮 fresh
`ResponsesReplayWriter`、`TritonCompiler`、`TritonGate2ExecutionAdapter` 与 `SCRCGateRunner`。
CLI 只负责选择 canary、构造已知 Harness 坐标和保存 sidecar，不复制认证逻辑。

### 2. 普通运行记录

每次运行至少保存：

```text
source.py
request.json                 # canary、c0、输入模式、模型/协议名；无 secret
attempt-N/artifact.json
attempt-N/replay.py
attempt-N/gate1.json
gate2.json                   # 若进入 Gate2
report.json
SUMMARY.md
```

provider request/response 的普通 receipt 可以记录模型、token、耗时与 response id；API key、授权头和
私有配置不得写入。已有 digest 若由 provider client 产生，只是 trace metadata，不参与判断。

### 3. 执行纪律

- TypeCheck 前拒绝不计正式 Gate1 round，但计真实 Lift/provider 调用并受 pre-Replay rejection 上限；
- 每个正式 round 完整 relift、完整 TypeCheck、新 Replay、新 Gate1；
- 最多三次正式 Gate1 round；三次不闭合为 `NO_CLOSURE`；
- Gate2 通过才是 `SCRC_CERTIFIED`；失败后该次运行立即结束；
- 不因 canary 失败修改论文 claim，先定位系统能力缺口并以通用负例/回归修正。

## 验收条件

1. canary CLI 的离线 catalog、Harness、保存逻辑和调用顺序有单测；
2. exact canary 至少完成一次真实 provider Lift，并给出明确 terminal 结果；
3. 若 exact 暴露通用缺口，修复有最小负例和全量回归，再重新运行；
4. exact 明确后依次执行 masked 与 reduction，不并发污染判断；
5. Task 02 最终汇总真实 TypeCheck、closure@1/2/3、Gate2、调用预算与失败分类；
6. Task 02 不外推到 28-kernel 成功率或性能结论。

## 后续

三个 canary 形成稳定证据后，再依据真实缺口创建 Task 03：扩展代表性 Kernel 范围、建立批量输入
清单，并逐步接回 28 个高性能 Kernel；性能工作仍在更后面的独立任务中处理。
