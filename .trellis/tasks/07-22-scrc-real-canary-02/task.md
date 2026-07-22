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
