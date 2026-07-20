# SCRC Gate2：Source／Replay 双腿动态等价

## 定位

Gate2 在同一批 canonical 输入上分别实际执行原 Kernel 与 Gate1 生成的重放
Kernel：

```text
O_src = 原 Kernel 实际执行
O_rep = 与最终 Gate1 artifact 精确对齐的 replay Kernel 实际执行
```

当前 Gate2 的执行协议为
`source_replay_dynamic_equivalence_v1`。IntentIR production lowering、归档 JSON
解释器、手写 MLIR 与任何 backend benchmark 都不参与 Gate2 verdict。

2026-07-20 前使用的 source/replay/production-lowering 三方方案已经归档到
[back/gate2_production_lowering_v1/scrc-gate2-three-way.md](back/gate2_production_lowering_v1/scrc-gate2-three-way.md)，
对应代码位于
`python/intentir/verify/back/gate2_production_lowering_v1/`。归档实现只能用于历史
报告复核，不得被活动 Gate2、live/held-out 入口或正式 campaign 导入。

## 裁决

Gate2 只形成一组逐输出比较 `source_replay`：

| 条件 | 状态 | 含义 |
|---|---|---|
| source、replay 均执行成功且全部输出通过比较 | `GATE2_PASS` | 在本次声明输入域内观察到动态一致 |
| 两腿均执行成功，但至少一个输出不一致 | `REPLAY_DIVERGED` | source 与 replay 的行为不一致 |
| source 或 replay 任一腿未完成 | `EXECUTION_FAILED` | 执行证据不完整，不计作语义 non-pass |

`REPLAY_DIVERGED` 是沿用的 operational 状态名，只表示
`source_replay.passed == false`。在没有第三条独立执行腿时，它不能机械区分问题来自
J、replay writer、编译器适配或其他环节；根因必须回到 Gate1 evidence 与单独诊断中
分析。

Gate2 PASS 同样不是全域证明，也不证明 IntentIR 能被某个 production backend 正确
lower。它只说明在已记录的 shape/value/seed/tolerance 矩阵内，原 Kernel 与该 replay
的实际输出一致。

## 执行隔离

- `TritonRunner` 对 source 与 replay 分别加载独立 module、分别 launch；
- 两腿从同一份 canonical NumPy 输入分别构造运行时 tensor，避免共享可变输出；
- 输出统一复制回 NumPy 后比较；
- source 或 replay 的加载、编译、launch、同步或输出回读异常统一保存为各自
  `ExecutionResult`，最终状态为 `EXECUTION_FAILED`；
- Gate2 不调用 backend lowering 作为 fallback，也不因 backend 支持情况决定某个
  replay 是否有资格进入动态比较。

## 输入矩阵

Gate2 继续覆盖三个正交维度：

- shape：1、block−1、block、block+1、非整除、多 tile，以及 case 明确声明的
  fixed-C profile；
- value：normal、uniform、positive、near-one、large-dynamic、IEEE special、
  integer boundary 等合法输入域；
- seed：matrix 默认使用主 seed 及其后两个整数，也可显式声明唯一 seed 集合。

`python/intentir/verify/gate2/inputs.py` 提供确定性输入生成。
`Gate2Suite` 必须运行所有 case；只有每个 case 都是 `GATE2_PASS` 时 suite 才通过，
不得用平均误差、多数票或跳过失败 case 掩盖边界不一致。

## 比较协议

逐输出比较协议保持 `exact_integer_tolerant_float_v2`：

- bool 与整数逐元素精确比较，用户提供的浮点 tolerance 不得放宽它们；
- 浮点按每个 output 的 `rtol`、`atol`、`equal_nan` 比较；
- shape 不一致、输出缺失或 NaN 位置不一致均明确失败；
- 每个 output 保存 pass、最大绝对误差与失败细节。

## Artifact／Replay 对齐

Gate2 使用的 replay 必须与最终 Gate1 artifact canonical digest 精确对齐。最终
artifact 改变但没有形成新 executable candidate 时，Gate2 记录 coverage gap，禁止
回退到旧 artifact 的 replay。该约束只保证 evidence 身份一致，不改变 Gate1 verdict。

Gate1 rejected sample 可以在存在与最终 artifact 对齐的已编译 replay 时运行独立
Gate2 diagnostic；该结果不得覆盖 Gate1 的结构或反泄漏拒绝。

## 报告与统计

单 case 与 suite evidence 必须携带
`execution_protocol=source_replay_dynamic_equivalence_v1`。旧三方报告即使也使用
`gate2_pass` 或 `replay_diverged` 字符串，只要没有该协议，就不得进入当前统计。

活动成对汇总 schema 为 `scrc_gate1_gate2_conditional_v3`，记录：

- Gate1 原始状态及是否 `REPLAY_CLOSED`；
- Gate2 evidence kind、execution protocol、是否尝试；
- source/replay comparison 是否完整完成；
- 单 case 或 suite 的原始状态、PASS/non-PASS 与 completion coverage。

只有协议匹配且完成 source/replay 比较的 `GATE2_PASS`、`REPLAY_DIVERGED` 进入
Gate1/Gate2 四象限。`EXECUTION_FAILED`、setup failure、协议不匹配和未运行均保留为
coverage gap，不得伪装成语义 non-pass。

## 活动代码边界

活动入口为 `python/intentir/verify/gate2/`：

- `triton.py`：source/replay 独立执行；
- `compare.py`：逐输出 exact/tolerant 比较；
- `inputs.py`：确定性输入矩阵；
- `model.py`、`orchestrate.py`：双腿结果模型与严格 suite 聚合；
- `metrics.py`：协议感知的 Gate1/Gate2 条件统计。

活动包不导出 `Lower2C*`、`CudaLowering*` 或 `GateIntentBundle`，也不存在
`lowering_runner`、`Gate2Case.lowering`、`source_lowering`、`replay_lowering` 等主线
字段。
