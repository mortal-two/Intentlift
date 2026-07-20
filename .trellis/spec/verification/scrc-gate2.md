# SCRC Gate2：最终 J 重放／标量 Lowering 动态差分

## 定位

Gate2 只接收 Gate1 已经硬闭合的最终 artifact。它在同一批 canonical 输入上执行
来自**同一个最终 J** 的两条独立路径：

```text
O_rep = final accepted J -> deterministic replay executable -> execute
O_sca = final accepted J -> typed IntentProgram
                         -> Intent MLIR -> linalg/loops -> LLVM CPU -> execute
```

活动执行协议为 `final_j_replay_scalar_differential_v2`，标量路径协议为
`intent_program_scalar_cpu_mlir_v1`。Gate2 PASS 只回答：在已记录的
shape/value/seed/tolerance 测试域内，最终 J 的重放实现与同一 J 的标量 MLIR
lowering 是否产生一致的可观察结果。

source/replay 的编译与 hard structural closure 属于 Gate1。原 source 的运行时结果若
另行采集，只能作为 Gate2 外部诊断 witness，不能替代标量腿，也不能改变 Gate2
verdict。历史 `source_replay_dynamic_equivalence_v1` 报告是 legacy evidence，不得
进入当前统计。

production CUDA/ROCm/RISC-V 等目标 backend lowering、target benchmark 与归档 JSON
解释器均不参与 Gate2 verdict。标量 CPU 路径是 verification/reference lower，不是
production target conformance；Gate2 通过不证明任一目标 backend 正确或高性能。

## 前置条件与不可绕过的身份链

活动 Gate2 只在 Gate1 状态为 `REPLAY_CLOSED` 后运行。Gate1 rejected、
`NO_CLOSURE`、`LEAKAGE_REJECTED`、预算耗尽或证据不完整的样本均不得进入 Gate2。
离线执行这些样本可以形成普通诊断，但不能命名为 Gate2、不能计入 Gate2 coverage，
也不能覆盖 Gate1 verdict。

每个 case 必须绑定以下 canonical 身份：

```text
final Gate1 artifact digest
  -> final Gate1 J digest
  -> IntentProgram J / contract / program digest
  -> ScalarLoweringSpec digest
  -> replay executable bound to the same final artifact digest
```

`Gate2Case.replay_artifact_digest` 必须等于
`ScalarLoweringSpec.final_artifact_digest`。scalar spec 的 input/output bindings 必须
与 typed `IntentProgram` 的 ABI 完全一致；最终 artifact 改变后，旧 replay、旧
IntentProgram 或旧 scalar spec 一律不能复用。digest 缺失、格式错误或不一致必须在
setup 阶段 fail closed。

## 裁决

Gate2 只形成逐输出比较 `replay_scalar`：

| 条件 | 状态 | 统计语义 |
|---|---|---|
| replay、scalar 均执行成功且全部输出通过 | `gate2_pass` | 完整 comparison，PASS |
| 两腿均执行成功，但至少一个输出不一致 | `replay_scalar_mismatch` | 完整 comparison，non-PASS |
| typed J 尚不能由标量 MLIR 路径诚实执行 | `scalar_unsupported` | coverage gap，不是语义 non-PASS |
| replay 或 scalar 的准备、编译、执行、同步或回读失败 | `execution_failed` | coverage gap，不是语义 non-PASS |

`scalar_unsupported` 不得回退到 NumPy/JSON interpreter、source runtime、手写 CUDA
或 production backend。没有真实 `replay_scalar` 比较对象、任一腿 `ok != true`、
input digest 缺失、零 case 或零 comparison evidence 时，均不能产生 PASS。

Gate2 PASS 是有限测试域证据，不是全域形式证明；它也不能补偿 Gate1 的任何 hard
mismatch。为了让样本通过，只能补齐 J/IntentProgram/标量 lowering 的真实表达与执行
能力，不能删除 comparison、扩大输入排除域或把 unsupported 当作通过。

## 执行隔离

- replay 与 scalar 从同一份 canonical NumPy 输入各自构造运行时值，避免共享可变
  output/inout buffer；
- replay runner 只执行与最终 Gate1 artifact 对齐的 replay，不在 Gate2 内重新生成或
  修复 J/replay；
- scalar runner 只消费 typed `IntentProgram`、显式 dims 与 ABI bindings；未知 op、
  unresolved B obligation、非法 broadcast/scope 或缺失 output 必须 fail closed；
- scalar 的规范路径必须经过项目真实 Intent MLIR lowering 与 LLVM CPU execution，
  不得用 kernel-name 分支、手写期望值或解释器伪装；
- replay/scalar 的异常分别保存为 `ExecutionResult`，不得吞掉异常后制造空输出通过；
- active Gate2 与 `backend/cpu` scalar 路径不得 import `intentir.backend.cuda`、ROCm
  或任一 production target registry，也没有 target fallback。

replay 可由外部 Triton runtime 执行；这是 replay executable 的宿主实现细节，不等于
Gate2 调用了 TianchenIR 的 CUDA production lowering。

## 输入矩阵

Gate2 覆盖三个正交维度：

- shape：1、block-1、block、block+1、非整除、多 tile，以及 case 明确声明且同时
  满足 source capture、replay ABI 与 typed J 合同的 profile；
- value：normal、uniform、positive、near-one、large-dynamic、IEEE special、
  integer boundary 等合法输入域；
- seed：matrix 默认使用主 seed 及其后两个整数，也可显式声明唯一 seed 集合。

`python/intentir/verify/gate2/inputs.py` 提供确定性输入生成。每个 case 记录按 dtype、
shape 与原始字节计算的 exact `input_digest`。`Gate2Suite` 必须运行所有声明 case；
只有每个 case 都为 `gate2_pass` 时 suite 才通过，不得使用平均误差、多数票、缩小矩阵
或跳过失败 case 掩盖边界不一致。

## 比较协议

逐输出比较协议为 `exact_integer_tolerant_float_v2`：

- bool 与整数逐元素精确比较，浮点 tolerance 不得放宽它们；
- 浮点按每个 output 的 `rtol`、`atol`、`equal_nan` 比较；
- output 与所有 observable inout 都必须进入绑定和比较；
- shape 不一致、输出缺失、dtype/NaN 位置不一致均明确失败；
- 每个 output 保存 pass、最大绝对误差与失败细节。

tolerance 是数值语义政策，不是过门工具。不得针对失败 kernel 临时扩大 tolerance；
任何修改都必须有 dtype/算法级依据，并由独立回归覆盖。

## 报告与统计

单 case 与 suite evidence 必须携带
`execution_protocol=final_j_replay_scalar_differential_v2`。scalar result 必须携带
`protocol=intent_program_scalar_cpu_mlir_v1` 以及 final artifact、Gate1 J、
IntentProgram 和 scalar spec digest。

活动汇总 schema 为 `scrc_gate1_gate2_conditional_v4`：

- 只有 Gate1 `REPLAY_CLOSED` 的样本有资格形成 Gate2 observation；
- 只有协议匹配、两腿均成功、input digest 合法且实际 `replay_scalar` comparison
  完成的 `gate2_pass`／`replay_scalar_mismatch` 进入完成统计；
- `scalar_unsupported`、`execution_failed`、setup failure、协议不匹配与未运行分别
  保留为 attempted-but-incomplete 或 unattempted coverage；
- case id 必须非空且唯一，`status_counts` 与顶层 `passed` 必须能由逐 case evidence
  精确重建；任何矛盾都使 suite incomplete。

历史 source/replay v1 或三方报告即使也使用 `gate2_pass` 字符串，也不得静默升级为
v2 PASS。

## 活动代码边界

活动入口为 `python/intentir/verify/gate2/`：

- `triton.py`：最终 replay executable 的隔离执行；
- `scalar.py`：digest-bound `ScalarLoweringSpec` 与真实 scalar CPU MLIR adapter；
- `compare.py`：逐输出 exact/tolerant 比较；
- `inputs.py`：确定性输入矩阵；
- `model.py`、`orchestrate.py`：双腿结果模型、身份检查与严格 suite 聚合；
- `metrics.py`：v2 协议感知的 Gate1/Gate2 条件统计。

活动包不导出 `Lower2C*`、`CudaLowering*`、production backend selector 或旧
`GateIntentBundle`；也不提供 source runtime、rejected-sample Gate2 diagnostic、
causal intervention 或 target-backend fallback。旧 source/replay/production-lowering
三方规范与实现见
[归档说明](back/gate2_production_lowering_v1/scrc-gate2-three-way.md)，只能用于历史
报告复核。
