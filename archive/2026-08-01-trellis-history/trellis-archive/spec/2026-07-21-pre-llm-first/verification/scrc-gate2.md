# SCRC Gate2：Source-anchored 重放／标量 Lowering 双差分

## 定位

Gate2 只接收 Gate1 已经硬闭合的最终 artifact。它在同一批 canonical 输入上执行
三个真实程序：

```text
O_src = captured source kernel -> frontend runtime -> execute
O_rep = final accepted J -> deterministic replay executable -> execute
O_sca = final accepted J -> typed IntentProgram
                         -> Intent MLIR -> linalg/loops -> LLVM CPU -> execute
```

活动执行协议为 `source_replay_scalar_differential_v3`，标量路径协议为
`intent_program_scalar_cpu_mlir_v1`。Gate2 有且只有两个必需的、以 source 为锚的
动态差分：

```text
source <-> replay(final J)
source <-> scalar-lower(IntentProgram(final J))
```

Gate1 的 source/replay TTIR hard structural closure 与 Gate2 的 source/replay runtime
differential 是两个不同证据，前者不能替代后者。`replay <-> scalar` 可以派生为诊断，
但不能单独形成 Gate2 verdict，也不能替代任一 source-anchored comparison。历史
`source_replay_dynamic_equivalence_v1` 和 `final_j_replay_scalar_differential_v2` 报告
均是 legacy evidence，不得进入当前统计。

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
  -> captured source identity / launch contract
  -> replay executable bound to the final artifact digest
  -> IntentProgram J / contract / program digest
  -> ScalarLoweringSpec digest
```

`Gate2Case.replay_artifact_digest` 必须等于 scalar request 的
`final_artifact_digest`。可执行 scalar spec 的 input/output bindings 必须与 typed
`IntentProgram` 的 ABI 完全一致；最终 artifact 改变后，旧 replay、旧 IntentProgram
或旧 scalar spec 一律不能复用。

typed `IntentProgram` handoff、predicate parser 或 scalar shape binding 失败不是 Gate0、
Gate1.5 或 Gate2 之前的新准入。它必须作为当前 Gate2 case 的 scalar leg unavailable
证据保存，同时仍执行并保留 `source_replay`。只有三腿和两个比较都完整时才能 PASS。

## 裁决

Gate2 分别形成逐输出比较 `source_replay` 与 `source_scalar`：

| 条件 | 状态 | 统计语义 |
|---|---|---|
| 三腿成功，两个比较都通过 | `gate2_pass` | 完整双 comparison，PASS |
| 只有 source/replay 不一致 | `replay_diverged` | 完整双 comparison，语义 non-PASS |
| 只有 source/scalar 不一致 | `scalar_diverged` | 完整双 comparison，语义 non-PASS |
| 两个 source-anchored 比较都不一致 | `replay_and_scalar_diverged` | 完整双 comparison，语义 non-PASS |
| typed handoff、scalar spec 或诚实 scalar lower 不可用 | `scalar_unsupported` | 保留 source/replay；整体 coverage gap |
| 任一实际执行发生准备、编译、同步或回读失败 | `execution_failed` | 保留已完成比较；整体 coverage gap |

`scalar_unsupported` 不得回退到 NumPy/JSON interpreter、source runtime、手写 CUDA
或 production backend。没有真实 `source_replay` 和 `source_scalar` 两个比较对象、
任一腿 `ok != true`、input digest 缺失、零 case 或零 comparison evidence 时，均不能
产生 PASS。scalar 不可用也不得删除或覆盖已经取得的 source/replay 证据。

Gate2 PASS 是有限测试域证据，不是全域形式证明；它也不能补偿 Gate1 的任何 hard
mismatch。为了让样本通过，只能补齐 J/IntentProgram/标量 lowering 的真实表达与执行
能力，不能删除 comparison、扩大输入排除域或把 unsupported 当作通过。

## 执行隔离

- source、replay 与 scalar 从同一份 canonical NumPy 输入各自构造运行时值，避免共享
  可变 output/inout buffer；
- source runner 执行被 capture identity 与 launch contract 绑定的原 Kernel；它是正式
  Gate2 verdict leg，不是 production backend conformance；
- replay runner 只执行与最终 Gate1 artifact 对齐的 replay，不在 Gate2 内重新生成或
  修复 J/replay；
- scalar runner 只消费 typed `IntentProgram`、显式 dims 与 ABI bindings；未知 op、
  unresolved B obligation、非法 broadcast/scope 或缺失 output 必须 fail closed；
- scalar 的规范路径必须经过项目真实 Intent MLIR lowering 与 LLVM CPU execution，
  不得用 kernel-name 分支、手写期望值或解释器伪装；
- scalar module 只绑定 typed IntentProgram、声明 dtype/shape 与 dims specialization，
  runtime tensor 值不得序列化为 `dense` 常量或进入编译 cache key；所有逻辑输出必须
  通过一个 `llvm.emit_c_interface` ranked-memref entry 在一次 native call 中产生；
- scalar native code 在隔离 worker 中执行；signal、timeout、ABI 校验失败、编译／链接
  失败均是显式 execution failure，不得降级为数值 mismatch、unsupported 或 fallback；
- 三腿异常分别保存为 `ExecutionResult`，不得吞掉异常后制造空输出通过；
- replay Triton launch 使用 `triton_name_keyed_complete_abi_v1`：runtime tensor/scalar
  按完整 Python ABI 名称绑定，signature 中间的 constexpr 从冻结 C.schedule 按名补入，
  compiler-only slot 仍作为 launch option；未知、缺失、重名或 runtime/schedule overlap
  都是 execution failure，不能用压缩后的位置参数跨过 constexpr；
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

`python/intentir/verify/gate2/inputs.py` 提供确定性输入生成。每个 case 的同一 value
mapping 必须复制给三腿，并记录按 dtype、
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
`execution_protocol=source_replay_scalar_differential_v3`。scalar result 必须携带
`protocol=intent_program_scalar_cpu_mlir_v1` 以及 final artifact、Gate1 J、
IntentProgram 和 scalar spec digest。scalar result 还应保存
`intentir_cpu_execution_evidence_v1` provenance：ranked-memref ABI、静态 scalar module、
lowering pipeline、toolchain、cache key 与 shared object digest，以及一次调用／隔离执行
记录。cache path、runtime pointer 不得进入报告；`cache_hit` 只是非确定 telemetry，
不得进入任何 semantic evidence digest、comparator、tolerance、supported 分类或 verdict。

活动汇总 schema 为 `scrc_gate1_gate2_conditional_v5`：

- 只有 Gate1 `REPLAY_CLOSED` 的样本有资格形成 Gate2 observation；
- 只有协议匹配、三腿均成功、input digest 合法且实际 `source_replay` 与
  `source_scalar` 都完成的四种语义终态进入完成统计；
- `scalar_unsupported`、`execution_failed`、setup failure、协议不匹配与未运行分别
  保留为 attempted-but-incomplete 或 unattempted coverage；
- case id 必须非空且唯一，`status_counts` 与顶层 `passed` 必须能由逐 case evidence
  精确重建；任何矛盾都使 suite incomplete。

历史 source/replay v1、replay/scalar v2 或旧三方报告即使也使用 `gate2_pass` 字符串，
也不得静默升级为 v3 PASS。

## 活动代码边界

活动入口为 `python/intentir/verify/gate2/`：

- `triton.py`：最终 replay executable 的隔离执行；
- `scalar.py`：digest-bound `ScalarLoweringSpec` 与真实 scalar CPU MLIR adapter；
- `compare.py`：逐输出 exact/tolerant 比较；
- `inputs.py`：确定性输入矩阵；
- `model.py`、`orchestrate.py`：三腿／双差分结果模型、身份检查与严格 suite 聚合；
- `metrics.py`：v3 协议感知的 Gate1/Gate2 条件统计。

活动包不导出 `Lower2C*`、`CudaLowering*`、production backend selector 或旧
`GateIntentBundle`；也不提供 rejected-sample Gate2 diagnostic、causal intervention
或 target-backend fallback。旧 source/replay/production-lowering
三方规范与实现见
[归档说明](back/gate2_production_lowering_v1/scrc-gate2-three-way.md)，只能用于历史
报告复核。
