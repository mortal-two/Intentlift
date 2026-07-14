# SCRC Gate2：源／重放／Lowering 三方实跑

## 定位

只有 Gate1 `REPLAY_CLOSED` 的正式 artifact 进入 Gate2。Gate2 对同一批输入实际
执行三条独立腿：

```text
O_src = 原 Kernel 实际执行
O_rep = Gate1 重放 Kernel 实际执行
O_low = J 经项目 production lowering 实际执行
```

Gate2 不调用归档 JSON 解释器充当第三条腿。

## 当前 lower2c 原型

仓库已经存在真实链路：

```text
IntentIR JSON → intent MLIR → linalg/loops → LLVM → mlir-runner
```

新入口 `python/intentir/verify/gate2/lower2c.py` 直接复用这条链。当前名字沿用
“lower2c 原型”的任务称呼，但实际产物是 CPU LLVM JIT，不伪称已经输出 C 文本。
若未来确需字面意义的 C source，应另行论证并获批；当前 SCRC 只要求一条真实
production lowering 执行腿。

## 拆分执行

- `TritonRunner` 对 source 与 replay 分别加载独立模块、分别 launch；
- 两腿从同一份 canonical NumPy 输入各自构造 GPU tensor；
- `Lower2CRunner` 从同一输入绑定到 IntentIR 参数并执行 lowering；
- 三路输出统一回到 NumPy，再按 dtype/case 的 tolerance 比较。

机械归属：

| 两两一致关系 | Gate2 状态 | 含义 |
|---|---|---|
| 三方一致 | `GATE2_PASS` | 本测试域闭合 |
| 源≈重放，lower 偏离 | `LOWERING_DIVERGED` | lowering 问题 |
| 源≈lower，重放偏离 | `REPLAY_DIVERGED` | Gate1/replay 漏检 |
| 重放≈lower，源偏离 | `LIFT_DIVERGED` | J 与重放自洽地错 |
| 三对均不一致 | `ALL_DIVERGED` | 需按三腿错误继续定位 |

有限 tolerance 可能不满足传递性；出现“两对通过、一对失败”时返回
`INCONCLUSIVE`，不强行归属。

## 输入

沿用主方案的三个维度：

- shape：1、block−1、block、block+1、非整除、多 tile；
- value：随机、零/负值、大动态范围、需要时的 inf/NaN；
- seed：每组至少 3 个。

这些输入属于 Gate2，不参与 Gate1 TTIR 裁决。

`python/intentir/verify/gate2/inputs.py` 提供显式、确定性的 normal、uniform、
positive、near-one、large-dynamic 与 IEEE-special tensor 生成。真实中间-stage
Kernel 必须按其合法输入域选分布，例如 L2 final 的 partial sum 非负，prod final
用 near-one 避免普遍下溢。`InputCaseTemplate × seeds` 生成稳定 case id 与输入
metadata；`Gate2Suite` 完整运行所有 case，只有每个子 case 都是 `GATE2_PASS`
时 suite 才通过，不能用平均误差或多数票掩盖一个边界失败。

live/batch 的 `--gate2-matrix` 已接入这一严格聚合；未显式给 seed 时使用主 seed
及其后两个整数。matrix enable、shape/value profiles、seeds、distribution、scale
和 tolerance 全部进入 batch identity。2026-07-15 首个真实矩阵为 FlagGems
`min_kernel_2`：固定源 C `BLOCK_MID=64` 下，shape `{1,63,64}` × value
`{normal,uniform,large_dynamic,IEEE-special}` × 3 seeds，共 36 个 source/replay/
Lower2C case，全部 PASS 且三对最大误差均为 0。该矩阵仍是 Gate1-rejected 样本的
9.4 离线对照，不覆盖 Gate1 verdict。

## 当前实现

代码入口：`python/intentir/verify/gate2/`。

- source/replay 实际拆分执行已实现；
- production lowering 执行已实现；
- 三方 tolerance 比较与机械归属已实现；
- `Gate2Suite` 的严格全 case 聚合、逐 case metadata/status/error 保存已实现；
- `python/tests/test_gate2.py` 已真实跑通 add、row-sum、softmax、RMSNorm、matmul、
  FlagGems dot、FlagGems ones、FlagGems mean-final 以及 FlagGems
  max/min/prod/L2-final 的
  source/replay/lowering 三方；
- `test/scrc_live_add_e2e.py` 已进一步跑通真实 LLM lift + 真实 source-blind
  replay 后的完整三方 add：`N=1003, BLOCK=256`，source↔replay 最大误差 0，
  source/replay↔lowering 最大误差约 `5.01e-6`，状态 `GATE2_PASS`；
- 其余 op 能否进入 Gate2，由现有 A/B/C 表达和 production lowering 的真实
  支持情况决定；不引入“verify-ready op”概念。

2026-07-15 首批真实外部样本中，
`fg_dot/fg_addr/fg_ones/fg_sum/fg_mean/fg_max/fg_min/fg_prod/fg_vector_l2` 都因
masked-J 在 Gate1 被拒；按方案的 rejected-sample 诊断例外离线执行 Gate2 后，
九例均三方 PASS。该结果只进入 `P(Gate2 pass | Gate1 rejected)`，不覆盖 Gate1。

为这些真实样本补齐的 production Lower2C 接缝包括：

- 无输入、symbolic shape 的 f32 const 输出；
- CPU main 对 rank-0/任意 rank i1/i8/i16/i32/i64 输入发射 typed dense constant；
- tracked integer/i1 与 f32/int 之间发射真实 `intent.cast`；
- 没有 surviving axis 的 full reduction 默认得到 rank-0 结果；C++
  `ReduceLowering` 使用 leading-unit parallel reduction 后的
  collapse→extract→`tensor.from_elements` 生成 rank-0 tensor，源码、重建后的
  `intent-opt` 与 MLIR regression 已一致。
- Lower2C 输出不再经有限精度的 `printMemrefF32` 十进制文本读回；CPU wrapper
  使用可 bufferize 的逐元素 f32→i32 bitcast 打印精确 IEEE-754 位模式，再恢复
  f32。max/min 的零容差三方比较因此不会被文本舍入伪造 divergence。
- Lower2C 输入 dense literal 同样改为 f32 十六进制位模式，因而能精确注入普通
  数值、`-0.0`、`±inf` 与 NaN；不再用非法的 `inf/nan` 文本或 8 位十进制近似。
- 通用 `ReduceLowering` 的 max/min combiner 使用 `arith.maxnumf/minnumf`，与
  Triton TTIR 和项目 GPU row-reduction 的 NaN 行为一致。真实 max 矩阵最初曾
  机械归属为 `LOWERING_DIVERGED`（source/replay=`+inf`，lowering=NaN），修复后
  normal/uniform/large-dynamic/IEEE-special × 3 seeds 全部通过；min 的
  IEEE-special 回归也输出 `-inf`。
- `reduce(fn=prod)`、`reduce(max/min)` 与 `sqrt(reduce(sum))` 的 rank-0 final-stage
  已经在真实 FlagGems source/replay/lowering 三腿上执行通过。

当前明确未闭合的既有能力也必须原样报告：通用 CPU Lower2C 尚无 iota/gather
lowering，integer output 仍没有通用 read-back；因此 `fg_arange/fg_trace` 继续
queued。`fg_eye` 还缺合法的二维 index/predicate shape realization。本轮没有为
这些 queued 样本擅自增加新 op 或回退解释器。

若 lower 当前不支持某个真实 LLM 提取 artifact，Gate2 返回
`LOWERING_UNSUPPORTED`。这是一条需要补 A/B/C 或 lowering 的具体失败证据，
不能退回解释器后伪报通过。
