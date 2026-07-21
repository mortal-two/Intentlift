# 28 Real Kernels 04：正式 LLM Lift/Replay/Lowering/Performance 批次

> 状态：进行中
> 优先级：P0，Typed Relations 03 后的首个正式系统批次
> 流程：普通 Markdown 工作项；不运行 Trellis workflow

## 最终目标

用 28 个固定 digest 的真实高性能 Kernel，完成并保存：

```text
real source
  -> 1× LLM Lift: canonical J + pure C + source realization π
  -> 1× source-blind LLM Replay
  -> Gate1 normalized compiler-topology closure
  -> Gate2 source/replay/E_J relation validation
  -> deterministic J -> Intent MLIR -> CUDA lowering
  -> independent backend conformance + reproducible performance evidence
```

正式结果不得由 hand-written J、π、Replay、Intent MLIR、kernel-name recognizer、RuleCard、
composer 或 source-specific deterministic patch 代替。历史 PK/benchmark 只用于环境与 backend
回归，不能计作 28 个 LLM artifact。

## 里程碑 0：先冻结 manifest，不先消耗 LLM 调用

建立唯一 `test/real_llm_batch_28.json`，每项必须固定：

- `case_id`、真实 repo/path、symbol、source SHA-256、frontend、license/source revision；
- specialization、logical inputs/outputs/inout、shape/dtype envelope、C 值；
- Gate1 observer level；Triton 强路径默认 Level-2 normalized TTIR hard closure；
- Gate2 deterministic/random cases、seed commitment、J-owned result relation；
- CUDA target、correctness oracle、source baseline、benchmark protocol 与预先冻结的性能门槛；
- 预计需要的 IntentOp/executor/backend capability，不用 kernel family 名充当能力证明。

28 项按机制分层各 7 项，避免用大量同构 elementwise 凑数：

1. compositional expression、fusion、reduction；
2. copy/layout/gather/dynamic indexing；
3. contraction/attention/semantic recurrence；
4. logical update、duplicate/atomic realization、dynamic/observable state。

必须包含已审计锚点：LayerNorm mask-before-square、xFormers index-select、xFormers
scaled-index-add、Liger embedding backward、FlashAttention online recurrence、CuTe async copy
以及 typed racing-overwrite relation。manifest、环境 probe 和 source digest 全部通过后，才可
开始正式调用。

## 在线调用与时间预算

每个 primary artifact 固定：

```text
1 × source-aware Lift LLM
1 × source-blind normal Replay LLM
若 normal Gate1 closure：最多 1 × masked-J Replay 检测
Gate2 / lowering / benchmark：0 次 LLM
```

- masked-J 是唯一在线附加检测；其他 mutation/swap/dropout/multi-model 属离线论文消融；
- normal Replay 不看源码、source/replay diff 或旧候选；
- failure recovery 只能提交 fresh `(J,π,C)` 后重新 source-blind Replay，不做局部 repair；
- first-attempt 与 recovery 指标、调用数、token、墙钟必须分开保存，不能用 recovery 覆盖；
- 在 manifest 中冻结总调用上限；超预算不是通过，需保留未完成状态并解释。

## Gate1 与 Gate2

### Gate1

- source 与 Replay 使用同一冻结 specialization/C/compiler/observer；
- 参数名、SSA 名先 alpha-normalize；数据流、predicate/fill/effect、reduction 与 realization
  topology 严格闭合；
- π 由 Lift LLM 直接输出，只引用 J/C 匿名 handle，不复制 combiner、predicate、constant；
- Gate1 不用数值 Gate2、production backend 或 capability allowlist 授权 closure。

### Gate2

- 只接收 Gate1 `REPLAY_CLOSED` 的最终 artifact；
- source、Replay、E_J 使用 committed input pre-state 的独立副本；
- exact/tolerance/one-of-updates 全部从同一个 canonical J 读取；
- `E_J` 只能走正式 J-only scalar semantic path。若 CPU scatter、recurrence 或 observation
  lowering 尚未闭合，先以独立提交补 executor；不得使用 legacy NumPy interpreter；
- 28 项必须完成完整三腿，`SCALAR_UNSUPPORTED`、缺腿或 comparator incomplete 不计 PASS。

## Production lowering 与性能

production correctness 与 Gate1/Gate2 分开记录：

```text
Lower_cuda(J, C?, sm_120a) -> executable
```

- backend 直接消费完整 J；未实现 B relation 在 codegen 前 fail closed；
- 每项保存 J/C digest、Intent MLIR、selection/probe、binary hash、实际 B consumption；
- target 输出分别对 source 与 E_J/J relation 做 conformance，不能用“编译成功”代替；
- benchmark 使用同一 device、clock/power policy、warmup、同步、CUDA event、重复次数与输入；
- 保存 raw samples、median、分位数、有效带宽/FLOP/s（适用时）、source 与公开 baseline；
- 默认性能门槛为 target median latency 不高于真实 source 的 `1.10×`；无法使用该门槛的
  case 必须在 manifest 冻结替代效率门槛。运行后不得改门槛制造 PASS；
- correctness 失败、测量不完整或超过冻结性能门槛的 case 均使 Task04 保持未完成。

## 产物

每个 case 保存机器可审计 receipt：

- source identity 与匿名 compiler view；
- LLM provider/model/prompt/schema、调用次数/token/latency；
- canonical J/C/π、normal/masked replay packet 与源码；
- source/replay compiler artifact、observer、Gate1 diff/verdict；
- Gate2 input digest、三腿原始输出摘要、逐 relation verdict；
- target lowering/conformance/performance raw data；
- failure class 与 fresh-attempt lineage，不保存任何看答案后的 patch。

最终生成一份 28 行主表和按机制/observer/first-attempt 的统计；不把不同 observer level
混成一个无条件 Gate1 pass rate。

## 提交节奏

1. manifest + environment baseline 独立提交；
2. 每个必要 executor/backend 能力闭合独立提交；
3. 每完成一个 7-case 机制分层，保存 receipt 并提交；
4. 最终 28-case report/performance evidence 独立提交。

用户已有修改始终保留；不得用 reset/checkout 清除失败证据或历史 receipt。

## 验收标准

1. 28/28 均来自冻结的真实 source/symbol/digest，且不是手写 semantic artifact。
2. 28/28 有完整 primary Lift + source-blind Replay 调用账本。
3. 28/28 在声明 observer level 上 Gate1 hard-close；Triton 强路径必须 Level 2。
4. 28/28 完成 Gate2 三腿并按同一 J relation PASS。
5. 28/28 完成 CUDA backend conformance，完整消费 B。
6. 28/28 有可复现 raw performance evidence并达到预冻结门槛。
7. first-attempt、recovery、masked-J、墙钟/token/调用数统计完整且不混淆。
8. 不新增 composer、RuleCard、IntentProgram、LogicalABI、per-kernel deterministic rule。
9. 全量回归通过；任务归档，并按上述阶段提交。

## 当前第一步

只做 manifest 候选审计与环境 probe：从真实仓库解析 28 个精确 symbol/specialization，核对
source digest、可导入/可编译性、observer 支持、运行输入和基线计时。此阶段不调用 LLM，
也不实现缺口；先产出“可进入正式批次/明确阻塞”的逐 case 表，再冻结 Task04 manifest。

## 里程碑 0 实际进展（2026-07-22）

- `test/real_llm_batch_28.json` 已冻结 28 个 exact path/symbol/file digest/symbol
  digest，四个机制分层各 7 项；manifest canonical digest 为
  `12e2507433f23d6405cfa1dd0f1ff1169671a664822c07b0f5ecfbdc93570154`。
- source/repository/ABI/protocol/budget/1.10x performance threshold validator 已闭合；此阶段
  `online_llm_calls=0`。
- RTX 5090 D、Triton 3.6、Intent/MLIR/PTX toolchain 与 provider 配置可用。
- 新 compiler epoch 对 26/26 Triton source specialization 真实生成 TTIR/TTGIR；原始 digest
  保存在 `artifacts/real_llm_batch_28/preflight.json`。
- 两个 CuTe async-copy anchor 的源码 pin 已通过，但本机尚无 `cutlass.cute`，且系统尚无
  CuTe Level-2 observer；因此 overall readiness 是 26/28，正式 LLM 调用仍被 fail-closed。

下一步不是删掉 CuTe anchor 或把它冒充普通 copy，而是闭合通用 CuTe frontend/compiler
observer，再重新生成 28/28 preflight receipt。
