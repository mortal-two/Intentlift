# SCRC schema-30：fixed-C 全 Family Relift 与 Gate1/Gate2 四象限

日期：2026-07-18
协议：`gate1_fixed_c_family_diff_guided_pi_relift_v10`
held-out identity：schema 30

本轮没有运行 Trellis task/workflow；只更新现有实现、测试、报告和 Trellis spec。
没有修改 π 拓扑、IntentIR op、C.strategy family、RuleCard 或 Gate2 verdict。

## 1. 为什么进入 schema 30

schema-29 已完成 scope-aware TTIR SSA、单 specialization 的确定性 stagnation、首次
Gate1 hard diff 指导下一轮 Lift，以及结构化 `NO_CLOSURE`。但 fixed-C family 仍在主
Gate1 循环之后执行 `same_replay_diagnostic_v1`：secondary profile 失败只能进入报告，
不能阻止 primary closure，也不能指导下一轮 Lift。

这与既定阶段 B 的要求不一致：一份 J/C/π、一份 replay、固定 C，必须覆盖完整 runtime
profile family；任一 profile 失败都要完成全 family sweep 后统一回流 Lift。

## 2. 当前 fixed-C family 正式 normal 流程

schema-30 的 normal replay round 为：

```text
source-aware Lift 产生一份 J/C/π
  → writer 产生一份 source-blind replay
  → primary source/replay compile + hard compare
  → 复用 primary compile observation
  → 同一 replay 对其余 fixed-C profiles 完成 source/replay compile + hard compare
  → all profiles hard-close：进入现有 anti-leakage
  → 任一 profile 失败：聚合一次 family diff，调用一次 source-aware Relift
  → 下一 artifact round 重新生成 replay 并重新检查全部 profiles
```

约束如下：

- writer 不按 profile 重采样；每个 artifact round 只有一份 replay；
- 不在第一个 secondary failure 后提前退出，完整 sweep 后才 Relift；
- 不存在 profile-local replay repair；
- 不使用平均 score、多数票或模糊比例；
- primary compile observation 被 family runner 直接复用，避免重复编译和可能的观测漂移；
- source specialization 编译失败归为 reviewed contract/harness failure，不伪装成 J 缺口；
- provider 在 replay candidate 形成前失败时仍没有 family diff，不调用 Relift。

活动 policy：

- family execution：`fixed_c_family_full_sweep_relift_v1`；
- family hard diff：`exact_ordered_fixed_c_family_hard_diff_v1`；
- family semantic guidance：`deterministic_fixed_c_family_hard_diff_guidance_v1`。

## 3. Family hard diff 与 Relift guidance

完整且可比较的 family sweep 才会生成 family hard-diff digest。canonical payload 按
`profile_id` 和 hard category 稳定排序，每个 profile 只贡献现有 comparator 的精确硬
差分状态。compile/setup failure 不生成 digest，因此不能触发语义 stagnation。

family guidance 包含：

- `affected_profiles`：该 hard category 仍失败的 profiles；
- `confirmed_profiles`：该 category 已通过的 profiles；
- `profile_differences`：去内部 node hash 后的逐 profile grouped differences；
- 全 family 共同的 J semantic surfaces 和已审查 strategy families；
- `hard_closed_profiles` 作为下一次 Lift 的不可回归约束；
- `failed_profiles` 作为本轮必须复查的 runtime specialization。

Relift history 只保留 profile id、failure stage、compile/pass 状态和 family digest；完整
per-profile comparator payload 不跨入 Relift。精确、净化后的差异只通过
`ReliftGuidance` 传递。replay code、raw TTIR 和 compiler source snippet 始终不可见。

## 4. Family 级确定性 stagnation

family stagnation 状态为：

```text
(
  canonical J/interface/C.schedule/C.strategy/registry/π digest,
  canonical complete-family hard-diff digest
)
```

只有连续两个完整 fresh replay family sweep 的上述状态字节级相同，才返回
`deterministic_stagnation`。任一 profile 的差异变化都会改变 family digest并继续循环。
replay compile、interface、parser 或 setup failure 没有完整 family digest，会打断连续
stagnation 链。

## 5. 确定性随机与边界回归

单元回归覆盖：

- secondary profile hard mismatch 阻止 primary Gate1 closure；
- 三个 profile 全部执行完后才进行一次 Relift；
- Relift 后下一轮重新执行全部 profiles；
- 一个 secondary replay compile failure 不会阻止后续 profile 被执行；
- 同一 family 打乱输入顺序后 canonical family digest/guidance 不变；
- 连续两轮相同 artifact + family diff 只调用一次 Relift，第二轮触发 stagnation；
- 12 个固定 seed 随机 runtime extents 在 integrated Gate1 family path 上 strict all-pass；
- Relift history 不包含 raw per-profile comparison，writer 调用数等于 artifact round 数。

## 6. 真实 `fg_ones` 六 profile 二次检验

使用 schema-26 正式批次保存的真实 `gpt-5.6-sol/xhigh` LiftArtifact 与 source-blind replay，
不调用 Lift、Relift 或 writer model，重新执行当前 schema-30 stack。这是保存真实观测的
诊断迁移，不冒充 schema-30 fresh LLM 结果。

fixed-C family：

- `one = 1`；
- `block_minus_one = 1023`；
- `block = 1024`；
- `block_plus_one = 1025`；
- `non_divisible = 2051`；
- `far_multitile = 8209`。

结果：

- 一份 artifact、一份 replay、固定 `BLOCK_SIZE=1024`；
- family source/replay compile：6/6；
- normal hard comparison：6/6；
- family status：`REPLAY_CLOSED`；
- family hard-diff digest：
  `725023b9590fd46d0910ac04abca26ea4d6ee4f6d8772928acf042b290310f49`；
- 当前 Gate2 matrix：6 shapes × 3 seeds = 18 cases，18/18 `gate2_pass`；
- source↔replay、source↔Lower2C、replay↔Lower2C 全部最大绝对误差为 0。

证据：`/tmp/scrc_schema30_family_v10_recheck_20260718/fg_ones.json`。

## 7. Gate1/Gate2 四象限 v2

`summarize_gate1_gate2` 已升级为 `scrc_gate1_gate2_conditional_v2`，直接输出四象限：

- `gate1_closed_gate2_pass`；
- `gate1_closed_gate2_nonpass`；
- `gate1_not_closed_gate2_pass`；
- `gate1_not_closed_gate2_nonpass`。

只有完成三方 Gate2 comparison 的观测才能进入象限。`LOWERING_UNSUPPORTED`、setup
failure、execution incomplete 和未运行 Gate2 进入 `quadrant_coverage`，不会被错误计为
Gate2 non-pass。

对现有 14 个保存真实观测重新聚合：

- Gate1 closed + Gate2 pass：13；
- Gate1 not closed + Gate2 pass：1（`fg_arange`）；
- 其余两象限：0；
- eligible completed：14/14，completion coverage = 1.0。

该字段只改报告，不改变 Gate1 或 Gate2 verdict。

## 8. 当前仍未完成

schema-30 完成的是 fixed-C **normal replay** 的全 family Gate1 条件、统一 Relift 和 family
stagnation。以下仍不能宣称完成：

- paired J/B intervention 尚未对每个 profile 复用同一份 mutant replay；
- actual-π masked-J 尚未在所有 profiles 逐项形成合法可比指纹和 hard mismatch；
- 因而报告明确保存 `anti_leakage_scope=primary_profile_pending_family_wide`；
- schema-30 fresh `gpt-5.6-sol/xhigh` Lift/replay 仍受 provider cooldown 阻断；
- 新增 `fg_diag_extract`、`fg_skip_layer_norm`、`fg_index_select` 仍只有真实 source
  compile/fingerprint contract，没有 fresh family/Gate2 结果；
- Gate2 多输出、scan/atomic/dynamic-output Lower2C 仍按现有计划后续推进，本轮没有扩张。

下一主线是 family-wide paired intervention 与 actual-π masked-J。完成后再对
`fg_log_softmax → fg_arange → fg_diag_extract → fg_skip_layer_norm → fg_index_select`
执行 schema-30 fresh SCRC。

## 9. 回归结果

- Gate1 TTIR/orchestrate/specialization、SCRC Lift/live/batch/保存观测聚焦回归：
  159 passed、16 warnings；
- 完整 `python/tests`：661 passed、3 skipped、3 failed；
- 两个 `test_splitk_reduce_bwd_vs_xformers` 仅在整套 suite 中因 Triton binder 的
  `BLOCK_SIZE` 重复绑定失败，原样隔离重跑 2/2 PASS；
- 旧 universe `count_nonzero` manifest 仍记录 `ok`，当前 saved replay 为
  `lowering_fail`，隔离重跑仍失败；
- 本轮没有修改 split-k/xFormers 或旧 universe 数据路径。

## 10. schema-31 后续状态

本文保留 schema-30 当时的真实边界：normal family 已接通，paired/masked 尚为
primary-only。后续 schema-31 已在不修改 π schema/拓扑的前提下，将两条反泄漏通道
扩展到完整 fixed-C family。当前协议和新证据见
`doc/SCRC_schema31全Family反泄漏接入与验证_2026-07-18.md`。
