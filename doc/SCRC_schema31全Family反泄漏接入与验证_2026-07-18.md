# SCRC schema-31：全 fixed-C Family 反泄漏接入与验证

日期：2026-07-18
协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`
held-out identity：schema 31

本轮没有运行 Trellis task/workflow，没有修改 π schema、π 拓扑、C.strategy
family、RuleCard 或 Gate2 verdict。主变化是将现有 paired intervention 和
actual-π masked-J 从 primary profile 扩展到已声明的完整 fixed-C family；随后
又补上 family 证据身份绑定、Gate2 suite 完成性 fail-closed，以及最终 artifact
与 Gate2 replay 的精确对齐约束。这些收紧不改变 Gate1/Gate2 的正交关系。

## 1. schema-30 留下的缺口

schema-30 已要求同一 normal replay 覆盖完整 fixed-C family，但反泄漏只检查
primary profile。因此旧流程可能错过：

```text
primary masked-J hard mismatch
+ secondary/boundary masked-J hard closure
```

同样，mutant replay 在 primary 上出现注册 exact delta，不代表它在其他 runtime
shape/grid 上也真正响应了 J 变化。

## 2. Paired intervention 全 family 流程

每个注册 intervention 执行：

```text
mutate J/B
  → deterministic recompose π
  → writer 只生成一份 mutant replay
  → 同一 mutant replay 编译全部 fixed-C profiles
  → 逐 profile 比较 normal replay fingerprint ↔ mutant fingerprint
  → 逐项要求注册 exact delta
```

活动策略：`fixed_c_family_exact_delta_all_profiles_v1`。

裁决：

- 全部 profiles 的注册 delta 精确通过：该 intervention PASS；
- 任一 profile 忽略 J、方向错误或 off-target drift：`LEAKAGE_REJECTED`；
- writer/compile/fingerprint 等未形成因果证据：结构化
  `intervention_evidence_incomplete`，Gate1 `NO_CLOSURE`；
- 首个失败不会中止 family sweep；
- 不使用 score 阈值、多数票或 profile-local replay repair。

family evidence 按 `profile_id` canonical 排序。digest 不再哈希 score、rank 或
诊断文本，而是显式绑定 baseline/intervention J digest、normal/mutant code
SHA-256、完整 fixed-C family contract，以及逐 profile 的精确算子/常量/fill/
store/reduction 与 hard-component 差分。混合出现语义失败和基础设施不完整时，
结果明确记为 `semantic_failure_and_incomplete`，且语义失败优先决定 Gate1 拒绝。

## 3. Actual-π masked-J 全 family 流程

masked-J 依然保留 normal Lift 产生的 actual π，不对 masked J 重新 compose：

```text
writer(masked J, same actual π, same interface/C/rules)
  → 只生成一份 masked replay
  → 恢复 opaque ABI
  → 同一 masked replay 编译全部 profiles
  → 逐 profile 与对应 source fingerprint 做 hard comparison
```

活动策略：
`actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1`。

严格裁决：

- 所有 profiles 都形成可比 fingerprint 且 hard mismatch：允许保留 normal
  closure；
- 任一 profile hard-close：`LEAKAGE_REJECTED`；
- 任一 profile 编译、ABI 恢复、parser 或 source baseline 不完整：
  `masked_evidence_incomplete`，Gate1 `NO_CLOSURE`；
- family 模式不再把 masked compile failure 当成反泄漏通过证据。

masked family digest 同时绑定 actual π digest、masked/restored code SHA-256、
完整 family contract、每个 profile 的 source/replay fingerprint digest 与精确
hard-difference digest；诊断文本和模糊 score 不参与身份。

实现中曾发现一个重要边界：masked replay 可能保留正确函数签名，但不再使用
output pointer。这种 access 变化本身是 hard `interface/effect` 差异，应交给
comparator，不应被 normal artifact-interface 前置校验误判为不可比。当前 family
路径已与原 single-profile masked-J 语义统一。

## 4. 组合 closure 条件

fixed-C family 的正式 closure 条件为：

```text
normal family: all hard-close
AND every registered paired intervention: all profiles exact-delta PASS
AND actual-π masked-J: all profiles legal comparable hard mismatch
```

组合策略：
`normal_paired_and_masked_fixed_c_family_all_required_v1`。

Gate2 仍然正交独立；Gate1 的 leakage/no-closure 不被 Gate2 PASS 升级。

此外，Gate2 现在只能选择 `candidate.artifact_digest` 与最终 Gate1 artifact
canonical digest 完全一致的 replay。若最后一次 Relift 已改变 artifact、但尚未
产生与之绑定的可执行 replay，Gate2 必须报告 coverage gap，不能回退到旧轮 replay
并与新 artifact 的 Lower2C 混用。specialization evidence 使用相同对齐规则。

Gate2 suite 指标也改为 fail-closed：所有 cases 都必须有合法终态；一旦 suite
使用 `case_id`，则每个 case 都必须提供非空且唯一的 ID；`status_counts`（若有）
必须与实际 cases 一致；顶层 `passed` 必须等于逐 case 推导结果。缺失或矛盾证据
只计 attempted-but-incomplete，不计 Gate2 nonpass。

## 5. 确定性随机与边界回归

新增回归包含：

- `1/63/64/65/66/127/128/129` 边界与固定 seed 随机 extents；
- writer 调用数只随 normal/intervention/masked 实验臂增长，不随 profile 数增长；
- primary intervention PASS、secondary `IGNORED_J` 时仍完成全 sweep 并拒绝；
- secondary intervention compile failure 返回 incomplete，不冒充 leakage；
- primary masked mismatch、secondary hard-close 时完成全 sweep 并拒绝；
- secondary masked compile failure 返回 incomplete，不冒充 non-leakage PASS；
- masked ABI 无法恢复时所有 profiles 明确记为 incomplete；
- profile 输入顺序变化不改变 paired/masked family digest 和 verdict；
- actual π 不变，不存在 masked replay repair 或 profile-local writer。

## 6. 保存真实 `fg_ones` 迁移

使用 schema-26 正式批次保存的真实 `gpt-5.6-sol/xhigh` 数据：

- LiftArtifact：真实 LLM Lift 产物；
- normal replay：保存的 source-blind writer 输出；
- mutant replay：保存的 paired intervention writer 输出；
- masked replay：保存的 actual-π masked-J writer 输出。

当前 schema-31 stack 不调用模型、不修 replay，直接迁移得到：

- normal fixed-C family：6/6 hard-close；
- `output_const:n0:1_to_0` paired intervention：6/6 exact-delta PASS；
- actual-π masked-J：6/6 可比 hard mismatch，0 hard closure，0 incomplete；
- intervention family digest：
  `78b2f2675b01da37b4dd5dadb025d444b3b7505c7a01d7b078bd3df1ae03c51e`；
- masked family digest：
  `37d6cd0cb7fe4d89967063fe396a6b94b5c7b4f9e673612c42c231f43a373c55`；
- fixed-C family contract digest：
  `ecbefaec05d06f85b719e2739ec9a6a5c81ee6fa93595573cc896019f1844eef`；
- 最终 artifact 与 Gate2 candidate 对齐 digest：
  `9330bbd3d0ab9eda3777c2ba4a67dd897725d711f641ef4f396cfaead4f97d86`；
- masked arm 保留的 actual π digest：
  `8fc19fd64e4f28f17619dc8b875edaf18e65f58091eebd6d6ea07ca27193cbd4`；
- Gate2：6 shapes × 3 seeds = 18/18 PASS。

证据：
`/tmp/scrc_schema31_family_antileak_v11_recheck_20260718/fg_ones.json`。

这是保存真实 writer 输出的当前协议迁移，不是 schema-31 fresh LLM 实验，
也不能外推为其他 family 的通过率。

## 7. π 拓扑结论边界

本轮仅扩大监测范围，不优化 π 拓扑。`fg_ones` 六个 profiles 中没有出现
masked hard closure，因此没有“大量 π 拓扑泄漏”的新证据。这不证明 π 在整个
Kernel universe 中无泄漏；后续只有在跨多个真实 family 的 fresh 批次中出现成批
hard closure，才按约定报警并提交修改 π 的必要性报告。

## 8. 当前仍未完成

- schema-31 fresh `fg_log_softmax` 已完成：2 份 normal replay 均编译，但完整
  fixed-C family 均在 `op_multiset + dataflow + predicate_effect` 不闭合；Gate2
  `single_n_tile/exact_tiles` 6/6 PASS，`multi_n_tail/multi_m_and_n_tail` 6/6
  `replay_diverged`。该实验直接暴露 source 2 loops、replay 0 loop、当前 J 0 loop/
  0 carry 的 Relift 指导缺口；活动 guidance 已升级为
  `deterministic_gate1_hard_diff_guidance_v2` /
  `deterministic_fixed_c_family_hard_diff_guidance_v2`，详见
  `doc/SCRC_schema31_fresh_logsoftmax与Relift语义指导改进_2026-07-18.md`；
- `fg_diag_extract`、`fg_skip_layer_norm`、`fg_index_select` 仍只有真实 source
  compile/fingerprint contract；
- family-wide 反泄漏目前只有 `fg_ones` 保存真实 writer 输出证据；
- 跨 C family、scan/atomic/sort/dynamic-output/multi-output 仍按原路线后续处理。

由于该 fresh 单例耗时约 37 分钟，后续在线顺序调整为：

```text
fg_arange
→ fg_diag_extract
→ fg_skip_layer_norm
→ fg_index_select
```

## 9. 回归结果

- TTIR/Gate1 intervention/orchestrate/specialization、SCRC Lift/live/batch/保存观测、
  Gate2 扩展聚焦回归：333 passed、33 warnings；
- 完整 `python/tests`：692 passed、3 skipped、3 failed、77 warnings，总计 698；
- 两个 `test_splitk_reduce_bwd_vs_xformers` 仅在全套顺序下因 Triton binder
  `BLOCK_SIZE` 重复绑定失败，隔离重跑 2/2 PASS；
- 旧 universe `count_nonzero` manifest 仍记录 `ok`，当前 saved replay 为
  `lowering_fail`，隔离重跑仍失败；
- 本轮没有修改 split-k/xFormers 或旧 universe 数据路径。
