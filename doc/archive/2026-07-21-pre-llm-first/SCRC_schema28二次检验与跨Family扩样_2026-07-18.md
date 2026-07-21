# SCRC schema-28/29/30 二次检验与跨 Family 扩样

日期：2026-07-18
当前协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`
held-out identity：schema 31
模型配置：正式 fresh 路径固定为 `gpt-5.6-sol/xhigh`

本轮没有运行 Trellis task/workflow；只更新了项目实现、测试、文档和 Trellis spec。
主线限定为：scope-aware TTIR、确定性 stagnation、Gate1 差分驱动 Relift、现有样本
二次检验和非同质真实 family 扩样。没有修改 π 拓扑模型。

## 1. scope-aware TTIR SSA

旧 parser 用全函数 SSA name map，因而会把两个 sibling region 合法复用同名 SSA
误判为 duplicate definition。当前 `triton-ttir-3.6-gate1-v2` 使用 lexical region
scope：

- 当前/父 region 逐层解析 operand；
- sibling region 的同名局部 SSA 使用不同内部 identity；
- 同一 region 的 duplicate definition 仍 fail-closed；
- bounded normalization 遇到跨 scope 重名时不再覆盖式改名。

真实 `fg_log_softmax` source TTIR 已成功提取指纹：102 ops、2 个 `scf.for` loop、
2 个 masked read、1 个 masked write。旧 `%n_offset` parser false reject 已消失。

## 2. 确定性 stagnation stop

当前 policy 为 `exact_artifact_and_hard_diff_v1`。连续状态定义为：

```text
state_i = (
  canonical J/interface/C.schedule/C.strategy/registry/pi digest,
  exact hard-component difference digest
)
```

只有完成初始 normal replay、至少一次 Relift、第二次 fresh normal replay 且
`state_i == state_{i-1}` 时，Gate1 才以 `NO_CLOSURE` 停止。artifact 相同本身不会在
writer 前提前停止；generation/compile/parser 不可比较失败会打断连续链。score、自然
语言诊断、raw TTIR 和 replay source 均不进入 signature。

### `fg_arange` 当前机制复验

由于 provider cooldown，本次使用 schema-26 正式实验中真实 `gpt-5.6-sol` 生成的
J/π 和 replay code，重新走当前真实 source/replay Triton compile 与 Gate1 hard
comparison；这不是 fresh LLM 结果，也不替代正式 batch。

结果：

- candidate round：`[0, 1]`；
- 两轮 replay 均真实编译成功；
- 两轮失败类别均为 `op_multiset + dataflow`；
- canonical artifact digest：
  `fc0aea8e4e1be87af41fd8feb51611bcf5c3f9d1f30f64e6d6708e3ce185c5c5`；
- hard-diff digest：
  `aae365e71b92fbe3725d7b749bca1cf6128bbfc06eb050bc3d779d17e6ed51ef`；
- 第二次完整 replay 后触发 stagnation，只调用一次 Relift，不再消耗下一次
  Relift/writer 预算。

该结果证明无变化的 Relift 不再被当成无限 replay 重采样通道。

## 3. Gate1 hard diff 驱动的下一轮 Lift

本轮明确不以“提高初始 Lift 一次响应率”为目标。新增
`deterministic_gate1_hard_diff_guidance_v1`，只在 source/replay 已完成 hard compare
后生成 `ReliftRequest.semantic_guidance`。其内容包括：

- exact hard-diff digest；
- failed/confirmed hard categories；
- 去内部 node hash 后、按 producer/consumer/operand role 分组的 dataflow delta；
- 每个失败类别应复查的 J semantic surface；
- 允许考虑的、已经注册的 C.strategy family；
- 当前 J 的 op/loop-carry/boundary/scope/storage/strategy 摘要；
- `J gap / reviewed strategy gap / no expressible change` 三分决策合同。

`fg_arange` 的首次差分被压缩为 5 组 op delta 和 22 组 dataflow delta。Relift prompt
明确要求：已确认 hard category 必须保持；TTIR op count 不能直接当作 J op；没有语义
依据时返回 J 不变并交给 stagnation 停止，不能为获得另一份随机 replay 而制造修改。

纯 provider timeout/quota/model-cooldown 没有 replay candidate 或 hard diff，不再调用
Relift。合法候选被 AST policy 拒绝、replay compile 失败或 hard mismatch 仍按现有
source-free evidence 回流 Lift；旧 replay code 始终不可见。

## 4. `fg_log_softmax` fresh 二次检验

schema-27 曾完成一次无缓存 `gpt-5.6-sol/xhigh` 运行：

- source preflight 成功；
- fresh Lift 使用 2 个模型响应；第一个响应因
  `storage 'b0' has unknown member(s): output_ptr` 被 Lift schema 拒绝，第二个得到完整
  J/C/π；这是 LiftArtifact 自身校正，不是 replay repair；
- initial Lift 约 483.5 秒；
- scope-aware source fingerprint 成功，流程首次真正进入 normal writer；
- writer 随后收到 provider `HTTP 429 model_cooldown`，未形成 replay candidate；
- 因而本轮没有 Gate1 hard diff，也没有 Gate2 语义结果，最终 `NO_CLOSURE` 不能解释成
  J/π 失败。

该运行还暴露旧 schema-27 会在纯 provider failure 后浪费一次 Relift；schema-28 已
禁止这种无语义证据的调用。provider 恢复后必须重新 fresh 执行，schema-27 报告不能
resume 为当前结果。

### schema-31 fresh 结果

当前协议已完成新的无缓存 `gpt-5.6-sol/xhigh` 运行：initial Lift 成功，2 份 normal
replay 均编译，但四个 fixed-C profiles 全部在
`op_multiset + dataflow + predicate_effect` 不闭合。source 使用两段 runtime `N`
loop，其中 recurrence loop 携带 2 个状态；两份 replay 都错误退化为 single-tile
实现。独立 Gate2 对 `single_n_tile/exact_tiles` 6/6 PASS，对两个 multi-N profiles
6/6 `replay_diverged`，且 source↔Lower2C 全部通过。

第一次 Relift 虽执行成功，但返回的 J/C/π semantic digest 完全不变。基于该真实失败，
family guidance 已改为 exact consensus/profile-specific 压缩，并新增 loop/reduction/
effect 结构语义坐标；活动 policy 为 `deterministic_gate1_hard_diff_guidance_v2` /
`deterministic_fixed_c_family_hard_diff_guidance_v2`，Relift prompt 不再重复完整
source edge graph 与 comparison。
本例 guidance 从 68,494 bytes 降为 33,470 bytes，相关 prompt view 组件估算从约
194.6 KB 降至约 39.4 KB。另修复 loop `init_value` 对象触发 Python TypeError 的
失败路径，现稳定返回可修复 SchemaError。完整分析见
`doc/SCRC_schema31_fresh_logsoftmax与Relift语义指导改进_2026-07-18.md`。

## 5. 14-case 保存观测的 schema-29 Gate1/Gate2 二次检验

新增诊断工具 `test/scrc_saved_observation_recheck.py`。它只复用正式批次已经保存的真实
LLM LiftArtifact 与 replay source，重新执行当前 artifact validation、source/replay
Triton compile 和 normal hard comparison；可选地对同一份不可变 replay observation
执行当前 Gate2 source/replay/production Lower2C 三方诊断。它不调用 Lift/Relift/writer
model、masked-J 或 intervention，因此不能替代 fresh SCRC，也不会把旧 replay 反馈给
writer。Gate2 结果明确标记为 `diagnostic_never_upgrades_gate1`。

对 schema-26 批次中仍有完整 artifact+replay 的 14 个 case 逐例复验：

- source compile：14/14；
- replay compile：14/14；
- current normal hard-close：13/14；
- `fg_arange` 仍为 `op_multiset + dataflow` hard mismatch；单轮诊断预算下结构化终止原因为
  `relift_budget_exhausted`；
- `fg_addr` normal replay 仍 hard-close，但这不覆盖其 historical actual-π masked-J
  `LEAKAGE_REJECTED`；
- `fg_eye/fg_trace` 没有合法 LiftArtifact，`fg_log_softmax` 旧报告没有 replay candidate，
  三者按证据缺失排除。

同一批 14 个 case 的当前 Gate2 二次检验结果为：

- Gate2：14/14 `gate2_pass`；source、replay、Lower2C 三条执行腿均成功；
- Gate1/Gate2 四象限：`Gate1 pass + Gate2 pass = 13`，
  `Gate1 no-closure + Gate2 pass = 1`，其余两个象限为 0；
- 唯一后一象限是 `fg_arange`：Gate1 因 `op_multiset + dataflow` 不闭合，Gate2 的
  source/replay 数值却逐元素完全一致；这不是互相覆盖，而是两道门分别报告“结构闭合”
  和“本次动态输入行为一致”；
- 14 例 source↔replay 最大绝对误差均为 0；source/replay↔Lower2C 的全批最大绝对误差
  为 `3.814697265625e-06`（`fg_sum`），仍在声明容差内；
- 每例独立冷进程 compile 观测中，source 平均/中位数为 3252.2/3108.9 ms，replay
  平均/中位数为 2635.6/2521.7 ms。该数字只描述当前 Triton compile 诊断，不含 fresh
  Lift、Relift、writer 或完整 Gate2 墙钟，不能解释为端到端性能。

Gate1-only 输出位于 `/tmp/scrc_schema29_saved_recheck_20260718/summary.json`；带独立 Gate2
诊断的输出位于 `/tmp/scrc_schema29_saved_gate2_recheck_20260718/summary.json`。该复验证明
parser v2、结构化 termination、当前 comparator 与当前 Lower2C 没有破坏已有真实保存
观测，但不证明新协议下的 fresh LLM Lift/replay 成功率。

## 6. 非 reduce 真实 family 扩样

held-out manifest 从 25 candidates / 17 ready 扩为 27 candidates / 20 ready。

### `fg_index_select`：two-axis indirect indexed gather

使用未改写的 FlagGems `index_select_kernel`：

- primary shape：input `[19,23]`，index `[17]`，output `[19,17]`；
- 同时包含二维 program id、index tensor 读取、上下界谓词、间接地址计算和二维尾部 mask；
- fixed C：`BLOCK_M=8`、`BLOCK_N=32`；
- primary source compile 与 scope-aware TTIR fingerprint 成功；
- primary fingerprint：58 ops、2 个 masked load、1 个 masked store、无 loop/reduction；
- digest：`1843ac9c5ecb31d00e6e659192feebc1726d4950bef31e0cd856f2206b2a201e`；
- fixed-C profiles：`single`、`rectangular_tail`、`two_axis_multitile`，三者 source
  compile/fingerprint 全部成功；最后一例把 grid 扩为 `[3,3]`，不是追加同类 reduce。

### `fg_diag_extract`：strided gather

使用未改写的 FlagGems `diag_2d_to_1d_kernel`：

- primary shape：input `[11,13]`，output `[11]`，`diagonal=2`；
- runtime stride：`stride0=13, stride1=1`；
- fixed C：`BLOCK_SIZE=128`；
- source compile 与 TTIR fingerprint 成功；
- fingerprint：28 ops，1 masked read，1 masked write；
- digest：`1e3b6451e4ef5e3150b7062ebc8f8b0f81fc1f85f80e71fbf100ede1c97a8763`；
- fixed-C profiles：`single`、`rectangular_tail`、`multi_block`，三者 source compile/
  fingerprint 全部成功，覆盖 grid 1→2 和 128 边界。

### `fg_skip_layer_norm`：fused norm

使用未改写的 FlagGems `skip_layer_norm_kernel`：

- shape：`Y/X/R=[7,257]`，`W/B=[257]`；
- 覆盖 residual add、mean、variance、rsqrt、affine、尾部 mask；
- `eps` 保持 runtime do-not-specialize scalar；
- fixed C：`BLOCK_SIZE=512`；
- source compile 与 TTIR fingerprint 成功；
- fingerprint：48 ops，4 masked reads，1 masked write；
- digest：`c3b17ae3b95bc2e1a56dcfc0bf50c02010e72044e2da400493d2430fcf10d323`。

当前 20 个 ready source 均已完成 compile + scope-aware fingerprint；本轮三个 non-reduce
contract（indirect indexed gather、strided gather、fused norm）也各自独立复验。它们尚未
获得 schema-31 fresh LLM Lift/replay，因此只能称为 reviewed compiler contracts，不能
称为 Gate1/Gate2 支持。

`fg_rotary_embedding` 具有较高后续价值，但当前是双输出；现有 live Lower2C leg 只接受
一个 output，本轮没有为扩样擅自扩大 Gate2。`fg_cumsum_part` 是完整 cumsum 的内部
scan/fix-up 阶段，`fg_masked_select` 涉及动态输出和多阶段 prefix-sum/scatter；两者暂不
伪装成完整公开 op 的 ready case。

## 7. π topology policy

本轮不优化 π 拓扑。保留 canonical compose equality、paired intervention 与 actual-π
masked-J 终检，并在批次报告中持续统计 masked hard closure。只有未来跨多个 family
出现成批 π topology leakage，而不是孤立 `fg_addr`，才向用户报警并提交独立必要性
报告；在此之前不改变 π schema、composer 或规则卡。

## 8. 验证与未完成项

schema-29 增加 `structured_no_closure_termination_v1`。活动 Runner 的
`NO_CLOSURE` 不再依赖诊断字符串推断，而是明确记录：

- `deterministic_stagnation`；
- `replay_provider_unavailable`；
- `relift_budget_exhausted`；
- `relifter_unavailable`；
- `relift_failed`；
- `masked_evidence_incomplete`。

每条记录保存 artifact round、normal candidate 数、剩余 artifact-round budget 和失败
hard categories，不保存 replay code/raw TTIR。live/batch 直接聚合该字段。回归明确覆盖
`Gate1 NO_CLOSURE/deterministic_stagnation + Gate2 diagnostic PASS`，Gate2 仍是独立
证据，不升级 Gate1。

当前 TTIR/Gate1/Relift/live/batch/保存观测/Gate2 扩展聚焦回归：333 passed，
33 warnings。完整 `python/tests` 本轮共 698 项：692 passed、3 skipped、3 failed。
三项失败均不在本轮 SCRC
改动面：

- `test_splitk_reduce_bwd_vs_xformers` 的两个参数例在整套测试中因 Triton 动态 binder
  报 `BLOCK_SIZE` 重复绑定；随后原样单独重跑均 PASS，属于 suite-order/shared-state 问题；
- `test_saved_replay_reproduces_failure_class` 的旧 universe `count_nonzero` manifest 仍记录
  `ok`，当前 saved replay 为 `lowering_fail`；该项单独重跑仍失败，是既有 manifest 漂移。

本轮遵守主线边界，没有修改 split-k/xFormers 或旧 universe 数据路径。环境仍报告 Triton
autotuner 的既有 deprecation warnings。

## 9. schema-30 fixed-C family 与四象限

schema-30 将 fixed-C normal family sweep 从 post-hoc diagnostic 接入 Gate1 外层循环：

- 每个 artifact round 仍只生成一份 replay；
- 必须完成全部 profile 后才决定 closure/Relift；
- secondary failure 进入一次性 family hard-diff guidance；
- 已闭合 profile 是下一轮不可回归约束；
- 连续两轮相同 artifact + 完整 family diff 才触发 stagnation；
- compile/setup failure 没有 family digest，不伪造语义停止证据。

保存真实 LLM `fg_ones` artifact/replay 已完成当前迁移复验：6/6 fixed-C normal
hard-close；当前 Gate2 6 shape × 3 seed 共 18/18 PASS，三组 pairwise 最大误差均为 0。
详细报告见 `doc/SCRC_schema30_fixedC全FamilyRelift与Gate2四象限_2026-07-18.md`。

Gate1/Gate2 汇总升级为 `scrc_gate1_gate2_conditional_v2`，直接输出四象限及
eligible/incomplete/unattempted coverage。unsupported 或未完成 Gate2 不进入 semantic
non-pass 象限。

schema-31 后续已把 paired intervention 和 actual-π masked-J 也接入同一 fixed-C
family。保存真实 `fg_ones` 重放结果为normal 6/6、paired exact delta 6/6、
masked-J 可比 hard mismatch 6/6、Gate2 18/18 PASS。该证据是保存 writer 输出
的当前协议迁移，不是 schema-31 fresh LLM 批次。

后续在线执行顺序限定为：

1. schema-31 fresh `fg_arange`，确认真实 LLM Relift worklist 与 stagnation；
2. schema-31 fresh `fg_diag_extract`；
3. schema-31 fresh `fg_skip_layer_norm`；
4. schema-31 fresh `fg_index_select`；
5. 再按 family 分层重跑当前 20 ready，而不是继续追加同类 reduce。

在 provider cooldown 解除前，不用缓存结果冒充 fresh LLM 证据，也不切换到手写 J/MLIR。
