# SCRC schema-31 fresh `fg_log_softmax` 与 Relift 语义指导改进

日期：2026-07-18
模型：`gpt-5.6-sol`
reasoning effort：`xhigh`
协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`

本轮没有运行 Trellis task/workflow，没有修改 π schema、π 拓扑、Gate1/Gate2
verdict 或 RuleCard。实验目标是用一个非同质、带 runtime streaming recurrence 的
真实 Kernel 检验：第一次 Gate1 hard diff 能否为下一次 Lift 提供足够明确的语义指导。

## 1. 实验与证据

fresh 命令使用：

```text
case=fg_log_softmax
model=gpt-5.6-sol
effort=xhigh
relift_rounds=3
semantic_interventions=3
gate2_matrix=true
```

正式输出：

`/tmp/scrc_schema31_fresh_cross_family_20260718/fg_log_softmax.json`

离线 current-stack guidance 重检：

`/tmp/scrc_schema31_logsoftmax_guidance_recheck_20260718/fg_log_softmax.json`

离线重检不调用 Lift/Relift/writer model，不修改 replay，只重新编译保存的真实
artifact/replay，并生成当前确定性 family guidance。

## 2. fresh 结果

- initial Lift：1 个真实响应，成功产生完整 J/C/π artifact；
- normal replay：2 个真实 source-blind writer 响应；
- 两份 replay 均真实编译成功，compile rate 2/2；
- fixed-C family：`single_n_tile`、`exact_tiles`、`multi_n_tail`、
  `multi_m_and_n_tail` 四个 profiles 均未通过 Gate1；
- 两轮共同失败 hard categories：
  `op_multiset + dataflow + predicate_effect`；
- `interface` 与 `reduction` hard category 均保持通过；
- 最终 Gate1：`NO_CLOSURE/relift_failed`；
- 未进入 paired intervention、masked-J 或 π ablation，因为 normal family 从未闭合。

两轮 artifact semantic digest 完全相同：

```text
673462f07722c00de876f52119ea8c6c9738adbbd81bf998314614f8befb7692
```

但两份 fresh replay 的 hard-difference digest 不同：

```text
round 0: 6415175aee12844196b00f8ff1f288a0fc468cf60f2c886775337f6fc34e091e
round 1: 9c1574523cf2c0256719112e5029d81b5d4267cf04558dbb69231abf17120edd
```

因此 deterministic stagnation 没有触发：artifact 未变，但连续 hard diff 不同，不能
把两个不同 writer observation 误判为同一停止状态。

## 3. Gate1 为什么正确拒绝

真实 source 使用两段 runtime `N` 循环：

1. 第一段按 `BLOCK_N` 流式扫描，携带 `m/z` 状态，计算稳定 online log-sum-exp；
2. 第二段再次扫描并写回最终 log-softmax。

两份 replay 都把整个 `N` 维直接当成一个 `BLOCK_N` tile，执行两次 load 后直接
`max/sum/log`，没有 runtime loop。该实现只在 `N <= BLOCK_N` 时成立。

Gate2 的 12 个独立动态 case 精确验证了这一点：

- `single_n_tile`：3/3 PASS；
- `exact_tiles`：3/3 PASS；
- `multi_n_tail`：3/3 `replay_diverged`；
- `multi_m_and_n_tail`：3/3 `replay_diverged`。

六个失败 case 全部满足：

```text
source ↔ replay: false
source ↔ Lower2C: true
replay ↔ Lower2C: false
```

因此 Gate2 不是在否定 J/Lower2C 的 denotational log-softmax 语义，而是在证明
replay 缺少多 tile recurrence。Gate1 的结构拒绝与 Gate2 的行为归因相互一致，
但二者仍是独立 verdict。

## 4. 旧 Relift 指导暴露的问题

第一次 Gate1 diff 后，Relift 通道确实常开并执行：

- 第一次 Relift 用 2 个响应完成；第一个响应因 portable scope effect 缺失被修复；
- 接受后的 artifact 与 initial artifact 的 J/interface/C.strategy/π digest 全部相同；
- 第二份 fresh replay 再次缺少 runtime loop；
- 第二次 Relift 的 3 个响应依次失败为：
  1. `q0` 缺少 Boundary effect `e0`；
  2. `q1` 缺少 captures `input/z_init`；
  3. loop carry `init_value` 对象触发旧验证器
     `TypeError: unhashable type: 'dict'`。

旧 family guidance 的主要问题不是没有差分，而是差分组织不适合 Lift：

- 四个 profiles 重复展开大量相同的底层 TTIR edge；
- guidance JSON 为 68,494 bytes；
- source fingerprint 约 14,841 bytes；
- 单个完整 comparison 约 111,280 bytes；
- prompt 中 history 又重复 comparison，关键 loop-carry 事实被约 194 KB 的重复
  结构证据淹没；
- 当前 J 的 `intent_loop_count=0`、`loop_carry_count=0` 没有与
  `source loop=2 / replay loop=0` 形成置顶的确定性语义检查项。

这解释了为什么第一次 Relift 虽然收到了 hard diff，却返回了语义 digest 完全不变的
artifact：通道存在，但指导没有把“多 tile 正确性依赖 semantic recurrence”表达得足够
直接。

## 5. 已实现的改进

### 5.1 family exact diff 共识压缩

活动 family policy：`deterministic_fixed_c_family_hard_diff_guidance_v2`。

同一 category 的 difference item 只保存一次：

- 出现在所有失败 profiles：`consensus_differences`；
- 只出现在部分 profiles：`profile_specific_differences`；
- counts 相同时只存一组 exact count + profile ids；
- counts 不同时保存确定性的逐 profile counts；
- 完整 hard-diff digest 和 Gate1 audit comparison 保持不变，不使用模糊比值。

### 5.2 结构语义坐标

新增 `deterministic_fingerprint_structure_guidance_v1`，从 source/replay
fingerprint 提取紧凑且确定性的：

- loop bounds、carry/yield 数量、loop body effects；
- reduction structure；
- memory effects；
- mask predicate count；
- dot structure。

fixed-C family 将结构完全相同的 profiles 合并成一个 group。当前
`fg_log_softmax` 离线重检得到一个覆盖全部四 profiles 的 group：

```text
mismatch_coordinates = loop_structure
                     + memory_effect_structure
                     + mask_predicate_count
source_loop_count = 2
replay_loop_count = 0
current J intent_loop_count = 0
current J loop_carry_count = 0
```

其中 source 的 recurrence loop 明确有 2 个 init values 和 2 个 yielded values。
内部 TTIR node hash 已从该提示视图移除。

### 5.3 Relift prompt 去重复

活动 single-profile policy：`deterministic_gate1_hard_diff_guidance_v2`。

Gate1 内部报告仍保留完整 immutable comparison；发给 source-aware Lift 的 history
改成 bounded prompt view，只保留：

- hard pass/category 摘要与 difference count；
- exact hard-diff digest/categories；
- sanitized diagnosis；
- compact specialization stage/pass evidence。

source fingerprint prompt view 也不再重复完整 dataflow edge graph。exact worklist 由
digest-bound semantic guidance 承载。对本例的 JSON 组件估算为：

```text
旧重复证据约：194.6 KB
当前 prompt views：source 约 2.8 KB
                  history 约 3.1 KB
                  guidance 约 33.5 KB
合计约：39.4 KB（下降约 79.7%）
```

这是 JSON byte 数，不是模型 token 数，也不进入 Gate1 verdict。

### 5.4 只在 Relift 路径强化 loop 决策

没有修改 generic initial Lift 目标。只有第一次 Gate1 已观察到 structural loop gap 时，
Relift prompt 才明确要求：

- 从 SOURCE 判断 carry/yield recurrence 是否对任意 runtime extent 的语义正确性必要；
- 必要时用 J loop/carry/yield 表达；
- 纯 schedule tiling 则保持 J 不变；
- 不允许为了模仿 TTIR op count 任意制造 loop。

### 5.5 loop `init_value` 失败关闭

wire schema 现在只允许 string/number/boolean/null loop init；对象和数组在本地也稳定
返回 `SchemaError: init_value must be a scalar literal or a visible value reference`，
不再泄漏 Python `TypeError`。这样 Lift repair 可以得到精确、可操作的 schema 反馈。

## 6. 性能观察

本次 fresh 单例墙钟：

- preflight compile：3.840 s；
- initial Lift：235.244 s；
- Gate1 loop：1,973.749 s；
- Gate2 12-case diagnostic：4.496 s；
- pipeline total：2,213.654 s（约 36 分 54 秒）；
- held-out total：2,222.743 s（约 37 分 03 秒）。

记录的模型层逻辑调用：initial Lift 1 个响应、Relift 共 5 个响应、normal writer 2 个
响应。当前吞吐瓶颈明显在外部 Lift/writer 响应，而不是本地 TTIR compare 或 Gate2。

## 7. 当前结论边界

- 这次 fresh 结果证明 Gate1 正确发现了 single-tile replay 对 multi-tile family 的结构/
  行为失真；
- 也证明旧 Relift 通道虽然常开，但对 loop-carry 根因的语义聚焦不足；
- 当前改进已通过本地和保存 observation 重检，但尚未再次支付约 37 分钟的 fresh
  在线调用来证明下一次模型响应一定会补出正确 J loop；
- 没有观察到新的 π topology leakage，按约定不修改 π；
- 其余 fresh family 暂不并发发起，避免把 provider wall-clock 拥塞误当 Gate1 性能。

下一在线顺序仍保持：`fg_arange → fg_diag_extract → fg_skip_layer_norm →
fg_index_select`，但应在 provider 吞吐恢复后执行。
