# SCRC schema-31 fresh `fg_arange` 确定性 Stagnation 验证

日期：2026-07-18
模型：`gpt-5.6-sol/xhigh`
协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`
Relift guidance：`deterministic_gate1_hard_diff_guidance_v2`

本轮没有使用 Trellis task/workflow，没有使用 Lift/replay cache，没有修改 π、规则库、
C.strategy vocabulary 或 Gate1/Gate2 verdict。

证据：`/tmp/scrc_schema31_fresh_cross_family_20260718/fg_arange.json`。

## 1. 实验结果

- initial Lift：2 个真实响应，得到完整 J/C/π artifact；
- normal writer：2 个真实 source-blind replay；
- replay compile：2/2 成功；
- 第一次 hard mismatch 后执行 1 次真实 source-aware Relift；
- 第二次 replay 后触发 `deterministic_stagnation`；
- Gate1：`NO_CLOSURE/deterministic_stagnation`；
- Gate2 diagnostic matrix：3 shape profiles × 3 seeds = 9/9 PASS；
- normal Gate1 未闭合，因此 paired/masked/π ablation 均未运行。

这是明确的 `Gate1 no-closure + Gate2 pass` 象限，不允许合并成 overall pass，
也不允许用 Gate2 覆盖 Gate1。

## 2. Stagnation 的精确证据

两轮 candidate 的 canonical artifact digest 完全相同：

```text
846a8eff9f8c8490f63f085c140f937d30fbb2b89ef515dfcb63cfef3caae5e3
```

两轮 exact hard-difference digest 也完全相同：

```text
aae365e71b92fbe3725d7b749bca1cf6128bbfc06eb050bc3d779d17e6ed51ef
```

失败 hard categories 均为：

```text
op_multiset + dataflow
```

`interface`、`reduction`、`predicate_effect` 均保持 hard pass。第二轮完成 fresh writer、
真实 compile 和 hard compare 后才停止；不是因为 artifact 未变化就在 writer 前提前
退出。停止记录为：

```text
first_relift_round = 0
repeated_relift_round = 1
consecutive_observations = 2
remaining_artifact_round_budget = 1
```

因此本次提前停止真实节省了剩余一个 artifact/writer round。

## 3. Relift 是否改变了 J/C/π

没有。initial artifact 与 accepted Relift artifact 的下列 digest 全部相同：

- J digest：`5a1056ba512556da6021128245dbfb2b443e78a73408cb59c56359d7f41c06a8`；
- interface digest：`732538786a33481276d21c5773a77717b3103697ab8d38d568a81eba68019dc5`；
- schedule digest：`9626adf97770f9ab777750c5264de9aa14a6a5e1556d5c62b849910600507b18`；
- strategy digest：`7e2694b0f6f973fe3f10d96318156dcc5b91abac6bdaf1c5fcc0658b9938533c`；
- π plan digest：`c5accef469dd3bb11039118d068493539b222e804c053d401b0441acbf63b1e5`。

这不是 Relift 通道关闭。模型真实接收了第一次 Gate1 hard diff，并返回完整 artifact；
只是 source-aware Lift 判断当前 editable semantic fields 没有应当修改的内容。

## 4. 为什么 Gate1 仍不闭合

source 的值表达式按 block/lane 分组：

```text
step_offset = pid * BLOCK_SIZE * step
value = lane * step + step_offset + start
```

两份 replay 都使用全局 ordinal：

```text
global = pid * BLOCK_SIZE + lane
value = float(global) * step + start
```

二者对当前数值域代数等价，因此 Gate2 9/9 PASS；但 TTIR 的算子多重集和 producer/
consumer edge 不相同。第一轮差分包括：

- source 比 replay 多 1 个 `arith.addf`；
- 多 1 个 `arith.mulf`；
- 多 1 个 `arith.sitofp`；
- replay 多 1 个 `arith.trunci`；
- source 多 1 个 `tt.splat`；
- dataflow 有 23 条 exact edge count 差异。

结构 guidance 的 loop/reduction/effect/mask/dot coordinates 完全一致，
`mismatch_coordinates=[]`。当前 J 使用全局 `iota(size)`、cast、mul、add，表达的是正确
whole-tensor arange 语义。按既定合同，Relift 不得把它改写成 block/lane 分解，只为模仿
source TTIR association。

## 5. 对 guidance v2 的结论

本例证明 guidance v2 的两个必要行为：

1. hard diff 存在时 Relift 通道真实常开；
2. 没有可证明的 J gap 或已注册 strategy gap 时，允许 artifact 不变，并由 exact
   stagnation 停止，而不是随机重采样 writer 或强行改变 J。

这与 `fg_log_softmax` 不同：后者 structural guidance 明确显示 source 2 loops、replay
0 loop、J 0 loop/carry，应优先复查 semantic recurrence；`fg_arange` 没有结构坐标缺口，
仅有代数 association 差异。

## 6. 是否增加新 normalization/C.strategy

当前不增加。仅凭一个 `fg_arange` 不足以证明需要新的 value-association strategy 或
TTIR 代数规范化；Gate1 的保守 false reject 在设计上允许存在。

只有后续满足以下条件时才提交计划外必要性报告并等待批准：

- 多个不同真实 family 重复出现同一有限 association 模式；
- Gate2/source-aware 分析持续证明 J 语义正确；
- 可以用少量、可枚举、方向安全的规则规范化；
- 规则不会把 overflow、cast placement、整数/浮点 reassociation 等真实语义差异吞掉。

## 7. 性能

- preflight compile：3.167 s；
- initial Lift：283.280 s；
- Gate1 loop：213.148 s；
- Gate2 9-case matrix：2.402 s；
- pipeline total：498.840 s；
- held-out total：506.823 s，约 8 分 27 秒。

记录调用数：initial Lift responses=2、Relift responses=2、normal writer=2、
normal replay compile=2。与 `fg_log_softmax` 约 37 分钟相比，本例明显更快，但端到端
耗时仍主要来自外部模型响应。
