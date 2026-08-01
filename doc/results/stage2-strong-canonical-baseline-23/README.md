# Stage 2 强 Canonical Baseline 重跑结果

日期：2026-07-31

本结果替代 `stage2-new8-production-backend-22` 中 5090D/H100 的内部三柱归因。旧结果把完整消费 J 所需的
read-once row lowering 与 paired rotation 只放进 Full，造成 `G_full=52.201×/89.373×`；本轮按照 Fresh-14
口径，将这些基础生产 lowering 放入 C-Fixed/Schedule-Only。

固定定义：

```text
C-Fixed       = CompleteCanonicalDirectLower(J) + C 的一次机械 schedule 投影
Schedule-Only = 相同完整 target-program structure，仅搜索 schedule
Full          = Schedule-Only + 真正额外的规约树、同步/stage、materialization、
                数值执行状态或 contraction structure

G_schedule = C-Fixed / Schedule-Only
G_full     = Schedule-Only / Full
```

所有 executable 均来自 C++/MLIR。测量仍为相同 natural/model-level shape、code-warm/data-cold、CUDA event，
L2 eviction 位于计时区间外。

## 1. RTX 5090D

单位：微秒。

| Kernel | C-Fixed | Schedule-Only | Full | Source | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSD | 182.048 | 181.696 | 181.408 | 197.024 | 1.002× | 1.002× | 1.084× |
| TVD | 189.152 | 181.952 | 181.952 | 200.672 | 1.040× | 1.000× | 1.098× |
| xFormers RMSNorm | 1392.032 | 269.408 | 269.408 | 286.288 | 5.167× | 1.000× | 1.061× |
| CrossEntropy | 22.752 | 16.256 | 16.256 | 40.608 | 1.400× | 1.000× | 2.479× |
| GroupNorm | 16.000 | 16.000 | 15.680 | 32.240 | 1.000× | 1.020× | 2.011× |
| Liger RoPE | 54.912 | 40.800 | 40.800 | 54.336 | 1.346× | 1.000× | 1.333× |
| Qwen2VL MRoPE | 70.240 | 54.400 | 54.400 | 63.472 | 1.291× | 1.000× | 1.158× |
| Flash Linear | 240.416 | 152.736 | 142.464 | 194.832 | 1.574× | 1.072× | 1.365× |

八项几何均值：`G_schedule=1.460×`、`G_full=1.011×`、`Source/Full=1.382×`。8/8 Full 不慢于 Source。

Full 真正增加的内容：

- JSD：rank-0 host capture，边际 `1.002×`；
- GroupNorm：multi-accumulator tree、reciprocal evaluation 与 rank-0 capture，边际 `1.020×`；
- Flash Linear：canonical tiled contraction 之外的真实 `mma.sync`，边际 `1.072×`；
- 其余五项的 Full winner 就是 Schedule-Only winner，`G_full=1.000×`。

## 2. H100 80GB HBM3

单位：微秒；bundle 为 `sm_90a` 独立重新编译，未复用 5090D cubin/winner。

| Kernel | C-Fixed | Schedule-Only | Full | Source | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSD | 109.888 | 109.888 | 104.256 | 162.400 | 1.000× | 1.054× | 1.607× |
| TVD | 131.904 | 100.992 | 100.992 | 118.464 | 1.306× | 1.000× | 1.181× |
| xFormers RMSNorm | 919.744 | 144.224 | 144.160 | 196.560 | 6.377× | 1.000× | 1.090× |
| CrossEntropy | 19.328 | 15.648 | 15.648 | 45.008 | 1.235× | 1.000× | 2.777× |
| GroupNorm | 16.096 | 16.064 | 15.840 | 36.144 | 1.002× | 1.014× | 2.293× |
| Liger RoPE | 69.568 | 33.344 | 33.344 | 45.712 | 2.086× | 1.000× | 1.262× |
| Qwen2VL MRoPE | 89.568 | 47.616 | 47.616 | 50.224 | 1.881× | 1.000× | 0.911× |
| Flash Linear | 347.136 | 245.824 | 142.016 | 248.864 | 1.412× | 1.731× | 1.648× |

八项几何均值：`G_schedule=1.658×`、`G_full=1.080×`、`Source/Full=1.495×`。7/8 Full 不慢于 Source；
Qwen2VL MRoPE 为当前保留的负例，Full 比 Source 慢约 9.8%。

Full 真正增加的内容：

- JSD：FMA evaluation state，边际 `1.054×`；
- xFormers RMSNorm：multi-accumulator tree 被选中，但边际仅 `1.0004×`；
- GroupNorm：subgroup shuffle 与 rank-0 capture，边际 `1.014×`；
- Flash Linear：`mma.sync` 替代 canonical tiled contraction，边际 `1.731×`；
- TVD、CrossEntropy、Liger RoPE、Qwen2VL MRoPE 没有额外 Full 收益。

## 3. 旧结果为什么废弃

旧结果的最终 Full latency 基本真实，但 Schedule-Only 分母不符合 Fresh-14 的强 canonical baseline：

```text
旧 JSD: Schedule-Only generic materialized 149550.278 us -> Full 181.088 us
新 JSD: Schedule-Only read-once              181.696 us -> Full 181.408 us
```

因此旧 `825.843×` 主要衡量“generic fallback 到完整 J consumer”，新 `1.002×` 才衡量完整消费 J 后 Full 额外
带来的性能空间。相同修正适用于 xFormers RMSNorm、CrossEntropy、GroupNorm、Liger RoPE 与 Qwen2VL MRoPE。

Flash Linear 原本就采用正确边界：Schedule-Only 搜 canonical tiled contraction，Full 额外开放 `mma.sync`。
TVD 原本也是正确的单遍负例。

## 4. 实现与机器报告

本轮修正包括：

- `schedule_only_cuda_path_identity` 将 read-once row lowering 与 paired rotation 归入完整 canonical spine；
- rank-0 capture、规约树、multi-accumulator、FMA/approximation 等仍留在 Full；
- C-Fixed 只在 exact target compilation 真正形成的 canonical schedules 中投影 C，不用性能或 cost model替换非法状态；
- Flash Linear 保持 canonical tiled → Full `mma.sync`。

原始报告：

```text
doc/results/stage2-strong-canonical-baseline-23/5090d.json
doc/results/stage2-strong-canonical-baseline-23/h100.json
```

