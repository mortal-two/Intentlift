# Task 24：以真实数据复用提升 Stage2 `G_full`

> 状态：阶段性完成（22 项强基线已收口；5090D 全表 1.1× 需要新增有结构余量的 J/config）
>
> 范围：22 个 accepted J；优先 5090D/H100，保持阶段一、阶段二相同三柱与 cold-data 口径

## 目标

不削弱 `C-Fixed`，不改变 `Schedule-Only` 的完整 canonical direct lowering，在 `Full` 中增加由 J 的索引、只读输入和逐元素独立 effect 足以证明合法的结构搜索，使 5090D/H100 的阶段二 `G_full = Schedule-Only / Full` 至少达到 `1.1×`。

## 第一条高杠杆路径

Liger RoPE 与 Qwen2VL MRoPE 的角度只依赖 `(batch, sequence, rotary_pair)`，Q/K 数据还依赖 head。Canonical lowering 以 `(tensor, head, pair)` 为线程域，因此每个 head 重复加载同一组 cos/sin。新增 Full-only 状态改为：

```text
thread domain = (batch, sequence, rotary_pair)
load cos/sin once
apply the same angle to all Q heads and K heads
```

这会改变 traversal 与跨 head 数据复用，但不改变每个输出元素的浮点运算顺序，也不需要数值放宽。它必须由 C++/MLIR lowering 生成并通过同一 CUDA 编译、正确性和计时链路。

## 验收

- `C-Fixed`、`Schedule-Only` 不包含跨 head 复用状态；
- `Full` 能枚举、物化、计时该状态，并在 transformation trace 中明确记录；
- 真实 MLIR 中角度加载位于 head 展开之外，Q/K 每对元素仍先读后写；
- 5090D/H100 重新跑完整 22-kernel cold-data 三柱，名称仍为 `C-Fixed`、`Schedule-Only`、`Full`、`Source / Native`、`G_schedule`、`G_full`；
- 若单条路径不足以让阶段二两端 `G_full >= 1.1×`，继续增加第二条真实结构路径，而不是停止在当前数据。
- 不使用弱 baseline、Python target emission、kernel-name allowlist、哈希或冻结材料制造结果。

## 实际进展（2026-07-31）

### 已完成的真实 lowering

- paired rotation 的 canonical 发射顺序已恢复为原始 data-load → angle-load 顺序，避免
  Full-only 实验通过无意的 canonical 回归制造分母收益；Qwen2VL 的 paired-head 状态公平复测
  仅约 `1.044×`，Liger RoPE 约 `1.001×`。
- Flash Linear 已增加真实 `mma.sync` 结构内的二维 warp-N 分区：同一 block 的 N tile 由
  2/4/8 个 warp 子组承担，MLIR 中记录
  `cuda.contraction.mma-sync.warp-n-partition.v1`、`warps_n` 和 `group_m`；不是 Python
  发射或 kernel-name 分支。
- Flash Linear 的双 f16 输出/bias epilogue 已生成 `vector<2xf16>` packed load/store，保持
  每个元素原有 f32 add→truncate 顺序，记录
  `cuda.contraction.packed_bias_dual_output.v1`。
- 同一 MMA 路径把 L2 swizzle 的 `GROUP_M` 从硬编码提升为 1/2/4/8/16/32 的 target schedule
  坐标；Split-K 仍是真实两次 `gpu.launch`，但当前自然 shape 下不是赢家。

### 当前实测证据

5090D Flash Linear（natural-headline、code-warm/data-cold、CUDA event）最近一次复测在同一
强 canonical 三柱中为：

```text
C-Fixed       237.664 us
Schedule-Only 150.464 us
Full          128.672 us
G_full        1.169x
Source/Full   1.541x
winner        warps_n=4, group_m=8, BK=64, stages=2,
              packed_bias_dual_output
```

同一机制在 H100 独立 `sm_90a` bundle 上的最新复测为 `Full=127.040 us`、
`Schedule-Only=223.328 us`、`G_full=1.758x`；该次 H100 Source=250.640 us、
code-warm/data-cold，不能与旧的不同机器状态下的分母混用。

### 不能偷换的结论

按八项阶段二的完整强 canonical 几何均值，5090D 目前仍约 `1.03x`，不是 `1.1x`。JSD、TVD、
xFormers RMSNorm 等已达到约 1.3--1.5 TB/s 的 boundary-memory 实现，GroupNorm/CrossEntropy
受 launch floor 约束；Flash preprocess 的 Full 已接近 logical Roofline。把这些物理负例改名、
退回 generic materialization 或削弱 C-Fixed 都不属于真实提升。

### 失败路径与物理边界（不计入结果）

- 5090D CTA 内 Split-K（2/4 个 warp group）所有候选均通过 Q/D，但最佳约 199.94 us，
  明显慢于 128.64 us 的单阶段 winner；已从 backend 和 search 中删除。
- 5090D/H100 多 accumulator MMA chain 均通过 Q/D，但最佳分别约 136.51/154.91 us，
  都慢于各自单链 winner；已删除。
- 等价 cuBLAS `addmm + 双输出` 诊断中位数约 122.94 us，仅比 MLIR winner 快约 4.7%，
  不能提供把全 22 项几何平均从约 1.0545x 推到 1.1x 的乘积。
- 官方 PTX 能力表中 `tcgen05.mma` 的 f16 路径属于 SM100/101；SM120 当前 f16
  contraction 不能直接使用该数据中心 Blackwell 路径。

因此本阶段必须同时报告两层数字：完整 22 项 `G_full`（保留真实内存墙/launch-floor
负例）与结构适用子集的 `G_full`；不能把单个 Flash 的 1.169x 或 H100 的 1.758x
冒充 5090D 全表 1.1x。后续只有新增真实 J/config、且其 lowering 确实有未消费的结构空间时，
才继续提升 5090D 全表；不通过削弱 C-Fixed、改分母或引入 Python 发射制造数字。
