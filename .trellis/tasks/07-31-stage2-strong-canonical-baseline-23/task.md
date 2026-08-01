# Task 23：阶段二强 Canonical Baseline 与 Full 重新归因

> 状态：5090D/H100 已完成；RVV 沿用阶段一的 materialized-to-optimized 口径，待形成 22 项统一矩阵时再横向重跑
>
> 范围：阶段二 8 个 accepted J；优先 5090D/H100，随后同步 RVV 口径

## 目标

按照 Fresh-14 的基线原则重新生成阶段二三柱：完整、合理地消费 J 属于 DirectLower 基础能力，不能计为 Full
收益。Full 只统计在完整 canonical lowering 之上额外打开的优化空间。

固定名称不变：

```text
C-Fixed
Schedule-Only
Full
Source / Native
G_schedule = C-Fixed / Schedule-Only
G_full     = Schedule-Only / Full
```

## 机械边界

### C-Fixed

```text
CompleteCanonicalDirectLower(J) + C 的确定性 target schedule 投影
```

它必须完整消费 typed J 的计算、index/broadcast、effect、alias 与全部 observable outputs。不能为了制造差距退回
generic materialized fallback。

### Schedule-Only

与 C-Fixed 使用完全相同的 target-program structure，只搜索 tile、threads、warps、vector、pipeline 等 schedule。

### Full

完整包含 Schedule-Only spine，只额外开放能够在编译后记录中证明改变了规约树、同步、stage、materialization、
index graph、contraction structure 或数值执行状态的候选。仅仅从 generic fallback 进入能够完整消费 J 的专用
lowering，不再计为 Full。

## 阶段二重新分类

| Kernel | Canonical DirectLower | Full 可保留的额外空间 |
|---|---|---|
| JSD | 单 kernel read-once row reduction | compact tree、multi-accumulator、multi-row/stage、合法数值状态 |
| TVD | 单遍多输出 elementwise | 当前没有额外结构时如实 `G_full=1` |
| xFormers RMSNorm | read-once row reduction | reduction topology、multi-accumulator、FMA 等准入状态 |
| CrossEntropy | 完整 gather/effect/read-once row path | reduction topology、multi-accumulator、stage/materialization |
| GroupNorm | grouped read-once statistics | joint/tree、raw-moment、epilogue/load placement |
| Liger RoPE | N-D paired in-place rotation、双输出 | 当前没有额外结构时如实 `G_full=1` |
| Qwen2VL MRoPE | N-D indexed paired rotation、双输出 | 当前没有额外结构时如实 `G_full=1` |
| Flash Linear | canonical tiled contraction | 真实 `mma.sync`；尚未实现的 WS/SplitK 不计入 |

## 实施

1. 修正 CUDA `schedule_only_cuda_path_identity`，让完整 row/paired consumer 进入 canonical spine；
2. 用测试锁定 C-Fixed/Schedule-Only 的非 schedule 字段相同；
3. 检查 Full expanded 候选只剩真实额外状态；
4. 重新生成 5090D/H100 natural-headline bundle并完成 code-warm/data-cold CUDA-event 测量；
5. 更新阶段二与 22 项总览，旧 `52.201×/89.373×` 明确降级为已废弃的错误归因口径；
6. RVV 使用相同原则复核，不能把“第一次完整融合消费 J”自动算作 Full。

## 验收

- JSD/xFormers RMSNorm/CrossEntropy/GroupNorm 的 Schedule-Only 不再是 `generic:generic:default`；
- Liger RoPE/Qwen2VL MRoPE 的 paired rotation 已在 C-Fixed/Schedule-Only 中；
- Flash Linear 保持 canonical tiled → Full `mma.sync`；
- TVD 保持真实负例；
- 三柱所有 executable 继续来自 C++/MLIR；
- 5090D/H100 8 项重新实测，逐项说明 Full 在完整 J 消费之上具体增加了什么；
- 不用弱 baseline、kernel-name allowlist、Python target emission、哈希或冻结材料制造结果。

## 结果

| Target | `G_schedule` | `G_full` | Source/Full | Full 强 cell (`>=1.50×`) |
|---|---:|---:|---:|---:|
| 5090D | 1.460× | 1.011× | 1.382× | 0 |
| H100 | 1.658× | 1.080× | 1.495× | 1（Flash Linear `mma.sync`） |

逐项结果与机器报告见 `doc/results/stage2-strong-canonical-baseline-23/README.md`。旧
`52.201×/89.373×` 已废弃，不再计入阶段二或 22 项汇总。
