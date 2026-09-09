# 真实运行结果与 evidence

以下结果来自一次正式真实 Kernel closure，不是模拟输出。工作负载是
`high_performance:liger_softmax`，成功运行目录为：

```text
real-evidence/retry-02/liger_softmax/run-001/
```

权威文件：

- `report.json`：总状态、阶段计时和闭环汇总；
- `attempt-03/gate1.json`：最终成功候选的 Gate1 Source/Replay 结构证据；
- `attempt-03/artifact.json`：最终成功的 J/C/pi artifact；
- `attempt-03/replay.py`：最终成功的 source-blind Replay；
- `gate2.json`：Gate2 三边数值比较和分项计时；
- `source.py`、`source_projection.json`：Source 与 Harness/compiler 绑定视图；
- `report.json`：总状态、Lift 次数、阶段计时和闭环汇总。

## 总结

```text
main_status: SCRC_CERTIFIED
Gate1: replay_closed
Gate2: gate2_pass
Gate2 case: finite-hidden-1
accepted Lift call: 3 of 3
```

Gate1 的 formal round 为 1，最终 `gate1.json` 中的 hard comparison differences 为空。
Gate2 的三条比较边全部通过：

```text
Source–Replay: pass, max_abs_error = 0
Source–J:      pass, max_abs_error = 2.7939677238464355e-09
Replay–J:      pass, max_abs_error = 2.7939677238464355e-09
```

输出 `y` 为 `float32`，形状为 `(2, 127)`。Gate2 使用的 C slot 是
`BLOCK_SIZE=128`，规范化视图为 `c0=128`。

## 阶段耗时

| 阶段 | 状态 | 耗时 |
| --- | --- | ---: |
| Lift（3 次逻辑调用） | finished | 211.920 s |
| TypeCheck（3 次，最终 pass） | pass | 53.46 ms |
| semantic probe | pass | 2.352 s |
| BindSource | finished | 0.020 ms |
| Replay/Gate1 | `replay_closed` | 65.164 s |
| Gate1 search total | `gate1_closed` | 279.491 s |
| Gate2 total | pass | 168.577 ms |
| 命令端到端墙钟时间 | exit 0 | 280.78 s |

这些是流程各阶段的独立记录，不应简单相加作为总墙钟时间；`report.json` 的
`stage_metrics` 是各阶段自身计时，命令端到端时间还包含工作区、provider 响应和
证据写入。

## Gate2 分项耗时

| 项目 | 耗时 |
| --- | ---: |
| Source execution | 9.743 ms |
| Replay execution | 122.988 ms |
| J reference execution | 18.768 ms |
| 三边比较 | 4.432 ms |
| case total | 159.972 ms |
| suite total | 168.577 ms |

三条边使用同一 Harness 输入和 Gate2 tolerance policy。`Source–Replay` 为精确一致，
另外两条边的最大绝对误差远小于 `f32` 的 `atol=1e-6` 和 `rtol=1e-5` 要求。

## 地址表示闭合

最终 artifact 的两个 address-representation 绑定均把 row stride 绑定到 tensor
axis 0：

```text
a0 <- p3, axis_ref = x0
a1 <- p1, axis_ref = x0
```

最终 Replay 使用完整的二维地址关系：`row * stride + lane`。逻辑 extent `p4`
只用于 active-length predicate，不参与地址乘法；因此这次通过是地址语义修复的
结果，不是放宽 Gate2 容差。

Lift 上限保持 3 次，最终在第 3 次得到闭合候选。Replay 使用 Source/Harness 已
绑定的 `artifact.c`，没有为了 Replay 另外搜索或生成 `target_C`。

## 结果边界

这次 `SCRC_CERTIFIED` 只证明上述真实 workload 的完整闭环：

```text
Source -> Lift artifact -> TypeCheck -> BindSource
       -> source-bound Replay -> Gate1
       -> Gate2(Source–Replay, Source–J, Replay–J)
```

它不等价于所有活动 Kernel、所有前端或所有生产后端都已闭合。当前活动清单的整体
状态仍以 [`docs/final-task.md`](../final-task.md) 的当前合同和权威汇总为准；未完成
的 FlashAttention、xFormers 及其他 native 前端 case 不能用本例结果代替。
