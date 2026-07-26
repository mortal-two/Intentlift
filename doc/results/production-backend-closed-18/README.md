# 18 个闭合 J 的 production backend 结果

本目录冻结 2026-07-26 在同一 18-J production catalog 上得到的真实后端结果。所有 target
kernel 都由 C++/MLIR pipeline 生成；Python 只负责 artifact transport、搜索编排、driver ABI、
Q/D 执行、计时和报告，不生成目标计算。

## 第一性原理结论

J 的核心价值不是给几个局部 peephole 提示，而是提供完整 typed graph 与 B hard boundary，使后端
可以从同一 immutable J 构造、验证和搜索不同的 concrete MLIR states：

```text
同一 J
  -> materialized multi-kernel graph / whole-graph state
  -> sequential / subgroup / workgroup / hierarchical reduction state
  -> exact / FMA / target-supported precision-or-approximation state
  -> 每个状态独立调 schedule
  -> D_search 的 Target--RefJ 过滤
  -> 性能选择
  -> D_cert 的 Target--Source/Replay/RefJ 三条直接闭合（正式 CUDA 主实验）
```

在 RTX 5090D 的正式 E2/E3 报告中，E3 相对固定 concrete path 的 schedule-only E2，18 项
几何平均提升 **1.1302x**；E3 相对 Source 为 **1.0928x**，18/18 winner 都完成 held-out
三边 backend closure。更能直接暴露瓶颈的同臂图级对照中，6 个同时存在 materialized 和
whole-graph state 的 J，消除中间物化与多次 dispatch 的几何平均收益为 **17.1258x**，最高
PolyNorm 为 **45.0139x**。该大收益来自编译图和内存流量改变，而不是局部指令修补。

数值阶梯把其余收益拆开：N0→N1 的 reduction topology 在全 18 项上为 **1.0409x**、12 个
适用 J 上为 **1.0622x**；N1→N2 的 FMA 仅为 **1.0003x**，是明确负结果；N2→N3 的
target-supported precision/approximation 在全体为 **1.0676x**、2 个适用 J 上为 **1.1271x**。
这些 observation-changing states 没有改写 J，也没有从 tolerance 推导静态许可，而是统一经过
预先冻结的 Q/D evaluator。

## 结果边界

- RTX 5090D E2/E3 与 L0/L1 报告使用 opaque Replay manifest、互不相交的 `D_search/D_cert`，并
  以 `--require-backend-closure` 强制最终 Target--Source、Target--Replay、Target--RefJ 三条直接边。
- RTX 5090D numeric ladder 与 budget-8 C-prior 报告用于机制/搜索效率消融；它们自身没有正式
  `D_cert` receipt，主 E3 winner 的正式闭合由 E2/E3 报告承担。
- H100 portable runner 对 18/18 执行 Source--RefJ 与 Target--RefJ correctness，并做 Source/Target
  配对计时；没有 Replay 和三条 held-out direct edges，因此不写成 formal backend closure。
- RVV runner 对 18/18 执行 Target--RefJ correctness 与原生计时；尚无 Source/Replay `D_cert`，
  因此只报告同 J、同 target 的 E3/E2 内部因果增量。

## 文件

| 文件 | 内容 | 关键结果 |
|---|---|---|
| `cuda-e2-e3-budget32-formal-v4.json.gz` | RTX 5090D，E2/E3，budget=32，正式三边 closure | E3/E2 1.1302x；E3/Source 1.0928x；18/18 closed |
| `cuda-local-b-budget32-formal-v5.json.gz` | RTX 5090D，L0/L1，真实 post-pass B receipt | 全体 1.0162x；10 个适用 J 1.0295x |
| `cuda-numeric-ladder-budget32-v3.json.gz` | RTX 5090D，N0/N1/N2/N3 | tree 1.0409x；FMA 1.0003x；precision/approx 1.0676x |
| `cuda-c-prior-budget8-roleaware-v3.json.gz` | RTX 5090D，E3/Erase-C，budget=8 | C 的低预算 best-latency 优势 1.0195x |
| `h100-e2-budget32-v4.json.gz` | H100 fixed-path E2 portable execution | 18/18 Source--J、Target--J correct |
| `h100-e3-budget32-v4.json.gz` | H100 expanded-state E3 portable execution | E3/E2 1.1196x；E3/Source 1.0539x |
| `rvv-e2-e3-v3.json.gz` | RVV4，materialized/fused states | E3/E2 1.1817x；18/18 Target--J correct |
| `mechanism-summary-v1.json` | 上述报告的机器可读摘要与 SHA-256 | 不替代原始报告 |

解压示例：

```bash
gzip -cd doc/results/production-backend-closed-18/cuda-e2-e3-budget32-formal-v4.json.gz \
  | jq '.evidence_summary'
```
