# C-prior 固定预算消融

本目录比较同一个 `J`、同一个完整 concrete candidate universe 和同一个测量协议下的两种访问顺序：

- `C`：使用随 artifact 保留下来的 non-binding source prior 排序；
- `Erase-C`：擦除 C 后使用相同后端的无先验排序。

两侧的候选集合、B/target hard admission、编译器和 profiler 均相同。`C` 只改变访问顺序，不创建候选，也不改变
充分预算下的 oracle best。

## GPU

`gpu-eleven-kernel-b32.csv` 覆盖 RTX 5090D 与 H100 上 11 个 kernel；每一侧固定只消费前 32 个候选，
但为计算 candidates-to-95% 实际编译并测量了完整空间。当前 FlashAttention forward 空间包含 WGMMA/async
路径，分别有 252/263 个可执行候选；旧 56-candidate 结果已删除。

本轮新增七项的完整 target-side report 分别为
`cuda-current7-sm120a-b32-cold.report.json.gz` 与
`cuda-current7-sm90a-b32-cold.report.json.gz`。

| Target | Kernels | Erase-C/C best-at-B geomean | 到 95% oracle：C / Erase-C（中位数） |
|---|---:|---:|---:|
| RTX 5090D | 11 | 0.995× | 4 / 4 |
| H100 | 11 | 1.004× | 11 / 18 |

## RVV

`rvv-eight-kernel-b12.csv` 覆盖 VLEN128 单线程 RVV 的 8 个 natural-headline workload，预算为 12；每项 report
都包含完整 latency map。汇总 best-at-budget 为 1.000×，到 95% oracle 的中位候选数为 6.5 / 6。

## 字段

- `erase_c_over_c_speedup = erase_c_best_us / c_prior_best_us`，大于 1 才表示 C 在固定预算下更好；
- `*_candidates_to_95pct` 是该顺序首次达到完整空间最终最优性能 95% 所需的候选数；
- `oracle_best_us` 只用于定义 95% 阈值，不计入固定预算选择。

结论是 target-dependent：C 在 H100 上减少了早期命中所需候选数，在 5090D/RVV 汇总上近似中性；
Flash linear（5090D）和 Softmax（H100）是必须保留的反例。该结果支持 C 是可擦除 prior，而不是合法性权限或
可达空间的一部分。
