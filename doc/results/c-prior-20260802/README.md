# C-prior 固定预算数据

`gpu-five-kernel-b32.csv` 是 5090D 与 H100 上五个 kernel 的 cold-data 搜索顺序消融。每一行都使用同一个完整
concrete candidate universe；`C` 与 `Erase-C` 只改变访问顺序，不改变候选、合法性、编译器或 profiler。

- `candidate_budget=32`：两种顺序都只消费前 32 个候选；
- `erase_c_over_c_speedup = erase_c_best_us / c_prior_best_us`；
- `*_candidates_to_95pct`：在完整 latency map 上，首次到达最终最优性能 95% 所需的候选数；
- `oracle_best_us`：完整候选集合中的最低 latency，只用于定义 95% 阈值。

五项汇总：5090D 的 best-at-budget 几何平均为 `1.0080×`，候选数中位数为 `7 vs 9`；H100 分别为
`1.0113×` 与 `4 vs 15`。因此这组数据支持的是“C 通常提高搜索前期命中效率”，不是“C 扩大最终可达空间”。
Sparsemax 的 candidates-to-95% 是明确反例，必须与总体中位数一起保留。
