# RVV 新增 8-J 三柱结果

目标机为 RV64GCV（VLEN=128），单线程、CPU0，`symbolic-medium-v1` workload。
测量使用 cold-data，128 MiB working set 大于检测到的 64 MiB LLC；cache eviction
不进入计时区间。每批 4 次、7 批，Full 固定预算为 12 个候选。所有候选均来自
`J -> Intent MLIR -> LLVM/RVV object`，远端只负责链接、正确性与计时。

8 项几何平均为：

- `G_schedule = 1.126×`；
- `G_full = 1.577×`。

LayerNorm 的 `producer_consumer_fused_joint_vector_reduction` 给出 `5.356×`
Full/Schedule，PolyNorm 为 `2.207×`，Sparsemax 为 `1.679×`。DYT 的 `1.008×`
是明确负例，说明 Full 并非对所有图强行制造收益。

本表只回答内部三柱归因；没有把不适用的 Triton Source 或缺失的外部 intrinsic
硬填成 baseline。外部 LLVM-RVV C、Triton-CPU/RVV 与 ggml/Expert 的严格适用子集
仍见 `doc/RVV可迁移性与外部基线实施报告_2026-08-01.md`。

逐项数据在 [summary.csv](summary.csv)。
