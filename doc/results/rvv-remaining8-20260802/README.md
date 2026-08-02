# RVV 剩余 8-J 三柱结果

这组结果补齐此前尚无 RVV 三柱的 8 个可消费 J。统一目标为 RV64GCV、VLEN128、
单线程、cold-data，128 MiB working set 大于 64 MiB LLC，cache eviction 不计时，
Full 搜索预算为 12。

由于部分 J 没有 `symbolic-medium-v1` 所需的 batch/attention 维度，结果分成两组，
不跨 workload profile 混算几何平均：

- medium 四项：`G_schedule=1.150×`、`G_full=1.055×`；
- natural 四项：`G_schedule=1.732×`、`G_full=1.184×`。

GroupNorm 是最强的解耦案例：`5428.924 -> 985.764 -> 502.242 us`，即 retile
`5.507×`，Full 再提升 `1.963×`。Liger RoPE、Qwen2VL MRoPE 与 Flash Linear
则清楚显示现有 Full 尚未增加有效状态；其中模型级 Flash Linear 在单核 RVV 上
一次 Full 已约 12.75 秒，后续应另设 target-appropriate contraction headline，
但不能用缩小 shape 覆盖本次极限迁移负例。

加上原七项和上一批八项，目前 **23/24** 个正式 J 已有 RVV 三柱。唯一缺口
FlashAttention forward 在 intake 阶段因 `logical_update_relation` 尚未被 RVV
ContractView 消费而 fail closed；没有绕过 B 强行发射。

逐项数据在 [summary.csv](summary.csv)。
