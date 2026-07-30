# Stage 2 新增 8 个 J：Production Backend 结果

本目录记录 Task 22 的直接实验结果。固定名称如下：

```text
C-Fixed
Schedule-Only
Full
Source（5090D/H100）或 Native（RVV）
G_schedule = C-Fixed / Schedule-Only
G_full     = Schedule-Only / Full
```

所有 GPU 数字均来自同一 kernel、同一 natural/model-level shape、同一 dtype、同一 observable boundary 的
code-warm/data-cold CUDA-event 测量；L2 eviction 位于计时区间外。RVV 固定单线程、CPU 0、code-warm/data-cold。
Python 只负责输入绑定、Q/D、搜索编排与计时；表中 Full executable 均由 C++/MLIR pass 生成目标代码。

## 1. Natural workload

| Kernel | 输入 shape / dtype | observable 输出 |
|---|---|---|
| JSD | 两个 `f32[4096,4096]`，label `i64[4096]`，标量 | 两个 `f32[4096,4096]` |
| TVD | 两个 `f32[4096,4096]`，label `i64[4096]`，标量 | 两个 `f32[4096,4096]` |
| xFormers RMSNorm | `f32[4096,12288]`，weight `f32[12288]`，epsilon | `f32[4096,12288]` |
| CrossEntropy | logits `f32[1024,2048]`，label `i64[1024]` 及原 Source ABI auxiliary inputs | inout logits `f32[1024,2048]`，loss `f32[1024]` |
| GroupNorm | `f32[8,32,512]`，weight/bias `f32[256]` | `f32[8,32,512]`，mean/rstd `f32[8,32]` |
| Liger RoPE | Q `f16[4,2048,16,64]`，K `f16[4,2048,8,64]`，cos/sin `f16[1,2048,64]` | 两个 in-place f16 输出 |
| Qwen2VL MRoPE | Q/K `f16[8,1024,8,128]`，cos/sin `f16[3,8,1024,128]` | 两个 in-place f16 输出 |
| Flash Linear | A/B `f16[1024,8192]`，bias `f16[1024]` | 两个 `f16[1024,1024]` |

这些 valuation 只改变已符号化的 J extent 和对应 Harness runtime/C 参数；accepted J 本身未被改写。

## 2. NVIDIA RTX 5090D

单位：微秒。

| Kernel | C-Fixed | Schedule-Only | Full | Source | G_schedule | G_full | Source / Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSD | 149550.278 | 149550.278 | 181.088 | 197.232 | 1.000 | 825.843 | 1.086 |
| TVD | 189.568 | 182.176 | 182.176 | 201.744 | 1.041 | 1.000 | 1.109 |
| xFormers RMSNorm | 71937.981 | 71937.981 | 270.272 | 283.904 | 1.000 | 266.169 | 1.048 |
| CrossEntropy | 5211.520 | 5211.520 | 22.688 | 43.104 | 1.000 | 229.704 | 1.896 |
| GroupNorm | 463.296 | 463.296 | 15.296 | 33.328 | 1.000 | 30.289 | 2.174 |
| Liger RoPE | 4911.648 | 4911.648 | 40.320 | 55.312 | 1.000 | 121.817 | 1.370 |
| Qwen2VL MRoPE | 15384.416 | 15384.416 | 54.976 | 65.392 | 1.000 | 279.839 | 1.184 |
| Flash Linear | 238.720 | 150.880 | 142.656 | 203.392 | 1.582 | 1.058 | 1.422 |

八项几何均值：`G_schedule=1.064×`，`G_full=52.201×`，`Source/Full=1.364×`。去掉物理上没有额外
结构空间的 TVD 后，七项 structural-applicable subset 的 `G_full=91.847×`。

## 3. NVIDIA H100 80GB HBM3

单位：微秒。H100 为 `sm_90a` 独立重新编译并重新搜索，没有复用 5090D winner。

| Kernel | C-Fixed | Schedule-Only | Full | Source | G_schedule | G_full | Source / Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSD | 193985.352 | 193985.352 | 102.944 | 128.544 | 1.000 | 1884.377 | 1.267 |
| TVD | 134.400 | 101.536 | 101.536 | 118.304 | 1.324 | 1.000 | 1.173 |
| xFormers RMSNorm | 94019.554 | 94019.554 | 143.328 | 162.064 | 1.000 | 655.975 | 1.121 |
| CrossEntropy | 7376.384 | 7376.384 | 18.656 | 42.960 | 1.000 | 395.389 | 2.293 |
| GroupNorm | 731.232 | 731.232 | 15.872 | 35.664 | 1.000 | 46.071 | 2.203 |
| Liger RoPE | 7461.664 | 7461.664 | 33.152 | 45.040 | 1.000 | 225.074 | 1.337 |
| Qwen2VL MRoPE | 20961.504 | 20961.504 | 47.072 | 50.352 | 1.000 | 445.307 | 1.017 |
| Flash Linear | 312.448 | 250.560 | 138.912 | 250.160 | 1.247 | 1.804 | 1.660 |

八项几何均值：`G_schedule=1.065×`，`G_full=89.373×`，`Source/Full=1.445×`；七项
structural-applicable subset 的 `G_full=169.805×`。

## 4. RVV VLEN=128，单线程

单位：微秒。Native 是独立 C/RVV 实现，与 IntentIR 的 MLIR、搜索候选和 target object 无共享；runner 在计时前
逐输出与闭合参考做相同 Q/D。大 shape 项先采用一次 cold sample 完整遍历搜索空间，较小项采用三轮中位数；这不
影响同一行三柱的配对关系，论文 headline 前可统一增加重复数。

| Kernel | C-Fixed | Schedule-Only | Full | Native | G_schedule | G_full | Native / Full | Full concrete path |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| JSD | 4396340.920 | 3266021.966 | 1869142.271 | 2174966.827 | 1.346 | 1.747 | 1.164 | producer-consumer fused |
| TVD | 584868.125 | 322229.720 | 56346.699 | 43817.502 | 1.815 | 5.719 | 0.778 | producer-consumer fused scalable-vector |
| xFormers RMSNorm | 2442806.274 | 520851.389 | 107896.594 | 180319.352 | 4.690 | 4.827 | 1.671 | fused explicit vector reduction |
| CrossEntropy | 107725.774 | 73974.100 | 73974.100 | 56122.898 | 1.456 | 1.000 | 0.759 | materialized fixed-vector reduction |
| GroupNorm | 3687.497 | 1133.125 | 513.522 | 742.663 | 3.254 | 2.207 | 1.446 | fused joint vector reduction |
| Liger RoPE | 778428.254 | 618946.302 | 618946.302 | 272488.851 | 1.258 | 1.000 | 0.440 | materialized fixed-vector |
| Qwen2VL MRoPE | 1231349.434 | 882964.114 | 882964.114 | 53462.706 | 1.395 | 1.000 | 0.061 | materialized fixed-vector |
| Flash Linear | 12515014.181 | 12436121.849 | 12436121.849 | 92347719.883 | 1.006 | 1.000 | 7.419 | materialized scalable-vector contraction |

八项几何均值为 `G_schedule=1.769×`、`G_full=1.792×`、`Native/Full=0.863×`。RoPE/MRoPE 的负结果直接
说明 RVV 当前通用 materialized lowering 尚未达到 GPU 已有的 N-D indexed/in-place fused topology；它们不能
被写成 Full 收益。Flash Linear 保持自然 shape 实测，没有以缩小 K/M/N 换取快速完成；其
`G_schedule=1.006×、G_full=1.000×` 也明确表明当前 RVV contraction 尚无可用的专用 K 分块或结构路径。

## 5. 结构证据与归因

- JSD、CrossEntropy、GroupNorm、xFormers RMSNorm 的 GPU Full 都落到单 kernel read-once row-reduction，删除
  generic 路径中的中间 tensor、重复 traversal 和多次 dispatch。
- Liger RoPE 与 Qwen2VL MRoPE 的 GPU Full 从 typed access/index/broadcast 关系恢复 paired rotation；一个 C++
  elementwise lowering 直接产生两个 observable output，不物化 rotated-half、concat 或 gather tensor。
- Flash Linear 的 Schedule-Only 固定 canonical contraction 只搜索 M/N/K tile 与 stages；Full 额外进入真实
  `mma.sync` MLIR path。H100 winner 为 `64×128×64, stages=2`，5090D 独立选出自己的 MMA schedule。
- TVD 是保留的边界项：GPU canonical lowering 已是单遍 elementwise，Full 没有新的合法结构；RVV 则因通用
  producer-consumer fusion 将 14 个 Linalg op 降到 4 个而仍有 `G_full=5.719×`。
- RVV xFormers RMSNorm winner 将 17 个 Linalg op 降到 3 个，并生成真实 scalable-vector reduction；GroupNorm
  winner 将 54 个 Linalg op 降到 12 个并形成 joint vector reduction。
- RVV Flash Linear 的 27 个真实目标对象全部通过 correctness；winner 仍是 materialized scalable-vector
  contraction。通用 reduction tiling 不会被误当作 contraction K 分块，因此本项如实保留 `G_full=1.000×`。

原始机器可执行报告位于本目录的 `5090d/`、`h100/` 与 `rvv/`，其中包含每个候选的实际 pass、schedule、
launch/object、post-pass topology、correctness 和 timing 数据。
