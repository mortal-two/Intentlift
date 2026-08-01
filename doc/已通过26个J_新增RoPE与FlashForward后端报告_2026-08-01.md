# 已通过 26 个 J：新增 Llama4 RoPE 与 FlashAttention forward 后端报告

日期：2026-08-01

本文在已有 24 项总览基础上加入两个新闭合 J：

1. Llama4 RoPE：`/tmp/intentir-rope-auto-certified-20260801-final/liger_llama4_rope/gate2-recheck-001`
2. FlashAttention forward：`/tmp/intentir-flash-fwd-q-closure2-20260801/flash_attn_flash_attn_triton_fwd_kernel/gate2-recheck-001`

前 24 项的逐项矩阵继续以 `已通过24个J_三目标后端总览_2026-07-31.md` 为准。本文只增加真实后端能力、
新两项的完整 GPU 四柱结果和当前 RVV 状态，不增加与实验无关的冻结或摘要材料。

## 1. 实验口径

沿用既定四柱，不改名：

```text
C-Fixed      = J 经 deterministic direct lowering，并固定 C schedule
Schedule-Only = 同一 direct lowering，只解冻 tile/threads/stages
Full          = 在 Schedule-Only 可达状态之外，搜索 J 允许的 materialization、
                register lifetime、shared layout 与跨 head reuse 状态
Source        = 同一 workload 上的原 Triton Source
```

其中：

```text
G_schedule = C-Fixed / Schedule-Only
G_full     = Schedule-Only / Full
```

GPU 使用 `natural-headline-v1`、code-warm/data-cold、CUDA event single-shot；L2 eviction 不在计时区间内。
5090D 与 H100 各自搜索，不搬运 winner。RVV 固定单线程、CPU affinity、code-warm/data-cold。

## 2. Natural workload

| Kernel | dtype | headline shape | observable output |
|---|---|---|---|
| Llama4 RoPE | Q/K/frequency `f16`，`imag_sign` `f32` | Q `[8,1024,8,128]`，K `[8,1024,4,128]`，frequency `[1024,64,2]` | in-place Q、K |
| FlashAttention forward | Q/K/V/Out `f16`，LSE/TMP `f32` | Q/K/V `[8,1024,32,64]`，LSE/TMP `[8,32,1024]` | Out、LSE、TMP |

Flash 的 TMP 是 J 中可观察的 in/out boundary。Full 可以删除循环内部冗余 round-trip，但最终仍必须产生同一
TMP 观察；因此不能把 128-key 语义更新随意拆成改变 TMP 的两个 64-key 更新。

## 3. 新两项 GPU 结果

### 3.1 5090D

| Kernel | C-Fixed (µs) | Schedule-Only (µs) | Full (µs) | Source (µs) | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama4 RoPE | 54.816 | 38.016 | 38.016 | 65.920 | 1.442× | 1.000× | 1.729× |
| FlashAttention forward | 648.480 | 610.240 | 518.176 | 518.912 | 1.063× | 1.178× | 0.999× |

Flash 的 Full winner 为：

```text
BM=64, BN=128, 4 warps, stage=1
direct V ldmatrix.trans
shared V row padding=8
paired B-fragment load
streamed P-fragment lifetime
```

这条路径把 Schedule-Only 的约 610 µs 降至约 518 µs，并达到 Triton Source parity。

### 3.2 H100

| Kernel | C-Fixed (µs) | Schedule-Only (µs) | Full (µs) | Source (µs) | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama4 RoPE | 73.056 | 35.968 | 31.392 | 72.688 | 2.031× | 1.146× | 2.350× |
| FlashAttention forward | 1060.128 | 1011.296 | 836.064 | 352.464 | 1.048× | 1.210× | 0.338× |

H100 也选择 `direct V + row padding=8`，但选择 BM=32、2 warps、vectorized Q staging，未选择 paired B。
该 winner 的 dynamic shared memory 从未 padding 的 54,272 B 降至 38,912 B，寄存器为 168/thread；这正是
Full 收益的主要来源。H100 的 Full/Schedule 已达到约 1.21×，但与高度优化 Triton Source 仍有明显差距；
Source 交替确认阶段 Full latency 漂到约 1042 µs，因此本文使用 runner 的保守 paired Source/Full 数字，不把
候选筛选阶段的低点包装成外部竞争力。

## 4. J 实际打开的状态

### 4.1 Llama4 RoPE

后端从 A/B 中恢复 interleaved even/odd pair rotation，而不是按 kernel 名匹配。GPU 复用既有 paired-rotation
lowering，并增加 head-angle reuse；H100 winner 让两个相邻 head 共享同一 angle load。5090D 的成本结构不同，
搜索保留普通 paired path，所以 `G_full=1` 是真实负例而非路径缺失。

RVV 由 C++ MLIR pass 把相同 marker lower 为 `scf`/`memref`/`vector`。contiguous direct loop 是完整 J consumer，
因此属于 C-Fixed/Schedule canonical spine，不能再用 materialized 215-op fallback 人为抬高 `G_full`；Schedule-Only
可在同一路径内选择 scalar、固定 2W contiguous packet 和 scalable gather/scatter。Full 只额外开放 head-angle
reuse 这一不同 loop topology。Python 只枚举与运行 object，不发射计算 kernel。另有独立、完全相同 interleaved
arithmetic 的单线程 C/RVV baseline。

### 4.2 FlashAttention forward

后端消费完整 structured scan J：batch/head 显式 grid、QK/PV MMA、在线 softmax carry、LSE 与 TMP boundary。
三柱之间的关键差别不是换一个 kernel 名：

- C-Fixed：固定 BM/stage，保留 TMP round-trip 和 materialized V transpose；
- Schedule-Only：只搜索 BM/stage，仍保留同一 direct structure；
- Full：可删除内部 TMP round-trip、直接从 natural V 形成 PV fragment、改变 shared bank layout、配对装载相邻
  B fragments，并缩短 P fragment 生命周期。

Full 的 shared V row padding 不是新造轮子：它复用了 Flash backward `one_col_block` 路径已经验证过的 D-stride
padding思想。`one_col_block` 自身是 backward 专用的 query-column/update topology，不能假装成 forward；真正能
共享的 `ldmatrix.x4` B-fragment loader 已抽到 `FlashMMAFragments.h`，由 forward 和 backward 同时调用。

## 5. 26 项 GPU 总体状态

用旧 24 项几何均值与本次两项合并，当前近似总览为：

| Target | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|
| 5090D，26 项 | ≈1.264× | ≈1.057× | ≈1.293× |
| H100，26 项 | ≈1.373× | ≈1.141× | ≈1.360× |

这些合并值使用旧 24 项文档中的三位小数几何均值，因此标记为近似值；逐项结论以上述两张新表与旧 24 项矩阵
为准。新增 Flash forward 同时给出一个重要正结果和一个明确缺口：J 确实打开了 target-specific shared-layout
空间并带来约 1.18–1.21× Full 增益，但 H100 的 Source gap 说明 register/layout/synchronization 代码质量仍需继续
提升。

## 6. RVV 状态

Llama4 RoPE 已完成真实 MLIR/LLVM/RVV object 生成与独立 Native baseline；正式目标机结果见下表。
FlashAttention forward 当前没有成熟 RVV tensor-core 等价路径，也没有可直接比较的 Triton-CPU Source，因此暂记
`N/A`，不以普通 C 循环冒充迁移性结果。

| Kernel | C-Fixed (µs) | Schedule-Only (µs) | Full (µs) | Native (µs) | `G_schedule` | `G_full` | Native/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama4 RoPE | 19479.978 | 19155.706 | 19155.706 | 14075.963 | 1.017× | 1.000× | 0.734× |
| FlashAttention forward | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Llama4 的 C-Fixed 是 O3 fixed 4-pair packet，Schedule/Full winner 是同一 contiguous topology 的 O2 fixed
4-pair packet。head-angle reuse 已真实进入 Full 并通过 Q，但没有胜出。独立 Native 与 Full 的两个 in-place
输出均为 exact match；当前 1.36× 代码质量差距主要来自 LLVM-RVV C 更成熟的 interleaved pair vectorization，
不是未消费 J 或 Python launch overhead。

## 7. 验证

- `ninja -C build intent-opt`：通过；
- 新 J/三柱边界/RVV vector path、materialization 与 selection tests：224 passed；
- 复用 shared MMA fragment helper 后的 Flash backward GPU gate：2 passed；
- 5090D：88 个 Flash concrete states 均由 C++/MLIR 编译为 cubin，目标侧 Q 后搜索；
- H100：同一 88 个 concrete states 独立编译为 `sm_90a` cubin，目标侧 Q 后搜索；
- Llama4 RVV：7 个有效 MLIR/LLVM/RVV objects（5 个 canonical contiguous schedule、2 个 Full head-reuse），
  目标侧逐一 Q 后搜索；不再运输 15 个不能代表完整 J consumer 的 materialized fallback。
