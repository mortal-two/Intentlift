# Stage 3：Gather 与 RowMask 两个新增 J 的 Production Backend 结果

本批在既有 22 项上加入 `tilelang:gather2d` 与
`tilelang:rowmask_where2d`。两项均由 accepted J materialize 为 typed
Intent MLIR；目标代码来自 C++/MLIR lowering，Python 只负责输入、候选编排、
Q/D 与计时。

固定指标为：

```text
G_schedule = C-Fixed / Schedule-Only
G_full     = Schedule-Only / Full
```

GPU 的外部基线为相同计算与 launch contract 的 TileLang Source；RVV 为独立
单线程 C/RVV Native。全部结果均为 code-warm/data-cold，缓存驱逐在计时区间外。

## 1. Workload

| Kernel | 输入 shape / dtype | 输出 | accepted C |
|---|---|---|---:|
| Gather2D | data `f32[64,64]`，row/column index 各 `i32[256]` | `f32[256]` | threads=128 |
| RowMask Where2D | data `f32[65536,64]`，row mask `i1[65536]` | `f32[65536,64]` | threads=64 |

Gather 保留 closure Harness 的 `M=64,N=64,L=256`；其 J 只有 `M` 是符号维，
没有通过放大固定的 `L=256` 制造收益。RowMask 的 `M` 是 J 符号维，本批采用后端
natural workload 的 `M=65536`，没有改写 J、dtype 或固定宽度 `N=64`。

## 2. NVIDIA RTX 5090D

单位：微秒。

| Kernel | C-Fixed | Schedule-Only | Full | Source | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gather2D | 17.616 | 17.536 | 17.248 | 12.472 | 1.005× | 1.017× | 0.804× |
| RowMask Where2D | 41.456 | 26.672 | 21.712 | 36.560 | 1.554× | 1.228× | 1.648× |

两个结构适用项的 `G_full` 几何均值为 `1.118×`。RowMask winner 是
`cuda.elementwise.rowmask.zero-fill.v1`，选择 32 threads、`vector<8xf32>`；
Gather winner 是 64-thread composed-index path。

## 3. NVIDIA H100 80GB HBM3

H100 使用 `sm_90a` 独立编译和搜索，没有复用 5090D winner。

| Kernel | C-Fixed | Schedule-Only | Full | Source | `G_schedule` | `G_full` | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gather2D | 15.520 | 15.392 | 15.248 | 11.848 | 1.008× | 1.009× | 0.761× |
| RowMask Where2D | 53.456 | 21.760 | 19.824 | 46.264 | 2.457× | 1.098× | 2.328× |

两个结构适用项的 `G_full` 几何均值为 `1.053×`。RowMask winner 是
256 threads、`vector<4xf32>`；H100 的额外 Full 收益稳定接近、但没有包装成
`1.1×`。其主要收益已经出现在 C 解冻后的 target retile，Full 再减少 false-row
输入流量。

## 4. RVV VLEN=128，单线程

| Kernel | C-Fixed | Schedule-Only | Full | Native | `G_schedule` | `G_full` | Native/Full | Full winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gather2D | 46.030 | 36.750 | 36.750 | 20.511 | 1.253× | 1.000× | 0.562× | materialized，parallel tile=4 |
| RowMask Where2D | 6418.870 | 6410.100 | 6410.100 | 6737.480 | 1.001× | 1.000× | 1.050× | materialized fixed-vector |

RVV 搜索真实包含 RowMask 的四个 zero-fill object：pass 将 mask load 提到 outer
row loop，并分别生成 scalar 或显式 `vector<4xf32>` 的 true/false row path；所有
候选均通过 Q/D。它们没有胜出：在该单核目标上，完整输出写流量和每行分支成本
抵消了跳过一半输入读的收益。因此 Full 正确选择 canonical materialized path，
而不是为了与 GPU 一致强行选结构候选。

Gather 目前只有通用 Linalg/RVV 间接 load 路径；parallel retile 带来 `1.253×`，
但尚无 RVV vector-gather/composed-index state，所以 `G_full=1.000×` 是明确能力缺口。

## 5. 新增的真实实现空间

### Gather2D

- C-Fixed/Schedule-Only：逐级实现两次 typed gather；第一级行结果在 workgroup
  shared memory 中可见，Schedule-Only 只改变 threads。
- Full：复合两级 runtime index map，直接执行 `data[row[k], col[k]]`，删除
  256×64 的逻辑中间行和 block barrier。
- 两条路径擦除 schedule 后仍具有不同的 materialization/index topology。
- `L=256` 时两条 target kernel 都受 launch floor 主导，因此结构消除只呈现
  1–2%；TileLang Source 原本就是专门的 direct gather，当前 Full 仍慢于 Source。

### RowMask Where2D

- Canonical：严格执行 `data * cast(mask)`，保留 NaN/Inf×0 行为。
- GPU Full：按 J 的 rank-1 row broadcast 恢复统一行谓词；false row 写零且不读
  data，并搜索 packet width、threads 与 grid-stride logical tile。
- RVV Full 候选：outer row 只读一次 mask，inner feature loop 连续向量化；
  zero-fill 同样不读 false-row data。
- zero-fill 会改变 NaN/Inf×0，因此被明确标成 observation-changing，仅在 Full 中
  进入，并由每个 workload 的 Q/D 决定可接受性；它没有被伪装成 B 的静态许可。

## 6. 对 24 项总览的影响

沿用原 22 项汇总的几何均值并加入本批两项，GPU 约为：

| Target | 范围 | `G_schedule` | `G_full` | Source/Full |
|---|---|---:|---:|---:|
| 5090D | 24 项 | ≈1.266× | ≈1.055× | ≈1.291× |
| H100 | 24 项 | ≈1.366× | ≈1.138× | ≈1.409× |

5090D 为 23/24 Full 不慢于 Source；H100 为 22/24。新增负例均为小型
Gather2D；它说明“J 打开结构路径”不等于“该 workload 必然有足够工作量摊薄
launch floor”。

## 7. 机器报告

- `5090d.cold.report.json`
- `h100.cold.report.json`
- `rvv-vlen128-single-thread.cold.report.json`

报告包含每个候选的具体 MLIR pass、schedule、launch/object、correctness 与 timing。
