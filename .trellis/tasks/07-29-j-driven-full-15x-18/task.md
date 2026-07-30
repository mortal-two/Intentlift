# Task 18：J-Driven High-Leverage Full Search（真实 1.5× 结构机制）

> 状态：阶段性收口；三柱边界、首批 Sparsemax/RVV 结构与当前性能已落地，未完成的 1.5× 目标由 Task 19 接管
>
> 类型：production backend 能力扩展、CUDA/H100/RVV 单线程真实 MLIR lowering search
>
> 继承：Task 17 的 Fresh-14、三柱定义、Source baseline、cold-data GPU 测量和 Q/D；本任务只加强 Full 的有效实现空间

## 1. 第一性目标

当前后端已能从正式 J 生成真实 Intent MLIR、CUDA cubin 和 RVV object，但 5090D Fresh-14 的
`Full/Schedule-Only` 几何均值只有约 `1.055x`。候选数量已经很大，缺少的是能改变主瓶颈的实现：

```text
J 允许的合法实现空间
    远大于
当前后端真实可达的 MLIR 实现空间
```

本任务不继续堆叠布尔开关，而是扩展后端从 J 真正构造不同 dataflow、并行层次、规约/扫描结构和
materialization 边界的能力。目标是得到多个可复现的 `Full/Schedule-Only >= 1.5x` 结果，并证明收益来自
J 打开的实现自由度，而不是弱化 baseline、改变 workload 或修改 tolerance。

## 2. 三柱在本任务中的固定定义

```text
C-Fixed:
  DirectLower(J, Project_C(C,target), target)
  exactly one output；不枚举、不测量、不选路径

Schedule-Only:
  同一个 DirectLower
  只解冻 threads/warps、tile、vector、unroll、pipeline、grid 等 schedule

Full:
  完整包含 Schedule-Only
  额外枚举由 A/B/target/QD 准入的真实 MLIR transformations
  每个 transformation result 独立获得同等级 schedule search
```

`DirectLower` 可以是高质量、确定性的 compiler lowering；不能为了做大 Full 把它退化成一 op 一 launch。
Full 的增益只能来自 Schedule-Only 擦除 schedule 数值后仍无法到达的实现结构。

工程上先收直两点，但不把它们扩成独立子项目：

1. C-Fixed 由 C 的确定性 target legalization 直接构造，不能复用 schedule candidate enumeration 再取最近点；
2. DirectLower 按 typed J graph kind 唯一分派 lowering rule，不通过性能、cost model或多路径比较决定结果。

## 3. “1.5×”验收口径

不把物理上不可能的目标写成实验技巧。Fresh-14 中已单趟读写且接近 compulsory-memory traffic 的大
elementwise/normalization，未必存在 1.5× 余量。因此使用两层固定口径：

### 3.1 全量口径

- 5090D、H100、RVV 单线程均保留 Fresh-14 全体，逐项报告 C-Fixed、Schedule-Only、Full；
- GPU 同 shape/dtype/cache/timer 执行原始 Triton Source；RVV 执行独立 Native RVV baseline；
- 报告 all-corpus geometric mean、median、`#>=1.0/#>=1.05/#>=1.5/#<1.0`；
- 不允许删除负例、只汇总正例或由单个异常值支配结论；
- all-corpus 工程目标：`G_lowering >= 1.15x`，但未达到时保留真实结果。

### 3.2 高杠杆结构口径（1.5×硬门槛）

- 至少 **5 个 kernel-target cells** 达到 `Full/Schedule-Only >= 1.50x`；
- 覆盖至少 **3 个不同 kernel**、**3 种不同 MLIR transformation traces** 和 **2 个 target**；
- 5090D 与 H100 各至少有 1 个 `>=1.50x` 的非 schedule winner；
- RVV 单线程至少有 2 个 `>=1.50x` 的真实结构 winner；
- 预先由 J graph facts 判定的 structural-applicable subset，其 `Full/Schedule-Only` 几何均值目标
  `>=1.50x`；该 subset 单列但不能替代 Fresh-14 全量结果；
- 同一 kernel 的两个 GPU target可以形成两个 cells，但不能冒充两种 transformation；
- 每个 `>=1.50x` winner 必须通过相同 Q/D、fresh finalist remeasurement，并保留 Source/native 对比。

这些门槛是后端工程目标。若完成真实能力、profile 和完整搜索后仍不成立，任务如实失败，不能改
Source、shape、timed boundary、Q、tolerance或 direct lowering来达标。

## 4. 什么才算有效 Full 空间

Python 候选名、候选数量、pass开关数量都不计入能力。一个 Full state 必须满足：

1. 从同一个 immutable J/Intent MLIR 出发；
2. 由真实 C++/MLIR pass改变 dataflow、loop/reduction/scan topology、materialization或 stage graph；
3. pass admission只读取 A/B/target/QD和当前 typed IR，不读取 kernel 名、source path或 benchmark allowlist；
4. post-pass transformation record 与 launch plan可观察到变化；
5. 独立调 schedule 后仍形成不同的可执行状态；
6. target execution 与 RefJ/QD直接比较通过。

有效状态直接使用现有字段辨别，不增加 hash、冻结、receipt或 artifact 管理：

```text
pipeline / fast-path identity
lowering_path_kind
transformation_trace
reduction_topology
grid / block / shared memory
per-thread tile / vector width / stages / split factor
```

报告增加四个性能相关数字即可：raw admitted count、unique effective states、unique measured states、winning
state。相同 transformation/launch 的重复组合不算扩大 Full。

## 5. 当前物理判断

### 5.1 不作为 1.5× 主攻对象

- SwiGLU、GeGLU、DYT 的 direct lowering 已是单 launch、read-once、多输入直接融合，当前自然 shape接近
  logical bandwidth roof；其主要自由度属于 Schedule-Only；
- 已单趟读写、接近带宽下限的 RMSNorm/LayerNorm不能仅靠改 reduction spelling达到 1.5×；
- 当前 SplitK `D=256` 约 15 us，multi-warp merge受 launch/小流量下限限制；没有消除 partial O/LSE
  materialization时，继续调 merge geometry难以达到 1.5×；
- FMA、reciprocal、native transcendental 的现有收益多为 0--10%，只作 numeric state，不作为主要故事。

上述 kernel仍进入全量实验，但不为了门槛制造退化路径。

### 5.2 当前已有高杠杆信号

- Flash backward preprocess：padded subgroup-feature已在 5090D 达到约 `1.66x` Full/Schedule；
- Sparsemax：sorted-prefix execution-domain state在 5090D/H100约 `1.16--1.18x`，仍有减少 scan/tail
  work的空间；
- RVV Sparsemax：producer-consumer fusion相对 materialized Schedule约 `1.70x`，说明结构收益存在，但
  绝对性能仍远慢于 Native RVV；
- RVV 当前只做了普通 Linalg fusion/tiling和 LLVM scalable-vector flags，尚没有真正的 dynamic-VL
  strip-mined row lowering、vector reduction tree或prefix scan。

## 6. P0：GPU Sparsemax 与 Scan

目标：把现有 `1.16--1.18x` 扩展成真实 `>=1.5x` scan机制，并保留严格准入边界。

实现：

1. warp-local prefix + CTA-level segment-prefix，而不是普通 reduction伪装 scan；
2. support predicate、support count、supported-value sum和tau epilogue共用 register/shared frontier；
3. sorted-prefix early-stop以 active prefix为工作量，不扫描已证明无效的 tail；
4. 分别搜索 lanes-per-row、warps-per-row、elements-per-lane、shared frontier和CTA row grouping；
5. H100/5090D分别调度，不复用 winner；
6. RVV实现 vector prefix/segmented scan，不复用 CUDA物理结构。

准入：

- 普通 J 的显式 `scan(sum,axis)`足以开放保持相同scan语义的不同物理实现；
- sorted early-stop目前只能使用明确的 execution-domain/QD admission；现有 J/B没有
  `sorted_x=sort_descending(x)` relation，不能写成 contract-proven；
- `require-sorted-prefix` 对非匹配图继续 fail closed。

性能来源：减少 tail scan、barrier/shared traffic和串行 prefix depth；不能靠移除输出所需的完整 `x` read与
`y` store伪造收益。

## 7. P0：GPU Row Statistics / Softmax

目标：为当前一个 block/row 的 read-once direct lowering加入真正不同且有尺寸适用区间的结构。

实现：

1. **warp-per-row / multi-row CTA / persistent rows**：让一个CTA并发处理多行，减少宽线程块的同步与
   register pressure；
2. **CTA-per-row**：保留当前结构作为Schedule-Only基线；
3. **multi-CTA long-row partial/finalize**：只传递 partial statistics，不允许第二阶段重读完整 producer/input；
4. **joint statistics traversal**：mean/variance、sum/sumsq、max/sumexp等共享同一次输入 traversal和
   register/shared frontier；
5. **online Softmax state**：max、normalizer与epilogue形成真实online/streaming结构，并由数值Q/D准入；
6. **Welford / raw moment / hierarchical fold**：作为真实不同的 reduction graph，各自独立调 schedule；
7. producer、statistics和epilogue只在能减少现有materialization/重读时形成Full state。

优先 kernel：Softmax、Flash one-pass LayerNorm、fused-add RMSNorm、KLDiv、PolyNorm、xFormers RMSNorm-add。
普通 RMSNorm/LayerNorm若 compulsory-traffic profile显示无1.5×余量，则保留全量负结果，不继续堆开关。

## 8. P0：RVV 真实目标结构

RVV只做单线程，必须是真实 MLIR→LLVM→RVV object，不允许Python target emission。

### R1. Dynamic-VL strip mining

- 在bufferize前生成 `scf.for` + scalable vector mask/transfer的真实MLIR循环；
- VLEN/tail/strip由target schedule决定；
- post-pass LLVM必须出现 scalable vectors和相应 `vsetvli`，不能只设置LLVM flag。

### R2. Vector reduction tree/finalize

- 从J的reduce axis、dtype、identity构造vector partial accumulators；
- 使用显式horizontal reduction/finalize，不把普通Linalg loop tiling记作新结构；
- 支持ordered/reassociated状态时分别经过Q/D。

### R3. Producer--statistics--epilogue fused row loop

- 消除当前materialized Linalg graph的中间buffer和重复遍历；
- 读取B的readonly/non-alias/effect/output relation；缺失则fail closed；
- 面向RMSNorm、LayerNorm、Softmax、KLDiv、PolyNorm。

### R4. Vector prefix/segmented scan

- 对J中显式scan图构造RVV vector prefix state；
- threshold/reduction/epilogue融合；
- 首个目标为Sparsemax。

### R5. Unit-stride packing与不变量提升

- 只在J access/domain证明连续、无alias时collapse/pack；
- 形成真实vector load/store，减少无效 `vsetvli`和标量address work；
- 作为上述结构的target schedule/placement，不单独冒充headline transformation。

RVV除要求结构 `>=1.5x` 外，还必须改善绝对性能；至少3个代表kernel的Full延迟达到Native RVV的
`<=1.25x`，否则只能称结构消融成功，不能称后端有竞争力。

## 9. P1：Flash 与 SplitK

### Flash preprocess

- 将已获胜的 subgroup-feature推广为通用padded-reduction matcher；
- 搜索feature packet、subgroup count、CTA mapping与H100/5090D target schedule；
- 覆盖所有A/B匹配图，不按Flash kernel名触发。

### SplitK

- 保留当前warp-owned weighted merge；
- 只有profile表明D/parts存在并行余量时实现CTA-cooperative D partition与shared/register packet merge；
- LSE max、sumexp、weight与numerator每个split只计算一次并广播；
- 当前J boundary不包含上游partial producer时，不能声称消除了O/LSE materialization；若该能力需要更大J，
  明确记录为本批artifact不可达，不在后端猜测。

## 10. 搜索与消费规则

每个 transformation描述：

```text
A/typed-IR match
required B facts
target preconditions
current-state preconditions
resulting MLIR transformation
independent schedule domain
Q/D policy
```

禁止：

- kernel-name recipe；
- source-path/benchmark allowlist；
- Python生成目标kernel；
- 从tolerance反推任意数值放宽；
- 在Schedule-Only中混入fusion、materialization、reduction/scan topology或numeric变化；
- 用候选数量代替有效实现空间；
- 只给新结构一个固定schedule，再与充分调优的Schedule-Only比较。

C只用于：

1. C-Fixed的一次确定性合法投影；
2. Full/Schedule有限预算遍历顺序。

C不得改变准入集合或converged Full性能上限。

## 11. 执行顺序

### Phase A：收直 direct/schedule 边界

1. C-Fixed直接由C构造一个target schedule，不调用candidate enumerator；
2. typed graph kind唯一分派DirectLower；
3. 列出Schedule-Only真正开放的坐标及winner变化；
4. 将elementwise traversal-topology、SplitK broadcast/multi-warp等非纯schedule开关固定到Full。

### Phase B：先打三个高杠杆 vertical slices

1. CUDA Sparsemax vector/block prefix + early-stop；
2. RVV dynamic-VL fused row reduction + explicit vector tree；
3. CUDA multi-row/persistent statistics + online Softmax。

每条slice完成后立即在一个J上做A/B准入、post-pass MLIR、Q/D和target性能闭环；没有理论headroom或测量
低于1.2×的机制立即停止扩散，换下一个结构，不在14项横向复制微收益。

### Phase C：推广到Fresh-14

通过A/B graph facts匹配所有适用J；每种结构独立调schedule；5090D、H100、RVV单线程分别搜索。

### Phase D：正式实验

1. 5090D/H100使用natural workload、cold-data、CUDA event与Source/Full交替paired batches；
2. RVV单线程使用相同Fresh-14和独立Native RVV baseline；
3. 同时输出全体、structural-applicable subset和每种winning transformation的结果；
4. C-prior equal-budget结果保持独立，不进入三柱性能上限。

## 12. 完成标准

1. 达到第3.2节的真实1.5×结构门槛，或在全部高杠杆机制实施/验证后明确记录未达成；
2. 至少3种新结构由真实C++/MLIR pass生成，并在post-pass state中可见；
3. Fresh-14在5090D、H100、RVV单线程全部进入同一三柱controller；
4. 每个Full结构拥有独立、充分的schedule search；
5. 每个winner通过相同Q/D，GPU保持原始Triton Source对比；
6. 不弱化DirectLower、Schedule-Only、Source或Native RVV baseline；
7. 不修改accepted J、Q、tolerance、timed boundary或natural workload追求收益；
8. 不增加hash、冻结、receipt或与性能无关的artifact管理；
9. 全量测试通过，代码与task提交，工作区干净。

## 13. 当前实测进展（2026-07-29）

### 13.1 新增 CUDA Sparsemax 结构

已实现并进入 Full 搜索：

```text
sorted_prefix_multirow_warp_early_stop
sorted_prefix_warp_scan_wide_epilogue
sorted_prefix_multirow_warp_scan_wide_epilogue
上述 wide epilogue 的 Q/D-gated cross-multiply support 状态
```

其核心是把 active-prefix 扫描和完整输出行的并行度解耦：32 个线程负责 prefix/support，
64--1024 个线程负责整行 epilogue；多个逻辑行还可共享一个 CTA。交叉乘法状态以正 rank 的结构证明将
`z > (prefix-1)/rank` 改为 `rank*z > prefix-1`，明确作为数值观察可能变化的 Full 候选经过 Q/D，
不写回 J，也不混入 Schedule-Only。

同一 natural workload `[2048,4096]`、f32、cold-data：

| Target | C-Fixed | Schedule-Only | Full oracle | Full/Schedule | paired Full | Triton Source | Source/Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090D | 76.736 us | 71.936 us | 51.744 us | 1.390x | 52.400 us | 85.136 us | 1.625x |
| H100 | 51.712 us | 46.304 us | 37.504 us | 1.235x | 38.272 us | 57.584 us | 1.505x |

结论：该路径是真实、跨 target、绝对性能有效的新结构，但没有达到 `Full/Schedule >= 1.5x`，因此
不能作为 1.5x 门槛的完成项；停止继续堆叠 Sparsemax 局部变体，转向其他有更大 headroom 的图。

### 13.2 已有 RVV natural-shape 强基线

Task 16 的 native RVV、单线程、natural-headline 三柱结果仍是本任务必须保留并继续加强的能力基线。
其中 `Full/Schedule` 已达到：Flash backward preprocess `4.478x`、Liger RMSNorm `3.780x`、
PolyNorm `3.238x`、fused-add RMSNorm `2.728x`、Flash one-pass LayerNorm `2.658x`、LayerNorm
`2.474x`、KLDiv `1.778x`、Sparsemax `1.704x`、GeGLU `1.698x`、SwiGLU `1.516x`。

这些 winner 均由真实 MLIR pipeline 生成，`python_target_kernel_emission=false`；主要结构目前仍集中在
`rvv.linalg.producer_consumer_fusion.v2`。因此下一阶段的目标不是重复实现融合，而是增加第二、第三种
真实机制（显式 vector reduction/finalize、prefix scan、producer--statistics--epilogue row loop），并
改善 PolyNorm/LayerNorm/Sparsemax 等尚落后于 Native RVV 的绝对性能。
