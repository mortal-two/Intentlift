# Task 21：J 驱动的 Full 强结构路径扩展

> 状态：执行中
>
> 类型：production backend 能力建设、真实 C++/MLIR 结构搜索、Fresh-14 跨目标消融
>
> 继承：Task 20 已完成第一轮结构能力闭环；本任务以 5090D/H100 为主线，把 Full 扩展到
> normalization、scan 和 padded/multi-input reduction。RVV 保留已有能力与后续横向入口，但不占用当前
> GPU 强路径的主要调优时间。

## 1. 第一性目标

本任务不是把某个 baseline 人为做慢，也不是继续堆 tile、warps、VL 或 candidate label。我们要让同一个
J 能被 lower 成多个真实、可执行、结构不同的 MLIR 状态，并让 Full 搜索发现固定结构的 schedule search
到不了的目标程序。

一个变化只有满足下列至少一项才计为 Full 新路径：

- 删除一次或多次完整 tensor traversal；
- 删除中间 tensor/buffer 的 materialization 与对应 load/store；
- 改变 reduction、scan、同步或跨 stage 的计算拓扑；
- 把 producer、statistics 与 epilogue 合成一个新的 live-state 区域；
- 在 Q/D 允许时，用新的数值观察顺序打开上述结构变化。

如果擦除 tile、vector width、threads、warps、pipeline、VL/LMUL 后两个 executable 的 dataflow、遍历数、
materialization 与同步图相同，它们只是同一结构的不同 schedule，不计为 Full 扩展。

`Full / Schedule-Only >= 1.50x` 是本任务对高杠杆结构单元的目标线。它不强迫已经单趟 read-once、接近
带宽或 launch 下限的 kernel 违反物理上限；这类项必须保留在 Fresh-14 全表中作为真实负例。

## 2. 三柱与外部 baseline

```text
C-Fixed:
  DirectLower(J) + C 的一次确定性 target schedule 投影
  一个 executable，不搜索

Schedule-Only:
  与 C-Fixed 完全相同的 DirectLower(J) 和目标程序结构
  只开放该结构内部的 schedule 参数

Full:
  完整包含 Schedule-Only 候选
  额外构造由 typed J、B hard facts、target capability 与 pass precondition
  共同准入的具体结构 MLIR 状态
  每个状态再独立搜索 schedule
```

内部主指标固定为：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_full     = latency(Schedule-Only) / latency(Full)
```

GPU 外部 baseline 是原始 Source kernel；RVV 外部 baseline 是同 shape、dtype、observable outputs 与 timed
boundary 的单线程 Native RVV。外部 baseline 用于说明绝对竞争力，不能替代 `G_full`。

## 3. 当前事实

### 3.1 已经成立的强结构信号

RVV joint statistics 已经证明结构空间是真实存在的，而不是论文术语：

- PolyNorm `[2048,4096]`：`Schedule-Only / Full = 28.668x`；
- LayerNorm `[4096,4096]`：`Schedule-Only / Full = 11.258x`；
- Flash one-pass LayerNorm `[4096,4096]` 首轮：`Schedule-Only / Full = 3.354x`。

这些收益来自共同 root 的一次 packet load、多 accumulator 同时存活、raw-moment 重写以及 producer fusion，
不是来自把 C-Fixed 做弱。CUDA 当前的正式强结构证据来自 Flash backward preprocess 的 interleaved
multirow live-accumulator，以及 Sparsemax 的 prefix/frontier/epilogue live state；DYT 在模型级自然 shape 上是
带宽负例，不再沿用早期小规模筛选中的强收益表述。

### 3.2 目前真正缺少的路径

- RVV Softmax 仍将 max、shift/exp、sum、divide lower 成多个 traversal/materialization；
- RVV 的显式 vector reduction 已支持 sum/max，但单独 vectorize reduction 并不会删除 exp tensor 与阶段流量；
- RVV multi-input、mixed-dtype、masked/padded row reduction 还不能统一进入 packet state；
- CUDA Sparsemax 已完成 prefix/support/tau、X prefetch、frontier ballot 与 leader-tau 状态；其剩余下限是完整
  X/output 流量与短 active-prefix 控制；
- CUDA Flash preprocess 已由通用 typed padded-reduction matcher覆盖 5090D/H100，并让两卡独立选择不同
  interleaved row geometry；
- CUDA direct Softmax 已经把输入行和 `exp(x-max)` 保存在寄存器 SSA 中，不能虚构一个“删除 exp tensor”
  的重复路径；普通 CUDA Norm 也已非常接近 read-once，除非 profile 证明 barrier、spill 或重新物化存在，
  否则它们不是优先纵切。

### 3.3 2026-07-30 CUDA 自然负载横向结果

当前执行优先级已切到 5090D/H100；RVV 只保留已通过编译与测试的能力，不继续占用本轮主要调优时间。
以下均为 `natural-headline-v1`、code-warm/data-cold、CUDA event。每个三柱 winner 使用 5 个 measurement
rounds；最终 target 与 Source 再做 8 个交替批次、每批 5 个 L2-evicted samples。`Source/target` 使用交替确认
阶段的 paired latency；内部 `G_schedule/G_full` 使用同一次候选空间测量，避免跨阶段时钟漂移污染消融。

5090D 正式横向表：

| Kernel | C-Fixed | Schedule | Full | `G_schedule` | `G_full` | Source/target |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU | 1458.208 | 1382.912 | 1382.848 | 1.054x | 1.000x | 1.021x |
| RMSNorm | 49.376 | 49.376 | 48.960 | 1.000x | 1.008x | 1.295x |
| Fused Add RMSNorm | 96.320 | 95.072 | 93.888 | 1.013x | 1.013x | 1.182x |
| GeGLU | 2922.624 | 2751.712 | 2751.712 | 1.062x | 1.000x | 1.022x |
| LayerNorm | 94.304 | 94.304 | 94.304 | 1.000x | 1.000x | 1.122x |
| Softmax | 49.792 | 46.272 | 45.248 | 1.076x | 1.023x | 1.259x |
| DYT | 98.400 | 91.328 | 91.168 | 1.077x | 1.002x | 1.136x |
| Sparsemax | 76.544 | 71.232 | 47.840 | 1.075x | 1.489x | 1.741x |
| KLDiv | 98.592 | 97.984 | 94.688 | 1.006x | 1.035x | 1.196x |
| PolyNorm | 47.488 | 47.488 | 46.880 | 1.000x | 1.013x | 1.239x |
| Flash backward preprocess | 418.624 | 307.264 | 190.912 | 1.362x | 1.609x | 1.073x |
| Flash one-pass LayerNorm | 156.288 | 155.008 | 152.128 | 1.008x | 1.019x | 1.187x |
| xFormers RMSNorm-add | 2350.080 | 533.920 | 533.920 | 4.402x | 1.000x | 1.047x |
| SplitK reduce | 15.936 | 15.648 | 15.584 | 1.018x | 1.004x | 2.914x |

H100 正式横向表：

| Kernel | C-Fixed | Schedule | Full | `G_schedule` | `G_full` | Source/target |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU | 856.352 | 761.632 | 752.192 | 1.124x | 1.013x | 1.024x |
| RMSNorm | 33.952 | 33.248 | 32.864 | 1.021x | 1.012x | 1.517x |
| Fused Add RMSNorm | 56.160 | 55.104 | 55.008 | 1.019x | 1.002x | 1.346x |
| GeGLU | 1890.272 | 1569.760 | 1569.760 | 1.204x | 1.000x | 1.084x |
| LayerNorm | 57.728 | 57.728 | 56.096 | 1.000x | 1.029x | 1.301x |
| Softmax | 35.008 | 33.344 | 31.520 | 1.050x | 1.058x | 1.336x |
| DYT | 68.960 | 56.896 | 56.896 | 1.212x | 1.000x | 1.199x |
| Sparsemax | 53.728 | 48.832 | 32.480 | 1.100x | 1.503x | 1.784x |
| KLDiv | 63.104 | 61.824 | 61.696 | 1.021x | 1.002x | 1.251x |
| PolyNorm | 36.704 | 35.168 | 33.728 | 1.044x | 1.043x | 1.428x |
| Flash backward preprocess | 648.928 | 648.928 | 120.192 | 1.000x | 5.399x | 1.050x |
| Flash one-pass LayerNorm | 92.000 | 83.072 | 81.184 | 1.107x | 1.023x | 1.424x |
| xFormers RMSNorm-add | 1507.456 | 286.656 | 282.592 | 5.259x | 1.014x | 1.085x |
| SplitK reduce | 15.776 | 15.616 | 14.336 | 1.010x | 1.089x | 3.470x |

Fresh-14 全量几何均值：5090D 为 `G_schedule=1.168x`、`G_full=1.073x`、Source/target `1.261x`；H100
为 `G_schedule=1.197x`、`G_full=1.185x`、Source/target `1.373x`。Full 计数分别为
`#>=1/#>=1.05/#>=1.5/#<1 = 14/2/1/0` 与 `14/4/2/0`。只统计本轮真实新增结构适用的
Flash+Sparsemax，`G_full` 几何均值为 5090D `1.548x`、H100 `2.849x`；因此强结构结论成立，同时全表保留
已经 read-once、带宽/launch 下限主导的负例。Flash/Sparsemax 使用完整空间，其余项使用统一
96-candidate 预算。

Sparsemax 新增的 H100 winner 不是 schedule label：single-warp 状态用 frontier ballot 直接选出 `(rank,sum)`，
leader-only tau 继续留在 warp-local SSA。5090D 自动拒绝该图并选择更宽的 epilogue mapping，直接给出同一个 J
在两个 GPU target 上选择不同 concrete MLIR topology 的证据。

本轮还发现并修复了一个搜索正确性缺口：候选输出缓冲复用时，漏写候选可能继承前一候选的正确结果。
runner 现在在每个候选 Q 前分别写入两种 canary，并要求两次完整输出都通过；C++ matcher 同时拒绝不能覆盖
完整输出域的 packet geometry。每个 cold sample 还会在 CUDA event 计时区间外恢复输入，再做 L2 eviction，
保证 in-place Source/候选不会改变下一次测量的 workload。由漏写或状态串扰产生的表面加速一律作废。

### 3.4 Flash preprocess 八批稳定结果

在相同自然负载 `B=4, H=16, S=8192, D=128` 上，Full 现在搜索真实的多行 live-accumulator
MLIR 状态：同一 subgroup 的 2/4/8 个行累加器同时存活，feature load/multiply/add 按行交错发射，随后每行
独立完成 shuffle 与 store。它不改变单行内部的规约树；旧实现虽然接受 `rows-per-subgroup>1`，实际上仍是
完整做完一行再做下一行，不能算 row-level ILP。

| Target | C-Fixed | Schedule | Full | `G_schedule` | `G_full` | Source | paired Source/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5090D | 418.624 | 307.264 | 190.912 | 1.362x | 1.609x | 204.752 | 1.073x |
| H100 | 648.928 | 648.928 | 120.192 | 1.000x | 5.399x | 124.208 | 1.050x |

两卡均使用 code-warm/data-cold CUDA event、5 samples/batch、8 个 Source/target 交替确认批次。表内 Full
来自内部三柱的同轮候选测量；5090D 的八个 target 确认批次为 `190.560--191.264 us`，H100 后六批稳定在
`117.728--123.200 us`，paired Source/target 使用确认批次中位数。
两张卡从同一候选并集独立选择出不同 executable：

- 5090D：`1024 threads, 32 lanes/row, 1 row/subgroup`；
- H100：`256 threads, 32 lanes/row, 4 interleaved rows/subgroup`。

因此这里的跨目标复用不是携带一个 GPU winner，而是复用同一个 typed padded-reduction J 和同一个 C++/MLIR
变换集合，再由目标测量选择不同的具体执行图。Schedule-Only 仍只搜索 serial-row direct-lowering spine，
多行 interleaved 状态只存在于 Full 的 subgroup-feature 结构中。

### 3.5 Sparsemax prefix/frontier/epilogue live-state 八批结果

Sorted-prefix Full 现在包含三个可组合的独立物理坐标。第一项在 prefix/support/tau 之前发起每个输出线程的
coalesced X packet 读取，并把 packet 保存在 SSA/register 中；tau 就绪后直接执行 `max(X-tau,0)`，使完整 X
读取与串行 active-prefix 延迟重叠。第二项消费 descending typed precondition：support 必然是一个前缀，因此
用 `nvvm.vote.ballot.sync + ctlz + indexed shuffle` 直接取得最后 supported `(rank,sum)`，替代逐 lane pair 的
XOR tree；随后每个 warp 只由 leader 计算一次 tau，再广播 live scalar。该状态把实际 shuffle 从 16 降为 9，
并把每行/warp 的重复 tau division 从 32 次降为 1 次。第三项把 sorted-prefix packet width `1/2/4` 与完整
X/output 的固定 `vector<4>` 解耦：前者决定每个 early-stop chunk 覆盖 32/64/128 等多少个排序值，后者继续
保持 16-byte 合并访存。三项都只能由精确的 typed
`scan -> support -> (k,s) -> tau -> epilogue` matcher 准入，不读取 kernel 名；缺少 descending 证据时只可
作为 execution-domain candidate 进入 Q/D，不能转写成 B hard fact。

| Target | C-Fixed | Schedule | Full | `G_schedule` | `G_full` | Source | paired Source/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5090D | 76.544 | 71.232 | 47.840 | 1.075x | 1.489x | 83.056 | 1.741x |
| H100 | 53.728 | 48.832 | 32.480 | 1.100x | 1.503x | 57.184 | 1.784x |

两卡均为 5 samples/batch、8 个交替确认批次。5090D 选择
`64-thread + prefix packet 1 + prefetched X + frontier ballot`，确认批次为 `47.392--47.904 us`，中位数
`47.696 us`；H100 保留 `32-thread single-warp + prefix packet 4 + prefetched X + frontier ballot + leader tau`，
确认批次为 `31.872--32.416 us`，中位数 `32.048 us`。同一个候选并集因 target 不同选择了不同 packet/
thread executable；所有获胜状态都通过两次 canary Q。

本轮还修复了 multi-row sorted-prefix 的真实索引错误：第二个逻辑行过去错误使用 CTA 全局线程号索引
SortedX，导致全部多行候选被 Q/D 拒绝；改为 row-local scan thread 后，32 个多行状态全部正确可测，但两卡
都选择单行 winner。当前同轮三柱的稳定 `G_full` 是 `1.489x/1.503x`；5090D 仍按实报告略低于 1.50x，H100
已跨过 1.50x，收益来自更少的实际 prefix 工作与既有 frontier/epilogue live state，而不是削弱 Schedule。

### 3.6 SplitK 跨 CTA D 分片的跨目标裁决

Full 现在额外包含一个真实的 cross-CTA D partition：同一输出 row 的连续 D packet 由
`d_partitions={2,4,8}` 个 CTA 无重叠地写回，grid 的 z 维显式承载 partition；LSE 只由 `z=0` 的 row leader
写入。它改变 work decomposition，而不是增加一个候选标签；Schedule-Only 固定 `d_partitions=1`。

同协议复测中，H100 选择 `d_partitions=8`，得到 `C-Fixed=15.744 us`、`Schedule=15.616 us`、
`Full=14.432 us`，即 `G_full=1.082x`；5090D 的 Full 则保留原有单 CTA 多 warp 状态，
`C-Fixed=15.872 us`、`Schedule=15.424 us`、`Full=15.424 us`。这说明同一候选并集能够产生真实的跨目标
选择，但 SplitK 当前自然 shape 已接近 launch/短规约下限，因此该路径保留为后端能力和物理负例，不冒充
`>=1.50x` 的 headline 机制。两卡相对 Source 仍分别约为 `2.93x/3.61x`。

## 4. 本任务实现的四条强路径

### P1. RVV fused Softmax：先删除物化，再比较两种真实拓扑

从 typed graph 识别稳定 Softmax：

```text
row_max -> subtract -> exp -> row_sum -> divide
```

不得读取 kernel 名。首先实现 conventional fused state：

```text
pass 1: vector max(X)
pass 2: vector exp(X-max) + vector sum，同时把未归一化值写入最终输出 scratch
pass 3: 原位归一化 scratch
```

它必须删除独立 shifted tensor、exp tensor、max/sum broadcast tensor和分离的 elementwise loop。随后实现
online state：

```text
pass 1: 稳定 online (max,sum) recurrence
pass 2: 重新读 X，直接计算并写最终输出
```

两条路径都由 C++/MLIR pass 产生，分别拥有独立 VL/LMUL/unroll schedule。online state 若改变浮点观察
顺序，作为 observation-changing candidate 进入统一 Q/D；不能由 benchmark tolerance 反向生成 hard fact。

P1 的结构成功条件不是 flag 被打开，而是 post-pass 中可直接观察到：

- max 与 sum 都使用真实 scalable-vector accumulator；
- conventional state 不再存在 shifted/exp/statistics tensor；
- online state 只有两次完整 row traversal；
- Python 不发射目标 kernel。

### P2. Statistics-to-epilogue live state

在现有 joint statistics/raw-moment 基础上，把 statistics finalize 与最终 normalization/epilogue 放进同一个
target region，避免统计 tensor、broadcast loop 和不必要的 root producer 重算。适用 typed graph包括
RMSNorm、Fused Add RMSNorm、LayerNorm、PolyNorm、Flash one-pass LayerNorm 与 xFormers RMSNorm-add。

允许两类目标实现：

- row 能驻留时，输入 packet/unnormalized output 保存在 register、shared 或 vector scratch，统计完成后直接
  normalize；
- row 不能驻留时，保留必要的第二次输入读取，但删除所有 statistics/broadcast materialization。

是否驻留由 target capacity 和静态 shape决定，不由 kernel 名决定。若现有 target lowering 已经做到同样的
read-once/live-state，matcher 返回 `already_realized`，不能把它重复计为新 Full 路径。

### P3. Prefix/support/tau 融合 scan

为 Sparsemax 实现通用 prefix/segmented-scan target state：

- CUDA：warp prefix、CTA segmented prefix、compact shared frontier；
- RVV：vector prefix/segmented scan；
- prefix、support predicate、supported-value sum、tau 与 epilogue 共用 live state；
- 只有 typed domain 中存在可证明的排序/active-prefix 前提时才开放 early stop，否则执行完整 scan。

P3 的收益必须来自缩短串行依赖、减少 barrier 或删除 prefix/support/tau 的中间物化；单纯改变 block size 或
vector width 仍属于 Schedule-Only。

### P4. Multi-input、mixed-dtype 与 padded reduction

把 RVV row packet state 扩展为同时读取多个同域输入、显式 cast 到 accumulator dtype、融合 map/select/mul
producer，并只在 row 末做一次 horizontal finalize。CUDA 将 Flash preprocess 已验证的 padded/
subgroup-feature reduction按 typed map/select/cast/reduce 结构通用化。

CUDA 部分已进一步实现多行 interleaved live-accumulator 状态，并把 `rows-per-subgroup={1,2,4,8}` 纳入同一
完整搜索；C++ post-pass 明确记录 `rows_per_subgroup`、实际 shuffle 数与 multirow trace，而不是由 Python
候选标签声称命中。

优先覆盖 KLDiv、Flash backward preprocess、Fused Add RMSNorm 与 xFormers RMSNorm-add。目标是删除独立
cast/mul/mask tensor 与标量 feature traversal，并让 5090D、H100、RVV 分别选择自己的 schedule。

## 5. Fresh-14 横向策略

| 类别 | Kernel | 本任务角色 |
|---|---|---|
| 已融合 elementwise | SwiGLU、GeGLU、DYT | 全量报告；自然 shape 下均作为带宽主导的物理负例 |
| normalization/statistics | RMSNorm、Fused Add RMSNorm、LayerNorm、PolyNorm、Flash one-pass LayerNorm、xFormers RMSNorm-add | P2 与 P4 |
| stable normalization | Softmax | P1 |
| prefix/support | Sparsemax | P3 |
| multi-input/padded reduction | KLDiv、Flash backward preprocess | P4 |
| boundary-limited merge | SplitK reduce | producer不在 J 内时保留负例，不消除不可见 stage |

14 个 kernel 都进入同一 controller 和结果表，但 `>=1.5x` 的机制结论只在结构实际适用的 cell 上统计；
不能用 applicable subset 隐藏全量负例。

## 6. 搜索与实现约束

每条路径必须登记并执行以下内容：

```text
typed MLIR pattern
B hard preconditions
current-state preconditions
target capability
concrete C++/MLIR rewrite
exact-preserving 或 observation-changing
该结构自己的 schedule domain
```

并满足：

- Full 完整包含 Schedule-Only spine；
- structural candidate 必须根据实际 post-pass MLIR/record确认命中，请求 flag 不算命中；
- backend pass 不读取 kernel 名、Source 路径、provider、历史 winner 或 benchmark label；
- 目标计算只来自真实 C++/MLIR/LLVM pipeline，Python 只编排搜索与测量；
- 不修改 accepted J、Source baseline、workload shape/dtype、observable outputs 或 Q/D；
- 数值 state 只在它能打开真实 reduction/scan/online topology 时作为主要机制，不以单独 FMA/reciprocal
  的几个百分点冒充强 Full 收益。

## 7. 执行顺序

2026-07-30 调整：先执行 CUDA 强路径与 5090D/H100 横向覆盖；RVV Phase 保留但后移。

### Phase A：CUDA 强路径

1. 将 Flash preprocess padded/subgroup-feature state收敛为通用 typed-pattern rewrite；
2. 在 5090D 与 H100 分别搜索，不跨 GPU 复用 winner；
3. 完成 Sparsemax warp/CTA scan 与 support/tau fusion；
4. 仅在 post-pass/profile发现真实 barrier、spill或重复 materialization时扩展 CUDA Softmax/Norm。

### Phase B：GPU Fresh-14 能力横向

1. 对 reduction、scan、multi-input 与 normalization 逐类寻找会删除 traversal/materialization 或改变同步图的状态；
2. 所有新状态在 5090D/H100 上使用同一候选并集独立搜索；
3. 同轮产出 C-Fixed、Schedule-Only、Full 与 Source，并报告全表负例；
4. 若 profile 已证明 read-once/带宽或 launch 下限，不用局部算术开关伪造结构收益。

### Phase C：RVV 后续横向覆盖

1. 完成 P2 statistics-to-epilogue live state；
2. 完成 P3 RVV prefix scan；
3. 完成 P4 multi-input/mixed-dtype/padded row state；
4. 对 Fresh-14 全量重新搜索并保留不适用/无收益项。

### Phase D：三目标结果

1. 5090D、H100 使用 code-warm/data-cold CUDA event；
2. RVV 使用固定 CPU、单线程、code-warm/data-cold；
3. 三柱与外部 baseline使用相同 natural shape、dtype、timed boundary 与 correctness policy；
4. 同时报告全量 Fresh-14 与 structural-applicable subset。

## 8. 验收标准

### 8.1 强机制

- 至少 6 个 kernel-target cells 达到 `G_full >= 1.50x`；
- 覆盖至少 4 个 kernel、3 条真实结构路径以及 5090D/H100/RVV 三个 target；
- 每个 target 至少一个 `>=1.50x` cell；
- P1 Softmax、P3 scan、P4 padded/multi-input reduction各至少一个代表跨过 1.50x，或者留下已经实现目标
  traversal/topology 后仍受物理下限约束的直接证据；
- 达标 cell 的 Full 不慢于对应 Source/Native。

### 8.2 可归因性

每个 headline cell 同时报告：

```text
C-Fixed latency
Schedule-Only latency
best Full latency
best Full-without-this-mechanism latency
Source/Native latency
完整 traversal 数
中间 materialization 数
关键 barrier/reduction/scan/target 指令数
```

因此既能回答“Full 比固定结构 schedule 快多少”，也能回答“新增机制本身贡献多少”。

### 8.3 全量后端能力

- Fresh-14 在三个 target 上都进入统一控制器；
- 报告 `G_schedule`、`G_full`、external ratio、几何均值、中位数及
  `#>=1/#>=1.05/#>=1.5/#<1`；
- structural-applicable subset 的 `G_full` 几何均值目标为 `>=1.50x`；
- Fresh-14 全量 `G_full` 几何均值目标为 `>=1.15x`；
- Full 因包含 Schedule-Only spine，在同一次 complete-space timing 中不得系统性回退。

## 9. 完成条件

1. P1--P4 中至少三条由真实 C++/MLIR pass产生并进入统一 Full search；
2. MLIR、LLVM IR 与 target assembly能直接看到所声称的 traversal/materialization/topology变化；
3. 每条新路径有正向、边界和 fail-closed 测试，并通过现有 backend/search回归；
4. Fresh-14 三目标三柱与外部 baseline结果完整；
5. 达到第8节目标，或对未达标机制给出已经消除目标瓶颈后的物理解释；
6. 不使用 Python target-kernel emission，不通过削弱任何 baseline、改变输入或隐藏负例追求数字；
7. 代码、任务和结果提交，工作区干净。
