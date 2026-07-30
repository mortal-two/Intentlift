# Task 21：J 驱动的 Full 强结构路径扩展

> 状态：执行中
>
> 类型：production backend 能力建设、真实 C++/MLIR 结构搜索、Fresh-14 跨目标消融
>
> 继承：Task 20 已在 RVV 上完成 joint statistics 与 raw-moment 的第一轮闭环；本任务继续把
> Full 扩展到 normalization、scan 和 padded/multi-input reduction，并在 CUDA 与 RVV 上建立可重复的
> `Full / Schedule-Only` 强收益。

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
不是来自把 C-Fixed 做弱。CUDA 已有 DYT 的 `Full / Schedule-Only = 1.568x` 信号；历史 Flash backward
preprocess 的 padded/subgroup-feature lowering 也已经到 1.5x 附近。

### 3.2 目前真正缺少的路径

- RVV Softmax 仍将 max、shift/exp、sum、divide lower 成多个 traversal/materialization；
- RVV 的显式 vector reduction 已支持 sum/max，但单独 vectorize reduction 并不会删除 exp tensor 与阶段流量；
- RVV multi-input、mixed-dtype、masked/padded row reduction 还不能统一进入 packet state；
- CUDA Sparsemax 仍缺少完整的并行 prefix/support/tau live-state 路径；
- CUDA Flash preprocess 的强 padded reduction 尚未作为通用 typed-pattern state 覆盖两个 GPU target；
- CUDA direct Softmax 已经把输入行和 `exp(x-max)` 保存在寄存器 SSA 中，不能虚构一个“删除 exp tensor”
  的重复路径；普通 CUDA Norm 也已非常接近 read-once，除非 profile 证明 barrier、spill 或重新物化存在，
  否则它们不是优先纵切。

### 3.3 2026-07-30 CUDA 自然负载进展

当前执行优先级已切到 5090D/H100；RVV 只保留已通过编译与测试的能力，不继续占用本轮主要调优时间。
以下均为 `natural-headline-v1`、code-warm/data-cold、CUDA event；筛选数据用于决定实现方向，最终论文表仍按
第8节协议统一重测。

5090D 首轮横向表：

| Kernel | C-Fixed | Schedule | Full | `G_schedule` | `G_full` | Source/target |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU | 1465.344 | 1384.448 | 1382.272 | 1.058x | 1.002x | 1.017x |
| RMSNorm | 49.920 | 49.408 | 47.968 | 1.010x | 1.030x | 1.408x |
| Fused Add RMSNorm | 103.872 | 101.856 | 101.216 | 1.020x | 1.006x | 1.262x |
| GeGLU | 2927.264 | 2751.136 | 2751.136 | 1.064x | 1.000x | 1.025x |
| LayerNorm | 94.240 | 94.240 | 93.440 | 1.000x | 1.009x | 1.136x |
| Softmax | 53.536 | 47.552 | 46.400 | 1.126x | 1.025x | 1.290x |
| DYT | 97.984 | 90.912 | 90.912 | 1.078x | 1.000x | 1.177x |
| Sparsemax | 76.608 | 69.728 | 52.256 | 1.099x | 1.334x | 1.627x |
| KLDiv | 97.632 | 97.632 | 93.920 | 1.000x | 1.040x | 1.168x |
| PolyNorm | 48.704 | 48.704 | 48.640 | 1.000x | 1.001x | 1.341x |
| Flash backward preprocess | 377.792 | 312.672 | 187.584 | 1.208x | 1.667x | 1.092x |
| Flash one-pass LayerNorm | 147.520 | 143.392 | 140.992 | 1.029x | 1.017x | 1.235x |
| xFormers RMSNorm-add | 2356.864 | 536.672 | 535.072 | 4.392x | 1.003x | 1.042x |
| SplitK reduce | 18.688 | 15.680 | 15.680 | 1.192x | 1.000x | 2.954x |

H100 首轮横向表：

| Kernel | C-Fixed | Schedule | Full | `G_schedule` | `G_full` | Source/target |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU | 954.752 | 803.392 | 717.920 | 1.188x | 1.119x | 1.025x |
| RMSNorm | 35.136 | 35.136 | 33.088 | 1.000x | 1.062x | 1.586x |
| Fused Add RMSNorm | 56.128 | 56.128 | 54.912 | 1.000x | 1.022x | 1.425x |
| GeGLU | 1667.936 | 1403.360 | 1403.360 | 1.189x | 1.000x | 1.032x |
| LayerNorm | 57.184 | 56.736 | 54.976 | 1.008x | 1.032x | 1.354x |
| Softmax | 38.176 | 35.616 | 33.920 | 1.072x | 1.050x | 1.317x |
| DYT | 68.064 | 56.544 | 56.544 | 1.204x | 1.000x | 1.233x |
| Sparsemax | 51.008 | 47.168 | 36.288 | 1.081x | 1.300x | 1.584x |
| KLDiv | 70.688 | 68.544 | 66.624 | 1.031x | 1.029x | 1.257x |
| PolyNorm | 36.448 | 34.976 | 33.664 | 1.042x | 1.039x | 1.434x |
| Flash backward preprocess | 820.064 | 812.000 | 128.064 | 1.010x | 6.341x | 1.106x |
| Flash one-pass LayerNorm | 88.480 | 82.400 | 80.608 | 1.074x | 1.022x | 1.452x |
| xFormers RMSNorm-add | 1470.336 | 276.512 | 276.512 | 5.317x | 1.000x | 1.083x |
| SplitK reduce | 15.936 | 15.456 | 14.464 | 1.031x | 1.069x | 3.453x |

首轮几何均值：5090D 为 `G_schedule=1.179x`、`G_full=1.069x`、Source/target `1.286x`；H100 为
`G_schedule=1.199x`、`G_full=1.199x`、Source/target `1.379x`。Flash/Sparsemax 使用完整空间；其余横向项
使用统一 96-candidate 预算，只有出现结构收益的项才升级为完整空间。最终论文数仍需按第8节增加测量轮数重测。

Sparsemax 新增的 H100 状态不是 schedule label：每个输出 warp 重算短 active prefix 并在 warp-local SSA 中保留
tau，因而删除 shared tau publication 与整块 `gpu.barrier`。5090D 搜索会自动拒绝其性能，而 H100 选择它，
直接给出同一个 J 跨目标选择不同 concrete MLIR topology 的证据。

本轮还发现并修复了一个搜索正确性缺口：候选输出缓冲复用时，漏写候选可能继承前一候选的正确结果。
runner 现在在每个候选 Q 前分别写入两种 canary，并要求两次完整输出都通过；C++ matcher 同时拒绝不能覆盖
完整输出域的 packet geometry。每个 cold sample 还会在 CUDA event 计时区间外恢复输入，再做 L2 eviction，
保证 in-place Source/候选不会改变下一次测量的 workload。由漏写或状态串扰产生的表面加速一律作废。

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

优先覆盖 KLDiv、Flash backward preprocess、Fused Add RMSNorm 与 xFormers RMSNorm-add。目标是删除独立
cast/mul/mask tensor 与标量 feature traversal，并让 5090D、H100、RVV 分别选择自己的 schedule。

## 5. Fresh-14 横向策略

| 类别 | Kernel | 本任务角色 |
|---|---|---|
| 已融合 elementwise | SwiGLU、GeGLU、DYT | 全量报告；DYT保留已有强信号，其余作为物理负例 |
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

### Phase A：RVV Softmax 纵切

1. 先将 max 与 sum 的 scalable-vector reduction作为 P1 的必要能力，而不把其约 1.06x 单独算作成功；
2. 实现 conventional fused state并确认删除中间 tensor/traversal；
3. 实现 online two-pass state；
4. 在 RVV4 单线程 natural workload上完成 complete-space C-Fixed/Schedule-Only/Full/Native 比较；
5. 若 `G_full < 1.5x`，直接依据 exp 指令、访存字节与 traversal profile选择下一步，不增加局部开关。

### Phase B：CUDA 强路径

1. 将 Flash preprocess padded/subgroup-feature state收敛为通用 typed-pattern rewrite；
2. 在 5090D 与 H100 分别搜索，不跨 GPU 复用 winner；
3. 完成 Sparsemax warp/CTA scan 与 support/tau fusion；
4. 仅在 post-pass/profile发现真实 barrier、spill或重复 materialization时扩展 CUDA Softmax/Norm。

### Phase C：RVV 横向覆盖

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
