# Task 27 实验报告：J-only CUDA/TileLang gather2d

> 状态：`PAIR_CLOSED`（2026-08-01）
>
> 机器可读指标：`gather_pair_metrics.json`

## 1. 实验问题

本实验在 add2d 之后只增加一个有意义的难度维度：data-dependent indirect address。主代理直接从
当前 Dataset Source、Harness 和编译观察手工作者化两份 `(J,C,π)`，不调用 Lift agent；两个 fresh、
source-blind Codex 分别只消费 `J + MaskC(C) + U_f`，生成完整 TileLang/CUDA，再经过各自 Gate1 与
Source–Replay、Source–J、Replay–J 三边 Gate2。

它回答的是这一对真实 gather case 的可接入性与 J 消费能力，不回答 J necessity、性能等价或大规模
泛化。正式成功 run 均为 `gpt-5.6-sol/xhigh` 单次 provider call，π、Source、Harness valuation、Gate
observation 与 prior Replay 均不在输入中。

## 2. 两份 J：共同核心相同，完整语义不强行相同

共同的 valid-index 数学关系是：

```text
out[k0] = input[row_idx[k0], col_idx[k0]]
0 <= row_idx[k0] < x0
0 <= col_idx[k0] < 64
```

两份 J 都有三个逻辑域、两个 typed gather 和两个 `index_bounds`，π 也都是95 canonical bytes的
`d0 -> program axis 0`。但 full J 不是 byte-identical，而且不应该被强行写成一致：

| 真实差异 | TileLang | CUDA |
|---|---|---|
| Source 语义 | TIR 中直接间接访问 | 显式 row/col bounds，越界取0 |
| J body | 2个 gather | 3 const + 7 map_binary + 2 gather + select，共13 op |
| predicated boundary access / fill | 0 / 0 | 1 / 1 |
| ABI | 4个 TileLang tensor | 4个 pointer + scalar `M` |
| full artifact canonical | 2696 B | 4381 B |
| J canonical | 2422 B | 4109 B |

因此最科学的论文表述是“共享间接 gather 数学核心，同时完整 J 保留 frontend ABI 与 bounds-to-zero
delta”，而不是“跨前端 J 完全一样”。

## 3. 正式闭合结果

| 指标 | TileLang gather2d | CUDA gather2d |
|---|---:|---:|
| accepted run | run-003 | run-002 |
| TypeCheck | PASS | PASS |
| Replay | 12 display lines / 297 B | 18 lines / 426 B |
| Replay episode | 161.91 s | 120.78 s |
| Gate1 total | 181.23 s | 124.18 s |
| Gate2 | 0.20 s | 6.29 s |
| provider calls | 1 | 1 |
| construction tool calls | 5 | 3 |
| Gate1 hard | PASS | PASS |
| lower/raw diagnostic differences | 5项 PTX | 3项 raw PTX |
| Gate2三边 | 全部 exact PASS | 全部 exact PASS |
| max absolute error | 0 | 0 |

TileLang Replay最终直接生成：

```python
with T.Kernel(256, threads=c0) as v2:
    p3[v2] = p0[p1[v2], p2[v2]]
```

CUDA Replay最终使用`blockIdx.x`作为唯一output coordinate，保留四个bounds、zero fill和间接load。
两者都由 agent 作者化完整目标程序，不是 fixed code 填空、kernel模板或旧 winner 拼接。

## 4. 为什么 TileLang 看 TIR、CUDA 看语义 PTX

TileLang 的公开前端边界是 TIR，所以 hard Gate比较interface、TIR structure、topology和operation
multiset；其生成 CUDA/PTX 仅作诊断。本次 TileLang Source/Replay的5项 PTX诊断不同，但TIR hard字段
全部闭合，三边数值也完全一致。若反过来要求 raw PTX相等，就会把后端合法lower差异错误归咎于
TileLang Replay。

CUDA Source本身位于CUDA C++边界，因而使用PTX semantic-effects v6：ABI、semantic value/memory、
control、address coordinate、target等为hard，raw instruction string/opcode/cache spelling只诊断。本次
3项 raw PTX不同而所有semantic hard字段通过，直接说明CUDA也不应使用raw PTX文本相等。

## 5. 三个失败基线与机制归因

### 5.1 TileLang run-001：program mapping

Agent恢复了正确的间接数学，却选择`T.Kernel(1)+T.Parallel(256)`；当前identity TIR profile要求首个
output-covering J domain直接映射到program grid，因多出control loop而拒绝。修复是通用、J-derived
canonical mapping，不是 gather 模板。

### 5.2 TileLang run-002：pure intermediate materialization

Agent把两级逻辑 gather 逐级物化成 local allocation和copy loop。J没有该storage/effect，且中间值只有
一个consumer；把“纯单消费者中间值默认SSA融合”写成完整编译职责后，fresh run生成直接复合地址。

### 5.3 CUDA run-001：未授权 lane coordinate

Agent正确恢复了guards、zero fill与gather，但擅自使用
`blockIdx.x * blockDim.x + threadIdx.x`。真实launch会额外枚举128倍坐标并越界，这不是合法schedule
差异；`coordinate_uses` hard Gate正确在Gate2前拒绝。通用修复规定：没有独立J-owned lane axis时，
外部`launch_block_x`不授权`threadIdx/blockDim`进入J地址。observer没有被放松。

三个机制commit合计只改通用Replay production Python：+47/-10，net +37；新增kernel-specific
production lowering为0 LoC，也没有`gather2d` case matcher。522行手工作者化脚本与271行共享runner是
实验审计材料，不是production frontend接入成本。

## 6. 论文可以声称什么

当前证据支持：

1. J-only Replay在TileLang TIR与CUDA semantic PTX两个不同编译观察边界上，均能闭合data-dependent
   gather；
2. π与Source对Replay不可见时，共同数学核心仍能复用，真实frontend guard/ABI delta也能保留；
3. Gate1既容忍lower层raw差异，也能在Gate2前拒绝真实的coordinate错误；
4. 新case没有增加per-kernel production recipe，失败只沉淀为3条通用J消费/ownership机制。

当前证据不支持：

- J必要或最小；本实验没有运行fresh Mask-J；
- 两前端full J完全一致；证据恰好表明它们有真实差异；
- Replay性能与Source相同；这里只验证结构与有限域正确性；
- 任意新frontend或任意kernel都能零修改接入；目前主表是add2d和gather2d两对；
- 手工作者化耗时优于某个Lift baseline；本轮没有记录可比wall time。

该最小mutation实验现已完成：两个frontend的6个主mutation均compile PASS后被Gate1 hard拒绝，Gate2
调用为0；同时发现CUDA semantic-effects v6对单元素constant address displacement存在已知漏检。机器
证据与不外推边界见`mutations/add2d-v1/mutation_metrics.json`和`mutation_report.md`。
