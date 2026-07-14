# SCRC 真实 softmax 探针报告

日期：2026-07-15

## 1. 结论

softmax 已经在共享 live harness 中完成了一次真实
`gpt-5.6-sol / xhigh` 试验。本轮得到的准确结论是：

- Lift 在一次 schema repair 后成功；
- normal source-blind replay 可编译，与 source TTIR 在所有硬项上
  完全闭合，score 为 `1.0`；
- masked-J replay 也可编译且完全闭合，score 为 `1.0`；
- 泄漏比 `L=1.0 > 0.5`，所以最终 Gate1 状态是
  `LEAKAGE_REJECTED`，而不是 `REPLAY_CLOSED`；
- 正式 live Gate2 按顺序没有运行。独立的 production lower2c
  三方预检已 PASS，它只证明该 case 的 lowering 第三腿具备能力。

因此，这轮关闭了 softmax 的 normal 重放工程路径，同时暴露了
当前 masked-J 实验设计的新阻塞：π 的有序机制拓扑结合仍公开的
接口和结构信息，已足以让模型猜回完整 softmax。

## 2. 实验配置

```text
model = gpt-5.6-sol
reasoning effort = xhigh
M = 4
N = 37
BLOCK_N = 64
num_warps = 4
num_stages = 1
grid = (4,)
dtype = f32
Gate2 tolerance = rtol 2e-4, atol 2e-5
```

`N=37, BLOCK_N=64` 故意保留 tail，同时真实覆盖 masked load 的
`-inf` fill、两次 reduction、归约标量对 row tile 的 broadcast 和 masked
store。首个闭环仍限定一个 program 持有完整一行，没有引入 online
softmax 或跨 program 归约。

## 3. 两次 normal replay 差异如何收敛

### 3.1 第一次：全局轴被错当成局部 tile 轴

首次 Lift 产物把 J 中 `[M,N]` 张量的全局 reduce axis `1`直接复制为
Triton 一维 row tile 的归约轴，并把全局语义 broadcast 强制实现为
本地 `expand_dims`。access predicate 还被从源码的 `col < N` 加强成
`0 <= col < N`。

结果是 replay 可编译，但 Gate1 `NO_CLOSURE`，score 为
`0.3358333333`。这是 J 全局语义与 B 中局部前端 tile 机制的契约缺口，
不是需要放宽 comparator 的等价性问题。

修复后的契约是：

- J `reduce.axes` 始终表示全局逻辑轴；
- `Boundary.reductions.tile_axis` 表示前端局部 tile 的真实轴；
- 语义 broadcast 不等于必须本地插入维度；
- predicate 精确复制源码实际执行的条件，不额外增强。

### 3.2 第二次：地址算术的 staging 不同

契约修复后，replay 的语义、归约和 predicate/effect 都对齐，但仍然
`NO_CLOSURE`，score 为 `0.7542903226`。稳定 diff 是：

```text
source: pointer + row*N + col
replay: pointer + (row*N + col)

arith.addi: replay 多 1
tt.addptr: source 4, replay 2
tt.splat: replay 多 1
```

这不是一般代数规范化，而是 π 没有区分两种前端地址物化机制。
因此规则库新增最小通用卡 `materialize_pointer_tile`，与原有
`materialize_index_tile` 分工：

- 前者保留 staged `pointer + scalar base + lane`；
- 后者先构造完整 integer offset tile，再与 pointer 相加。

新卡不包含 softmax 算法、算子种类或可复制的专用源码。

## 4. 最终 normal 与 masked 结果

新契约和规则卡生效后，第三次完整运行的 Lift 在一次有界 schema
repair 后成功：

```text
repair diagnostic:
SchemaError: ReplayPlan references unknown J identifier(s):
denominator, numerator, row_max, shifted
```

normal replay 首个候选即编译成功，与 source 之间：

- interface 相同；
- op multiset 零差异；
- dataflow 零差异；
- 两次 reduction 结构相同；
- load fill、load/store predicate 与 effect 相同；
- 总 score `1.0`，所有硬项通过。

masked packet 已经对函数名、参数名、constexpr、compiler slot 和 grid
名称做统一不透明改名，但仍保留方案规定的 π 机制序列。模型据此
完整恢复了：

```text
masked load(-inf)
  -> max
  -> subtract
  -> exp
  -> sum
  -> divide
  -> masked store
```

恢复原 ABI 后，masked replay 的 TTIR 也在所有硬项上与 source 零差异，
所以 `masked score=1.0`、`L=1.0`。Gate1 按现有方案 fail-closed 拒绝是
正确行为。

## 5. Gate2 production lowering 预检

`python/tests/test_gate2.py` 新增固定 softmax J 的真实三方测试：

```text
source Triton
replay Triton
committed IntentIR J -> intent MLIR -> linalg/loops -> LLVM -> mlir-runner
```

三条腿对同一份 `M=4,N=37` 随机输入分别独立执行，并按
`rtol=2e-4, atol=2e-5` 做三组两两逐元比较，结果为 PASS。这证明
现有 IntentIR primitive composition 和 production lower2c 能支撑该 softmax case，
不需要新增 softmax 专用 op。

但该预检使用仓库中已提交的固定 J 和 replay，不是本轮 live Lift 产物。
因此它不能覆盖 Gate1 的 `LEAKAGE_REJECTED`，也不能表述为本轮已完成
正式 `Lift -> Gate1 -> Gate2` 全闭环。

## 6. 可复核证据

| 阶段 | response id / 本地证据 | 结果 |
|---|---|---|
| Lift | `resp_06c991a807d02279016a567a78a9a08199b54735a58e73572e` | 2 attempts，1 schema repair |
| normal replay | `resp_0344d49f6a8f19df016a567b0333188199a70cab5e8db3d434` | compile PASS，score 1.0 |
| masked replay | `resp_0c7cfec07fab4718016a567b16fc6c819a8e6418e53d79809a` | compile PASS，score 1.0 |
| Gate1 | `/tmp/scrc_live_softmax_result_v3.json` | `LEAKAGE_REJECTED`, L=1.0 |
| Gate2 preflight | `test_gate2_real_softmax_source_replay_lowering_three_way` | three-way PASS |

`/tmp` 中的 live 结果不纳入仓库；本报告保留可复核的 response id、确定性
配置、差异性质和最终裁决。本轮没有调整 `leakage_threshold=0.5`，
也没有新增任何方案外反事实裁决。
