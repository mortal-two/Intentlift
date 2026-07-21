# SCRC 真实 matmul 探针报告（v7 历史归档）

日期：2026-07-15

## 1. 结论

matmul 已通过共享 live harness 完成真实
`gpt-5.6-sol / xhigh` Lift 和 source-blind replay。最终结果是：

- Lift 首轮成功，无 schema repair；
- normal replay 首候选可编译，TTIR score `1.0`，全部硬项零差异；
- masked-J replay 也可编译，但没有完全复现源码结构，score 为
  `0.8045180723`；
- 泄漏比 `L=0.8045180723 > 0.5`，最终 Gate1 为
  `LEAKAGE_REJECTED`；
- 正式 Gate2 按门禁顺序没有运行；
- 按方案 9.1 对被拒绝 live 样本做的离线 Gate2 对照三方 PASS，但不改变
  Gate1 结论。

这轮完成了五核探针中最后一个 kernel 的正常重放闭合。matmul 不需要新增
IntentIR op：J 使用一个 `contraction("mk,kn->mn")` 即可；源码中的 K 分块循环
是该 contraction 的 Layer-C/前端实现，不是额外的 J `loop`、`reduce` 或
`broadcast`。

## 2. 实验配置

```text
model = gpt-5.6-sol
reasoning effort = xhigh
M = 17
N = 13
K = 19
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
num_warps = 4
num_stages = 2
grid = (2, 1)
A dtype = f16
B dtype = f16
Out dtype = f32
Gate2 tolerance = rtol 2e-3, atol 2e-3
```

三个逻辑维度都不是相应 block 的整数倍，且 `K=19 > BLOCK_K=16`，所以该
case 同时覆盖 M/N 尾块、第二个 K tile、A/B 两侧独立 mask、零填充和二维
output store guard。

## 3. J / B / π 的正确分层

最终 live J 的 A 层只有一个 whole-tensor contraction：

```text
inputs:  a f16[M,K], b f16[K,N]
output:  out f32[M,N]
body:    out = contraction(a, b, subscripts="mk,kn->mn")
```

B 层保存三次真实访问：

```text
A: offs_m[:,None] * K + offs_k[None,:]
   mask = (offs_m[:,None] < M) & (offs_k[None,:] < K), fill = 0
B: offs_k[:,None] * N + offs_n[None,:]
   mask = (offs_k[:,None] < K) & (offs_n[None,:] < N), fill = 0
O: offs_m[:,None] * N + offs_n[None,:]
   store guard = (offs_m[:,None] < M) & (offs_n[None,:] < N)
```

`boundary.reductions` 和 `boundary.broadcasts` 都为空。局部
`[:, None] / [None, :]` 只是构造二维 access tile，不是 whole-tensor 语义
broadcast；K tile 累加是 contraction 的重放机制，不是第二个 A 层归约。

最终 π 只选择：schedule slot、runtime extent、二维 program domain、staged
pointer tile、masked access、`realize_k_tiled_dot` 和 output mapping。具体
`mk,kn->mn`、dtype、地址、mask、fill 与输出都从 J/B 引用读取。

## 4. 三次 live 运行暴露的真实问题

### 4.1 v1：source n-gram 扫描误伤合法 J.B

v1 Lift 首轮产出的 J/B 已包含真实 domains、index expressions 和 predicates，
但 Gate1 在重放前把这些合法 B 字段也拿去做源码四 token n-gram 扫描，因而
直接报 `LEAKAGE_REJECTED`。典型命中是：

```text
pid_m * BLOCK_M + ...
k_tile * BLOCK_K + ...
```

这是实现 false reject。按 SCRC 信息归属，J.B 本来就必须保存从源码提取的
边界、mask 和 index expression；禁止复制源码片段的静态 n-gram 约束针对的是
π。修复后：

- frontend API 拼写仍同时扫描 J 与 π；
- source n-gram 只扫描 π；
- 对 v1 artifact 的本地复扫无 violation。

该修复没有放宽 normal TTIR 裁决，也没有改变 masked-J 阈值。

### 4.2 v2：J 正确，但 π 和 contraction 规则不足以约束源码实现

v2 的 J 已正确使用单个 `contraction("mk,kn->mn")`，三个 normal replay
候选也都能编译，但三者固定停在 `score=0.5771492030`，没有闭合。组件为：

| 硬项 | score |
|---|---:|
| interface | `1.0` |
| op multiset | `0.7752808989` |
| dataflow | `0.0883720930` |
| reduction | `1.0` |
| predicate/effect | `0.25` |

主要原因不是缺少 matmul 语义 op，而是：

1. π 在 `realize_k_tiled_dot` 之外又错误选择了 `realize_reduce`，把 schedule-only
   K 循环重复解释成语义归约；
2. 规则卡没有明确 accumulator/result dtype、K 的语义 extent、K tile slot、
   两侧 access/mask/fill 和输出 tile；
3. 重写把源码的 `pointer + row-term + column-term` 重组为 base pointer 加完整
   offset，改变了 `tt.addptr`、broadcast/expand-dims 与整数算术结构；
4. prompt 没有明确局部 rank insertion 不是语义 broadcast，也没有钉死向量
   predicate 必须使用 `&`。

本轮只完善已有 `materialize_pointer_tile`、`realize_masked_access` 和
`realize_k_tiled_dot` 卡及 Lift 分层契约，没有新增 matmul 专用卡。

### 4.3 v3：normal 零差异，masked 仍超过阈值

v3 Lift 首轮成功，π 不再包含 `realize_reduce`。normal replay 首候选保持了
源码的二维 program ids、K tile 循环、A/B/out staged pointer association、
mask 与 accumulator，编译后：

- interface score `1.0`；
- op multiset score `1.0`；
- dataflow score `1.0`；
- reduction score `1.0`；
- predicate/effect score `1.0`；
- 无 difference，总 score `1.0`。

该 normal 候选没有显式写出源码中的 `out_dtype=tl.float32`，但在本次固定
Triton 环境和 f16 operands 下生成的 TTIR 仍完全一致。这只是当前实测事实，
不登记为跨版本通用的省略规则。

masked packet 已不公开 J 节点内容，并对函数、ABI 参数、constexpr、C 槽和
grid 名称统一做不透明改名。模型仍根据二维接口、π 的 K-tiled-dot 机制和规则卡
猜出一个标准尾块 matmul。它没有复现源码的全部地址 staging，主要差异包括：

```text
arith.muli: source 6, masked replay 8
tt.addptr:  source 6, masked replay 12
以及 broadcast / expand-dims / dataflow 差异
```

因此 masked 硬项并未全过，但总 score 仍为 `0.8045180723`：

| 项目 | score | hard pass |
|---|---:|---:|
| interface | `1.0` | 是 |
| op multiset | `0.8433734940` | 否 |
| dataflow | `0.3433734940` | 否 |
| reduction | `1.0` | 是 |
| predicate/effect | `1.0` | 是 |
| schedule-only | `1.0` | 是 |

normal score 为 1.0，所以 `L=0.8045180723`。现有阈值要求 `L<=0.5`，Gate1
必须拒绝。这里没有用“masked 没有硬闭合”绕过方案规定的泄漏比裁决。

## 5. Gate2 证据

### 5.1 production preflight

`python/tests/test_gate2.py` 对 committed canonical matmul J 增加了真实三方测试：

- `M=17,N=13,K=19`，A/B 为 f16，输出为 f32；
- seeds `0/1/2` 三组随机输入全部 `GATE2_PASS`；
- 故意把 replay 的 B 地址交换为错误布局时，机械归属为
  `REPLAY_DIVERGED`，且 source 与 lowering 仍一致。

Lower2C 接收已经量化为 f16 的 A/B 数值并生成 f32 输出。但当前 production
MLIR emitter 仍把这条路径统一发射为 f32 运算，因此这些测试证明的是该 case
的数值语义第三腿，不是“mixed-f16/f32 dtype-faithful lowering 已实现”。本轮
没有未经批准扩展 lowering backend。

### 5.2 真实 live 被拒绝样本的离线对照

正式流程在 `LEAKAGE_REJECTED` 处停止。按方案 9.1 的 rejected-sample 评估
例外，复用 v3 真实 live J 和 normal replay 离线执行 Gate2：

| 比较 | 结果 | 最大绝对误差 |
|---|---:|---:|
| source ↔ replay | PASS | `0.0` |
| source ↔ lowering | PASS | `5.960464477539062e-7` |
| replay ↔ lowering | PASS | `5.960464477539062e-7` |

机械归属为 `GATE2_PASS`。这是一份
`P(Gate2 pass | Gate1 = rejected)` 对照证据，不能把 Gate1 改判为通过。

## 6. 可复核证据

| 阶段 | response id / 证据 | 结果 |
|---|---|---|
| v1 Lift | `resp_0bd20bea0c62305d016a568c68150c8196b7f52a96f62216c9` | 1 attempt；旧 n-gram 实现误拒 |
| v2 Lift | `resp_0dc88b5cbca70e41016a568d64f65c8190a035fe83062736ae` | 1 attempt；J 正确、π 多选 reduce |
| v2 normal 1 | `resp_0627c5ead76f09e4016a568e11e1c4819692aed2e545a845f5` | compile PASS，score 0.577149 |
| v2 normal 2 | `resp_0887ea6841a1e290016a568e47a2e881968ef9618b83e192c1` | compile PASS，score 0.577149 |
| v2 normal 3 | `resp_0800b8d3dcea933e016a568ea0e2748195ba19a4bba87163a2` | compile PASS，score 0.577149 |
| v3 Lift | `resp_04a7daf45b1296f1016a568fec35b8819aa8ef4b2911722276` | 1 attempt，0 repair |
| v3 normal | `resp_0b9bba8b83ab72aa016a5690639ab88199a03a8cf6f0dcf691` | compile PASS，score 1.0 |
| v3 masked | `resp_03b3891a30d66008016a5690833348819bb7efddbdcbf70180` | compile PASS，score 0.804518 |
| Gate1 | `/tmp/scrc_live_matmul_result_v3.json` | `LEAKAGE_REJECTED` |
| live Gate2 diagnostic | `/tmp/scrc_live_matmul_gate2_diagnostic_v3.json` | three-way PASS |
| fixed-J Gate2 regression | `test_gate2_real_matmul_f16_tail_three_way` | 3 seeds PASS |
| wrong-B negative | `test_gate2_matmul_attributes_wrong_b_address_to_replay` | `REPLAY_DIVERGED` |

临时 JSON 不纳入仓库。本轮没有新增 IntentIR op、新 verifier、解释器路径、
matmul 专用规则卡或方案外裁决，也没有修改 `leakage_threshold=0.5`。
