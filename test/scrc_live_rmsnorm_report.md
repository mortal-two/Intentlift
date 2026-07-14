# SCRC 真实 RMSNorm 探针报告

日期：2026-07-15

## 1. 结论

RMSNorm 已经通过共享 live harness 完成真实
`gpt-5.6-sol / xhigh` Lift 和 source-blind replay。最终结果是：

- Lift 在一次 schema repair 后成功；
- normal replay 首候选可编译，TTIR score `1.0`，全部硬项零差异；
- masked-J replay 同样可编译且 score `1.0`；
- 泄漏比 `L=1.0 > 0.5`，最终 Gate1 为 `LEAKAGE_REJECTED`；
- 正式 Gate2 按门禁顺序没有运行；
- 按方案 9.1 对被拒绝样本做的离线 Gate2 对照三方 PASS，但不改变
  Gate1 结论。

这轮证明 RMSNorm 不需要新的 IntentIR op，现有
`mean + sqrt + recip + map_binary + shape=[] scalar` 已足够表达并走通
production lower2c。当前阻塞仍是 masked-J 可识别性，不是 RMSNorm
语义或 lowering 能力不足。

## 2. 实验配置

```text
model = gpt-5.6-sol
reasoning effort = xhigh
M = 7
N = 1003
BLOCK_N = 1024
eps = runtime f32 1e-5
num_warps = 4
num_stages = 2
grid = (7,)
dtype = f32
Gate2 tolerance = rtol 3e-4, atol 3e-5
```

`N=1003, BLOCK_N=1024` 保留尾部 lane，能检查零填充是否进入平方和、
均值除数是否错用 `BLOCK_N`、weight/output mask 以及归约标量对 row
tile 的广播。`eps` 是真实 runtime scalar，不是 constexpr 或 C 槽。

## 3. IntentIR 与 Gate2 接缝

### 3.1 runtime semantic scalar

以前的 live case 主要用 scalar 表示 M/N 形状长度。RMSNorm 第一次要求
一个 scalar 真正参与数值计算。当前契约明确为：

```text
J input: eps, dtype=f32, shape=[]
interface: eps, kind=scalar, dtype=f32, tensor_ref=eps
Gate2 values: eps=np.float32(runtime_value)
```

M/N 类只提供 shape extent 的 scalar 仍不进 J input。schema 与 compiler-view
交叉检查会拒绝 scalar `tensor_ref` 指向输出、非标量 shape 或 dtype
不一致的 artifact。

### 3.2 mean 与 rsqrt canonicalization

源码中：

```text
square_sum = sum(x*x)
mean_square = square_sum / N
inv_rms = rsqrt(mean_square + eps)
```

J 中规范为：

```text
mean_square = reduce_mean(x*x, global_axis=1)
rms = sqrt(mean_square + eps)
inv_rms = recip(rms)
```

`realize_reduce` 重放 mean 时使用局部 `tile_axis=0`，但除数从全局 J
轴语义长度取 `N=1003`，绝不是 `BLOCK_N=1024`。Gate1 的已登记
局部规范化负责对齐源 `rsqrt` 与重放 `1/sqrt`。

production lower2c 不需要特殊 RMSNorm 分支：`mean`、rank-0 scalar
broadcast、`sqrt` 和 `recip` 都经现有 IntentIR→MLIR→LLVM 链路执行。

## 4. 三次 live 运行暴露的真实问题

### 4.1 v1：规则库 API token 提取误伤合法 J/B

完善 elementwise 卡时曾把 `+ - * /` 也放入 `api_mapping`。旧
`RuleRegistry.api_tokens()` 直接把所有 mapping value 当成前端 API，导致
B 层合法地址表达式 `row*N+col` 被静态泄漏扫描拒绝。

这是本地 Gate1 回归，不是模型泄漏。修复后 `api_tokens()` 只提取
`tl.*` 等真正的限定 API 拼写或独立 API identifier；通用算术符号和
构造 recipe 不参与 API 泄漏扫描。对 v1 artifact 重跑静态扫描已是
`passed=true`。

### 4.2 v2：提升官错选已有地址机制卡

v2 的 J、runtime `eps`、mean、broadcast、mask 全部正确，三个重写候选也
都编译成功，但三者固定停在 `score=0.7703589986`。核心 diff：

```text
source: pointer + row*N + lane
replay: pointer + (row*N + lane)

arith.addi: replay 多 1
tt.addptr: source 5, replay 3
tt.splat: replay 多 1
```

原因不是缺卡，而是提升官对内联源表达错选了
`materialize_index_tile`。Lift 契约随后明确：

- `load(ptr + row*N + lane)` / `store(ptr + row*N + lane, ...)` 选
  `materialize_pointer_tile`；
- 只有源码先赋值完整 `offset = row*N + lane`，后续再使用
  `ptr + offset`，才选 `materialize_index_tile`。

### 4.3 v3：normal 零差异，masked 也零差异

v3 Lift 正确选中 x/weight/output 三处 `materialize_pointer_tile`。normal replay
首候选中：

- interface score `1.0`；
- op multiset score `1.0`；
- dataflow score `1.0`；
- reduction score `1.0`；
- predicate/effect score `1.0`；
- 无 difference，总 score `1.0`。

masked packet 已将函数名、参数名、constexpr、compiler slot 和 grid 名称
全部不透明改名。但它仍公开 ABI 结构和 π 的有序规则拓扑，模型因而
完整猜回：

```text
x*x -> sum/N -> +runtime_scalar -> sqrt -> recip -> *x -> *weight
```

地址 staging、mask、fill、归约和数据流都与 source 一致，masked score
`1.0`、`L=1.0`。因此 Gate1 按当前方案拒绝是正确行为。

## 5. Gate2 证据

### 5.1 production preflight

`python/tests/test_gate2.py` 对 committed canonical RMSNorm J 运行了两类真实测试：

1. production lower2c 单腿分别传入 `eps=1e-5` 与 `eps=1e-2`，两次
   都与 NumPy 诊断 reference 一致，且输出明确发生变化；
2. source Triton / committed replay Triton / production lower2c 对两个 eps
   都三方 PASS；
3. 把 replay 中的 `eps` 故意烘成 `1e-5`，但运行 case 传入 `eps=1e-2`，
   Gate2 机械归属为 `REPLAY_DIVERGED`：source 与 lowering 一致，replay 偏离。

这证明 `eps` 没有被烘成常量，也没有被伪装成 shape dim。

### 5.2 真实 live 被拒绝样本的离线对照

正式流程在 `LEAKAGE_REJECTED` 处停止。按方案 9.1 对被 Gate1 拒绝样本
离线跑 Gate2 的评估例外，本轮复用 v3 的真实 live J 和 normal replay：

| 比较 | 结果 | 最大绝对误差 |
|---|---:|---:|
| source ↔ replay | PASS | `2.384185791015625e-7` |
| source ↔ lowering | PASS | `5.125999450683594e-6` |
| replay ↔ lowering | PASS | `5.0067901611328125e-6` |

机械归属为 `GATE2_PASS`。这是
`P(Gate2 pass | Gate1 = rejected)` 的一个真实对照样本，表明当前 Gate1
因反作弊实验损失了一个行为正确的 RMSNorm。它不是把 Gate1 改判为通过。

## 6. 可复核证据

| 阶段 | response id / 证据 | 结果 |
|---|---|---|
| Lift | `resp_0823d6e4ae6a822d016a5686106720819b82d4fe1c4316a8c8` | 2 attempts，1 schema repair |
| normal replay | `resp_07d0fd24317b9eab016a5686c170f88197a117b0351f44083c` | compile PASS，score 1.0 |
| masked replay | `resp_06e2034caad6e160016a5686d57ac8819487a53ffde964051b` | compile PASS，score 1.0 |
| Gate1 | `/tmp/scrc_live_rmsnorm_result_v3.json` | `LEAKAGE_REJECTED`, L=1.0 |
| live Gate2 diagnostic | `/tmp/scrc_live_rmsnorm_gate2_diagnostic_v3.json` | three-way PASS |
| fixed-J Gate2 regression | `test_gate2_real_rmsnorm_runtime_eps_three_way` | two eps values PASS |
| baked-eps negative | `test_gate2_rmsnorm_attributes_baked_eps_to_replay` | `REPLAY_DIVERGED` |

本轮没有新增 RMSNorm 专用 op、专用规则卡、新 verifier 或解释器路径；
也没有改动 `leakage_threshold=0.5`、重试到 masked 失败，或让 Gate2 覆盖
Gate1 的反作弊裁决。
