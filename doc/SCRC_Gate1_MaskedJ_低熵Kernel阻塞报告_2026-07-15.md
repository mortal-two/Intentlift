# SCRC Gate1 masked-J：低熵 Kernel 与 π 结构可识别性阻塞报告

日期：2026-07-15

## 1. 结论先行

当前 Gate1 没有实现错误放行；相反，它按 SCRC 方案正确地拒绝了最新的
`add`、`row_sum`、`softmax`、`rmsnorm` 和 `matmul`：五者的正常
source-blind replay 最终都达到 TTIR 硬闭合，但遮盖 J 后，
`π + 接口结构 + 规则卡 + 模型先验` 仍能重构出
过于接近、甚至完全等同于源结构的实现。

因此当前真实状态是：

```text
add:
  Lift 成功
  normal replay score = 1.0，硬项全过
  masked replay score = 1.0，硬项全过
  leakage ratio L = 1.0 > 0.5
  Gate1 = LEAKAGE_REJECTED

row_sum:
  Lift 成功
  normal replay score = 1.0，硬项全过
  masked replay score = 0.778676，硬项未全过
  leakage ratio L = 0.778676 > 0.5
  Gate1 = LEAKAGE_REJECTED

softmax:
  Lift 在一次 schema repair 后成功
  normal replay score = 1.0，硬项全过
  masked replay score = 1.0，硬项全过
  leakage ratio L = 1.0 > 0.5
  Gate1 = LEAKAGE_REJECTED

rmsnorm:
  Lift 在一次 schema repair 后成功
  normal replay score = 1.0，硬项全过
  masked replay score = 1.0，硬项全过
  leakage ratio L = 1.0 > 0.5
  Gate1 = LEAKAGE_REJECTED

matmul:
  Lift 成功
  normal replay score = 1.0，硬项全过
  masked replay score = 0.804518，硬项未全过
  leakage ratio L = 0.804518 > 0.5
  Gate1 = LEAKAGE_REJECTED
```

按现有方案，Gate2 不能覆盖 Gate1 的反作弊失败，所以这五次最新运行都没有
作为正式流程进入 Gate2。add、row-sum 和 softmax 的独立三方预检仍能证明
production lower2c 与三方执行链可工作，但在更完整的不透明消融实现下，它们
不构成“当前 Gate1 已接受”的证据。

RMSNorm 和 matmul 还分别对真实 live J 与 normal replay 按方案 9.1 的评估
例外离线跑了 Gate2：source/replay/production lowering 三方 PASS。这只是
`P(Gate2 pass | Gate1 rejected)` 的对照样本，不改变 Gate1 结论。

本轮没有调低阈值、没有重试到 masked 偶然失败、没有给简单 Kernel 加白名单，
也没有让 Gate2 把 Gate1 救回来。

## 2. 本轮先排除的实现侧信道

旧 masked packet 虽然遮掉了 J 的节点内容，却仍暴露：

- `add_kernel` / `row_sum_kernel`；
- `a_ptr`、`x_ptr`、`out_ptr`；
- `N`、`BLOCK_N` 等维度与 C 槽名称；
- 原 grid 表达式中的名字。

这不满足方案第八章“不透明重命名后做遮盖消融”的要求。本轮已经补齐：

1. masked packet 按声明位置把函数、参数、constexpr、compiler slot 和 grid
   标识符统一改成 `scrc_k0 / scrc_pN / scrc_cN`；
2. J、π、interface、schedule 中的引用使用同一映射；
3. writer 保留模型真实返回的 opaque code 与 hash；
4. Gate1Runner 在 masked compile 前统一做 AST 级确定性恢复；
5. 恢复前精确检查 opaque 参数顺序、数量和 constexpr 位置；
6. 若模型局部变量与原 ABI 名冲突，先确定性改名，再恢复 ABI；
7. 恢复后再次经过现有安全 AST 校验，再用原 CompileCase 编译；
8. ABI 没有正确复现时，消融记为不可用并 fail-closed，不能把编译失败误当成
   “无泄漏”。

所以最新失败已经不是函数名、参数名或 C 槽拼写造成的侧信道。

## 3. 五个真实样例说明了什么

### 3.1 add：不是相似，而是完整猜中

最新 masked packet 只公开位置式 ABI 和 π 的机制步骤。模型仍生成：

```text
两次同 mask load
  -> 两个值相加
  -> 同 mask store
```

恢复原 ABI 后，它的 interface、op multiset、dataflow、reduction、
predicate/effect 全部与源一致，`L=1.0`。这说明问题不能归咎于当前 score
把固定接口权重算进去；masked 候选确实完整猜中了 add。

原因是该 Kernel 的剩余假设空间太小：两个同 shape 输入、一个输出、
`realize_elementwise`、两次 load 和一次 store 已把常见答案高度压缩，模型又有
强烈的“两个输入默认做加法”先验。

### 3.2 row_sum：名字遮掉后仍接近源

row_sum 的 masked 候选没有完全闭合，主要差异在地址构造与 dataflow，但它仍
正确猜到：

- 单行 tile；
- tail mask 与零填充；
- reduction；
- sum combiner；
- 标量行输出。

π 中 `realize_reduce`、`realize_masked_access` 等机制本来就按方案保留；接口还
公开一个 rank-2 输入、rank-1 输出和 grid 结构。即使语义名字全部不透明，这些
信息加上模型先验仍足以得到 `L=0.778676`。

### 3.3 softmax：π 的机制序列已经接近语义骨架

softmax 先经历了两个真实、可枚举的正常 replay 修复：

1. 区分 J 的全局 reduce axis=1 与一维 Triton row tile 的本地
   `tile_axis=0`，同时禁止把语义 broadcast 自动实现为 `expand_dims`；
2. 新增 `materialize_pointer_tile` 规则，让 π 能表达源码的 staged
   `pointer + row-base + lane`，而不是先计算完整 integer offset。

修复后 normal replay 首轮达到 `score=1.0`。但 masked packet 中仍保留同一 π：

```text
masked load
  -> reduce
  -> elementwise
  -> broadcast
  -> elementwise
  -> reduce
  -> elementwise
  -> broadcast
  -> masked store
```

结合规则卡里的 reduce/elementwise 实现集合和一个 rank-2 输入/输出接口，模型
完整猜回 `max -> sub -> exp -> sum -> div`，连地址 staging、mask、fill 与 TTIR
dataflow 都完全一致，`L=1.0`。

这推翻了“只有 add/row-sum 等低熵 Kernel 才有问题”的初步假设。即使 softmax
本身语义更复杂，源实例特定 π 的规则序列和拓扑也可能形成高度可识别的语义
骨架。按当前第②档定义，这就是 π 能独立重构 J 的直接证据，Gate1 必须拒绝。

### 3.4 RMSNorm：runtime scalar 和更长图仍没有打破可识别性

RMSNorm 比 softmax 多了一个权重输入、runtime `eps`、`mean`、
`sqrt -> recip` 以及两次输出广播。本轮首先用现有 IntentIR
primitive 补齐了通用契约：

- 参与计算的 scalar 以 J `shape=[]` 输入表示，由 interface
  `tensor_ref` 连到 runtime `eps`，不烘成常量；
- 源码 `sum / N` 在 J 中规范成 `reduce mean`，重放时除数必须是
  全局语义长度 `N`，不是 `BLOCK_N`；
- 源码 `rsqrt` 在 J 中使用既有 `sqrt + recip`，Gate1 沿用已登记的
  局部 `1/sqrt <-> rsqrt` 规范化。

第一次有效 normal replay 又暴露提升官错选了已有的
`materialize_index_tile`；明确“源码内联 `ptr + row*N + lane` 属于
staged pointer tile”后，新 artifact 正确选中三次
`materialize_pointer_tile`，normal replay 首候选达到 `score=1.0`。

但 masked packet 仅凭三个 pointer、一个 runtime float scalar、归约/广播/
elementwise 的有序拓扑和规则库，仍然完整恢复了
`x*x -> sum/N -> +eps -> sqrt -> recip -> *x -> *w`，包括地址 staging、
mask 和 runtime `eps`。恢复 ABI 后 TTIR 零差异，`L=1.0`。

### 3.5 matmul：masked 不必完全闭合，也会超过泄漏阈值

matmul 首先暴露了一个实现 false reject：source n-gram 扫描曾覆盖整个 J，
而 B 层依法必须保存源码提取的 domain、index expression 和 predicate，导致
合法 J.B 被当成 π 复制源码。修复后 API 拼写仍扫描 J+π，source n-gram 只
扫描 π；v1 artifact 本地复扫已通过。

随后 v2 的 J 已正确表达单个 `contraction("mk,kn->mn")`，但 π 又选择了
`realize_reduce`，已有 dot 卡也没有明确 K tile domain、result dtype、两侧
access/mask/fill 和 staged pointer association。三个 normal 候选都能编译，
却固定停在 `0.577149`。补全现有卡与 Lift 分层契约后，v3 normal 首候选
达到 TTIR 零差异。

masked matmul 没有像 add/softmax/RMSNorm 那样完全猜中：地址 staging 与
dataflow 仍不同，硬项未全过。但二维接口、K-tiled-dot 机制、三组 tail mask
和标准 contraction 先验仍让它得到 `0.804518`，高于 `0.5`。这说明当前阻塞
并不要求 masked replay 精确复原源码；只要 π/C/接口与模型先验把剩余空间压缩
到足够接近，现有泄漏比也会 fail-closed 拒绝。

## 4. 根因不是一个普通 bug

现有第②档消融把以下三种情况合并成同一个观测：

1. π/C 真的偷藏了 J 语义；
2. π/C 没写出具体语义，但剩余假设空间很小，模型凭先验偶然或稳定猜中；
3. π 没写算子参数，却用规则 id 的顺序、重复次数和引用拓扑暴露了足以识别
   整体算法的机制骨架。

softmax、RMSNorm 和 matmul 的实测表明不能再假设“高熵 Kernel 遮盖 J 后自然
会崩掉”。对 add 等
低熵 Kernel，模型先验与 π 泄漏仍不可区分；对 softmax/RMSNorm，π 的机制
拓扑结合接口结构已经足以精确恢复语义；对 matmul，即使没有精确恢复地址
staging，也足以超过当前泄漏阈值。反过来，只做一次随机 masked generation，
若模型这次主动失败，也不能充分证明没有泄漏。

因此现有一次性 masked-J 存在三个问题：

- **不可识别性：** 无法区分泄漏与先验猜测；
- **单样本不稳定性：** 一次模型采样可能左右最终接收结果；
- **π 结构泄漏：** 合法机制卡的有序拓扑可能已成为语义的替代编码。

Gate1 当前 fail-closed 是正确实现，但若不修改实验设计，不仅低熵 Kernel，连
softmax/RMSNorm/matmul 这类复合图也可能无法通过最终接受，且单次消融
证据仍不够强。

## 5. 现在可以选择的路线

### 路线 A：保持现方案，不改（最严格、覆盖率最低）

把 add、copy、部分简单 reduction 以及 π 可独立识别的
softmax/RMSNorm/matmul 标记为
`J_NON_IDENTIFYING` / 当前已有的 `LEAKAGE_REJECTED`，不宣称其
replay-closed。Gate2 可作为独立诊断运行，但不能改变 Gate1 结论。

优点：完全忠于当前文档；不引入新裁决。

缺点：大量简单 Kernel 和 π 拓扑可识别的复合 Kernel 都无法提供“语义因 J
而来”的单样本证据，规模化闭合率会显著下降。

### 路线 B：增加预注册的多次 masked 统计（改动较小，不推荐单独使用）

对同一 `(masked J, π, C)` 固定运行 K 次，报告 masked 闭合率与 score 分布，
用预先写死的统计阈值代替单次结果。

优点：能降低一次随机失败造成的 false-safe。

缺点：成本高；仍无法区分“π 泄漏”与稳定模型先验，也无法消除
softmax/RMSNorm/matmul 已
观察到的 π 拓扑泄漏。不能采用“重试到失败就通过”的做法。

### 路线 C：加入成对语义干预（建议进一步论证后采用）

不再只问“J 空了还能不能猜中”，而是同时问“J 中唯一改变的语义是否会驱动
重放发生对应改变”。例如保持接口、拓扑、π、C 和 opaque 名不变，仅构造：

```text
add  <-> sub / mul
sum  <-> max / min
exp  <-> log
```

正常 J 与干预 J' 分别交给同一重写员。Gate1 需要一个确定性的、范围受限的
J→TTIR 语义投影，检查 replay 是否跟随干预改变对应的 TTIR 语义项，同时保持
其余结构不变。

这比空白遮盖更接近因果问题：若模型总默认 add，它会在 J'=sub 时暴露；若输出
随 J 改变，才说明语义内容真正控制重放。

但该路线会新增“反事实 J 与预期 TTIR 差异”的裁决逻辑，属于 SCRC 方案的明确
扩展，不应在未经批准时直接实现。需要先限定可干预 op、变换合法性、浮点语义、
规则卡覆盖和接受阈值。

### 路线 D：把 masked-J 降为数据集级实验，而非逐 Kernel 接收门槛

构造接口/π/C 相同或近似的平衡语义族，在 held-out 集上报告 masked 预测准确率
与泄漏比；单个 Kernel 的 Gate1 只要求 source/replay 结构闭合，反作弊结论在
整个数据集上给出。

优点：能统计区分模型先验与 artifact 泄漏。

缺点：改变了当前“每个 artifact 最终接受前必跑一次”的接收定义，属于更大的
方案修改。

## 6. 建议与待批准事项

当前建议分两步：

1. 在获得批准前采用路线 A：五核探针保持各自的
   `LEAKAGE_REJECTED`，不篡改阈值；下一步只按既定主线进入首批
   20～40 个真实 LLM extraction kernel，继续收集 held-out 证据；
2. 单独评审路线 C，把“成对语义干预”作为第八章消融的补充或替代，并在方案
   文档中先写死适用 op、裁决方式与失败语义，再实现代码。

请求确认的问题只有一个：

> 是否允许把 Gate1 第②档从“单次空白 masked-J”扩展为“空白 masked-J +
> 成对语义干预”，并以成对干预作为低熵或 π 拓扑可识别 Kernel 的主要 J
> 因果证据？

在得到确认前，本项目不会自行加入该裁决，也不会把最新
add/row_sum/softmax/rmsnorm/matmul 标成 Gate1 闭合。

## 7. 可复核证据

最新 `gpt-5.6-sol / xhigh` 调用：

| Kernel | Lift response | Normal replay response | Masked response | Normal | Masked | L | Gate1 |
|---|---|---|---|---:|---:|---:|---|
| add | `resp_09756db56f798306016a56726741508190a5de16f83c99e5f0` | `resp_0d7755f9c6965b91016a5672c94000819498870cf2542f2a02` | `resp_0c7ede99b0c80499016a5672d7a5d88195b673d75e0956bf71` | 1.0 | 1.0 | 1.0 | `LEAKAGE_REJECTED` |
| row_sum | `resp_06a7127cd99a3713016a566f62f8c8819ba4b009051c465f46` | `resp_0a5bdeb327c00b91016a566fbfe1cc819bb2761cff2ceb2e6c` | `resp_05c1df0cfaaec727016a566fd31dc8819b8413f368ba41231c` | 1.0 | 0.778676 | 0.778676 | `LEAKAGE_REJECTED` |
| softmax | `resp_06c991a807d02279016a567a78a9a08199b54735a58e73572e` | `resp_0344d49f6a8f19df016a567b0333188199a70cab5e8db3d434` | `resp_0c7cfec07fab4718016a567b16fc6c819a8e6418e53d79809a` | 1.0 | 1.0 | 1.0 | `LEAKAGE_REJECTED` |
| rmsnorm | `resp_0823d6e4ae6a822d016a5686106720819b82d4fe1c4316a8c8` | `resp_07d0fd24317b9eab016a5686c170f88197a117b0351f44083c` | `resp_06e2034caad6e160016a5686d57ac8819487a53ffde964051b` | 1.0 | 1.0 | 1.0 | `LEAKAGE_REJECTED` |
| matmul | `resp_04a7daf45b1296f1016a568fec35b8819aa8ef4b2911722276` | `resp_0b9bba8b83ab72aa016a5690639ab88199a03a8cf6f0dcf691` | `resp_03b3891a30d66008016a5690833348819bb7efddbdcbf70180` | 1.0 | 0.804518 | 0.804518 | `LEAKAGE_REJECTED` |

本地临时报告：

- `/tmp/scrc_live_add_result_v2.json`；
- `/tmp/scrc_live_row_sum_result_v3.json`；
- `/tmp/scrc_live_softmax_result_v3.json`；
- `/tmp/scrc_live_rmsnorm_result_v3.json`；
- `/tmp/scrc_live_rmsnorm_gate2_diagnostic_v3.json`；
- `/tmp/scrc_live_matmul_result_v3.json`；
- `/tmp/scrc_live_matmul_gate2_diagnostic_v3.json`。

临时文件不纳入仓库。最新 harness 报告已经移除原始源 kernel，只保留 source
摘要，并保存所有真正被 Gate1 验证过的 normal/masked source-blind 候选代码。
