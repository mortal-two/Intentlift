# SCRC Gate1：规则库与数百 Kernel 扩展计划

## 1. 当前 Gate1 不是哪里有问题就加检查

Gate1 的职责固定为：

```text
J + π + 接口 + C 槽 + 规则卡
  → 源码隔离重写
  → source/replay 同前端 compile-only
  → TTIR 结构闭合
```

它不应该继续长成：

- JSON 解释器；
- 独立 schema/obligation verifier；
- 编译环境资格审查；
- 对齐、缓存、版本等政策的总门禁；
- 人工判断“这个 op 是否 verify-ready”的体系。

当前结论必须保守且明确：**Gate1 还没有覆盖全部可能的
Kernel 形态**。现有证据是一个可执行的 Triton SCRC MVP，不是一般
Triton 语义等价判定器：

- 早期 add/row-sum/softmax/RMSNorm/matmul 手写 J/π 只用于机制探针；
- 后续已将这五个族和 11 个 FlagGems 样本重新走真实
  `gpt-5.6-sol/xhigh` Lift，16 个 normal replay 均达到 TTIR 硬闭合；
- 16/16 最终都因 masked-J `L > 0.5` 被拒绝，因此正式
  `REPLAY_CLOSED` 真实 LLM artifact 样本仍为 0；
- 24 个真实 FlagGems 队列完成 11 个、剩余 13 个 queued；11/24 是
  进度，不是覆盖率。

## 2. 当前规则库

实现位置：`python/intentir/verify/gate1/rules.py`。

当前 19 张 Triton 规则卡分为：

1. domain/index：native/wide program domain、完整 integer offset tile、分阶段
   pointer tile；
2. memory：masked access、分阶段 mask tile、unmasked access、output mapping；
3. A 算子：elementwise、显式轴 reduce、flattened reduce、broadcast、cast、select；
4. contraction/loop：K-tiled dot、outer product、loop carry；
5. 接口/C：runtime scalar、schedule slot。

规则卡只提供“如何在 Triton 中实现某种机制”，具体算什么必须从 J 读取。
π 只保存 rule id 与 J/B 引用。

原先新增的 `materialize_pointer_tile` 不是预先为 softmax 写的专用模板。它来自
真实 softmax normal replay 的 TTIR diff：源码保留两次 staged
`pointer + scalar base + lane`，而重放先合成完整 integer offset，导致
`tt.addptr` 与 `tt.splat` 多重集不同。新卡只表达这个地址物化机制，
不包含 softmax 算子序列。

RMSNorm 证明这张卡是通用机制而非 softmax 特例：提升官首先错选
integer-offset card 时，normal replay 三个候选都固定停在
`score=0.770359`；明确源码内联 `ptr + row*N + lane` 的选卡契约后，
同一 pointer card 让 RMSNorm 达到 TTIR 零差异。

RMSNorm 没有新增规则卡，而是完善三张已有通用卡：

- `bind_runtime_scalar`：区分纯 shape extent 与由 `tensor_ref` 连到 J
  `shape=[]` 输入的语义 scalar；
- `realize_reduce`：`mean` 实现为局部 sum 除全局 J 轴的逻辑长度，
  不得除 C 的 tile 大小；
- `realize_elementwise`：补全 `recip` 和已有 binary/unary 构造映射，
  语义仍只从 J 节点读取。

matmul 同样没有新增专用卡，而是用真实 v2 diff 完善三张已有卡：

- `materialize_pointer_tile` v2：预计算的 m/n/k coordinate component 不是完整
  integer offset，必须保留 `pointer + row-term + column-term` 的结合方式；
- `realize_masked_access` v3：二维向量 predicate 使用 `&`，不得改写为标量
  `and` 或 `&&`；
- `realize_k_tiled_dot` v3：从 contraction/J.B 读取 subscripts、operand 顺序、
  result dtype、K 语义 extent、K tile slot/domain、两侧 access/mask/fill 和
  output tile，且禁止把普通 matmul 的 schedule-only K loop 再建成 J
  `reduce/loop/broadcast`。

同一轮还收紧了 artifact 接缝：interface `tensor_ref` 的 dtype 必须与引用的
J tensor 一致，并与可信 compiler view 的真实 tensor dtype 一致。它防止提升官
把 f16 matmul 输入漂成 f32 后仍进入重放，不是新的独立 verifier。

五核之后的首批真实 FlagGems 扩展样本又由稳定 TTIR diff 触发四张通用卡：

- `bind_wide_program_domain`：仅在 π 声明源码把 program coordinate 显式扩到
  i64 时，保留 `tl.program_id(axis).to(tl.int64)`；
- `realize_flattened_reduce`：区分源码省略局部 axis 的整 tile flatten reduction
  与显式 `tile_axis` reduction；
- `materialize_mask_tile`：先计算并复用一维 predicate component，再对 boolean
  value 做局部 rank insertion，禁止在二维 tile 上重复比较；
- `realize_outer_product`：`m,n->mn` 没有 contracted label，只通过两侧 rank
  insertion 和乘法实现，不引入 dot、K loop 或 reduction。

`declare_schedule_slot` 同时升级为 v2：只有 interface `kind=constexpr` 的槽属于
kernel 签名；`kind=compiler` 的 `num_warps/num_stages` 等槽只在 launch/compile
时注入，不能成为函数参数或 kernel body 变量。

后续 final-stage `fg_prod` 没有触发新卡，而是把已有 `realize_reduce` 升到 v4：
当 J 明确为 `reduce(fn=prod)` 时，规则卡要求在同一 replay 模块定义纯
`@triton.jit` 二元乘法 combiner 并交给 `tl.reduce`。相应安全策略只允许经过同一
AST 白名单验证、调用图无递归的 sibling JIT helper，仍拒绝任意 global/free
name。`fg_max/fg_min/fg_vector_l2` 直接复用 flattened-reduce 与 elementwise 卡。

`fg_vector_l1` 也没有新增规则卡，而是由真实 replay-policy 失败把已有
`realize_elementwise` 升到 v4：补齐 IntentIR 已有 `abs/pow` 到
`tl.abs/libdevice.pow` 的 Triton 映射。AST 策略只放行受信
`triton.language.extra.cuda.libdevice.pow`，其他 `libdevice.*` 仍拒绝。
同时，source-aware relift 现在只在已经产生可编译 source/replay comparison
且存在 normal 硬差异时触发；纯 replay policy/生成失败不能伪装成 J/π 问题。

这些卡分别由 `fg_dot` 和 `fg_addr` 的真实失败触发。因此两例证明规则库已经能
沿真实外部 Kernel 的 diff 增长并最终让 normal replay 闭合，但不能被表述为
“冻结规则库上的独立 held-out 验证”。冻结后的复用能力仍要由后续 queued 样本
检验。

检索过程已经显式化：

```text
π.steps
  → RuleRegistry 按首次使用顺序去重
  → 生成 selection trace
  → 仅把被选中的 cards 交给重写员
```

## 3. 规则库如何增长

每张新卡必须由真实失败驱动，流程固定：

1. 真实 Kernel 经 LLM 生成 J/π/C；
2. 重写失败或 TTIR 不闭合；
3. 判断是：
   - π 没选对已有卡；
   - 现有卡描述不够；
   - 缺一张新的实现机制卡；
   - J/A/B/C 本身有问题；
4. 只有第三种才新增卡；
5. 新卡必须记录适用的既有 J/B 字段、读取字段、最小示例和常见错误；
6. 用触发它的失败 Kernel 和至少一个不应受影响的 Kernel 回归。

不能因为“未来可能有某种复杂 Kernel”就先写大量猜测规则。

## 4. TTIR 规范化规则如何增长

规范化规则与重放规则卡不是同一件事：

- 重放规则卡帮助 LLM 生成前端代码；
- TTIR 规范化规则帮助纯代码裁判吸收稳定的等价写法。

当前只登记两个真实观测模式：

- `x + 0`；
- `1/sqrt(x) ↔ rsqrt(x)`。

新增规范化必须同时满足：

1. source/replay 实际 diff 可重复；
2. Gate2 在边界与随机输入上确认同行为；
3. 规则是局部、可枚举的 SSA 模式；
4. 原有错误变异仍被 Gate1 拒绝。

这比把 comparator 做成模糊相似度更忠于 SCRC。

## 5. 数百 Kernel 的正式样本来源

正式样本统一要求：

```text
真实源码 → LLM extract_scrc_artifact → Gate1 → Gate2
```

以下材料可以保留，但不能计入端到端闭合结论：

- 手写 `.mlir` lowering fixture；
- 手写/固定 Intent DAG 的 PK；
- 人工 J/π 五核探针；
- 直接从模板生成的重放答案。

当前 corpus 的 111 份 provenance 中，110 份声明来自 provider/LLM，只有
`corpus/kernels/layer_norm` 明确 hand-authored；但高级 GPU PK 中仍有大量手写
Intent/MLIR 起点。因此扩展时优先选择“有真实 source_file、能重新调用 LLM、
能执行 source”的样本，而不是直接复用 PK 输入。

## 6. 覆盖复杂特性的方式

只沿用项目已经规划的 A/B/C 和 op-signatures，不另造体系。

当前覆盖矩阵如下。“有规则卡”、“有手写单测”和“有真实 LLM
Lift 端到端证据”是三个不同层次，不能互相代替。

| 分层 | 已有真实证据 | 部分覆盖 | 尚无端到端证据 |
|---|---|---|---|
| A：算子/值数据流 | 常量 fill、基础 elementwise、sum/mean/max/min/prod、softmax/RMSNorm 固定链、matmul/dot/outer product | cast/dtype promotion、广播、多步融合、常量敏感性、混合精度 | RNG/dropout、scan/sort/top-k、量化/位运算组合、稀疏、多输出、大型 attention/跨 Kernel 图 |
| B：域/索引/mask/effect | 一维 tail mask、基础二维仿射索引、masked load/store、fill/guard、row reduction、staged mask tile、axis-0 take gather | 多 program axis、非末轴/多重归约、动态 stride、转置访问、复合/因果 mask、整数宽度、一般 along-axis gather | online/structured loop、scan carry、atomic、scatter、block pointer/descriptor、别名敏感副作用顺序 |
| C：实现/调度 | 单个 reviewed profile 的 constexpr block、num_warps/num_stages 槽、K tile、wide program id | `do_not_specialize` 的实际 TTIR 影响、grid 合同、编译属性 | 同一 Kernel 的多 C/多 specialization 聚合门禁、descriptor ABI、架构专属约束、跨 Triton 版本系统规范化 |
| 前端/裁判基础设施 | Triton compile-only、FlagGems wrapper、rank-0 pointer、TTIR interface/op/dataflow/reduce/predicate/effect 比较 | 复杂 region/loop 指纹、编译失败的安全差异摘要、masked-J 可识别性 | TileLang TIR、CUDA AST/PTX 正式 Gate1、atomics/内存序指纹、shape-parametric 等价 |

当前 comparator 更准确的定位是：对“无复杂结构化循环、仿射索引、
基础 mask/归约/dot”的 TTIR 结构闭包工具。`fg_log_softmax` 使用 online
结构化循环，不能因为名字中有 softmax 就认为已被当前 softmax 样本覆盖。

### A 层

通过真实样本逐步覆盖现有计划中的：

- elementwise 图；
- reduce/scan；
- contraction；
- structured loop 与 carry；
- 已有 gather/scatter/sort/RNG/attention 等表示。

### B 层

从源码真实提取：

- index expression；
- load mask/fill；
- store guard；
- reduction identity/masked contribution；
- output mapping 与已有 effect。

### C 层

保存源实现真实出现的 tile/warp/stage/loop hint；重写员只看槽，值由注入器
后填。不得把 A/B 语义塞进 C。

如果某个真实 Kernel 不能用这些现有结构表达，先出 A/B/C 扩展报告，经批准
后再改，不能由 Gate1 实现自行补 op。

### Gate1 normal 差异的外层重提循环

Gate1 不是只判一次就丢弃 artifact。对可编译但尚存任何硬差异的
normal source/replay，当前实现已按下列循环工作：

```text
原始 source
  → SCRC Lift 产生 J/π/C/接口/定位表
  → 源码隔离 normal replay
  → source/replay 同前端编译与纯代码硬比较
  → 内层有界 replay repair 仍未闭合
  → 提取 interface/op/dataflow/reduction/predicate-effect 结构差异
  → 原始 source + source fingerprint + 差异历史 + 当前 J/π
      交回可看源码的 SCRC Lift
  → 只修改 J/π/location_map，冻结 interface 与 C
  → 从新 artifact 重新进行源码隔离 replay
  → normal 硬闭合或重提预算耗尽
```

这里的“差异较大”不引入新的模糊分数阈值：只要任一 Gate1
硬项仍失败，内层小修耗尽后就进入外层重提。score 只用来保留最优
失败候选，不决定是否闭合。

回流给 Lift 的内容不包含 replay 源码；它包含原始 source、源指纹和
经白名单序列化的纯代码差异。Gate2 输出不进入该循环，不能用动态
PASS 覆盖 Gate1 差异。

normal 闭合后仍要执行 masked-J 反作弊终检。当前不把
`LEAKAGE_REJECTED` 自动伪装成 normal source/replay 语义差异；如需建立
专门的信息重分配循环，必须先在方案中单独定义反作弊反馈与防刷分
约束，不与本次 normal diff relift 混在一起。

## 7. 分批推进

### 第一批：20～40 个 Triton

- live add、row-sum、softmax、RMSNorm、matmul 的 normal replay 已结构闭合，
  五者都因 masked-J 消融被最终拒绝；add 的历史正式 Gate2 与五者的独立
  production lower2c 三方预检均已跑通；
- RMSNorm 和 matmul 还对真实 live 被拒绝样本离线跑通 Gate2，记入方案 9.1
  的 rejected 对照，不改变 Gate1 裁决；
- 五核探针到此只完成机制探路，不作为数百 Kernel 覆盖率；
- 已建立 24 个真实 FlagGems Triton kernel 的显式队列；当前 `fg_dot`、
  `fg_addr`、`fg_ones`、`fg_embedding`、final-stage `fg_sum`、final-stage `fg_mean`、
  final-stage `fg_max/fg_min/fg_prod/fg_vector_l2` 与 first-stage
  `fg_vector_l1` 已完成 compiler/execution contract 并实际运行，其余 13 个仍为
  queued；
- 十一例 normal replay 均达到 TTIR 硬闭合；除 embedding 外原十例 masked score
  依次为 `1.0`、
  `0.763387`、`0.925`、`1.0`、`1.0`、`0.507692`、`0.507692`、
  `0.509470`、`0.917544`；`fg_vector_l1` 的两次独立 masked-J 为
  `0.906140` 与 `1.0`，`fg_embedding` masked-J 为 `1.0`，所以正式状态全部是
  `LEAKAGE_REJECTED`；
- 十一个 rejected 样本的离线 Gate2 三方执行均 PASS，只计入
  `P(Gate2 pass | Gate1 rejected)` 对照，不覆盖 Gate1；
- Gate2 已从单 case 扩成严格 suite；首个 `fg_min` live matrix 覆盖固定 C 下
  `{1,63,64}` 三种 shape、四种值域和三个 seed，共 36 个三方 case 全部零误差
  PASS。该结果增强行为对照，但不把单一 Gate1 compile specialization 偷换成
  “整个 Kernel 已结构闭合”；
- `fg_dot/fg_addr` 是新增规则卡的增长样本；`fg_prod` 只补齐既有 reduce 卡的
  已规划 prod 映射，`fg_vector_l1` 只补齐既有 elementwise 卡的 abs/pow 映射与
  受限 replay 安全路径，`fg_embedding` 和其余六例没有新增卡并复用了已有 19 张
  卡；embedding 的数据依赖 access expression 来自 J/B，不把具体拓扑塞进规则卡。
  这开始提供冻结后复用证据，但仍不足以报告总体准确率；
- `fg_vector_l1` 的 rejected-sample Gate2 还覆盖
  `M={1,1023,1024}`、`ord={1.0,1.5,2.0}`、normal/uniform 与 3 seeds，
  18/18 三方 PASS；source/replay 始终零误差，两者对 Lower2C 最大绝对误差
  `0.005859375`；
- `fg_embedding` 在静态 `indices[17]`、`weight[97,13]` J 下运行 2 个合法
  index/weight value profile × 3 seeds，6/6 三方零误差 PASS；因为 compiler view
  不提供 runtime shape scalar，不能把这个证据扩大成动态 shape 支持；
- `fg_arange/fg_trace/fg_eye` 继续 queued：production Lower2C 仍分别缺 iota、
  gather+reduce 的端到端 contract 和二维对角 predicate realization；本轮加入的
  通用 take/along-axis gather lowering 只消除了 `fg_embedding` 已实际触发的接缝，
  没有替未运行候选制造结果；
- 覆盖五核同类和现有 corpus 中可执行 source；
- 全部重新走 LLM J/π/C；
- 跑小修、重提、规则检索、Gate2；
- 建立第一版真实失败集。

首批统计必须单独记录 Lower2C 能力边界。例如当前 matmul 第三腿会消费已经
量化为 f16 的输入值并输出 f32，但 production MLIR 仍统一发射为 f32；在没有
获批的 A/B/C 或 lowering 扩展报告前，不把它表述为 mixed-dtype faithful。

### 第二批：约 100 个

- 加入现有计划内的 fused、structured loop、数据依赖访问；
- 只按真实失败补规则；
- 检查规则是否复用于未参与编写的 Kernel。

### 第三批：数百个

- 批处理与缓存只做工程加速，不改变 Gate1/Gate2裁决；
- 自动汇总编译失败、结构不闭合、J 重提、lowering unsupported 与三方分歧；
- 形成论文需要的 SCRC 指标和失败案例。

## 8. 进入下一批前必须满足的条件

不新增自造指标，沿用 SCRC 文档已有观察：

- 重放可编译；
- Gate1 闭合；
- `P(Gate2 pass | Gate1 closed)`；
- `P(Gate2 pass | Gate1 rejected)` 对照；
- 小修/重提与规则卡成本；
- 泄漏消融。

最重要的质量线不是“手写 fixture 全过”，而是同一套规则卡能否让未参与规则
编写的真实 LLM 提取 Kernel 完成 Gate1，并在 Gate2 三方实跑中与源一致。
