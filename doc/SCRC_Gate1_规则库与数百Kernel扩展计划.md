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

## 7. 分批推进

### 第一批：20～40 个 Triton

- live add、row-sum、softmax、RMSNorm、matmul 的 normal replay 已结构闭合，
  五者都因 masked-J 消融被最终拒绝；add 的历史正式 Gate2 与五者的独立
  production lower2c 三方预检均已跑通；
- RMSNorm 和 matmul 还对真实 live 被拒绝样本离线跑通 Gate2，记入方案 9.1
  的 rejected 对照，不改变 Gate1 裁决；
- 五核探针到此只完成机制探路，不作为数百 Kernel 覆盖率；
- 已建立 24 个真实 FlagGems Triton kernel 的显式队列；当前 `fg_dot`、
  `fg_addr`、`fg_ones`、final-stage `fg_sum`、final-stage `fg_mean` 已完成
  compiler/execution contract 并实际运行，其余 19 个仍为 queued；
- 五例 normal replay 均达到 TTIR 硬闭合；masked score 依次为 `1.0`、
  `0.763387`、`0.925`、`1.0`、`1.0`，所以正式状态全部是
  `LEAKAGE_REJECTED`；
- 五个 rejected 样本的离线 Gate2 三方执行均 PASS，只计入
  `P(Gate2 pass | Gate1 rejected)` 对照，不覆盖 Gate1；
- `fg_dot/fg_addr` 是规则增长样本，后三例没有新增卡并直接复用了已有 19 张卡；
  这开始提供冻结后复用证据，但仍不足以报告总体准确率；
- `fg_arange/fg_trace/fg_eye` 继续 queued：现有 production Lower2C 分别缺 iota、
  gather 和二维对角 predicate realization，本轮没有越过计划擅自补新 op；
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
