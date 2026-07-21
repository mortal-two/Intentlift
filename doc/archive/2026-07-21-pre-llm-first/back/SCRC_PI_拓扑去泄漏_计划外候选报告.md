# SCRC π 拓扑去泄漏：严格偏序计划外候选报告（已取代归档）

> 状态：**仅报告，未实现；当前推荐已被更上层的“J 语义完备 + 确定性 π”候选取代。**
>
> 日期：2026-07-16
>
> 当前正式协议：`j_complete_deterministic_pi_relift_v8`
> 约束：批准前不得修改 `ReplayPlan` schema、Lift/replay prompt、Gate1 verdict、
> cache identity 或正式实验口径；不得使用 Trellis task/workflow。

> 2026-07-16 更新：用户进一步提出 π 应完全由 `J + C + 规则库` 确定性生成。
> 该方向比单独把 ordered π 改成严格偏序更根本，详见
> `doc/back/SCRC_J语义完备与确定性PI重构_候选方案.md`。本报告保留为备选设计记录，
> 不再是当前首选实现路线；上层方案现已实现。

## 1. 问题是什么

当前 π 已经移除了源码文本、具体前端 API、语义字面量和来源提示型名称，并将
J/B/π 引用统一改写为 `n*/a*/t*/p*/c*`。这解决了**命名泄漏**，但没有消除
**拓扑泄漏**。

当前 `ReplayPlan` 是完整线性序列：

```text
step[0] -> step[1] -> ... -> step[N-1]
```

即使每个 step 只含 `{rule, node_ref, refs}`，下列信息仍可能识别算法：

- 规则实例的种类序列；
- 同一规则的重复次数；
- target/context ref 的复用形状；
- load/reduce/elementwise/store 的全局顺序；
- 某些 ref 在多个 step 中形成的局部依赖图。

历史 masked-J 实验已经给出直接证据：softmax、RMSNorm 等复合 Kernel 在 J 内容
被遮盖后，仍可借助 π 的有序机制骨架、接口和模型先验恢复接近甚至完全相同的
结构。因此只做 opaque rename 不足以解决拓扑通道。

## 2. 不能采用的简单修补

### 2.1 随机打乱当前 π

当前顺序包含真实施工依赖。无约束 shuffle 会把 load、mask、loop carry、store 等
顺序打坏，使 replay 失败；这种失败不能证明泄漏降低。

### 2.2 给所有 Kernel 填充统一 dummy step

dummy/no-op 会破坏“π 是严格、真实施工合同”的定义，还可能要求 writer 忽略部分
π，重新引入隐藏 baseline。

### 2.3 只把 rule id 改成 `r0/r1`

opaque rule handle 可以消除 `realize_reduce` 等直接词面提示，但被选中的规则卡正文
和实例拓扑仍然存在，因此只能作为辅助，不是完整解决方案。

### 2.4 把 π 拆成多个 shard 后分别交给多个 writer

这会引入代码片段合并、变量作用域、重复 load/mask 和跨 shard 依赖的新正确性来源，
并不能阻止多个 shard 合起来恢复 J。已有多分面候选报告已经将其列为未批准方向。

### 2.5 把 refs/J 映射塞入 C

C 只拥有调度 slot/value，且在 relift 中冻结。J/π 引用随新 J 改变，不能由 C
拥有；否则 C 会成为新的语义与拓扑通道。

## 3. 推荐候选：完整机制实例 + 严格偏序 π

推荐把“严格”从“每两个 step 都有线性先后”改为：

> 每个必要机制实例都显式存在；所有必要先后约束都显式或由其唯一信息所有者
> J/B 确定性给出；独立 step 不携带任意顺序。

候选逻辑模型：

```json
{
  "steps": {
    "s0": {"rule": "m0", "target_ref": "a0", "refs": ["d0"]},
    "s1": {"rule": "m1", "target_ref": "n0", "refs": ["a0"]},
    "s2": {"rule": "m2", "target_ref": "a1", "refs": ["n0"]}
  },
  "must_precede": [
    {"before": "s0", "after": "s1", "reason": "materialization_dependency"},
    {"before": "s1", "after": "s2", "reason": "effect_dependency"}
  ]
}
```

这只是逻辑示意，不是已批准 schema。

### 3.1 信息所有权

| 信息 | 唯一所有者 |
|---|---|
| 算子、操作数、结果、语义 DAG | J.A |
| access、mask、effect、output mapping | J.B |
| 每个 target 选择什么实现机制 | π step |
| J/B 已经表达的依赖 | J/B，不在 π 重复 |
| J/B 无法表达的前端 staging 依赖 | π `must_precede` |
| 规则卡正文和前端 API | 规则库 |
| 调度值 | C |

π 不再复述完整 J 拓扑，只保存“机制赋值”和真正的实现剩余依赖。

### 3.2 确定性 composer

纯代码 composer 输入：

```text
J/B dependency graph
+ π complete mechanism instances
+ π explicit residual edges
+ fixed phase lattice
```

固定 phase lattice 只表达通用施工约束，例如：

```text
bind/control -> address/mask -> access -> compute -> output/effect
```

composer 执行稳定拓扑排序，生成 writer 使用的唯一线性 `π_materialized`。稳定 key
只能使用 phase、opaque target class 和声明序号，不使用源码行号、原始名称或语义
值。

composer **不能**：

- 自动选择d缺失规则；
- 从 J 推断 Lift 没有写出的 mechanism；
- 添加新 step；
- 用后一条边覆盖冲突；
- 在环或多重冲突时猜测顺序。

缺 step、非法 edge、环或冲突全部 fail-closed。

### 3.3 residual edge 约束

每条显式边必须属于预注册的非语义原因集合，例如：

- `materialization_dependency`；
- `reuse_dependency`；
- `loop_carry_dependency`；
- `effect_dependency`；
- `frontend_scope_dependency`。

每个 reason 只定义允许连接的 rule family/owner kind，不携带算子、轴、表达式或
源码提示。能够由 J/B 数据流推出的边不得再写入 residual edge，以避免 π 重复编码
J 图。

### 3.4 writer-facing rule handle

在 writer packet 中可进一步把选中的规则卡确定性改名为 `m0/m1/...`，π 与
`rule_cards` 同时使用该 handle。持久化 registry id 不变，报告和裁判仍能审计真实
规则。该措施只减少词面提示，不能替代偏序化。

## 4. 为什么仍属于严格重放规则

候选没有把 π 改成“建议”或“可选提示”：

- 每个必要 target 仍必须有且只有一个完整 mechanism 实例；
- override family 仍禁止重复/冲突；
- 所有必要依赖必须来自 J/B 或显式 residual edge；
- composer 结果唯一、可缓存、可审计；
- writer 必须执行 materialized π 的全部 step；
- 空 π、缺 step、缺依赖仍 fail-closed。

改变的是：不再让 Lift 用任意线性顺序为独立 step 编码额外信息。

## 5. 反作弊验证必须怎么改

### 5.1 保留 paired J/B intervention 为主要单样本证据

冻结 π、C、interface 和规则卡，只改变一个 J/B 语义坐标；replay TTIR 必须出现
预注册的精确局部 delta。它直接检验 J 是否因果控制 replay，不依赖“masked 后模型
会不会猜中”。

### 5.2 增加独立 step 置换不变性

对偏序中互不依赖的 step 使用不同合法拓扑序；composer 应归一到同一个
`π_materialized`，或两次 replay 必须得到相同 Gate1 硬指纹。否则仍存在任意顺序
通道。

### 5.3 residual-edge dropout

只删除一条显式 residual edge：

- 若 composer 仍生成同一计划，该 edge 是冗余信息，应拒绝或删除；
- 若计划发生预期的局部结构变化，记录为该 edge 的因果证据；
- generation/compile failure 仍是 inconclusive，不允许修 replay。

### 5.4 masked-J 继续作为诊断，不单独裁决

blank/masked-J 同时混合模型先验、接口可猜性、规则卡与 π 信息。即使偏序化后也
不能把单次 masked closure 直接解释为 π 泄漏；它继续用于发现异常，不覆盖 paired
causal evidence。

## 6. test-only 探针（获批后才能实施）

第一步不直接迁移生产 schema，而是对现有 ordered π 构造只用于试验的偏序 view：

1. 使用 add、row-sum、softmax、RMSNorm、matmul 的 fresh LLM artifact；
2. 从 J/B 提取已拥有的依赖；
3. 将当前 step 中不可由 J/B 解释的顺序列为 residual edge 候选；
4. 用纯代码 composer 物化；
5. 比较当前 ordered π 与偏序 π 的 normal Gate1、paired intervention、π audit；
6. 再选择至少两个未参与设计的真实 FlagGems Kernel；
7. 不使用旧手写 J/π 作为正式结果。

探针必须回答：

- normal replay 是否仍可编译并硬闭合；
- π step 完整性是否保持；
- 独立顺序是否被确定性归一；
- masked-J 的拓扑可识别性是否下降；
- paired J/B 因果 delta 是否保持；
- residual edge 是否真实必要而非新的信息藏匿点。

## 7. 获准进入正式实现的条件

至少同时满足：

1. 五核与选定 held-out 的现有 ordered π 对照可复现；
2. 偏序 view 的 normal Gate1 不因 composer 漏步骤而系统退化；
3. 所有 step coverage、override 和规则选择仍由纯代码精确检查；
4. 独立 step 置换得到唯一 materialization；
5. 没有把机制选择自动移入规则库或 composer；
6. paired intervention 因果覆盖不下降；
7. protocol/cache/report schema 明确升级，旧 v7 结果不得静默复用。

## 8. 当前决定

本轮只形成候选报告：

- 不修改当前 `ReplayPlan.steps`；
- 不修改 Lift/replay prompt；
- 不改变 Gate1/Gate2 verdict；
- 不加入 dummy step、随机 shuffle 或多 writer；
- 不把 refs/J 映射放进 C；
- 等待用户审查后，才决定是否做 test-only 严格偏序 π 探针。

因此，当前推荐不是“删掉 π 拓扑”，而是：

> **让语义拓扑只住在 J/B，让 π 只保留完整机制选择与不可由 J/B 推出的严格实现
> 偏序；再用确定性 composer 消除独立步骤的任意线性顺序通道。**
