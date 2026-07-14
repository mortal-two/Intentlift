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

当前 14 张 Triton 规则卡分为：

1. domain/index：program domain、index tile；
2. memory：masked access、unmasked access、output mapping；
3. A 算子：elementwise、reduce、broadcast、cast、select；
4. contraction/loop：tiled dot、loop carry；
5. 接口/C：runtime scalar、schedule slot。

规则卡只提供“如何在 Triton 中实现某种机制”，具体算什么必须从 J 读取。
π 只保存 rule id 与 J/B 引用。

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

- 先让首个 live add 的完整 J/π/C artifact 在有界修复内闭合；
- 覆盖五核同类和现有 corpus 中可执行 source；
- 全部重新走 LLM J/π/C；
- 跑小修、重提、规则检索、Gate2；
- 建立第一版真实失败集。

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
