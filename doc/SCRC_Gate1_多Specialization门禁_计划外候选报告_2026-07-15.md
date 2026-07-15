# SCRC Gate1 多 Specialization 门禁：计划外候选报告

日期：2026-07-15

## 1. 当前决定

本轮**没有**把 Gate1 改成多 specialization 门禁，也没有修改正式 Gate1 verdict。
当前边界仍是：

```text
Gate1：一个 reviewed source/replay compile profile，同一 C 下比较 TTIR
Gate2：可对已得到的 replay 执行严格 shape × value × seed suite
```

Gate2 matrix 属于 SCRC 主方案 Phase 3；Gate1 跨 profile 聚合没有在主方案中定义，
因此按用户要求先报告，等待审批后再决定是否实施。

## 2. 为什么会出现这个候选需求

当前 11 个 ready FlagGems Kernel 的 normal replay 都在一个具体 specialization 上
闭合。新 Gate2 suite 已证明同一 replay 可以进一步覆盖：

- `fg_ones`：1、BLOCK−1、BLOCK、BLOCK+1、非整除、远多 tile；
- `fg_min`：固定 `BLOCK_MID=64` 下 3 个 shape × 4 个 value profile × 3 seeds，
  共 36 个三方 case 全部通过。
- `fg_vector_l1`：固定 `BLOCK_SIZE=1024` 下
  `M={1,1023,1024}`、runtime `ord={1.0,1.5,2.0}`、2 个 value profile ×
  3 seeds，共 18 个三方 case 全部通过。
- `fg_embedding`：静态 `indices[17]`、`weight[97,13]` J 下，2 个合法索引/权重
  value profile × 3 seeds，共 6 个三方 case 全部零误差通过；尝试把 K 改为
  `{1,15,16,17}` 时，matrix builder 因静态 J extent 不同而 fail-fast。

这些证据增强行为可信度，但不能反推“所有 specialization 的 source/replay TTIR
都已结构闭合”。constexpr、runtime scalar specialization 和不同 grid 可能改变
TTIR；若论文要把结论从“该 specialization 闭合”提升为“该 Kernel 闭合”，需要
额外定义 Gate1 的 profile 语义。

`fg_embedding` 还给出更具体的边界：其 compiler view 没有 runtime K/V/D scalar，
Lift 因而正确生成静态 tensor extent。Gate2 value matrix 可以验证数据依赖地址与
词表上下界，却不能仅靠替换 host tensor shape 把静态 J 变成动态 J；这类 shape
扩展必须先有不同 compiler profile 或重新定义 artifact/profile 关系。

## 3. 不能直接实现的三个未决条件

### 3.1 profile 是否允许改变 C

- 固定 C、只改变 runtime shape/scalar：仍接近当前同一 C 控制实验；
- 改变 BLOCK/num_warps/num_stages：会进入多 C，source/replay 的结构和 masked-J
  信息量都可能变化。

主方案只规定一次比较中 source/replay 使用同一 C，没有规定一个 Gate1 verdict
是否要跨多个 C 聚合。

### 3.2 normal 闭合如何聚合

最保守的候选规则是：同一 replay source 在所有 profile 上都必须编译且 hard
closure，任一失败则整个 Gate1 不闭合。不能用平均 score 或多数票。但还需决定：

- 一个 profile 失败时，repair feedback 应只针对该 profile，还是重新生成整个
  replay；
- 不同 profile 的修复要求冲突时如何保留最优候选；
- Lift 的 J/π 是否允许因 profile 改变，还是必须共享一个 J/π。

### 3.3 masked-J 如何聚合

当前主方案定义单次 `L = masked_score / normal_score`，阈值为 `0.5`。多 profile
后至少有三种不同政策：

1. 每个 profile 都要求 `L <= 0.5`；
2. 取最大 L；
3. 报告分布但不参与逐 Kernel verdict。

它们会显著改变现有 `LEAKAGE_REJECTED` 的含义，不能由实现自行选择。

## 4. 若获批准，建议的最小实现范围

只建议先做“固定 C、多 runtime specialization”，不立即跨 C：

1. `Gate1SpecializationProfile`：runtime argument shape/value、grid、稳定 id；
2. Lift 和 normal replay generation 只执行一次；
3. 同一 replay source 对每个 profile 分别编译 source/replay 并提取 TTIR；
4. 保存逐 profile compile/comparison，不做平均；
5. normal verdict 采用 all-pass；
6. masked-J 聚合政策必须由审批明确后再编码；
7. batch identity 绑定完整 profile matrix；
8. 首个实验只用 `fg_ones` 的固定 `BLOCK_SIZE=1024` runtime-shape profiles。

该方案不需要新增 IntentIR op，也不恢复旧 verifier；它只扩展 Gate1 的实验控制
维度。但在第 3 节条件未获确认前，不应进入代码。

## 5. 本轮已完成但不越界的部分

- production Gate2 `Gate2Suite` 与严格 all-case 聚合；
- shape/value/seed metadata 和 batch identity；
- 首个真实 36-case live matrix；
- IEEE-special 输入触发并修复 Lower2C `maxnum/minnum` 语义；
- Gate1 仍使用一个 reviewed compile profile，masked-J 阈值和 verdict 未改。
