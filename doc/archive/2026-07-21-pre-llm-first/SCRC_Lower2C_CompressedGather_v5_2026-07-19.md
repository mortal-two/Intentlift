# SCRC Lower2C compressed along-axis gather v5

日期：2026-07-19

## 1. 本轮目标

补齐 Gate2 中既有 `gather(mode="along_axis")` 的一种合法 IntentIR 形态：数据为
rank-R，而 index 与结果省略被选 axis，均为 rank-(R-1)。典型例子是二维矩阵按每一行
给定列下标抽取，J 直接表达 `[N, M] + [N] -> [N]`，不需要在 J 中制造无语义的单位轴。

本轮不修改 J schema、π、C、Gate1 或 replay；只在 JSON→Intent MLIR 边界做确定性
legalization。

## 2. 确定性 lowering

对于 compressed along-axis gather：

1. 在 index 的 gather axis 插入 size-1 维；
2. 复用现有等 rank `intent.gather`；
3. 将结果中的该 size-1 维确定性折叠，恢复 J 声明的 rank-(R-1) 结果。

完整等 rank along-axis gather 和 `take` 模式保持原语义。以下情况继续 fail-closed：

- index rank 既不是 data rank，也不是 data rank 减一；
- gather axis 越界；
- 当前尚无可表达 reassociation 的 rank-0 compressed index/result。

活动 Lower2C 协议为
`intent_linalg_cpu_symbolic_iota_integer_readback_boundary_broadcast_compressed_gather_v5`，
held-out identity 升为 schema 34。

## 3. 已完成验证

定向测试覆盖：

- axis 0 compressed gather；
- axis 1 compressed gather；
- rank-0 compressed gather fail-closed；
- embedding 使用的 `take` gather 不回归；
- Boundary broadcast 与 Gate1/Gate2/harness 联合回归。

联合核心回归结果为 `235 passed`。

## 4. 真实 `fg_diag_extract` 的当前证据边界

schema34 的全新 Lift 没有生成 compressed gather，而是生成了：

`[N] i32 column_index × [1] i32 unit_axis -> [N,1] i32 contraction`

再执行完整等 rank gather。当前 Intent contraction lowering 明确只支持浮点 contraction，
因此该 artifact 在 Gate2 中为 9/9 `execution_failed`；Gate1 也没有闭合，Relift 随后失败。
这次结果没有反证 compressed-gather legalization，但也不能被记作真实 family 闭环成功。

后续应优先让 Gate1 首轮差异后的 Relift 指导 J 使用已存在的 compressed gather 语义，
或使用规则库中已经受支持的确定性整数索引构造。不能为了迁就一次错误 Lift，擅自扩大
主线去增加整数 contraction。
