# Task 28：J-only 多前端七组各十个闭合实验

> 状态：`P0_REFROZEN`（2026-08-01）
>
> 上游正式证据：Task27 的 `add2d` / `gather2d` 两个 pair；本 Task 在统一批量协议下重新计量。

## 0. 用户要求与数据集硬边界

用户要求的是 **7 个 cohort、每个 cohort 10 个 kernel pair**，不是“新增 7 个 pair 后让每个前端达到
10 个”。

current 数据集机械枚举得到：

- TileLang factory：31 个；
- TileLang-export CUDA snapshot：31 个；
- 依据 CUDA descriptor 的 `origin_case_id`，恰有 31 个 same-origin pair；
- 另有 4 个 handwritten CUDA case，但没有 TileLang same-origin Source，不能伪造配对。

因此数据集不可能提供 70 个互不重复 pair。本实验冻结为：

- 7 个预注册语义/实现 cohort，每组恰好 10 个 pair membership；
- cohort 间允许同一 pair 重叠，组内不得重复；
- 七组合计 70 个 membership，覆盖全部 31 个 unique pair；
- 每个 unique pair 的 TileLang/CUDA 各运行一次独立正式闭合，共 62 个 unique frontend runs；
- cohort 表只引用这些 exact run；不得把重用后的 70 个 membership 宣称成 70 个 unique kernel。

论文必须同时报告 `70 memberships / 31 unique pairs / 62 unique frontend runs` 三种口径。

## 1. 冻结七组 × 十个

### G1：pointwise / selection（10）

1. `add2d`
2. `relu2d`
3. `floor2d`
4. `exp2d`
5. `where2d`
6. `add_bias2d`
7. `clamp2d`
8. `rowmask_where2d`
9. `any_kernel_dim`
10. `copy2d_divmod`

### G2：reduction / statistics / normalization（10）

1. `row_sum`
2. `row_max`
3. `grouped_row_sum2d`
4. `softmax_inner`
5. `masked_softmax2d`
6. `groupnorm`
7. `layernorm`
8. `rms_norm2d`
9. `layer_norm_residual2d`
10. `rms_norm_residual2d`

### G3：contraction / epilogue（10）

1. `gemm`
2. `matmul_relu2d`
3. `matmul_bias_relu2d`
4. `matmul_fused_epilogue2d`
5. `mlp2d`
6. `add_bias2d`
7. `relu2d`
8. `transpose2d`
9. `copy2d_divmod`
10. `where2d`

### G4：attention / masking（10）

1. `_attn_fwd`
2. `flash_attention2d`
3. `masked_attention2d`
4. `masked_softmax2d`
5. `softmax_inner`
6. `row_sum`
7. `row_max`
8. `transpose2d`
9. `rowmask_where2d`
10. `any_kernel_dim`

### G5：indexing / layout / image（10）

1. `gather2d`
2. `transpose2d`
3. `copy2d_divmod`
4. `upsample_bicubic2d_aa`
5. `any_kernel_dim`
6. `rowmask_where2d`
7. `where2d`
8. `add2d`
9. `add_bias2d`
10. `clamp2d`

### G6：fusion / composition（10）

1. `layer_norm_residual2d`
2. `rms_norm_residual2d`
3. `matmul_fused_epilogue2d`
4. `matmul_bias_relu2d`
5. `matmul_relu2d`
6. `mlp2d`
7. `groupnorm`
8. `masked_attention2d`
9. `_attn_fwd`
10. `flash_attention2d`

### G7：cross-feature stress（10）

1. `floor2d`
2. `exp2d`
3. `gather2d`
4. `grouped_row_sum2d`
5. `layernorm`
6. `rms_norm2d`
7. `gemm`
8. `mlp2d`
9. `masked_attention2d`
10. `upsample_bicubic2d_aa`

每个 stem 对应 `tilelang:<stem>` 与 `cuda:tilelang_<stem>`。七组并集必须机械等于数据集的 31 个
same-origin stem；任何成员变化必须先修改并提交本冻结文档，不能按 Gate 结果挑 case。

## 2. 核心不变量

1. 主代理读取 current Source、descriptor 与 compiler observation，直接手工作者化 `(J,C,pi)`；
2. `lift_agent_calls=0`：不得调用 Lift agent，也不得把历史 Lift 输出冒充手工作者化；
3. Replay payload 根恰为 `j,c,u_f`；`pi`、Source、valuation、Gate diff、prior/sibling Replay 不可见；
4. 每个 frontend run 使用 fresh source-blind Codex episode；不得跨 case 复用 conversation/candidate；
5. TileLang hard observer 是 TIR semantic structure，PTX 仅 diagnostic；CUDA hard observer 是 semantic PTX，raw PTX 仅 diagnostic；
6. J 记录数学/边界/effect；C 记录公开 ABI/config；紧凑 `pi` 只供认证与审计，不能给 Replay；
7. 失败只允许沉淀 kernel-independent J schema、consumer、U_f、observer 机制；禁止 kernel id/path/source matcher；
8. Gate1 闭合后才运行三边 Gate2；pair 两边独立，不借用另一 frontend 的 PASS；
9. 不恢复 `python/tests`，不以 mock/smoke/compile-only 代替真实 TypeCheck/Gate1/Gate2；
10. 所有数字只认 current exact run 机器证据。

## 3. 执行顺序

- [ ] 生成并校验七组矩阵：每组10、组内唯一、并集31；
- [ ] 为31个unique stem建立Source/ABI/J-feature清单；
- [ ] 手工构造62份frontend-specific artifact并全部 Decode/TypeCheck；
- [ ] 建立artifact-only、no-Lift批量runner；run目录必须原子分配；
- [ ] provider episode有界并行且每case fresh；GPU/Gate2单独限流；
- [ ] 先闭合G1十个pair，验证批量机制与简单语义覆盖；
- [ ] 按G2至G7推进尚未闭合的unique pair，不重复运行已闭合pair；
- [ ] 每个失败保存exact artifact/candidate/compiler observation/hard category；
- [ ] 生成31-pair唯一结果表、7×10 cohort表、前端delta表与论文claim boundary。

## 4. 每个正式 run 的最小记录

- artifact/J/C/pi canonical bytes与digest；
- `lift_agent_calls=0`、Replay payload keys与`pi_visible_to_replay=false`；
- TypeCheck结果；Replay model、provider calls、wall time、candidate bytes；
- Gate1 profile、hard/diagnostic结果与exact compiler observation；
- 三边Gate2、valuation数与最大误差；
- Source/descriptor revision、frontend ABI与C；
- 对应的cohort memberships。

每个 pair 额外记录共同数学关系、full-J真实差异、generic production LoC 与 kernel-specific production LoC。

## 5. 完成条件

- 七个cohort各恰好10个membership，且并集覆盖31个unique same-origin pair；
- 31个pair各有TileLang/CUDA独立current TypeCheck/Gate1/Gate2证据，失败也保留真实证据；
- 每个新run均`lift_agent_calls=0`且`pi_visible_to_replay=false`；
- 汇总不把70 membership、31 unique pair、62 frontend runs混为同一分母；
- 论文claim只覆盖measured cohort，不外推任意frontend/kernel；
- 提交后工作区clean。
