# Task 25：Stage 3 两个新增 J 的真实后端消费

> 状态：已完成
>
> 范围：在既有 22 项之上加入 `tilelang:gather2d` 与 `tilelang:rowmask_where2d`，形成 24 项后端集合

## 目标

把两个新增 accepted J 转化为真实 C++/MLIR 后端能力，而不止是收入语料：

- `gather2d`：完整消费 J 中连续的两级 typed gather。`C-Fixed` 使用确定性的逐级计算结构；`Schedule-Only` 只解冻其线程布局；`Full` 额外搜索由两级 index map 复合得到的 direct gather，消除未被最终结果使用的中间行物化。
- `rowmask_where2d`：让 CUDA/RVV elementwise lowering 正确支持 `[M]` 布尔行掩码向 `[M,N]` 的 typed broadcast；在 Full 中搜索由 J 与执行域检查共同约束的 uniform-row 分支和无效行数据读取消除。

所有 target code 必须由 MLIR lowering 产生；Python 仅负责候选编排、编译、正确性检查和测量。

## 实验口径

- GPU：5090D 与 H100；`Source` 为 lifted TileLang Source；指标固定为 `C-Fixed`、`Schedule-Only`、`Full`、`Source`、`G_schedule`、`G_full`。
- RVV：单线程；指标固定为 `C-Fixed`、`Schedule-Only`、`Full`、`Native`、`G_schedule`、`G_full`。
- 使用原 J 的 dtype 与自然/model workload；符号维可以取自然 valuation，J literal 不擅自改写。
- code-warm/data-cold，GPU 使用 CUDA event；三柱共享输入、正确性策略和可比搜索预算。

## 验收

- 两个 J 都能通过 production materialization、CUDA/RVV lowering、编译和正确性检查；
- gather 的 canonical 与 composed 路径在擦除 schedule 后仍有不同的 index/materialization topology，且 composed 路径只进入 Full；
- rowmask 的 rank-1 行广播来自 tensor shape/broadcast relation，不使用 kernel-name allowlist；
- 重新生成 5090D/H100/RVV 的完整 24 项表，并同时报告全表与结构适用子集；
- 新增路径对适用项应产生可解释的 `G_full`，若未达到 `1.1x`，继续定位真实瓶颈并扩充结构空间，不削弱 C-Fixed 制造结果；
- 完成相关 MLIR/Python tests、提交代码并保持工作区干净。

## 完成结果

- 两项 accepted J 已进入 24 项 production catalog，并在 5090D、H100 与 RVV 单线程完成真实编译、Q/D、搜索与测量；
- Gather 的逐级 materialized 与 composed-index 两条 CUDA MLIR 路径均已落地，Full 在两卡选择 composed path；
- RowMask 的 typed rank-1 broadcast 已被 CUDA 与通用 bufferized MLIR 正确消费；GPU Full 选择 zero-fill，RVV 则真实评估后保留 materialized winner；
- 5090D RowMask `G_schedule=1.554×、G_full=1.228×、Source/Full=1.648×`；H100 为 `2.457×、1.098×、2.328×`；
- 完整结果见 `doc/results/stage3-new2-production-backend-24/README.md`。
