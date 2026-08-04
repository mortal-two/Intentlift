# 后端消费算法结构而非算子名：名称干预实验

日期：2026-08-04

## 结论

这组实验直接补上以下证据缺口：

> 当前 target consumer 的准入与搜索由 J 的 Intent 图、typed contract 和 target capability 决定，而不是由 Source/算子函数名决定。

正式 24-J 集合上的结果为：

| 检查 | 结果 |
|---|---:|
| CUDA：换成另一结构的误导性算子名后，canonical lowering path 不变 | **24/24** |
| RVV：换成误导性算子名后，完整 search plan 不变 | **24/24** |
| CUDA：6 类代表图的 Full admitted candidate space 完全不变 | **6/6** |
| 同一函数名下，6 类图被正确结构 pass 准入 | **6/6** |
| 同一函数名下，错误结构 pass 被拒绝 | **30/30** |

这不是源码 grep，也不是只比较候选标签。CUDA 的 path 与 Full-space 检查实际运行了注册的 C++ `intent-opt` whole-graph probes；只有完整消费图、无残留 `intent.*` 并产生合法 launch graph 的 path 才算 admitted。

完整机器可读结果见 [report.json](report.json)。复现实验：

```bash
PYTHONPATH=python python3 tools/verify_algorithm_name_independence.py \
  --out doc/results/algorithm-not-operator-name-20260804/report.json
```

## 1. 干预变量

对每个 formal J，只修改 materialized MLIR 的 entry function symbol：

```text
原图、J、B、C、shape、dtype、target 均保持不变
              ↓
把函数名替换成来自另一 lowering 结构的真实误导名
```

示例：

- elementwise 图使用 `liger_rms_norm`；
- row-reduction 图使用 `swiglu`；
- padded-row 图使用 `flash_linear`；
- contraction 图使用 `splitk_reduce_varargs`；
- Split-K 图使用 `flash_attention_forward`；
- attention 图使用 `liger_rms_norm`。

如果 consumer 按算子名发射，误导名应改变 path 或搜索空间。如果 consumer 按算法/图结构匹配，结果应保持不变。

## 2. 正干预：24 个 J 的路径与 RVV 搜索计划

CUDA 对原名和误导名分别运行 production `schedule_only_cuda_path_identity`。24 项的最终 path 均保持一致：

```text
13 个 row graph          -> cuda.row_reduction.normalization_loss.read_once.v1
 7 个 elementwise graph -> cuda.elementwise.row.vectorized.v1
 1 个 padded-row graph  -> cuda.padded_row_reduction.masked_short_dot.v1
 1 个 contraction graph -> cuda.contraction.tiled.v1
 1 个 Split-K graph     -> cuda.splitk.reduce-varargs.v1
 1 个 attention graph   -> cuda.flash_forward.structured_scan.direct.v1
```

结果：

```text
CUDA path invariance = 24 / 24
```

RVV 对原名和误导名分别构造完整 `RiscvSearchPlan`，逐字段比较：

- candidate universe；
- Schedule-Only 的 Erase-C/C 顺序；
- Full 的 Erase-C/C 顺序；
- C-Fixed candidate；
- C score 与 target projection。

结果：

```text
RVV exact search-plan invariance = 24 / 24
```

## 3. Full 搜索空间，而不只是默认路径

从 6 类结构各选一个代表 J，对原名和误导名分别：

1. 枚举完整 Full raw candidates；
2. 对每个注册 path 运行 C++ whole-graph probe；
3. 比较 admitted path、candidate ID、pass argument、schedule、trace、B requirements 与顺序。

| 结构 | 代表 J | raw candidates | admitted candidates | admitted paths | 名称干预后 |
|---|---|---:|---:|---:|---:|
| elementwise | SwiGLU | 1,763 | 292 | 3 | 完全相同 |
| row | RMSNorm | 2,537 | 894 | 3 | 完全相同 |
| padded-row | Flash backward preprocess | 1,725 | 192 | 3 | 完全相同 |
| contraction | Flash Linear | 1,629 | 147 | 4 | 完全相同 |
| Split-K | SplitK reduce | 1,947 | 64 | 3 | 完全相同 |
| attention | FlashAttention forward | 1,631 | 256 | 8 | 完全相同 |
| **合计** | **6 类** | **11,232** | **1,845** | **24** | **6/6 完全相同** |

因此不能用“只有 canonical path 恰好没变”解释结果。对当前六类结构，误导性名字没有改变任何一个 admitted Full candidate。

## 4. 负对照：同名不等于同 lowering

只做名称不变性还不够，因为“所有 pass 都接受所有图”也会产生表面上的不变。为此增加 6×6 交叉矩阵：

- 六个代表图全部改成同一个函数名 `same_operator_name_for_all_graphs`；
- 每个图分别尝试六类 C++ whole-graph pass；
- 对角线是结构匹配，非对角线是结构不匹配。

结果：

```text
正确结构： 6 / 6 admitted
错误结构：30 / 30 rejected
```

矩阵等价于：

| graph \ pass | Elem | Row | Padded | Contract | Split-K | Attention |
|---|---:|---:|---:|---:|---:|---:|
| Elementwise | ✓ | × | × | × | × | × |
| Row | × | ✓ | × | × | × | × |
| Padded-row | × | × | ✓ | × | × | × |
| Contraction | × | × | × | ✓ | × | × |
| Split-K | × | × | × | × | ✓ | × |
| Attention | × | × | × | × | × | ✓ |

这证明了两个方向：

1. **名称不是充分条件**：相同名字无法让错误 pass 准入；
2. **名称不是必要条件**：换成错误名字仍然由正确结构 pass 准入。

## 5. 论文中应怎样解释

建议实验章把该实验命名为：

```text
Operator-Name Intervention / Structure-Selection Test
```

它回答的不是性能问题，而是 backend attribution 问题。可以使用下面的表述：

> We intervene on the only operator-name-bearing compiler input by replacing every entry symbol with a name borrowed from a different algorithmic structure, while holding J, C, shape, dtype, target, and the Intent body fixed. CUDA preserves the selected lowering path for all 24 kernels, RVV preserves the exact search plan for all 24 kernels, and all six representative CUDA Full spaces remain identical (11,232 enumerated and 1,845 admitted candidates in total). Conversely, under one shared function name, the six structurally matched passes are admitted while all 30 mismatched graph–pass pairs are rejected. Thus names are neither necessary nor sufficient for target-path admission.

中文：

> 我们只干预编译输入中的算子名称：将每个入口符号替换为来自另一算法结构的名称，同时固定 J、C、shape、dtype、target 和 Intent body。CUDA 的 24 个 lowering path 全部不变，RVV 的 24 个完整搜索计划全部不变，六类代表图的 Full 空间也完全一致，共比较 11,232 个枚举候选与 1,845 个准入候选。反向负对照中，六个结构匹配 pass 全部准入，而 30 个结构不匹配组合全部拒绝。因此算子名既不是 target-path 准入的必要条件，也不是充分条件。

## 6. 证据边界

本实验可以证明：

- entry symbol/算子名不控制当前 CUDA/RVV path admission；
- path 与搜索空间由 materialized J 的算法图、typed contract 和 target facts 决定；
- 当前 24 个 J 不是按同名 Kernel recipe 分流。

本实验不能单独证明：

- backend 没有结构特定的手写 C++ implementation；
- 六类结构覆盖所有未来算法；
- 任意新的 Intent operation 都无需新增 lowering；
- typed semantic markers 不重要——它们正是 J 的算法/合同信息，应当参与准入。

因此它需要与 consumer LoC/24→6 映射共同使用：名称干预证明“不是按名字”，LoC inventory 诚实量化“按结构实现”的工程成本。

## 7. 重复次数与置信区间

这是确定性的编译器干预实验，不是噪声性能测量：每个单元是 exact equality 或 exact admit/reject，因此不使用 latency 重复次数或置信区间。报告的是完整 formal cohort 的 `24/24`、六类完整搜索空间的 `6/6`，以及穷举 6×6 矩阵的 `6/6 + 30/30`，不是从随机 Kernel 样本外推总体比例。
