# 前端输入 Kernel 清单

本项目要提取的前端输入。**旧 intentir 的输入草率随意,本项目重新规定。** 这些 kernel 数据集放在仓库外的 **`/home/kingdom/phdworks/Dataset`**(不进 git)。

## Triton

| 来源 | 量级 | 备注 |
|---|---|---|
| 手写 Triton kernel | ~30 | |
| AI-bench kernel | 8 | 论文里 Triton-CPU 在 RISC-V 上跑的那批 |
| FlagGems | 100+ | 高性能库;入口需自己下载到 `Dataset/`(旧搬迁缺这个入口)。TianchenForge 用过同一批 |
| xformers / flash-attention / Liger-Kernel | 20+ | 真实高性能库 kernel |

## TileLang

| 来源 | 量级 | 备注 |
|---|---|---|
| 手写 TileLang kernel | ~30 | |

## CUDA

| 来源 | 量级 | 备注 |
|---|---|---|
| TileLang 生成 | ~30 | TileLang 可生成 CUDA |
| 高性能库 CUDA kernel | 待定 | **目前还没人利用过**,本项目想吃下这批 |

## 与能力的关系(关键)

这批输入(flash-attention / Liger / xformers / FlagGems / 高性能 CUDA)**几乎全是 atomic 密集 + 库级复杂**的 kernel——正是旧系统系统性躲开的(atomic 一律 OOS、宏只有一个手写)。本项目的能力设计必须正面覆盖它们:

- 带状态结构化循环(FlashAttention 等递推)→ 见 [../ir/structured-loop.md](../ir/structured-loop.md)
- atomic / 并发归约一等公民 → 见 [../ir/atomic-ops.md](../ir/atomic-ops.md)
- gather/scatter 等数据依赖寻址 → 见 [../ir/atomic-ops.md](../ir/atomic-ops.md)

完整能力 / 边界见 [../capability/capability-matrix.md](../capability/capability-matrix.md)。

## 待办(数据集获取)

- FlagGems 入口尚未就位:需下载 FlagGems 到 `Dataset/`,作为 Triton 高性能库输入来源。
