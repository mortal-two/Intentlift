# 2026-08-01 归档：pre-SCRC Python 代码与旧覆盖率语料

本目录保存 2026-07-20 前后那一轮"覆盖率扩张 + 性能对比"战役的代码与数据。它们**不在**当前
SCRC 闭合路径上：当前活动路径见 `.trellis/tasks/08-01-final-three-kernel-closure/task.md` 与
`doc/method/IntentIR_J-only_Replay与JC结构Gate1_2026-07-31.md`。

归档理由已按模块逐条核实（AST 依赖闭包 + 运行时导入验证），不是凭印象。

## 内容

| 路径 | 数量 | 归档理由 |
|---|---:|---|
| `python/intentir/benchmarks/` | 6 | 旧 vs-Triton/flag_gems 性能台架；其 `__init__.py` 自述 "not on the compile path" |
| `python/intentir/extraction/data/` | 7 | 231-module 覆盖率战役的 corpus/universe runner 与 coverage 生成 |
| `python/intentir/extraction/lifters/cuda_frontend.py` | 1 | 早期 CUDA lift 实验，自述 "retained as historical material" |
| `python/intentir/extraction/lifters/extract.py` | 1 | 仅由已归档模块调用；曾经只经 `extraction/__init__` re-export 而显得"活着" |
| `python/intentir/verify/back/` | 6 | `legacy_v0` NumPy 解释器；`verify/__init__.py` 早已声明不 re-export |
| `python/intentir/frontend/contract_expr_mlir.py` | 1 | 零引用，已被 `ir/canonical_j.parse_contract_expr` 取代 |
| `python/intentir/backend/cuda/` 除 `capabilities.py` | 17 | 旧 CUDA 执行栈 |
| `python/intentir/ir/layer_c.py` | 1 | 唯一引用者 `backend/cuda/schedules.py` 同批归档 |
| `corpus/` | 235 | 旧战役的 kernel/pilot/bulk 语料与报告 |

## 两处必须知道的边界

**1. `backend/cuda/` 只留 `capabilities.py`。**
它是活的：`verify/workload_manifest.py:18` 依赖它加载 `spec/lowering/high-performance-kernels.json`
里的 28-kernel registry。���档同批模块时，`backend/cuda/__init__.py` 已改为不再 re-export
`program.py`——否则 `capabilities` 的导入会连锁拉起一整套已归档的执行栈。

**真正的 production backend 不在本仓库**，而在
`TianchenIR-backend-certified-j-search` 分支（消费 accepted `J+C`，GPU/RVV 三目标）。

**2. `corpus/reports/PERF.md` 的数字不得进入论文。**
它停在 2026-06-26，其中 FlashAttention 三行是 **875×–983× 慢于 Triton** 的旧数据。
当前后端性能以后端分支的 `doc/已通过24个J_三目标后端总览_2026-07-31.md` 为准。

## 使用约束

历史材料只用于追溯与取证。不得作为当前 runtime、实验能力或论文数据来源；
不得从这里恢复模块以"补全"活动路径。
