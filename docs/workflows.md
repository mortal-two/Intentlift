# 工作流

## 1. 新 Kernel 的 SCRC 闭包

1. 在静态清单或多前端 Dataset 中登记真实 Source 与 Harness；Harness 只描述实际 ABI、输入、grid 和初始 `C`。
2. Lift 从 Source 生成完整 `(J,C,π)`，TypeCheck 对 artifact 失败关闭。
3. 为每个正式轮次创建全新的、Source-blind Replay；Replay 只消费 `J`、`MaskC(C)` 和固定工具环境。
4. Gate1 独立编译 Source 与 Replay，observer 分别取证，再用冻结 `π` 规范化和比较。
5. 只有 Gate1 闭合的候选才能进入隐藏 Gate2。
6. Gate2 在同一估值上执行 Source、精确 Replay 和 `RefJ`，保存 Source–Replay、Source–J、Replay–J 三条比较。
7. 三边全部通过才记录 `scrc_certified`；其他状态保留为失败证据，不得以修改 Gate、手写 per-kernel 规则或只通过 TypeCheck 替代。

批量入口支持清单验证、dry-run 和真实运行：

```bash
PYTHONPATH=python python tools/scrc_batch.py --validate-manifest --json

PYTHONPATH=python python tools/scrc_batch.py --dry-run \
  --case '<registered-case-id>' \
  --run-root '/path/to/evidence/scrc-batch'

PYTHONPATH=python python tools/scrc_batch.py --run \
  --case '<registered-case-id>' \
  --run-root '/path/to/evidence/scrc-batch'
```

## 2. 精确 artifact 的零模型能力复核

`scrc_gate2_recheck.py` 复用 parent 中已经接受的 artifact 与精确 Replay，不调用新的 Lift/Replay 模型。它重新运行当前 Gate1 编译，然后执行完整 Gate2：

```bash
PYTHONPATH=python python tools/scrc_gate2_recheck.py \
  --case '<registered-case-id>' \
  --artifact '/path/to/parent/attempt-N/artifact.json' \
  --parent-evidence '/path/to/parent/attempt-N' \
  --dataset-root '/path/to/Dataset' \
  --run-root '/path/to/evidence/recheck' \
  --json
```

Parent 必须包含 `gate1.json`、`replay.py` 与 `replay_receipt.json`；其本身或祖先运行目录还必须包含精确 `request.json`、`source_projection.json` 和 `source.py`。脚本在 lineage 不一致时失败关闭。

`scrc_gate1_recheck.py` 会创建新的 Replay，因此会调用模型。它是诊断工具，不进入 Gate2，也不改变 parent verdict；不能把它当作零模型复核或闭包进展。

## 3. 多前端工作流

统一入口 `tools/scrc_frontend.py` 根据登记项选择 Triton、TileLang 或 CUDA 路径。前端负责把真实 Source 投影到各自 compiler/observer；`J` 的所有权、Replay 可见性、Gate 顺序和 Gate2 三边规则不随前端变化。

```bash
PYTHONPATH=python python tools/scrc_frontend.py --list --json
```

列表输出是当前可运行能力的权威视图。Dataset 缺失或本机前端依赖不可用时，入口应明确报告 unavailable，而不是以模拟 Source 代替。

## 4. 生产后端工作流

生产后端严格按以下顺序工作：

```text
accepted J
  -> deterministic Intent MLIR materialization
  -> attach accepted/experimental C
  -> target lowering
  -> structural candidate admission
  -> compile and execute
  -> correctness audit
  -> target timing/search report
```

CUDA 和 RVV 共用规范化 `J` 与 `C` 边界，但各自拥有目标 lowering、候选空间和执行器。算法名称独立性检查用于证明候选选择由结构驱动：

```bash
PYTHONPATH=python python tools/verify_algorithm_name_independence.py \
  --out '/path/to/evidence/algorithm-name-independence.json'
```

性能报告必须保留 workload profile、目标信息、候选空间、正确性结果和测量设置。只有真实目标执行结果可以支撑性能结论。

## 5. 证据与文档

- 公共 `docs/` 只描述当前系统合同、可复现入口和唯一最终任务。
- `docs/final-task.md` 是唯一活动执行记录；旧状态不追加成日志，而是被当前事实替换。
- `local/internal/scrc/`、`local/internal/frontend-extension/`、`local/internal/backend/` 保存不提交 Git 的内部理解和重要真实结果。
- `local/archive/` 保存合并前快照、旧文档、Trellis 与历史测试；这些内容不得重新成为活动流程。
- 大型 evidence 建议保存在仓库外，并在最终任务中记录绝对路径、case、Gate 状态和 parent lineage。
