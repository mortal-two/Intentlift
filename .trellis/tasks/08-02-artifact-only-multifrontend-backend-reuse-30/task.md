# Task 30：Artifact-only 多前端后端复用实验

> 状态：`COMPLETE`（2026-08-02）
> 后端同步点：`6d764cff`（合并 `309e99fb` search-J backend）
> 共享 floor consumer：`64317b5e`
> 上游证据：Task28 31/31 same-origin pairs、62/62 exact accepted frontend artifacts
> 目标：验证同一个 production backend 能否只消费 J+C，而不按 TileLang/CUDA 增加后端分支

## 1. 论文问题

本 Task 只回答一个有限问题：

> 两个前端已经各自闭合后，它们的 exact accepted J 是否能被同一套既有后端真实 Decode、Emit、
> Contract-admit 并生成至少一个目标代码状态？新增前端是否需要 backend LoC 或 per-kernel recipe？

它不重复 Lift/Replay，不把 frontend Gate closure 冒充 backend lowering，也不运行性能搜索。

正式分母是 Task28 的全部 31 对 / 62 份 accepted artifacts。Task29 的 Primary-20 只是历史上的详细分析
选择，本 Task 不使用它作为分母，也不丢弃任何 unsupported/failure。

## 2. 隔离协议

实验拆成两个进程边界：

```text
Task28 accepted evidence
  -> export/provenance process
       -> backend_inputs_current.json
          only {anonymous run_id, ProductionSnapshot(J,C), extents, constexprs}
       -> provenance_current.json
          {stem, frontend, case, Source/Gate provenance}; never passed to backend

backend_consume.py
  -> strict snapshot digest decode
  -> target-neutral materialize_j / Emit Intent MLIR
  -> Attach_C + Erase_C invariant
  -> CUDA/RVV ContractView
  -> real CUDA cubin / RVV object
```

Backend 输入 schema 是 exact whitelist；未知字段立即拒绝。`ProductionSnapshot` 自身也只允许
`schema_version,j,c,j_digest,c_digest,snapshot_digest`。后端不可见：

- Source、pi、Replay；
- frontend、case_id、kernel catalog name；
- Gate1/Gate2 evidence；
- 历史 winner、benchmark diff 或 target recipe。

这里禁止的是显式 frontend/case identity；J 中数学与 ABI 所必需的 tensor/pointer parameter kind 仍被
诚实保留，因此可以与 frontend 统计相关。实验不要求“猜不出 frontend”，只要求 backend 没有 J 之外的
identity/Source 旁路。`pi` 只存在于 accepted certification artifact；production export 后停止，不进入 lower。

## 3. Stage 定义

| Stage | 本 Task 的精确定义 |
|---|---|
| Decode | `ProductionSnapshot.from_dict` 严格解析并复核 J/C/snapshot 三个 digest |
| Emit | `materialize_j` 从 J 与 concrete logical extents 生成非空 target-neutral Intent MLIR |
| Legal | `attach_c` 后满足 Erase_C，且对应 target ContractView 完整消费 Layer-B |
| CUDA Lower | generic materialized path 经 Intent→Linalg→NVVM/PTX→ptxas，产生至少一个 ELF cubin |
| RVV Lower | C-Fixed canonical candidate 经 Intent→Linalg→LLVM→RVV opt/llc，且 requested schedule 被真实实现并产生 object |

Lower 是真实目标编译，不是字符串 emission 或 mock；但本 Task 没有执行这些 binary，也没有 Q/D、
correctness recheck 或 timing。因此 Lower PASS 不能单独写成 runtime correctness/performance。行为闭合仍引用
Task28 的 frontend-specific Gate1 与 three-leg Gate2。

## 4. 估值规则

accepted artifact 的 J 使用 canonical `x0..x3`，Harness adapter 保存 finite source dimensions。导出端只做
三条确定性估值：

1. 静态 J：空 extent map；
2. 单动态维：唯一 source dimension → `x0`；
3. 矩阵/MLP：`x0=M, x1=N, x2=K, x3=H`。

该映射只生成 J materialization 所需的公开 logical valuation；它不选择 target schedule，不进入后端代码，
也不携带 frontend/case identity。

## 5. 真实运行与共享 feature delta

第一次全量运行：

- Decode 62/62；
- Emit 62/62；
- CUDA Legal 62/62；
- RVV Legal/Lower 62/62；
- CUDA Lower 60/62。

两个 CUDA Lower failure 恰好是 `floor2d` 的 TileLang/CUDA 两份 artifact，诊断完全相同：
`ptxas fatal: Unresolved extern function '__nv_floorf'`。这不是 frontend-specific failure，而是 CUDA backend
尚未消费共享 J feature `math.floor`：target-neutral emitter 已经正确生成 `math.floor`，旧 NVIDIA math pass
未把它改写为 native intrinsic。

修复只修改 `lib/Target/CUDA/LowerGpuMathToNVVM.cpp`：

- `math.floor f32` → exact non-FTZ `llvm.nvvm.floor.f`；
- `math.floor f64` → exact non-FTZ `llvm.nvvm.floor.d`；
- f16/bf16 复用已有 generic promote-to-f32 机制；
- +31 / -0 LoC，1 个 shared backend file；
- frontend-specific rules = 0；per-kernel rules = 0。

不采用 trunc/correction 算术近似，避免改变 NaN、±Inf、signed zero 与 subnormal 语义。重编译
`intent-opt` 后，同一批 62 个匿名输入重新全跑，不修改 J/C/valuation 或分母。

## 6. 最终机器结果

| Frontend | N | Decode | Emit | CUDA Legal | CUDA cubin | RVV Legal | RVV object |
|---|---:|---:|---:|---:|---:|---:|---:|
| TileLang | 31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 |
| generated CUDA | 31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 |

Pair-level “两边都通过”也是每个 stage 31/31。最终 failure taxonomy 为空。

工程归因：

| 归因 | LoC / rule |
|---|---:|
| 新 frontend 导致的 backend delta | 0 LoC / 0 rules |
| per-kernel backend recipe | 0 rules |
| 新共享 J feature 的 CUDA consumer | +31 LoC / 1 generic rule |
| 实验 runner / evidence | 不计 production backend |

这给出比“全部 PASS”更强、也更诚实的机制证据：

1. TileLang/CUDA ABI 与 J 差异不要求后端按 frontend 分叉；
2. 新前端在既有 J feature 上是 zero-delta reuse；
3. feature coverage 缺口按 J op 一次修复，两个前端同时获益；
4. 系统没有用 case template 把失败藏起来。

## 7. 复现命令

```bash
PYTHONPATH=python python \
  .trellis/tasks/08-02-artifact-only-multifrontend-backend-reuse-30/run_experiment.py \
  --phase export

PYTHONPATH=python python \
  .trellis/tasks/08-02-artifact-only-multifrontend-backend-reuse-30/run_experiment.py \
  --phase backend --workers 4

PYTHONPATH=python python \
  .trellis/tasks/08-02-artifact-only-multifrontend-backend-reuse-30/run_experiment.py \
  --phase summarize
```

## 8. Machine artifacts

- `backend_inputs_current.json`：backend 实际读取的匿名 J+C 输入；
- `provenance_current.json`：后处理映射，明确标记不进入 backend；
- `backend_results_current.json`：逐 artifact 的 MLIR/cubin/RVV object digest、stage 与 compiler receipt；
- `results_current.json`：论文汇总与 claim boundary；
- `results_current.md`：人类可读全 31-pair 表。

## 9. 完成条件

- [x] search-J backend 与当前 v13/J-only frontend 协议合并；
- [x] 31 对 / 62 artifact 全量、不筛选导出；
- [x] backend input 不含 Source/pi/Replay/frontend/case/Gate/history；
- [x] Decode/Emit/CUDA Legal/RVV Legal 全量运行；
- [x] CUDA 与 RVV 均产生真实 binary/object；
- [x] 共享 feature failure 按 J op 修复，不加 frontend/kernel 特判；
- [x] 报告 zero frontend-specific backend LoC 与 +31 shared feature LoC；
- [x] 不声称本 Task 已做 runtime correctness 或性能测量。
