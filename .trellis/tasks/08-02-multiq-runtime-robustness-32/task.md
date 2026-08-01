# Task 32：预注册 Multi-Q J-only Runtime 鲁棒性

> 状态：`IN_PROGRESS`（2026-08-02）
> 上游：Task31 31 pairs / 62 artifacts / exact accepted Q runtime 62/62
> 冻结 Q schedule：`6e06e3d7f4ae19da3174e0c0aecf89c5105bc79818e3fc49e5e0d083a269c8f3`

## 1. 有限问题

Task31 每份 exact J 只在一个 accepted Gate2 Q 上执行。本 Task 只问：

> 固定同一份 accepted J+C、logical extents 与 public comparison policy，加入四个在任何 runtime
> 结果产生前预注册的 deterministic Q 后，同一个通用 CUDA backend/runtime 能否 compile once，完成
> 62 × 5 = 310 次 CUDA-target--RefJ 执行比较？

这验证 fixed-shape value robustness，不验证新 shape、Source/Replay、RVV runtime、性能或任意输入域。

## 2. 预注册分母

- same-origin pairs：31；
- exact accepted artifacts：62（TileLang 31、generated CUDA 31）；
- 每 artifact：1 个 Task31 exact accepted Q + 4 个新 robustness Q；
- artifact-Q evaluations：310；
- pair-level：每 pair 两个 frontend artifact，共 10 次 Q evaluation。

新 seed 在 `q_schedule_current.json` 中冻结：

```text
seed = uint32(SHA256("intentir.task32.q-seed.v1:" + run_id + ":" + ordinal)[:4])
ordinal = 1,2,3,4
```

seed 只依赖 exact artifact 的匿名 content id 与 ordinal，不依赖 frontend/case、RefJ、CUDA output、
failure 或 error magnitude。运行后不换 seed、不删除失败、不改变 310 分母。

## 3. 隔离

source-aware export 进程读取 Task28/Task31 provenance，只负责按冻结 seed 生成合法 ABI/shape 的 Q，并
写出单独 provenance。`multiq_consume.py` 的 exact whitelist 只接收：

- anonymous run id 与 exact `ProductionSnapshot(J,C)`；
- fixed logical extents 与 public policy；
- 五个 content-addressed NPZ 的 q id、SHA256 与 logical digest。

consumer 不接收 Source、pi、Replay、frontend、case、Gate、stem、seed 或 accepted/robustness 标签。
每个 artifact 在打开任何 Q 文件之前完成唯一一次 `materialize_j + lower_to_cuda`；随后每个 Q 只重新
绑定已编译 cubin 的 host graph，不重新 lower target code。

## 4. 失败与输入策略

- accepted Q 原样继承 Task31 byte-stable NPZ；
- 新 Q 复用 production frontend workload generator 的 shape/ABI binding；
- rank-0 scalar 必须保持 shape `()`；
- export 可运行 RefJ precheck，但不得据此替换 seed；
- 若 index、denominator、RefJ 或 CUDA execution 失败，记录真实 failure taxonomy 并保留分母；
- 每个 artifact 独立进程，driver 使用临时 receipt、digest/run-id/returncode 校验与 300 s timeout。

## 5. 预注册统计

必须分别报告：

1. 62 个 compile-once receipts 与对 Task31 cubin digest identity；
2. 310 artifact-Q 的 input / RefJ / CUDA--RefJ pass；
3. output-comparison 分母，不把多输出冒充更多 kernels；
4. accepted Q 与 248 个新 Q 分层；
5. default 与 TF32 policy 分层，分别报告 exact/tolerance outputs 和 max-abs-error median/p95/p99/max；
6. TileLang/CUDA provenance 分层（只在 consumer 后处理）；
7. semantic-core exact 26 pairs 与 source-grounded non-exact 5 pairs；
8. 每 artifact 五个 logical input digest 是否 distinct；
9. 全部 failure taxonomy，不做成功样本筛选。

不做显著性检验或 frontend speedup；本 Task 没有性能假设。

## 6. 运行阶段

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multiq-runtime-robustness-32/run_experiment.py \
  --phase schedule

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multiq-runtime-robustness-32/run_experiment.py \
  --phase export

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multiq-runtime-robustness-32/run_experiment.py \
  --phase runtime --case-timeout-s 300

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multiq-runtime-robustness-32/run_experiment.py \
  --phase summarize
```

## 7. 完成条件

- [x] 在 export/runtime 前冻结 62 × 5 schedule；
- [ ] 导出 310 份 deterministic Q，分母与 digest 自洽；
- [ ] 62 个 artifact 均 compile once；
- [ ] 310 次真实 RefJ/CUDA execution 全量记录；
- [ ] accepted/new、policy、frontend、J-stratum 分层完整；
- [ ] failure 不筛选、不换 seed；
- [ ] 论文只采用机器结果支持的 value-robustness claim；
- [ ] 最终 commit 后工作区干净。
