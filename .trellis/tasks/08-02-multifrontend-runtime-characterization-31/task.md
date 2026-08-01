# Task 31：多前端 accepted J 的匿名 Runtime 闭合与生成物表征

> 状态：`COMPLETE`（2026-08-02）
> 上游前端证据：Task28（31/31 same-origin pairs，62/62 exact accepted artifacts）
> 上游 J 分层：Task29（26 个 semantic-core exact pair，5 个 source-grounded non-exact pair）
> 上游静态后端：Task30（62/62 CUDA cubin 与 RVV object）
> 通用窄整数 CUDA launch ABI：`88ffd63e`
> 目标：在不执行 Source/Replay、不给 runtime 显式 frontend 身份的条件下，验证 exact accepted J
> 生成的 CUDA graph 是否能在原 Gate2 Q 上与 canonical RefJ 闭合，并量化跨前端生成物的相同与差异。

## 1. 有限论文问题

本 Task 只回答三个可证伪问题：

1. Task30 的 62 份真实 cubin 是否不止“能生成”，而且能在 exact accepted Gate2 Q 上执行并与 RefJ
   对齐？
2. TileLang/CUDA 同源 pair 的 J 语义核心一致时，共享 backend 是否产生相同 target-neutral IR、CUDA
   binary、RVV object 与 runtime graph；J 有 Source 依据的差异时，backend 是否保留差异但仍正确执行？
3. 为完成这批 runtime 是否需要 frontend-specific/per-kernel backend recipe，还是只需要按共享 typed ABI
   补一个通用 consumer？

它不重新调用 Lift 或 Replay，不把 RefJ 比较冒充 Source--Replay Gate1，也不声称 RVV runtime、
Source/Replay speedup 或通用 handwritten-CUDA 覆盖。

正式分母固定为全部 **31 pairs / 62 artifacts**；不使用历史 Primary-20 过滤分母。

## 2. 两进程隔离与 exact Q 重建

```text
Task28 accepted artifact + adapter + Gate2 receipt
  -> source-aware export process
       reconstruct exact accepted Gate2 Q by input_digest
       project runtime names back to logical J input names
       -> runtime_inputs_current.json + deterministic NPZ
       -> runtime_provenance_current.json（never passed to consumer）

runtime_consume.py（fresh process per artifact）
  1. strict-decode anonymous ProductionSnapshot(J,C)
  2. materialize_j + generic lower_to_cuda + cubin digest
  3. only now load Q arrays
  4. canonical RefJ(J,Q)
  5. execute prepared CUDA whole graph
  6. compare CUDA target vs RefJ under the accepted public policy
  7. benchmark the already-prepared allocation-free/no-sync graph
```

`runtime_consume.py` 的 root/entry schema 是 exact whitelist。它只接收：

- anonymous content-addressed `run_id`；
- exact `ProductionSnapshot(J,C)`；
- concrete logical extents；
- Q 的 NPZ、SHA256 与 logical input digest；
- public comparison policy。

它拒绝并不可见 Source、pi、Replay、frontend、case id、Gate receipt、历史 winner 与 stem。显式身份只在
后处理 provenance 中恢复。编译阶段的 receipt 固定记录 `q_values_loaded_before_compile=false`，因此 Q
不能影响 materialization、variant selection、kernel topology 或 cubin generation。

Task28 的 accepted Gate2 runtime-key digest 与本 Task 的 logical-J-name digest 是两个不同层的摘要；它们
不应被误写成同一个 digest。62 份 Q 均通过 accepted digest 反查：52 份复用 accepted adapter seed，10
份 contraction 使用 current projection seed。确定性 NPZ writer 保留 rank-0 scalar，不把 `()` 提升成
`(1,)`。

## 3. Stage 与 PASS 定义

| Stage | 精确定义 |
|---|---|
| Decode | `ProductionSnapshot.from_dict` 严格复核 J/C/snapshot digest |
| Compile | exact J materialize 为 Intent MLIR，generic CUDA lower 产生至少一个 ELF cubin；Q 尚未加载 |
| Input | NPZ SHA256、logical input digest、J input name/type/shape 验证 |
| RefJ | `ScalarLoweringSpec.create(J, ...)` 与 `ScalarLoweringRunner` 完成 canonical J-only execution |
| Execute/Compare | 真实加载 cubin，执行完整 kernel graph，所有 J output 在同一 public Q 下与 RefJ PASS |
| Benchmark | 对已经 prepared 的 whole graph 使用 event timing；不在 timed closure 中分配或同步 |

每个 artifact 使用独立子进程，避免一个 CUDA context failure 污染后续分母。driver 只接受唯一临时
receipt；校验 result digest、run_id 与进程返回码后再原子替换正式结果，并设 300 秒 per-case timeout，
因此不会把旧 PASS 文件误当作当前结果。

## 4. 最终 Runtime 结果

| Evidence | Result |
|---|---:|
| Exact accepted Gate2 Q reconstructed | 62/62 |
| Decode / Compile / Input | 62/62 each |
| Canonical RefJ execution | 62/62 |
| CUDA target vs RefJ | 62/62 |
| Prepared graph benchmark receipt | 62/62 |
| Recompiled cubin sequence equals Task30 | 62/62 |
| Both frontend artifacts close per pair | 31/31 |

共比较 82 个 outputs，82/82 PASS：52 个 artifacts 使用 default policy（72 outputs，其中 8 个 exact、
64 个 tolerance），10 个使用 TF32 tensor-core policy（10 outputs）。default tolerance outputs 的最大观测
绝对误差为 `1.9073486328125e-6`；TF32 policy outputs 为 `2.384185791015625e-7`。这些是当前 Q 上的
observation，不是用来缩紧或重新选择 Gate2 policy 的依据。

运行环境：Python 3.10.12、PyTorch 2.10.0+cu130、NVIDIA GeForce RTX 5090 D、compute capability
12.0。计时使用 10 ms warmup / 30 ms repeat，未锁 GPU clocks；因此 compile/latency 只作描述性分布，
不作 frontend speedup 推断。

## 5. J correspondence 到 target artifact 的传递

| J stratum | N | Intent MLIR digest equal | CUDA cubin sequence equal | RVV object digest equal | Runtime graph topology equal | Output signature equal | Runtime both PASS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic core exact | 26 | 26/26 | 26/26 | 26/26 | 26/26 | 26/26 | 26/26 |
| Source-grounded non-exact | 5 | 0/5 | 0/5 | 0/5 | 0/5 | 4/5 | 5/5 |

这组结果比“62/62 都 lower”更能说明机制：

- 当 Layer-A semantic core、boundary 与 typed representation 足以一致时，两种 frontend 的差异在
  certification boundary 后消失，两个 target 的生成物达到 digest-level identity；
- 5 个 J 合理不一致 pair 没有被 canonicalizer 强行抹平，Intent MLIR、CUDA、RVV 与 graph topology
  都真实分叉；
- 这些分叉仍然全部与各自 RefJ 闭合，因此“保留差异”不等于 backend 不可复用；
- `any_kernel_dim` 的 output signature 也不同，其余 4 个 non-exact pair 保持相同 output signature。

静态多样性不是重复拷贝一个 binary：62 份 artifacts 共生成 401 个 cubin records、102 个 unique cubin
digests；62 个 RVV objects 中有 36 个 unique digests。

## 6. 共性机制缺口与工程归因

第一次诊断运行达到 RefJ 61/62、CUDA Execute/Compare 58/62：

- 3 个 CUDA-J artifacts 的 rank-0 `i8` launch operand 暴露 host orchestrator 只支持 i32/i64 的共享
  ABI 缺口；
- `gather2d` 的 rank-0 i32 输入被实验 NPZ writer 误提升成 shape `(1,)`，属于 evidence serialization
  缺口。

修复没有按 frontend 或 kernel 分支：

- production `orchestrate.py` 统一增加 `i1/i8/i16` exact-width launch ABI，commit `88ffd63e`，
  `+35/-4, net +31 LoC`；
- experiment writer 对 rank-0 保形、对高 rank 继续做 C-contiguous canonicalization；
- frontend-specific runtime rules = 0；per-kernel rules = 0。

同一 62 分母、同一 Q 和同一 J/C snapshot 重跑后为 62/62。Task30 的 `math.floor` +31 LoC CUDA
consumer 与本 Task 的 narrow-scalar +31 net LoC 是两个不同的共享 feature delta，论文中必须分开归因。

## 7. 描述性规模/时间数据

| Frontend provenance（仅后处理分组） | N | Serial compile median / p90 | CUDA kernels median | Cubin bytes median | Prepared graph median / p90 | RVV object bytes median |
|---|---:|---:|---:|---:|---:|---:|
| TileLang | 31 | 314.602 / 561.988 ms | 4 | 23,744 | 17.154 / 91.757 us | 4,592 |
| generated CUDA | 31 | 296.758 / 514.368 ms | 6 | 34,784 | 20.653 / 64.720 us | 4,592 |

这些 aggregate 混合了 26 个 exact 与 5 个 non-exact pair，并且两个 artifact 在独立进程中计时、未锁
GPU clocks。它们只刻画这批 J 的编译/生成物/graph 规模，不支持“一个 frontend 比另一个更快”。
对于 exact stratum，digest identity 比独立进程 latency ratio 是更强且更稳定的证据。

## 8. 复现命令

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multifrontend-runtime-characterization-31/run_experiment.py \
  --phase export

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multifrontend-runtime-characterization-31/run_experiment.py \
  --phase runtime --warmup-ms 10 --repeat-ms 30 --case-timeout-s 300

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python \
  .trellis/tasks/08-02-multifrontend-runtime-characterization-31/run_experiment.py \
  --phase summarize
```

连续两次 export 均得到同一个 input manifest digest：
`5fa636568bd03ef7072eebb07bdeff9c9907b9c9f8f3d3feb59fde3c25c18fbe`。

## 9. Machine artifacts

- `runtime_inputs_current.json`：consumer 实际读取的匿名 J+C/Q manifest；
- `runtime_values_current/*.npz`：62 份 byte-stable logical Q；
- `runtime_provenance_current.json`：Source/case/frontend 后处理映射，明确不进入 consumer；
- `runtime_runs_current/*.json`：62 份独立 compile/RefJ/execute/benchmark receipt；
- `runtime_results_current.json`：全量 stage 汇总；
- `results_current.json`：paired-J、numeric、artifact 与 claim-boundary 汇总；
- `results_current.md`：人类可读表。

当前 summary digest：
`2af0aea5d499f1184cd9592b021cbd45ff8400257bb31865205af8997e7517e0`。

## 10. 完成条件

- [x] 31 pairs / 62 artifacts 全量，不筛失败；
- [x] reconstruct 62/62 exact accepted Gate2 Q；
- [x] anonymous consumer 不见 Source/pi/Replay/frontend/case/Gate/history；
- [x] compile-before-Q-load；
- [x] canonical RefJ 62/62；
- [x] real CUDA graph vs RefJ 62/62，31/31 pair 两边闭合；
- [x] Task30 cubin digest identity 62/62；
- [x] exact/non-exact J strata 的 cross-target artifact identity 完整统计；
- [x] production 缺口只按 shared typed ABI 修复，无 frontend/kernel rule；
- [x] 不恢复或新增 active `python/tests`；验证只使用真实 Kernel closure artifacts；
- [x] timing、Source/Replay、RVV runtime 与 generated-CUDA generalization 边界明确。
