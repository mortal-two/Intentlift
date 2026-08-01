# TileLang / CUDA 多前端后端复用实验

> 日期：2026-08-02
> 论文用途：§5.2 multi-frontend extensibility / backend reuse
> 正式分母：31 same-origin pairs / 62 exact accepted frontend artifacts

## 一句话结论

31 个 TileLang kernel 及其 31 个 generated-CUDA counterparts 均已独立完成 TypeCheck、frontend-specific
Gate1 与 three-leg Gate2；把这 62 份 accepted artifacts 匿名投影为 J+C 后，同一套未经 frontend 分叉的
CUDA/RVV backend 对 62/62 完成 Decode、Intent MLIR Emit、Contract Legal 与真实 target binary lower。
新增 frontend 的 backend delta 为 0；唯一暴露的 `math.floor` feature gap 用一个 +31 LoC 的共享 CUDA
consumer 修复，同时闭合 TileLang/CUDA 两份 floor artifact，无任何 per-kernel rule。

## 论文主表建议

| Evidence | TileLang | generated CUDA | Pair-level |
|---|---:|---:|---:|
| Frontend SCRC closure | 31/31 | 31/31 | 31/31 both |
| Backend Decode | 31/31 | 31/31 | 31/31 both |
| Emit Intent MLIR | 31/31 | 31/31 | 31/31 both |
| CUDA Contract Legal | 31/31 | 31/31 | 31/31 both |
| CUDA real cubin lower | 31/31 | 31/31 | 31/31 both |
| RVV Contract Legal | 31/31 | 31/31 | 31/31 both |
| RVV real object lower | 31/31 | 31/31 | 31/31 both |
| Frontend-specific backend LoC/rules | 0 / 0 | 0 / 0 | — |
| Per-kernel backend rules | 0 | 0 | 0 |
| Shared J-feature consumer delta | — | — | +31 LoC, `math.floor` CUDA |

配套 correspondence 数字使用 full-31，而不是历史 Primary-20：signature-free semantic core 26/31、
op-kind set 29/31、boundary 26/31、carrier-normalized buffer profile 27/31；raw full J 为 0/31，因为
TileLang tensor ABI 与 CUDA pointer/runtime-scalar ABI 被诚实保留。换言之，复用不依赖强行让两个 J
字节相同。

## 实验为什么有说服力

后端不是通过 case_id 找模板。独立 backend process 只读取严格白名单：

```text
anonymous run id
ProductionSnapshot(J, optional C)
canonical logical extents
constexpr bindings
```

Source、pi、Replay、显式 frontend label、case_id、Gate evidence 与历史 winner 均不在输入中。生产
snapshot 还对 J、C 与整体内容分别做 digest 校验。J 必须保留 tensor/pointer 等真实 ABI，因而可能让人
统计上猜测 frontend；本实验禁止的是 J 之外的 identity/Source 旁路，而不是要求 frontend 零信息。
因此 62/62 lower 说明的是 accepted J 的真实可消费性，而不是 backend 偷看原 Source 或复用旧答案。

最有价值的观察不是“第一次就全 PASS”。第一次 CUDA Lower 是 60/62；两个失败都来自同一共享
`math.floor` feature，分别对应 TileLang/CUDA floor artifact，且诊断相同。我们没有添加两个 frontend
adapter，也没有添加 `floor2d` 模板，而是在通用 NVIDIA math pass 中新增 exact NVVM floor consumer。
同一修复让结果变成 62/62。这直接展示了期望的复杂度因子化：

```text
frontend diversity -> Lift/certification boundary
shared computation  -> J
target feature gap  -> one shared backend consumer
```

## 可以写进论文的 claim

- 31 对、62 份制品全部拥有完整 `(J,C,pi)`，且各自通过 TypeCheck/Gate1/Gate2；
- production backend 只消费匿名 J+C，不消费 pi 或 frontend identity；
- 同一 CUDA 与 RVV backend 均对 62/62 产生真实 target code state；
- 支持第二种 frontend 没有增加 backend code 或 per-kernel rule；
- 一个新 J feature 只需要一次共享 target consumer 增量，并同时服务两个 frontend；
- full-31 的 J correspondence/delta 与 backend reuse 是两组互补证据。

## 不能写的 claim

- 不能说本实验评价了自动 Lift：62 份 `(J,C,pi)` 是主 Codex 手工构造；
- 不能把 generated CUDA 外推为所有 handwritten CUDA；
- 不能把 binary generation 写成 runtime correctness；正确性证据仍来自 Task28 Gate1/Gate2；
- 不能把本 Task 写成性能实验：没有对 62 个 binary 做 Q/D、执行或计时；
- 不能借用 24 个 Triton J 的 backend speedup 作为这 62 个 frontend artifacts 的性能数据；
- 不能把 +31 shared feature LoC 隐藏后宣称“所有能力天然零修改”。

## 篇幅有限时怎么放

正文只需主表、一段 input-isolation 说明、一个 `60/62 -> +31 shared LoC -> 62/62` callout。J 的五个
source-grounded non-exact pair 不必逐个展开；正文举 `groupnorm`（core exact、ABI 不同）和 `gather2d`
（真实 boundary delta）两个例子即可，其余放 appendix/artifact。

机器证据：

- Task28 closure：`.trellis/tasks/08-01-j-only-replay-multifrontend-batch-28/results_current.json`；
- Task29 full-31 correspondence：`.trellis/tasks/08-01-tilelang-cuda-paired-j-analysis-29/results_current.json`；
- Task30 backend input/result：`.trellis/tasks/08-02-artifact-only-multifrontend-backend-reuse-30/`。
