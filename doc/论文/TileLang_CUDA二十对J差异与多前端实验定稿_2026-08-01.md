# TileLang / CUDA 三十一对 J 差异与多前端实验定稿

> 日期：2026-08-01
> 状态：Task28 closure、Task29 paired-J analysis 与 Task30 backend reuse 已完成
> 论文主分母：31 个 unique same-origin pairs / 62 accepted artifacts
> 历史详细选择：20 pairs，仅用于旧版逐行分析，不作为正文分母
> 文件名中的“二十对”仅为兼容既有引用而保留；正文与所有 aggregate 已统一为 full-31

## 1. 最终实验逻辑

TileLang case 与其 generated CUDA snapshot 构成一个受控 paired experiment：两者保留共同的算法来源，
但经过不同 frontend syntax、ABI、compiler representation 与 physical realization。我们不要求两边的
full J 相等，而分别完成：

```text
TileLang Source -> manual (J,C,pi) -> fresh J-only Replay
                -> TIR-hard Gate1 -> three-leg Gate2

CUDA Source     -> manual (J,C,pi) -> fresh J-only Replay
                -> semantic-PTX-hard Gate1 -> three-leg Gate2
```

然后只对已经独立接受的两份 exact artifact 做事后 deterministic J comparison。这样把三个问题分开：

1. 两个 frontend 是否都能闭合；
2. 闭合后的 J 哪些层稳定、哪些差异有 Source 依据；
3. 支持这些 frontend 需要多少 generic engineering，而不是多少 per-kernel recipe。

Task28 的闭合分母与本文所有 aggregate 始终是 31 pair / 62 frontend runs。Task29 曾冻结 20 对作为
篇幅可控的逐行选择；它现在只保留为历史分析便利，不作为论文主表、成功率或 robustness 对照。

## 2. Paired-J 结果

| Metric | Full 31 | 解释 |
|---|---:|---|
| Independent accepted artifacts | 62/62 | 两个 frontend 各自 TypeCheck/Gate1/Gate2 |
| Full canonical J exact | 0/31 | frontend identity 与 raw ABI 被保留；不是失败 |
| Identity-normalized J exact（仍含 signature） | 0/31 | tensor 与 pointer signature 始终不同 |
| Signature-free semantic core exact | 26/31 | 83.9% full-corpus exact |
| Layer-A typed intent exact | 26/31 | 与 core delta 同一五对 |
| Layer-A op-kind set exact | 29/31 | 93.5% 保留相同算子种类 |
| Boundary exact | 26/31 | 五对保留 source-grounded boundary/typed delta |
| Carrier-normalized typed buffer profile exact | 27/31 | 四对 i1↔i8 typed carrier 不同 |
| C slots + values exact | 31/31 | generated CUDA 保留同一组 source choices |
| Compact pi exact | 31/31 | 当前 program-axis witness 稳定 |

这里的 `semantic core exact` 删除的只有 `kernel_id`、IntentFunction name、node-id spelling 和 raw
signature；operator、operand、dtype、predicate、fill、access、effect、result relation、scope、storage 与
dimension facts全部保留。因此 26/31 不是模糊文本相似度，而是 canonical typed structure exact。

相反，0/31 full-J exact 是预期结果：TileLang 使用 tensor parameters，CUDA 使用 pointer parameters；
31/31 都有 carrier kind delta。27/31 CUDA signatures 还显式增加 runtime scalar，总计 38 个、median
每对 1 个。忽略 tensor/pointer carrier 与 scalar 名称后，typed buffer multiset 仍有 27/31 exact。

## 3. 五个 non-exact semantic core 到底差在哪里

| Pair | TileLang J | CUDA J | Source-grounded 原因 |
|---|---|---|---|
| `any_kernel_dim` | bool output 为 `i1`，`ne` 直接产生输出 | `signed char*` output；增加 `i1 -> i8` cast 与 runtime `M` | CUDA Source 明确存储 cast 后的 signed char |
| `rowmask_where2d` | mask `i1 -> f32` 后相乘 | mask carrier 为 `i8*`；显式 `i8 -> i1 -> f32` | CUDA Source 参数是 `const signed char* row_mask` |
| `masked_softmax2d` | mask buffer 为 `i1` | mask buffer 为 `i8*`，多一个 normalization cast 与 scalar extent | generated CUDA ABI 对 bool buffer 的具体表示 |
| `matmul_fused_epilogue2d` | 两个 mask buffers 为 `i1` | 两个 `i8*` mask carriers、额外 casts 与三个 runtime extents | CUDA Source exact typed ABI；TensorMap/TMA 仍只属于 physical Source |
| `gather2d` | 直接二维 gather，假设 index 有效 | 显式 `0<=r<M`、`0<=c<64`，越界 select `0.0`，并携带 `M` | 两份 Source 的 boundary 行为确实不同 |

前四对不能通过删除 cast 强制变成同一个 J，因为 J 必须精确描述各自 Source 的 typed observable
interface。`gather2d` 更不能强制相等：CUDA 对越界索引定义 zero-fill，TileLang Source 没有该保证。

每一对仍分别通过 Source--Replay、Source--J、Replay--J 三边 Gate2。科学结论不是“差异不重要”，而是：

> **共同算法结构在 26/31 对上达到 exact typed core；剩余五对的差异可被具体 Source ABI 或 boundary
> 行为解释，没有 unexplained J delta。**

五对逐行 Source/J 依据冻结在 Task29 `source_delta_audit.md`，其中包含 exact selected run 与源代码
行号；该分类不是由 pair comparator 猜测产生。

## 4. C 与 pi 应该怎样解释

`C exact = 31/31` 说明同源 generated CUDA snapshot 保留了相同的公开 source-choice slots/values。
它不能证明 C 在新 target 上最优，也不能证明 backend 已经消费这些 prior。

`compact pi exact = 31/31` 只说明当前 paired artifacts 使用相同 program-axis witness。π 不进入 Replay，
也不进入 production backend；因此这个数字不能被写成“J 相同”或“realization 相同”。事实上最后五对
CUDA Source 的 TensorMap/TMA、swizzle、256-thread physical launch 与 dynamic shared memory 均不在
J/Replay 中，而 TileLang 使用自己的 TIR realization。

## 5. Replay 规模与成本

| Frontend | Runs | Gate1 median | Gate1 mean | Replay bytes median | Provider calls |
|---|---:|---:|---:|---:|---:|
| TileLang | 31 | 198.701 s | 236.049 s | 690 B | 31 |
| CUDA | 31 | 128.694 s | 129.945 s | 916 B | 31 |

62 个 accepted run 均为一次 physical provider call，Lift 调用为 0，pi 对 Replay 不可见。Replay source
很小，但分钟级 wall time 仍然明显；因此论文可以声称 J-only Replay 能覆盖前端碎片化，不能声称该
agent path 已达到成熟编译器的编译速度。TileLang 与 CUDA 的时间差只作描述性结果，本实验没有隔离
模型服务负载、compiler latency 与 case complexity，不能直接归因于 frontend 本身。

## 6. 工程量如何报告

最后五个 TensorMap/TMA pair 从 checkpoint `b4f1b6da` 到 `df95382c` 的真实增量为：

| 类别 | Add / Del | Net | 论文归属 |
|---|---:|---:|---|
| Generic executable Python | +788 / -87 | +701 | ABI、observer、native execution、external Q |
| Declarative TensorMap sidecar | +71 / -0 | +71 | physical Source descriptor/launch data |
| Experiment artifact/runner/recheck | +472 / -5 | +467 | 证据；不计 production onboarding |
| Kernel-specific production semantic lowering | 0 rules | 0 | headline |

该数字是“从 26 对扩到最后 31 对”的最难 descriptor 增量，不是完整 TileLang/CUDA onboarding
成本。完整 frontend onboarding 仍需从可靠的 Triton-centered baseline 重新做 Git ownership diff，不能
把 +701 写成“接入 CUDA 的总 LoC”。

## 7. 论文正文最小呈现

篇幅有限时只保留一张主表和 Fig.2(a) 的一个 panel。

### 主表

合并六块证据：

1. closure：31/31 pairs、62/62 frontend runs；
2. correspondence：core 26/31、op-kind 29/31、boundary 26/31；
3. frontend deltas：raw signature 0/31、tensor/pointer 31/31、extra CUDA scalar 27/31；
4. backend reuse：CUDA/RVV 的 Decode、Emit、Legal、real Lower 均为 62/62；
5. engineering：frontend-specific backend 0 LoC/0 rules；共享 `math.floor` CUDA consumer +31 LoC；
6. frontend engineering：最难增量 +701 generic net、+71 sidecar、0 kernel-specific lowering rules。

### Fig.2(a)

主 panel 选择 `groupnorm` 或 `layernorm`：两边 signature-free core exact，但 TileLang tensor/TIR 与
CUDA pointer/scalar/semantic-PTX realization 不同。旁边用一个小 callout 展示 `gather2d`，说明系统不会
为了提高一致率删除真实的 CUDA bounds/zero-fill。

这样一张图同时说明：

- frontend diversity 不只是 parser syntax；
- J 可以稳定复用共同计算；
- J 仍诚实保留 Source-specific observable delta；
- physical realization 不需要泄漏给 Replay 或 backend。

## 8. 后端复用已补齐，仍缺什么数据

Task30 已补齐原 P0。剩余工作按论文价值只保留 onboarding ownership 与 external-validity stress；不再扩
重复 cohort。

### P0（已完成）：backend reuse matrix

search-J backend 已合入当前分支，并对 full-31 的 62 份 exact accepted J 记录：

```text
Decode / Emit Intent MLIR / Contract Legal / >=1 Lower State
```

backend process 只接收匿名 `ProductionSnapshot(J,C)` 与 logical extents，不接收 Source、pi、Replay、
显式 frontend/case identity、Gate evidence 或历史 winner。J 自身仍诚实保留 tensor/pointer ABI；这里
禁止的是 J 之外的 identity 旁路，而不是要求无法统计猜测 frontend。最终结果：

```text
Decode/Emit/CUDA Legal/CUDA cubin/RVV Legal/RVV object = 62/62
frontend-specific backend delta = 0 LoC / 0 rules
per-kernel backend recipe = 0
```

首次 CUDA Lower 为 60/62；两份 failure 是 TileLang/CUDA 的同一个 `math.floor` J feature，均报
`__nv_floorf` unresolved。通用 NVIDIA math pass 增加 +31 LoC exact NVVM floor consumer 后，同一批固定
输入达到 62/62。该增量按 shared J feature 报告，不伪装为零修改，也不归因给任一 frontend。

### P1：完整 onboarding ownership diff

冻结可靠 commits：Triton-centered baseline、TileLang onboarding、CUDA onboarding。按 adapter/runtime、
observer、public context、shared core、backend、sidecar、generated/evidence/docs 分类真实 Git numstat。没有
可靠历史基线时只报告 current ownership 和最难 descriptor 增量，不虚构 person-hours。

当前候选历史链为
`4712623b -> 0c6e4704 -> d30753e7 -> a21f90f2`；后续 `2495a931` 主要是 observer/audit
升级。这些 commits 混有 generated runs、历史 tests 与大规模同步，正式 LoC 必须做 path/ownership
过滤，不能直接引用 raw commit insertions。

### P2：external-validity stress

generated CUDA paired cohort 是有意控制算法来源的实验，但不能代表任意 handwritten CUDA。现有四个
handwritten CUDA cases 可作为 appendix stress：只报告其真实 prepare/closure 状态，不与 TileLang
强行配对，也不进入 31-pair correspondence 分母。

当前真实状态是：`cuda:naive_gemm`、`cuda:row_sum`、`cuda:transpose2d`、`cuda:vec_add` 只有
catalog/descriptor validation，尚无 TypeCheck、Gate1、Gate2 或 SCRC closure artifact。论文当前只能将
它们列为 `prepare PASS / closure N/A`；若要主张 handwritten-CUDA external validity，必须另开正式
J-only closure run。

不再花时间扩更多重复 cohort、比较 raw PTX 文本、追求 full-J byte equality或重跑 Lift。backend reuse
已经补齐；当前真正仍缺的是可靠历史基线下的完整 onboarding ownership，以及可选的 handwritten-CUDA
external-validity stress。

## 9. Threats and claim boundary

1. `(J,C,pi)` 是主代理手工作者化，本实验不评价自动 Lift quality；
2. generated CUDA 与 TileLang 同源，这使 paired difference 有控制变量优势，但限制了对独立 CUDA
   library 的外推；
3. canonical structural exactness 是描述指标，不是语义等价证明；最终行为依据仍是三边 Gate2；
4. 论文 aggregate 与 backend reuse 全部使用 full-31；Primary-20 只保留为历史逐行选择，不进入正文
   分母；
5. Task30 证明真实 target binary generation，但没有执行这些 binary、运行 Q/D 或计时，不能把 62/62
   Lower 写成 runtime correctness 或性能；
6. 24 个 Triton J 的 backend speedup 与本 62 份多前端 artifacts 是不同实验，不能混用分母或数值。

## 10. 机器证据

- Task28 closure：`.trellis/tasks/08-01-j-only-replay-multifrontend-batch-28/results_current.json`
- Task29 cohort：`.trellis/tasks/08-01-tilelang-cuda-paired-j-analysis-29/cohort20.json`
- Task29 analyzer：`.trellis/tasks/08-01-tilelang-cuda-paired-j-analysis-29/analyze_pairs.py`
- Task29 full result：`.trellis/tasks/08-01-tilelang-cuda-paired-j-analysis-29/results_current.json`
- Task29 readable table：`.trellis/tasks/08-01-tilelang-cuda-paired-j-analysis-29/results_current.md`
- 五个真实 delta 的 Source 审计：`.trellis/tasks/08-01-tilelang-cuda-paired-j-analysis-29/source_delta_audit.md`
- Task30 artifact-only backend reuse：`.trellis/tasks/08-02-artifact-only-multifrontend-backend-reuse-30/results_current.json`
- Task30 backend exact receipts：`.trellis/tasks/08-02-artifact-only-multifrontend-backend-reuse-30/backend_results_current.json`
- 每对 exact Source/J/Replay/Gate：Task28 `runs/*/run-*/`
