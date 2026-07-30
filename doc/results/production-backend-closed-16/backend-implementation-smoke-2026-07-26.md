# Closed-16 production backend implementation smoke（2026-07-26）

本记录只回答“新增后端机制是否已真实贯通”。它不把历史 closed-12 的性能 aggregate 重标为
16-kernel 结果，也不把单 case smoke 冒充全语料论文结论。持久语义输入始终是冻结的 `J=A+B`；
Python 负责 transport、候选编排、ABI、Q/D 与计时，目标 kernel 和 transformation 由 C++/upstream
MLIR pipeline 生成。

对应实现 revision：`4862ad4e8b7dbdd8de9c9eda69b6233f38ec4672`。

## 1. 冻结语料与编译覆盖

production catalog 已 append-only 扩展到 16 项：DYT、Flash backward preprocess、Liger fused-add
RMSNorm、GeGLU、JSD、Liger LayerNorm、Liger RMSNorm、PolyNorm、Softmax、SwiGLU、TVD、xFormers
RMSNorm、xFormers RMSNorm-add、Liger Sparsemax、Liger KLDiv、Flash one-pass LayerNorm。

本轮得到以下实现覆盖：

- 16/16 从冻结 J deterministic materialize，不读取 Source、Replay 或 kernel 名；
- 16/16 通过通用 `Intent -> Linalg -> GPU/NVVM -> cubin` 编译；
- 16/16 至少有一个 registered C++ whole-graph path 生成单 cubin；
- 新增 consumer 覆盖 inout/post-state、access-local guard、scan + reduction relation、loop
  scan/carry、runtime `value_if`、unused ABI carrier 和多输出 half-storage epilogue；
- Sparsemax generic baseline 使用真正的 MLIR `scf.for` inclusive scan；Flash runtime `value_if`
  保留为 `scf.if`，再由 typed CUDA pass 转为 pure predicated dataflow，不由 Python 选择分支。

这是 materialization/codegen coverage，不等于 16/16 已完成 target execution closure 或性能测量。

## 2. CUDA Q/D controller smoke

统一 `SearchAndCloseController` 已在 RTX 5090 D（`sm_120a`）上用 opaque accepted Replay 跑通：

```text
D_search(seed=2001): target -- RefJ
D_cert(seed=3001):   target -- Source
                     target -- Replay
                     target -- RefJ
```

`D_search` 与 `D_cert` 的 exact input digest 必须不相交；三条 certification edge 均直接比较，
不使用 tolerance 传递。Replay manifest 绑定 replay content hash；缺 Replay 且要求 formal closure 时
runner fail closed。

| case / arm | candidate budget | backend closure | MLIR μs | Source μs | 结果 |
|---|---:|---|---:|---:|---:|
| KLDiv / E3 smoke | 2 | PASS | 0.8311 | 0.8749 | 1.053x vs Source |
| Flash one-pass LayerNorm / E2 | 48 | PASS | 1.3782 | 1.3901 | 1.009x vs Source |
| Flash one-pass LayerNorm / E3 | 48 | PASS | 1.3734 | 1.3881 | 1.011x vs Source |

Flash E3 winner 的 post-pass receipt 确认实际生成 2 个 FP32 FMA；E2 禁用该数值 state。E3/E2 只有
约 1.0035x，当前没有显著性证据，因此这是必须保留的负结果，而不是 reduction/FMA search 已带来
稳定加速的证据。KLDiv 的短 budget 也只验证 controller 闭合，不用于性能结论。两份最新 runner
报告均绑定下述 machine-spec digest，而不是事后人工套用 arm 名。

## 3. RVV concrete-state 对照

同一 J-only MLIR 在本地编译为 riscv64 object，远端 `ssh rvv`（原生 riscv64、VLEN=128）只链接、
执行和计时。materialized 与 whole-graph fused 是不同 concrete MLIR topology，各自独立搜索 O2/O3
和 vectorize 状态。

| case | E2 materialized best μs | E3 best μs | E3/E2 | winner |
|---|---:|---:|---:|---|
| KLDiv | 14.1428 | 9.1056 | 1.553x | fused scalar O3 |
| Flash one-pass LayerNorm | 36.9453 | 36.9453 | 1.000x | materialized |

KLDiv 的 fused-vector state 含 120 条 RVV 指令并达到 9.3754 μs，但实测更快的是 fused-scalar O3；
搜索目标是 latency，不是“必须向量化”。当前 RVV runner 只有 target--RefJ current-case correctness，
尚未携带 Source/Replay 的独立 held-out `D_cert`，所以 1.553x 只解释为同 J、同 target 的结构 state
增量，不声称端到端 backend closure 或超越 Source。

## 4. 消融协议绑定

E0--E4、L0/L1、N0--N3 由
[`backend-search-ablation-v1.json`](../../../spec/lowering/backend-search-ablation-v1.json) 唯一定义，
不是 runner 内部的另一套硬编码。当前 canonical experiment spec digest：

```text
60ca646b34c29ce3542d3c4e6a9e08c39155bc9e2a27324844ab8fad80f050a7
```

每份新报告写入该 digest。E2 固定 structural/numeric trace 只搜索 schedule；E3 扩展 hard-safe
structural/numeric states 并让每个 state 独立调 schedule；E4 与 E3 的 universe、Q/D 和 budget 相同，
只擦除 C prior。L0/L1 只消融额外 B-conditioned local transformations；N0--N3 逐步加入 reduction
topology、FMA、precision/approximation states。当前 16 项没有 typed contraction，precision/TF32 轴应
记录 `unavailable/not_matched`，不能制造候选。

## 5. 当前结论与下一步测量边界

本轮已经闭合“冻结 J 能被真实后端消费、生成 concrete MLIR states、执行通用 Q/D、并按机器 spec
组织消融”的实现链。当前不能宣称的是 16-kernel 全量性能提升。下一轮应在同一 revision、target、
Q/D、profiler 和 equal-total-budget 下跑完整 E0--E4/L0--L1/N0--N3，同时报告 all-corpus 与
transformation-applicable subset、失败候选、search cost、time-to-best 及负结果。
