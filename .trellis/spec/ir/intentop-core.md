# IntentOp 健壮性要求（开放设计）

本文件定义下一阶段的审查目标，不冻结最终 op schema，也不声称当前代码已经覆盖。

## 目标

找到一套小型、可组合、LLM 友好、可执行的 J 语义核心，使其能表达而不是枚举：

- copy/reverse/transpose/gather 等 logical mapping；
- elementwise 与 reduction fusion 的组合 DAG；
- reduce/scan/contraction/segment aggregate；
- FlashAttention 等真正的 semantic recurrence；
- scatter/histogram/in-place/atomic realization 背后的 logical update；
- dynamic indexing、partial write、alias 和 observable conflict/order。

## 当前候选分解

当前研究假设是五类构造：

1. elementwise expression；
2. logical index/map；
3. aggregate；
4. semantic recurrence；
5. logical update。

它们只是 robustness audit 的候选基座。冻结前必须用代表性真实 Kernel 检查：

- 表达是否无损；
- 是否需要 per-kernel special case；
- J-only scalar语义是否可执行；
- Gate1 π 是否能只引用而不复制语义；
- CUDA/RVV lowering 是否能消费完整 J；
- 浮点重结合、duplicate update、atomic order 是否需要单值或允许结果关系。

## 归属原则

- logical computation、index relation、observable state transition 属 J；
- source fusion、barrier、atomic instruction、async copy、pipeline loop 等 realization 属 π；
- tile/vector/stage/prefetch/buffer 等可替换选择属 C；
- 只有会改变可观察结果的 conflict/order/tolerance 才进入 B。

不得因 reviewer 提到一个新机制就直接新增 `FusionContract`、`PipelineContract`、
`AtomicRule` 或同名大 op。支持 claim 必须来自真实 `Gate1+Gate2` artifact，而不是 dialect
中存在一个 op 名。
