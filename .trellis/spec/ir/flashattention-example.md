# Worked Example：FlashAttention 的 A/B/C 因式分解

本例说明旗舰用例如何落入 `J=A+B`，不规定尚在迁移中的具体 JSON 字段拼写。
权威 op/schema 见 [op-signatures.md](op-signatures.md) 与代码；本文件只固定语义所有权。

## Layer A：logical value semantics

```text
inputs:  Q[B,H,Sq,D], K[B,H,Sk,D], V[B,H,Sk,D]
outputs: O[B,H,Sq,D], optional LSE[B,H,Sq]

state:
  m    = -inf
  l    = 0
  Oacc = 0

semantic_loop axis=Sk:
  S      = contraction(Q, K, "bhqd,bhkd->bhqk")
  m_blk  = reduce_max(S, axis=k)
  m_new  = max(m, m_blk)
  P      = exp(S - m_new)
  alpha  = exp(m - m_new)
  l_new  = l * alpha + reduce_sum(P, axis=k)
  O_new  = Oacc * alpha + contraction(P, V, "bhqk,bhkd->bhqd")
  yield m_new, l_new, O_new

O = Oacc / l
LSE = m + log(l)  // 若接口声明该输出
```

A 保存的是 online-softmax recurrence，而不是 source K-tile loop 的具体语句数量、
pointer advancement 或 local rank insertion。

## Layer B：observable execution and legality

至少包括：

- causal 与 Sk tail predicate；
- masked score contribution；
- max/sum reduction identity 与 accumulator dtype；
- 允许的 reassociation/tolerance；
- Q/K/V/O/LSE 的 access domain 与 store guard；
- output value mapping；
- O/LSE 多输出、destination、alias/write coverage；
- observable state/effect/scope obligation。

示意：

```text
A.return_values:
  O   <- normalized_Oacc
  LSE <- m_plus_log_l

B.destination_bindings:
  O   -> output_O   { no_alias, full_write }
  LSE -> output_LSE { no_alias, guarded_write }
```

若 backend 不支持 LSE、partial writeback 或某个数值 obligation，应返回 unsupported，
不能只 lower O 后把整个 program 标为成功。

## Layer C：replaceable implementation strategy

```text
tile_q / tile_k
num_warps / num_stages
MMA variant
shared-memory staging
prefetch / software pipeline
vector width
query-outer vs other legal schedule
```

Gate1 可冻结同一 C 来比较 source/replay realization；目标 backend 可在保持 J 不变时
retune C。`tile_k` 不属于 semantic loop trip 的 portable 真值。

## Source-validation witness

以下仅属于 frontend RuleRegistry、π、fingerprint 或诊断：

- `tl.program_id` 与 lane 构造；
- `tl.expand_dims` 的本地 spelling；
- pointer addition association；
- source-specific temporary / SSA 名；
- compiler dump presentation topology。

它们不得成为 target backend 理解 FlashAttention 的前置条件。

## 这个例子证明什么

- online recurrence（A）与 tiling/pipeline（C）可以从同一个源循环中因式分解；
- causal/tail/multi-output/numerical obligation（B）不会被 `attention` 大算子吞掉；
- 不同前端可产生不同 realization topology，只要各自 `P_i ~ J_i` 且共享 backend
  能消费其完整 A+B 合同；
- backend conformance 与性能评价独立于 Gate1/Gate2 verdict。

## Quality Check

- [ ] K loop 中真正的 carry 是否在 A？
- [ ] tile/pipeline 是否只在 C？
- [ ] causal/tail/reduction identity 是否在 B？
- [ ] O 与 LSE 的 logical result、destination 与 writeback 是否分离？
- [ ] backend 是否消费完整 A+B，而不是只识别一个 FlashAttention 名字？
