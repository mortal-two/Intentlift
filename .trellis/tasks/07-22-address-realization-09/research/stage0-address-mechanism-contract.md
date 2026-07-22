# Task09 Stage 0：Address Base / Lane Mechanism Contract

## 权威反例

本任务只以已提交的 protocol-v5 证据为输入：

```text
artifacts/real_llm_batch_28_v5/recovery/liger_swiglu_forward/
```

- case receipt SHA：`5a286725e22f1fadc83771eb460a26e823e0c9675606bde311412563f2921b71`
- canonical artifact digest：`a3ec791ee55ffbac51aac2dc68fe06d7cb45a85a3a2b497e2bbfb77b931ff68e`
- Replay code SHA：`95951e292cc76b4281e97b6999ef3b616ae2b660bb3107b72327b043ebdd9c4f`
- strict Gate1：`no_closure`，score `0.6407238924050632`
- op delta：Replay 多 1 `arith.addi`/1 `tt.splat`，少 2 `arith.extsi`/3 `tt.addptr`

v5 artifact 已合法表达 p3/program instance 到三个 access-realize instances 的 routes、fanout、
sharing 和 realize→transfer chain。Replay 也使用 p3，但把 source 的 scalar pointer-base stage
折叠成 vector offset。因此本阶段不再讨论 graph 外键或 J/C 所有权。

## 三个相互正交的 stage

### `address_base`

一个 compiler-observable、在单个 program instance 内对 lanes 一致的 pointer/base realization
stage。它可以依赖匿名 pointer/interface parameter、program instance、runtime stride、pure-C
placement 等已有 anchors，但不记录如何计算。

它表示的是“源码/编译器在 lane broadcast 之前确实存在一个独立 base stage”，不是：

- `program_id * stride` 公式；
- scalar integer value；
- i32/i64、pointer width 或 cast operator；
- row-major、column-major 等 layout 语义；
- 某个 Triton/CuTe/CUDA API。

若公共 frontend 为实现该 base 自然需要 pointer-width conversion，Gate1 可以观察它，π 不复制
width/cast。没有独立 base stage 的 flat source 不得虚构 `address_base`。

### `address_lane`

一个 compiler-observable、形成 lane-varying final address collection 的 stage。它通常消费一个
`address_base` instance 和由 J 已定义的 access/domain anchors；若源码只有 raw pointer + lane，
它也可以直接依赖 pointer anchor，而不虚构 base stage。

它不记录：

- lane/index/offset formula；
- vector length、dtype 或 shape；
- mask/predicate/fill；
- gather/scatter axis；
- add/mul/cast 等 operator。

这些内容分别属于 J 或公共 frontend realization。

### transfer / update

`transfer_direct`、`transfer_async`、`update_atomic`、`update_privatized` 保持原义，描述 address
形成之后的数据移动/更新机制。它们不能替代 address stage，address tag 也不能替代 transfer、
atomic 或 synchronization。

典型 direct access 图为：

```text
pointer/program/stride anchors
  -> address_base
  -> address_lane (also anchored by J access/domain)
  -> transfer_direct
```

flat source 则可能只有：

```text
pointer/program/stride/lane anchors
  -> address_lane
  -> transfer_direct
```

二者的 π 必须不同。

## Source-neutral 正反例

合法 staged graph：

```json
{
  "instances": [
    {"id":"i0","semantic_ref":"p0","region":"r1","mechanism":"realize"},
    {"id":"i1","semantic_ref":"a0","region":"r1","mechanism":"address_base"},
    {"id":"i2","semantic_ref":"a0","region":"r1","mechanism":"address_lane"},
    {"id":"i3","semantic_ref":"a0","region":"r1","mechanism":"transfer_direct"}
  ],
  "routes": [
    {"source":"i0","target":"i1","slots":[0]},
    {"source":"i1","target":"i2","slots":[0]},
    {"source":"i2","target":"i3","slots":[0]}
  ]
}
```

合法 flat graph 删除 i1，让 pointer anchor 直接进入 i2。非法做法包括：

- 在 instance 增加 `formula="program*stride"`；
- 增加 `dtype="i64"`、`op="addptr"`、`api="tl.load"`；
- 把 logical index expression 从 J 复制到 π；
- 仅因目标 Gate1 需要而在源码没有独立 stage 时虚构 base。

## 对复杂机制的适用性审计

| 机制 | J 所有权 | address π | 其余 π |
|---|---|---|---|
| direct copy | identity/index relation | 可选 base/lane stages | `transfer_direct` |
| async/TMA copy | identity/index relation | 只标真实 base/lane；TMA descriptor 不强制 lane tag | `transfer_async` + synchronize |
| gather/scatter | logical index map、bounds、update relation | final lane address realization，不复制 indices formula | direct transfer/update |
| elementwise/reduction fusion | 完整 value DAG、access/effect | 每个真实 access 的 base/lane 与共享 | producer reuse/placement |
| atomic update | logical combiner/conflict/order relation | target address base/lane | `update_atomic` |
| privatized update | logical update relation | target address base/lane | `update_privatized` + later transfer |

address tags 对 source family 和 J opcode 完全正交。它们不要求为每个 kernel 写消费规则；Lift LLM
标注 source event，Replay LLM 根据完整 J/C/interface/π 选择公共 frontend 表达，Gate1 终检。

## 为什么不构成语义泄漏

知道一个 event 是 program-uniform base 或 lane-varying address，只能约束 compiler topology；它
不能恢复：

- 哪个 logical index 被访问；
- stride/offset 的公式和值；
- predicate、fill、effect 或 output mapping；
- load 后的 value DAG；
- update combiner 与 allowed-result relation。

所以 masked-J 后，address π 仍不足以生成正确 kernel。它与语义相关但不拥有 denotation；项目
要证明的是 ownership/non-recoverability，而不是统计独立。

## 阶段 1 实现决策

1. 保留 π JSON shape，只扩 `mechanism` closed enum；
2. 在 π 权威模块维护唯一 mechanism type declaration与共享文字 grammar，schema/parser/prompt
   从它投影；这是一份 closed type，不是按 kernel 匹配并产生代码的 RuleRegistry；
3. Replay contract 明确不得把 `address_base -> address_lane` 代数合并；
4. validator 不强制每个 transfer 都必须有 address tag，因为 descriptor/bulk/scalar transfer
   可能没有 lane-varying address stage；完整性仍由 Lift proposal + Replay + strict Gate1 检验；
5. 本任务不增加 width/cast/materialize tags。只有 address tags 正确出现后仍有新的独立反例，
   才能讨论下一次扩展。
