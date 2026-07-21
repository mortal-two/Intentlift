# Task07 Stage 0：所有权与信息流审计

## 结论先行

Protocol-v3 Swiglu 的 `no_closure` 是真实且正确的 Gate1 结果，但先前把根因整体命名为
`pi_realization_topology_incomplete` 过早。冻结证据实际证明的是：

```text
accepted packet 不能唯一决定 source Level-2 topology
```

它没有单独证明“当前 pi schema 的容量不足”。当前实现已经允许 pi instance 引用 `p3/c0/c1`；
v3 Lift 只是没有引用它们。与此同时，冻结 artifact 还存在两个更上游的所有权问题：

1. 源码的 `n_cols` 决定有效逻辑 extent，却被 manifest/Lift 当作 pure C；
2. source 以 `(row, col)` 访问，J 却被固定 compiler allocation shape 压成一维 `x0`。

因此正确修复顺序是：先闭合 J/C semantic partition，再以 strict Gate1 观察剩余 realization
residual；禁止让 pi 替遗漏的 J 传递语义。

## 冻结证据

权威 case 为 `artifacts/real_llm_batch_28_v3/recovery/liger_swiglu_forward`：

| 文件 | SHA-256 |
|---|---|
| `lift/artifact.json` | `a48329ea84c69d5d367c1abba38e5f42f7b7f2bd389c1161b706a3b9ba6ca6b6` |
| `calls/02_normal_replay/request.json` | `c2462624fcc38dee3ebc09178af6d8dd3c9b6a4af4c39b4952ab430657a357ff` |
| `calls/02_normal_replay/replay.py` | `7399b74d847735310ee8493975a20a521f71c836f1bcb149d103411409ae9350` |
| `gate1/receipt.json` | `1f6f62c5ee564d3e77b8441a14f593ddc736906b85c991f7deb617ccbbe3cf5f` |
| case receipt | `ae69d449cc70d7e812dbd4de4382aa164e837b6bcd8b1246211fc982de8df7c1` |

Canonical subobject SHA-256（`jq -cS`）为 J
`1d7a89894b5216dbd7d2f378fe4f0e05456e1ce2fc62452e019ccebbdd69cdfe`、C
`b6aa53f82f33b5072316151bf117f8add36a1a9a55d170428ac96ce8662d0bc9`、pi
`e02ef108628a4d8e0e5ef3f08960b994d8804b26b9a4c3d2f6be9b4cc88953e3`、interface
`16a97135884614662d11a16ca06c201791944454edc801d58304f61f363ca737`。

Source 保留：

```text
pid:i32 -> extsi:i64
p3:i32  -> extsi:i64
row_base = muli(pid64, p3_64)
base -> scalar addptr(row_base) -> splat -> vector addptr(col)
guard = col < 1024
```

Replay 生成：

```text
flat = pid * 1024 + col
base -> splat -> vector addptr(flat)
guard = flat < 131072
```

两者均可编译，但 strict comparator 报告 81 个 dataflow 和 6 个 predicate/effect 差异，
hard closure 正确失败；normal 未闭合，所以 masked-J 调用数为 0。

## 当前 pi 的真实消费路径

```text
Lift LLM response
  -> Responses JSON schema（closed shape）
  -> RealizationWitness parser（ID/ref/acyclic validation）
  -> Gate1Artifact cross-reference validation
  -> alpha-renamed Replay packet
  -> source-blind Replay LLM
  -> generated source
  -> compiler + strict observer/comparator
```

确定性代码不解释 pi 来生成 kernel，也不逐字段证明 Replay 使用了 pi。代码只能证明 pi 对
Replay **prompt-visible**；实际因果消费只能由最终 TTIR closure 和受控 intervention 提供证据。

当前 `RealizationWitness` 的 `semantic_ref` 接受 `p*` 与 `c*`，`Gate1Artifact` 也会从 J、
interface、C 收集这些 handle 做引用闭合。v3 pi 仅引用 `a0..a2/n0..n4`，没有 `p3/c0/c1`。
所以 v3 直接证明 artifact omission；不能把它写成 parser 禁止 interface/C reference。

## 所有权逐项分类

| source fact | owner | 理由 |
|---|---|---|
| Swiglu value DAG 与输出关系 | J.A/J.B | 改变可观察值 |
| 有效列 extent `n_cols` | J shape/domain | 改变合法元素集合，不能是 pure C |
| row/column logical factorization | J tensor/index domain | 是 whole-tensor 逻辑坐标，不是 compiler fake allocation shape |
| `BLOCK_SIZE`、num_warps | C | 在合法选择下只改变调度 |
| valid harness 中 `p3 == physical row stride` | harness precondition + pi reference | 对连续 view 是表示绑定；不是自由语义 scalar |
| `p3` 参与 source row-base realization | pi | 仅在 J 已有 row/column access 后记录实现依赖 |
| `extsi` 的位置、scalar/vector addptr 分层 | pi residual | Gate1 可观察，但不改变 J denotation |
| `col < n_cols` 的逻辑有效域 | J.B | predicate/bounds 语义归 J |
| guard SSA 的具体 materialization/share | pi residual | 只在 predicate 已由 J 定义后记录实现拓扑 |

若某个 envelope 允许调用者任意改变 `p3`，且这会选择不同逻辑元素，那么 `p3` 就不再只是
representation carrier：其逻辑访问关系必须进入 J。本任务不能用 frozen `p3=1024` 的事实
推广到任意 stride contract。

## 两类反例与边界

### Row-stride / flat

在冻结 precondition `rows=128, cols=1024, p3=cols` 下，两种源码具有相同 whole-tensor J
和 pure C，却有不同 Level-2 realization。一个充分的 pi 必须能区分 runtime `p3` row-base、
scalar/vector address boundary 和 flattened constant offset；但它不能首次定义有效元素或二维
逻辑 access。

当前 pi graph 的已定义标签只能说 `realize/transfer/order/share`。任意重复 instance 可以被当作
隐蔽编码，但 contract 没有规定其含义，Replay 不能可靠解码；这种 covert convention 不算
信息充分性。阶段 1 必须先测试“现有 graph + 明确全局消费语法”能否表达 residual，再决定是否
需要一个最小通用 relation。

### LayerNorm pre-mask / post-mask

`predicate/fill/select/mul/reduce` 的 operator 与 value edges 必须首先由 J 的 value DAG/B 表达。
若源码是 `select -> mul -> reduce`，Lift 不能只写一个 reduction masked-contribution，再期待 pi
创造缺失的 select。pi 只能引用既有 J node/value，说明同一个 selected value是否先物化并被
两个 operand 共享。

因此正确的两个测试是：

1. 改 `select/mul/reduce` value edge：J 必须变化；
2. 固定 J，只改 selected value 的 materialize/share：pi 可以变化，Gate1 可区分。

这避免重新引入 `mask_before_mul`、`reduction_mask_placement`、RuleCard 或第二份 operator IR。

## Fusion / copy / atomic / async 边界

- Fusion：逻辑 value/effect DAG 在 J；materialize、reuse、region placement 在 pi；tile/grouping 在 C。
- Copy：读写哪些逻辑值在 J；direct/vector/bulk/TMA event topology 在 pi；tile/vector width 在 C。
- Atomic：RMW、conflict、memory-order/scope 中会改变允许结果的部分在 J；local aggregation 与
  compiler materialization 才能在 pi。
- Async：producer-consumer correctness、visibility、可观察 happens-before 在 J；stage/prefetch
  在 C；issue/commit/wait/buffer-swap source topology 在 pi。一般 async contract 仍未闭合，
  Task07 不宣称一般支持。

## Stage 0 决策

1. 不改 Gate1 strictness，不改 v1/v2/v3 历史证据。
2. 不把 `n_cols` 或逻辑 index map搬入 pi。
3. 先修 semantic constexpr 与 pure C 的分类，以及 J logical shape 对 compiler allocation shape
   的错误绑定。
4. pi 保持 Lift LLM 直接写、Replay LLM 直接读；确定性核心只验证。
5. 只有离线 pair 证明现有图在定义良好的语法下仍无法区分 residual，才扩 schema。
6. normal closure 前不运行 masked-J；masked-J 只作一次 identifiability witness，不宣称普遍证明。
