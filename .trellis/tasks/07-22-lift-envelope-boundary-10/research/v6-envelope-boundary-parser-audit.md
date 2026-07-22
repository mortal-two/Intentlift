# Task10 阶段 0：v6 Envelope / Boundary Parser 真值审计

## 结论

v6 的唯一 Lift 调用返回了完整、未截断的 provider response，但模型把四个 root section 嵌进了
`j`；本地 parser 正确拒绝。仅在内存中诊断性 hoist 后，三个 access 又因
`index_expression=null` 被拒绝。这里有两个通用 wire 生成缺口，但没有新的 Kernel 语义类别，
也不授权 response repair。

Task10 的最小改动应是：把 root/J 字段声明和 access ownership grammar 各维护一份，并投影到
parser、Responses schema description 与 Lift prompt。固定代码继续只验证；源码中是否原本存在
mask/fill/domain，不能由一个孤立 JSON artifact 推断，最终仍由 source-aware Lift 提议与 Gate1
strict topology 检验。

## 冻结 v6 证据

路径：

```text
artifacts/real_llm_batch_28_v6/recovery/liger_swiglu_forward/
```

| 对象 | SHA-256 |
|---|---|
| receipt | `21fb0a2e548f75c619e0339664092defb554dacef5b6a132b54d5db0aed04b9d` |
| request | `cc44b107c1bad9ffa89f9489de539dc078bd81f167d297ec149f568f30cf3d9f` |
| parsed response text | `4ec0e7f4b8ef75ac0e0794ece900e8cb00e7e7575d2823a0aec74ce59f7d01fb` |
| raw provider response | `60c6242c419c79bbd2215842b8e88af8d93ef4fa0588915601fc669ffa1f0700` |

provider 状态为 `completed`，`incomplete_details=null`，`error=null`；唯一 Lift invocation 的
usage 为 input `10567`、output `18349`、reasoning `16335`、total `28916`。Replay、Gate1 和
masked-J 都是 0 次。

返回对象的实际 key sets 是：

```text
root = [schema, j]
j    = [kernel_id, intent, node_ids, boundary, scopes, storage,
        interface, schedule, pi, location_map]
```

`j.pi` 本身仍是有效的 `RealizationWitness`，包含 15 instances、15 routes，以及各三个
`address_base/address_lane/transfer_direct`。所以 v6 失败不能归因于 address tag capacity。

## 唯一 ownership 表

### LLM 可见 Gate1Artifact root

模型必须一次输出且只能输出：

```text
schema, j, interface, schedule, pi, location_map
```

`source` 和 `provenance` 是本地 harness/private persistence 字段：parser 可以接收，模型 response
schema 不允许输出。它们不属于上述六字段的 wire 同构范围。

### SemanticArchive `j`

canonical wire 必须一次输出且只能输出：

```text
kernel_id, intent, node_ids, boundary, scopes, storage
```

当前 Responses schema 和 `to_json()` 已输出全部六项，但 `SemanticArchive.from_json()` 只要求前
四项、把 `scopes/storage` 默认为空。这是实际 parser/schema 漂移。Task10 将 JSON wire 收紧为六项
必需；dataclass 直接构造时的默认值只是内部 Python API，不再扩大 persisted wire 接受域。

### Boundary

`Boundary.from_json()` 当前已经要求以下十项全部存在，并拒绝未知项：

```text
domains, accesses, broadcasts, aliases, effects, output_mappings,
reductions, index_bounds, updates, result_relations
```

其中 `domains` 必须是 object，其余九项必须是 array。数组或 object 可以为空；parser 不可能仅凭
空值知道源码曾经有何逻辑边界。

### BoundaryAccess

允许字段是：

```text
access_id, kind, tensor, predicate, fill_value, node_ref, index_expression
```

虽然 `from_json()` 表面只列 `access_id/kind/tensor` 为 required，`AccessKind` 的全集恰好是
`load/store/update`，而 `__post_init__()` 对这三类全部要求非空 `index_expression`。因此实际接受域
是每个 access 都必须提供：

```text
access_id, kind, tensor, index_expression
```

这与 rank 无关。当前 schema 把 `index_expression` 列入 required、但类型仍 nullable，故
`null` 会被 schema 接受后再被 parser 拒绝；Task10 应把 schema 投影收紧为非空 string。

`predicate/fill_value/node_ref` 均为结构可选字段。不能添加诸如“所有 load 都必须有 fill”或
“predicate 非空必有 fill”的伪全局规则，因为 unmasked load、store 和 update 的合法形态不同。

## 硬验证与 semantic ownership 的边界

固定 schema/parser 可以机械拒绝：

- root wrapper、缺字段、未知字段和 section misnest；
- J 缺 canonical 字段或出现 root-only sibling；
- Boundary 十项不完整或类型错误；
- 任意 load/store/update access 缺失、null 或空 `index_expression`；
- 已有 enum、引用、scope/storage、typed relation 的内部不一致。

固定 schema/parser不能从单个 artifact 机械证明：

- 源码是否原本有 logical domain 或 predicate；
- masked-load fill 是否被遗漏或改值；
- predicate 是否被错误简化成 tautology；
- frozen invocation 恰好全覆盖时，模型是否错误删除了逻辑 mask。

后四项的正确证据链是：

```text
source-aware Lift preserves logical boundary in J
  -> source-blind Replay consumes J
  -> Gate1 compares source/replay normalized compiler topology
```

π 禁止携带 domain、predicate、fill 或 index formula，也不能在 parser 后补回它们。normal Gate1
未 hard-close 时不运行 masked-J；masked-J 不是 source-boundary preservation 的替代证明。

## v5 正例与 v6 负例的用途

冻结 v5 response 只作为正向 conformance fixture：root 六项正确，J 六项正确，并含
`d0`、`[x0,x1]`、`x1 < p4` 与 masked-load fill `0`。它不能被当作 v6 repair 模板。

冻结 v6 response 只作为负向 fixture：原字节与 hash 不变；parser 必须先因 root misnest 拒绝。
诊断性 hoist 只能在测试内的深拷贝上暴露第二个 `index_expression` 拒绝，绝不能保存、Replay 或进入
正式 runner。

## 阶段 1 最小实现决定

1. 在 parser 所在模块声明 model-visible root 六字段、private root 两字段、J 六字段、Boundary 十字段、
   access 四个 wire-required 字段；parser 和 schema 从这些常量取值。
2. 提供一个短小 envelope renderer，明确 `j` 在 `storage` 后关闭，四个 section 是 root sibling；
   提供一个 boundary access renderer，区分结构硬规则与源码逻辑所有权规则。
3. 两个 renderer 同时进入 schema description 与 Lift prompt；不加入 Swiglu、LayerNorm、rank、源码名、
   operator 或 frontend 分支。
4. 不增加 hoist、alias、compatibility upgrade、response assembler 或 retry。
