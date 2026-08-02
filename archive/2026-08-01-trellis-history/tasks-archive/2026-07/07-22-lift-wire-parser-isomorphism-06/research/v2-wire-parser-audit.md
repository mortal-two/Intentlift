# Protocol-v2 Swiglu Wire / Parser 差异审计（冻结证据）

## 证据边界

本审计只读取 Task05 冻结文件，不修改或重写 response：

- request SHA-256：`96d97c02ac6957b77d91b52000cdd24632a2af8c7db72fdf7eb649b1cdc9ba57`
- response text SHA-256：`d33438060b0154787027c6bf6fa49a6bbcd1d9542c87ef615531b348c8a7bb35`
- raw envelope SHA-256：`25bf7dce740d7a6e756dbbafbc096ede82f756d5070554e5545547942a6abbb6`
- case receipt SHA-256：`82029abd631873847de48babdb1c9ebfad3b3006585fd95d8053f19d4d6e0901`

使用 request 中实际发送的 `text.format.schema` 运行 Draft-7 compatible validation，完整
response 的 error count 为 0。因此下面的拒绝不是“模型违反 wire”，而是 wire 接受域比
权威 parser 更宽。

## 实际 response 的全部结构差异

`j.intent.intent_function.body` 中五个 op 均不是 `loop`，却都声明了 `scope_id: "q0"`：

| 路径 | op | parser 要求 |
|---|---|---|
| `body[0].scope_id` | `cast` | non-loop 必须 absent |
| `body[1].scope_id` | `map_unary` | non-loop 必须 absent |
| `body[2].scope_id` | `map_binary` | non-loop 必须 absent |
| `body[3].scope_id` | `cast` | non-loop 必须 absent |
| `body[4].scope_id` | `map_binary` | non-loop 必须 absent |

v2 wire 的单一 generic IntentOp schema 把 `scope_id` 定义为所有 op 的 nullable optional
property；完整 parser 的 scope ownership 校验只允许 semantic `loop` 引用一个
`loop_region` scope。这五个路径是该 response 中全部 structural mismatch。

独立组件检查结果：原始 intent 可通过 `_strict_intent` 的局部 shape/type 检查；Boundary、
KernelInterface、ScheduleSpec、RealizationWitness、LocationMap、PortableScope 和 StorageObject
也都能独立反序列化。说明第一个错误来自全局 scope 合同，而不是其他对象的 JSON shape。

## 与结构漂移明确分开的语义拒绝

在内存深拷贝上仅移除上述五个 non-loop `scope_id`，用于继续诊断（该副本不保存、不参与
结果），完整 parser 依次发现：

1. `q0.definitions` 多声明 `t2`。function scope 的 derived definitions 是输入句柄与实际 SSA
   定义；逻辑 output handle 不能仅因是 destination 就成为 definition。
2. `b3` 至 `b6` 把 `v0..v3` 普通 SSA 临时值声明成 `write_once` storage，但它们没有唯一
   semantic write event。删除这四个诊断性 storage 后，artifact 可完整解析；`b0..b2`
   的 input/output storage 合同有效。

这两类是跨对象引用/derived-set/storage-lifetime 语义，JSON Schema 不能完整表达，应继续由
权威 parser fail closed。下一协议只能通过统一、全局 grammar 清楚说明，不得自动 repair。

## v2 Schema 中同时发现的潜在结构漂移

虽然本 response 未触发，当前 schema/parser 组合还存在以下全局接受域差异，必须一次收回：

- schema 允许 non-loop 携带 `carry/trip/scan/yield`；parser 拒绝；
- schema 允许 `op="loop"` 缺少 `result/scope_id/carry/trip/scan/body/yield`；parser 拒绝；
- schema 允许 loop result 为 scalar/null/空数组、carry 为空、scope/trip 为 null；parser 要求
  non-empty positional loop contract；
- schema 没有 `id`，而 canonical `IntentOp.to_json()` 与 parser 支持匿名 node id；这是
  parser-serializable → wire completeness 漂移；
- schema 与 parser 都曾允许 non-loop `body`，但 active structured-loop spec 没有第二种
  region-bearing variant。canonical 合同应只让 `loop` 暴露 region 字段。

## 结论

Task06 不能只在 prompt 增加一句“不要带 scope_id”。下一阶段必须以同一份 immutable
variant/field declaration 同时生成：

1. loop/non-loop structural validator；
2. recursive Responses JSON Schema；
3. 给 LLM 的全局 structural grammar。

operation-specific `attrs` 继续开放；合同只识别全局 discriminator `op == "loop"`，不是
operation registry，更不是 per-kernel rule。
