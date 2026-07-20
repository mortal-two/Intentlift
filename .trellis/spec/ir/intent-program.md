# IntentProgram 中立编译合同

`IntentProgram` 是 Gate、frontend 与各 target backend 之间唯一允许承载完整
`A+B+ABI(+optional C)` 的 active hand-off。它是 verification-neutral 编译输入，不是
Gate receipt，也不授权 backend 根据某个 Gate verdict 跳过合同检查。

## Canonical wire schema

```json
{
  "schema": "intentir.program.v1",
  "intent_program": {
    "function": {
      "name": "row_sum",
      "inputs": [{"name": "x", "dtype": "f32", "shape": ["M", "N"]}],
      "outputs": [{"name": "out", "dtype": "f32", "shape": ["M"]}],
      "body": [
        {
          "id": "n0",
          "op": "reduce",
          "result": "value",
          "operands": ["x"],
          "attrs": {"fn": "sum", "axes": [1], "accumulator_dtype": "f32"}
        }
      ],
      "returns": [{"output": "out", "value": "value"}]
    },
    "boundary": {
      "domains": [],
      "accesses": [],
      "broadcasts": [],
      "destinations": [],
      "aliases": [],
      "effects": [],
      "reductions": [],
      "scopes": [],
      "storage": []
    },
    "interface": {
      "function": "row_sum",
      "arguments": [],
      "outputs": ["out"]
    },
    "schedule_hint": {
      "values": {"BLOCK_N": 128},
      "grid": ["M"],
      "strategies": []
    }
  }
}
```

Canonical document 必须 strict parse：unknown top-level、unknown contract kind、unknown
predicate expression、legacy `op.layer_b`、destination 语义上的 `op.out`、function
`layer_c` 和 `trip.tile_hint` 均不得进入 active lowering 输入。

## A/B/C 与摘要

- `function.body + function.returns` 是 Layer A；logical permutation/data index、全局
  reduce axes、accumulator/result dtype 都是硬语义。
- `boundary` 是 Layer B；mask/fill、semantic broadcast projection、destination、alias、
  effect、write coverage/conflict、reduction corner case 与 scope/storage lifetime 都使用
  closed record。
- `interface` 是 target-neutral logical ABI；参数 tuple 顺序不可重排。
- `schedule_hint` 是可替换 Layer C；grid、tile、compiler value 与 source strategy 只提供
  hint，backend 可用 retuned C 替换。

实现提供三个不同摘要，禁止混称：

- `j_digest = SHA256(canonical(function + boundary))`；
- `contract_digest = SHA256(canonical(J + interface))`，backend conformance 必须绑定它；
- `program_digest = SHA256(canonical(J + interface + optional C))`。

只改变 C 时，`j_digest` 与 `contract_digest` 必须保持不变；logical index、return
mapping、accumulator/result dtype、destination、alias、conflict 或 reduction policy 任一
变化都必须改变前两者中的适用摘要。

## Predicate 与 index ownership

`domain/access/effect/destination` predicate 使用 `ContractExpr` 的封闭 AST。v1 只接受
常量、symbol、算术、布尔、比较、`ceil_div/min/max`；任意 call、attribute、source
helper 或 frontend API 均拒绝。历史自由字符串只有能无损解析成该 AST 时才可迁移。

每个 access 只记录 index ownership：

- `logical_node`：logical gather/scatter 等 index 已由所引用的 A node 定义；
- `realization_only`：source 的 pid/tile/lane/physical address expression 属 C/witness，
  不进入 J。

不得把 source `index_expression` 原样复制成 B 来伪装 logical semantics。

## Backend obligation ledger

每项 B 会展开为稳定的 obligation：

- `value_driving`：fill、broadcast projection、reduce axes/dtype/numerical policy；
- `legality`：predicate、bounds、write coverage、conflict、lifetime；
- `permission`：destination、alias、effect、reassociation/mutation permission。

Backend 对每个 required obligation 必须明确返回 `consumed`、`checked` 或
`unsupported`。缺项、额外项、重复项和 required `unsupported` 均在 codegen/launch 前
fail closed；默认忽略 unknown B 不存在。

## 依赖方向与 Gate adapter

```text
verify.gate1 -> intentir.ir
frontend     -> intentir.ir
backend.*    -> intentir.ir
```

禁止 `intentir.ir -> verify` 或 `backend.* -> verify.gate1/gate2`。Gate1 单向 adapter
只迁移 J、logical ABI 与 optional C；`kernel_id/node_ids` 的 archive 外壳、source、
location map、plan/replay、provenance/provider/Gate status 全部丢弃。无法无损迁移的
legacy correctness fact直接拒绝 hand-off。

