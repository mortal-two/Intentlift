# 算子签名(具体形态)

本文件把 [atomic-ops.md](atomic-ops.md) 的算子集落到**具体 JSON / dialect 形态**。这是规划稿,不是最终 TableGen(`.td` 属于代码);用于钉死 op 的 operands / results / attrs / 层归属。

## JSON 顶层结构

```json
{
  "intent_function": {
    "name": "flash_attention_fwd",
    "inputs":  [{"name": "Q", "dtype": "f16", "shape": ["B","H","Sq","D"]}],
    "outputs": [{"name": "O", "dtype": "f16", "shape": ["B","H","Sq","D"]}],
    "layer_c":  { "tile": {"Sk": "BK", "Sq": "BQ"}, "vector_width": 128 },
    "body":    [ /* op 列表,SSA 命名 */ ]
  }
}
```

- `layer_c` 是**非绑定** schedule hint(后端可覆盖)。
- `body` 里每个 op 产出一个 SSA 名;Layer B 约束挂在 op 的 `layer_b` 字段。
- destination-passing(决策二)在 JSON 里用可选 `out`/`in_place` 字段表达,转 dialect 时落成 `outs` 操作数。

## 单个 op 的通用形态

```json
{ "op": "<name>", "result": "<ssa>",
  "operands": ["<ssa>", ...],
  "attrs":   { /* op 专属 */ },
  "type":    { "dtype": "...", "shape": [...] },
  "layer_b": { /* 约束:bounds / mask / reduction_assoc / ... 可选 */ },
  "out":     "<ssa>"  /* destination-passing,可选 */ }
```

## 核心算子签名表

| op | operands | 关键 attrs | result | 层 |
|---|---|---|---|---|
| `map.unary` | `x` | `fn`: exp/log/sqrt/recip/... | tensor | A |
| `map.binary` | `lhs, rhs` | `fn`: add/mul/div/max/...; `broadcast` | tensor | A |
| `select` | `cond, a, b` | — | tensor | A(数据依赖值,图固定) |
| `cast` | `x` | `to_dtype` | tensor | A |
| `reduce` | `x` | `fn`: sum/max/...; `axes`; `assoc`: associative \| float_nondeterministic | tensor；声明 shape 可 keepdims 或 collapsed | A 语义 + B 结合性 |
| `contraction` | `a, b` | `subscripts`: einsum(如 `"bhqd,bhkd->bhqk"`) | tensor | A |
| `scan` | `x, init` | `fn`/`region`; `axis` | tensor | A |
| `transpose`/`reshape`/`broadcast`/`slice`/`concat`/`pad` | `x` | 形状参数 | tensor | A |
| `gather` | `data, indices` | `axis` | tensor | A(数据依赖地址) |
| `scatter` | `data, indices, updates` | `reduce`: none/add/mul/max | tensor | A + B 结合性 |
| `segment_reduce` | `data, segment_ids` | `fn` | tensor | A |
| `loop` | `<carry inits>` | `trip`{axis, tile_hint}; `carry`; `yield` | `<carry outs>` | A 语义 + C tiling |

## `loop` 的具体形态(承载递推)

```json
{ "op": "loop", "result": ["m","l","O"],
  "carry": [ {"name":"m","init":"NEG_INF"},
             {"name":"l","init":"0"},
             {"name":"O","init":"zeros"} ],
  "trip":  { "axis": "Sk", "tile_hint": "BK" },   // tile_hint 属 Layer C
  "body":  [ /* 用 carry 的 op 列表 */ ],
  "yield": ["m_new","l_new","O_new"] }
```

- `carry` = 跨迭代携带的状态(Layer A 递推语义)。
- `trip.tile_hint` = 切块大小,是 Layer C(非绑定);`trip.axis` = 被流式处理的轴。
- 这就是 FlashAttention 因式分解的落点:**递推在 carry(A),tiling 在 tile_hint(C)**。

完整走查见 [flashattention-example.md](flashattention-example.md)。
