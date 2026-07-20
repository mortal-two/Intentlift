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

对普通 tiled matmul，源码中的 K 分块循环、accumulator 和 pointer advancement
是单个 whole-tensor `contraction` 的 Layer-C/前端 realization。canonical J
只保留例如 `subscripts="mk,kn->mn"` 的一个 contraction，不再附加 `loop` 或
`reduce`；局部 `[:,None] / [None,:]` 只用于构造二维 access tile，也不产生
`Boundary.broadcasts`。只有循环携带了 contraction 本身之外、会改变 whole-
tensor 语义的递推状态时，才使用 J `loop/scan`。普通 contraction 的 K 累加
同样不产生 `Boundary.reductions`，A/B/out 的地址、mask、fill 与 store guard
分别保存在 Layer B。

`contraction` 不等于一定存在 contracted axis。对于
`subscripts="m,n->mn"` 这类没有任何输入 label 被消去的 outer product，A 层仍
使用现有 contraction op；前端 realization 是 `lhs[:, None] * rhs[None, :]`，
不得引入 K loop、`tl.dot` 或 reduce。是否存在 contracted label 必须直接由
subscripts 判定，不能只凭 op 名选择 K-tiled-dot。

SCRC 中 `reduce.attrs.axes` 始终是全局逻辑张量轴；前端局部 tile 的实际归约轴
记录在 `Boundary.reductions.tile_axis`。例如 `[M,N]` 的逐行归约在 J 中是
`axes=[1]`，但一整个 row 被物化为一维 Triton tile 时，本地归约调用使用
`tile_axis=0`。同理，`Boundary.broadcasts.axes` 描述全局语义 broadcast，不等于
强制生成本地 `expand_dims`；标量归约结果对同一一维 tile 可直接隐式广播。
当 reduction 覆盖输入的全部语义轴且没有显式声明 keepdims 结果时，canonical
结果为 rank 0；它仍可在后续 elementwise 中自然 broadcast。final-stage
`sum(partials) / M` 不得因为前端内部实现而被推成 shape `[1]`。

源码可以用 `scf.for`/Triton runtime range 分 tile 实现同一个 whole-tensor reduce。
当 loop carry 只是同一 reducer 的 partial accumulator 时，canonical J 仍只有一个
`reduce` 且没有 J loop；源码的 serial tile traversal 属于
`C.strategy reduction_tiling=multi_tile_accumulate`，其中 `tile_slot` 固定 tile step，
`accumulator_dtype` 固定 loop carry 类型。只有 carry 含 whole-tensor 语义所必需的额外
递推状态时才使用 J `loop/scan`。该 strategy 不能把全局 axes 当成本地 tile_axis，
且只覆盖 lower=0、upper=global semantic extent、step=单一 constexpr tile slot、
identity-seeded carry 的规范化循环；也不能因当前 specialization 只有一个 tile 而折叠
源码 runtime loop。

floating `reduce(fn=max/min)` 的 NaN 行为采用 number-preference 语义，与 Triton
TTIR 的 `arith.maxnumf/minnumf` 一致：只有一个 operand 为 NaN 时选择数值 operand，
两者均为 NaN 时结果仍为 NaN。production CPU/GPU lowering 必须保持这一语义，
不得改成传播单侧 NaN 的 `maximumf/minimumf`。这项约束由独立 backend regression
与历史 IEEE-special 三方实验固定，不引入新 op；当前 Gate2 不裁决 lowering。

JSON `const` 表示字面量或按声明 shape 的 splat materialization，例如
`{"value":1.0,"dtype":"f32","shape":["N"]}`；它直接降为标准
`arith.constant`，不是新增库级 named op。shape 中的符号必须由具体 lowering／执行
调用显式绑定，零输入函数同样允许产生该输出。

所有 interface `tensor_ref` 的 dtype 必须与被引用 J tensor 的 dtype 一致。
pointer 所指 tensor 可以是 rank 0，此时 compiler view 使用 `shape=[]`；它仍是
按地址读写的 tensor binding，不是按值 scalar 参数，也不得携带 scalar
`value`。
参与数值计算的 runtime scalar 使用 J 的 `shape=[]` 输入表示，并由
Gate1 interface scalar 参数的 `tensor_ref` 连接；只提供 symbolic shape
extent 的 M/N 类参数不是 J 输入。`reduce(fn=mean)` 在前端本地 tile
重放为 sum 除全局归约轴的语义长度，不得用 Layer-C tile 大小作除数。
对 SCRC canonical J，源码 `rsqrt` 使用既有 `sqrt` 后 `recip` 组合表达；
Gate1 只沿用已登记的局部 `1/sqrt <-> rsqrt` TTIR 规范化。
如果 runtime semantic scalar 的真实 ABI dtype 是整数，而数值图需要浮点值，
J 保留整数 `shape=[]` 输入并显式使用现有 `cast(to_dtype=f32)`；不得把接口 i32
偷偷声明成 f32，也不得把当前 specialization 烘成常量。

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
