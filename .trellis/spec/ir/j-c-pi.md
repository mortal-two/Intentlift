# J、C、π 所有权合同

## J

```text
J = A + B
```

J 是唯一具有 Kernel denotation 的对象：A 描述逻辑值/状态关系，B 描述真正改变可观察
结果的边界。J 使用按声明顺序规范化的匿名 handle，不保存源码 spelling。

## C

C 是独立、可替换的纯调度提示。改变合法 C 不得改变 J denotation。C 不允许有
`strategy`、source topology、predicate、operator、API spelling 或 per-kernel policy。
canonical C 只有 `{slots, values}`；每个 slot 由匿名名、kind、source-neutral 调度 `role`、
value type 和 required bit 构成。`role` 可以表达 `tile_extent/num_warps/vector_width`，不能
表达 mask placement、operator、predicate 或 source family。

`constexpr` 只描述前端绑定时机，不决定所有权。会改变 logical shape/domain/predicate/
result relation 的 constexpr 是 J 的 semantic specialization parameter：它在匿名 interface
中使用 `p*` handle，具体冻结值只留在 harness，绝不进入 C。只有不改变 J denotation 的
tile/block/stage 等 constexpr 才使用 `c*` 并成为 C slot。

## π

π 是 Intent 外的 Gate1 witness，由 Lift LLM 直接产生。它可以表达匿名 realization
instance、region、route、operand slot、order、sharing、placement，以及区分源码实现所
必需的 source-neutral mechanism tag（如 async/direct transfer 或 atomic/privatized
realization）。mechanism tag 不得携带逻辑 operator、combiner、predicate 或数值。

π 不允许声明：

- operator/function；
- constant、dtype 或 shape 语义；
- predicate/fill；
- reduction combiner/identity；
- logical index map；
- output/destination/conflict policy；
- 具体 frontend API、源码名或自由文本实现片段。

固定代码只做 schema、唯一性和引用存在性检查。它不得根据 J/C 推导 π、补步骤、排序、
选择规则或规范化成某种施工算法。

## HarnessContext

源码函数名、真实参数名、具体值、device、grid 和 invocation metadata 只存在于 harness。
逻辑参数按位置连接 J；不建立第二份 ABI 语义对象，也不把实名映射交给 Replay。
Gate1 observer 同样按位置 alpha-normalize 源与 Replay 的参数 spelling；arity、kind、rank、
dtype、constexpr 位置与 C 值仍是硬合同。

Gate1 编译时，semantic constexpr 的具体值按源码声明位置从同一冻结 invocation 复制到
Replay；这只是调用绑定，不是 C，也不建立 `LogicalABI`。其 logical consequence 必须已经在
J 中，pi 无权补写。
