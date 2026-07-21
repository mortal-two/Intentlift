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

`constexpr` 只描述前端绑定时机，不决定所有权。非 C constexpr 在匿名 interface 中使用
`p*` handle，具体冻结值只留在 harness：会改变 logical shape/domain/predicate/result
relation 的 consequence 进入 J；只影响 layout/address realization 的 use 只能由 pi 匿名引用。
只有不改变 J denotation 的 tile/block/stage 等 constexpr 才使用 `c*` 并成为 C slot。

## π

π 是 Intent 外的 Gate1 witness，由 Lift LLM 直接产生。它可以表达匿名 realization
instance、region、route、operand slot、order、sharing、placement，以及区分源码实现所
必需的 source-neutral mechanism tag（如 async/direct transfer 或 atomic/privatized
realization）。mechanism tag 不得携带逻辑 operator、combiner、predicate 或数值。

一个 instance 表示一个 compiler-observable realization event，不是一份新语义值。同一
`semantic_ref` 可以有多个 instance，以表达同一匿名 handle 的多级 realization；route 是
instance 间匿名 def-use，order 是必要偏序，sharing 表示一个实际 producer 被多个 consumer
复用。源码中参与 address/transfer/synchronization/placement/share 的 interface/C handle
必须在 pi 中出现并连接到受影响的 event。Replay 必须保留 event 数量、链、fanout、region
和 sharing，不能把它们压平。

这些结构不能当作无定义的 covert code。instance 的意义只来自上述全局合同和它锚定的
J/C/interface handle；缺少 operator/formula 时，Replay LLM 负责从完整 J/C/interface 推理出
前端写法，最终由 Gate1 strict observer 验证，而不是由固定 composer 解码。

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

Gate1 编译时，non-C constexpr 的具体值按源码声明位置从同一冻结 invocation 复制到
Replay；这只是调用绑定，不是 C，也不建立 `LogicalABI`。其 semantic consequence 必须已经
在 J 中；pi 只能引用 representation use，不能补写遗漏语义。
