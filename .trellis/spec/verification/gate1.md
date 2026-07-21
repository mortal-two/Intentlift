# Gate1：source-blind 构造闭合

设 `R_f=ReplayLLM(J,π,C,Context_f)`，Gate1 在声明 observer `F_f^(k)` 下要求：

```text
F_f^(k)(Source; C) = F_f^(k)(Replay; C)
```

同时必须保存 source isolation、compiler version、C、observer level、两份 dump/fingerprint
和 comparator receipt。

源码参数 spelling 不属于语义，也不要求 Replay 恢复。observer 在比较前按声明位置把两侧
Python 签名归一成同一组匿名 handle；但参数个数、runtime/constexpr 位置、kind、rank、
dtype 与 C 的具体值都必须 hard-close。constexpr C 按声明位置绑定，frontend compiler
option 只可由纯 C slot 的 source-neutral `role` 映射到固定公共选项。

## Observer 等级

- Level 0：可构建、接口与 specialization；
- Level 1：AST/intrinsic/mask/atomic/barrier 等必要结构；
- Level 2：规范化中层数据流、predicate/fill/effect/realization topology。

当前 Triton strong path 使用 Level 2 hard closure。新前端可以以 Level 0/1 接入，但不得
把较弱 observer 的结论命名为 Level 2 等价。observer 强度只影响 Gate1 claim，不改变
J、π、Replay 或 Gate2。

Level 2 的 alpha-normalization 只消除函数/参数/SSA spelling 等表示噪声；它不放宽规范化
后的 op multiset、数据流边、reduction、predicate/fill/effect 拓扑硬一致性。

## 在线 anti-leakage

仅在 normal Gate1 已闭合后，允许一次 masked-J Replay。它保留匿名 π、纯 C 和相同的
公共前端文档/工具能力，遮蔽 J 语义；若仍 hard-close，则 representation partition
审计失败。动态检索 trace 必须进入 receipt，但核心系统不能按当前 op/pattern 预选材料。

masked-J 不得看到 selected rules、C.strategy、semantic rule ID 或 kernel-specific API
mapping。其他 intervention/ablation 不属于在线 Gate1。

## 不属于 Gate1

Gate1 不执行数值语义，不以 Gate2 为标签，不由 production lowering 授权，也不通过
composer/RuleCard 建造 Replay。
