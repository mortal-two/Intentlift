# Layer A / B / C 三层表示

## 1. 完整语义对象

IntentIR 从优化 Kernel 的 SPMD realization 中恢复 portable whole-tensor program。
最终不变量是：

```text
A = logical value semantics
B = observable execution and legality contract
C = replaceable implementation strategy
J = A + B
```

因此 A 单独不是完整 Kernel 含义。A 与 B 都是绑定的；C 不改变 J，可由目标后端
重调，也可在 Gate1 中作为同前端、同 C 结构比较的冻结实验条件。

## 2. Layer A：逻辑值语义

A 回答“在合法输入上逻辑输出值是什么”，至少包括：

- 输入／输出、dtype、symbolic shape 与 semantic axes；
- tensor value DAG；
- elementwise、reduction、contraction、scan／semantic loop；
- transpose、reverse、reshape、slice、pad、concat 等 logical view；
- gather/scatter 等 data-dependent logical indexing；
- reduction axes、结果 shape、accumulator/result dtype 与会改变值的数值 policy；
- semantic recurrence 的 init/carry/yield/result；
- 逻辑返回值与 SSA value 的映射。

判断标准：修改该字段是否会改变 logical tensor denotation。若会，它必须位于 A，
或位于 B 中一个明确、可执行且所有消费者都必须遵守的 value-driving contract；不得只
存在于自由字符串、source pointer 表达式或 schedule hint 中。

## 3. Layer B：可观察执行与合法性合同

B 回答“实现 A 时必须满足哪些跨目标正确性条件”，至少包括：

- logical access domain 与 bounds obligation；
- load predicate/fill 与 store guard；
- operand-to-consumer projection；
- destination binding、alias、mutation、partial/conditional writeback；
- read/write/atomic effect 与可观察 ordering；
- duplicate-index、conflict 与 atomic requirement；
- reduction identity、masked contribution、reassociation/order freedom；
- observable scope、storage identity、lifetime 与跨 iteration state；
- 不可被 C 擅自改变的 numerical tolerance/order contract。

B 不是注释，也不是 optimization hint。correctness-relevant B 必须使用闭合、类型化
schema；extension map 不得承载正式语义或 backend obligation。

## 4. Layer C：可替换实现策略

C 保存不改变 J 的 realization／性能选择，例如：

- tile/block/thread/warp 数；
- vector width、unroll、pipeline stages；
- shared-memory staging、prefetch、double buffer、warp specialization；
- source-local rank insertion、address association、mask materialization；
- target-local physical layout 与已审计的实现 variant。

C 是 non-binding schedule／strategy。Backend 可以不用某个提示，但不能借“重调 C”
改变 A 或违反 B。若某种 fast math 会改变可观察数值，允许范围必须先由 A/B 定义；
C 只能在该范围内选择 realization。

## 5. 三类 index 必须分离

一个 `index_expression` 不能同时拥有三种语义真值：

| 信息 | 归属 | 示例 |
|---|---|---|
| logical tensor mapping | A | `y[i] = x[N-1-i]`、transpose、slice |
| data-dependent logical address | A | `y[i] = x[index[i]]` |
| access legality | B | `0 <= index[i] < N`、mask/fill/store guard |
| program/tile 到逻辑坐标的 realization | C / witness | `pid*BLOCK+lane` |
| shared/local physical address | C | swizzle、padding、pointer staging |

推荐使用受控的 affine/data index map，而不是 portable 自由字符串。Portable A/B 中
禁止出现 `tl.program_id`、`threadIdx`、warp/lane id、前端 API spelling 或 source
变量名。

copy 与 reverse 的 logical map 不同，因此 A 必须可区分；不能只靠 op multiset 或
外层 output address 字符串区分。

## 6. 逻辑返回值与 destination 分离

以下是两个不同事实：

```text
A.return_values:
  output Y <- value y

B.destination_bindings:
  output Y -> destination X
  alias_kind = must_alias
  write_coverage = partial
  preserve_unwritten = true
```

单个 `out` 兼容字段不足以表达多输出、in-place、共享 storage、partial/conditional
writeback。新 schema 不得再次把逻辑值和物理写回合并成同一个真值来源。

## 7. Semantic loop 与 schedule loop 分离

- whole-tensor reduction/contraction 的 source tiling loop 属 C／witness，canonical A
  仍是 reduce/contraction；
- online-softmax、scan、RNN state 等真正改变结果的 recurrence 属 A；
- tail mask、state effect、数值顺序义务属于 B；
- tile size、unroll、pipeline 属 C。

canonical J 的 semantic trip 只描述语义轴与 carry。历史 `trip.tile_hint` 只能作为
兼容输入或 lowering attribute，不得继续被解释为 A 的持久化语义。

## 8. Backend 消费规则

每个 B 字段必须登记为以下一类：

| 类型 | backend 责任 |
|---|---|
| value-driving | 必须生成对应行为 |
| legality obligation | 必须证明、生成检查或返回 unsupported |
| optimization permission | 可以保守不用，但不得违反 |

核心纪律：backend 可以不用 permission，但不能忽略 obligation。未知或未消费的
correctness-relevant B 必须 fail closed，并在 capability/conformance record 中标记
`consumed / checked / unsupported`。

## 9. Source-validation witness 不是第四层

以下信息可以进入 RuleRegistry、确定性 replay plan、fingerprint 或诊断记录，但不属于
portable J：

- source span/location；
- 前端本地 `expand_dims`、pointer arithmetic association；
- compiler dump presentation topology；
- source transformation history；
- Kernel 名、provider response、dataset identity、Gate verdict。

π 仍是由 J、ABI、C 与 RuleRegistry 确定性派生的施工计划，不是 Layer D，也不拥有
独立语义真值。

## 10. 典型因式分解

### Row softmax

- A：reduce-max、subtract、exp、reduce-sum、divide；
- B：tail predicate、max/sum masked contribution、store guard、数值合同；
- C：block size、warps、vector width。

### Tiled matmul

- A：`contraction("mk,kn->mn")` 与 accumulator dtype；
- B：M/N/K bounds、tail contribution、destination、允许的 reassociation；
- C：BM/BN/BK、MMA、staging、pipeline。

源码 K-tile loop、pointer advancement 与局部 rank insertion不是额外 A loop。

### FlashAttention

- A：QK contraction、online max/sum recurrence、output carry、normalization；
- B：causal/tail mask、LSE/output mapping、数值与多输出合同；
- C：Q/K tile、MMA、shared staging、software pipeline。

一个源 K 循环可同时承载 A recurrence 与 C tiling，必须因式分解，而不是整体归入
任一单层。

## 11. Quality Check

- [ ] 完整语义对象是否明确为 `J=A+B`？
- [ ] logical view/index 是否位于 A？
- [ ] bounds/mask/fill/store guard 是否位于 typed B？
- [ ] pid/tile/lane 与 physical layout 是否留在 C／witness？
- [ ] output value、destination、alias、writeback 是否分离？
- [ ] semantic recurrence 与 tile loop 是否分离？
- [ ] backend 对每个 B obligation 是否 consumed/checked/unsupported，而非静默忽略？
