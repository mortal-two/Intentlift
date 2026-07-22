# J / IntentOp 语义核心

> 状态：当前派生规范。历史实现审计见 [IntentOp 健壮性审计](../../../doc/source-notes/IntentOp_健壮性审计_2026-07-21.md)，但审计材料不具有规范权威。

## 目标

J 使用小型、可组合、可执行的语义构造表达 Kernel denotation，而不是枚举 Kernel family 或
前端 API。当前目标构造包括：

1. elementwise expression；
2. logical index/map；
3. aggregate；
4. semantic recurrence；
5. logical update。

支持 claim 必须同时来自真实 parser/TypeCheck consumer、J-only RefEval、Gate1/Gate2 与相关
backend conformance；schema 能保存字段不等于系统已经支持该能力。

## 所有权

- logical computation、index relation、observable state transition 与 signature 属 J；
- source fusion、materialization、barrier、atomic instruction、async copy 与 pipeline topology 属 π；
- tile、vector、stage、prefetch、buffer 等可替换选择属于 C；
- conflict/order/rounding/tolerance 等会改变允许结果的事实属于 B。

具体 atomic、CAS、privatization 或 frontend spelling 不得进入 J。B 可以授权某种 realization，
但 π 只能引用该 authorization，不能复制其语义。

## Typed boundary

当前语义核心保留以下通用关系方向：

```text
IndexBounds(node, data, indices, axis)
LogicalUpdate(node, base, indices, values, axis,
              combiner, duplicate_policy, order)
ResultRelation(output,
               exact
               | tolerance(parameters)
               | allowed_result(relation))
```

这些关系必须由 RefEval、comparator 或 production backend 实际消费。缺少必要 boundary、consumer
不支持、关系参数非法或 authorization 不完整时必须 fail closed，不能退回 case-specific Python
oracle 或旁路 metadata。

## Comparator class

B 只能从系统支持的封闭 comparison class 中选择并声明必要语义合同。系统比较协议根据 dtype、
reduction extent、reassociation 权限等构造实际 comparator，并约束 tolerance、NaN/Inf、
allowed-result 与 atomic 相关参数。Lift 不得写入任意 comparator 或自行放宽及格线。

## 禁止恢复的旧旁路

- per-op `layer_b` 或第二份 B；
- per-op `out` 绕过函数级 result/destination relation；
- J 中的 schedule tile、atomic/barrier/async realization；
- 以 named op、RuleCard 或 Kernel recognizer 代替可组合语义；
- 只记录“已消费”而实际忽略 B relation。

新增语义构造必须先给出 denotation、typed operands/results、授权边界、RefEval 行为与 fail-closed
能力，再进入代表性 canary；不得根据单个失败样本直接扩充 ontology。
