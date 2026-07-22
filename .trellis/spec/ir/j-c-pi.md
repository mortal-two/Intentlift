# J、C、π 与 TypeCheck

> 状态：当前派生规范。上位设计见 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## J：唯一语义对象

```text
J = A + B
```

J 拥有 logical values/state relation、domain、predicate、fill、logical index、effect、alias、
result relation 以及 logical signature。signature 中的 input、output、runtime scalar、destination、
dtype、rank、shape、alias/inout/effect 都有 J-owned opaque handle。

B 同时拥有会改变允许结果关系的 typed/effect authorization 与 comparison class 声明。它不包含
任意可执行 comparator。

## C：一个 implementation-selection 对象

C 只包含在不改变 J denotation 时可替换的匿名 slot、binding kind、value type、concrete
valuation 与必要的 source-neutral selection metadata。C 不得包含 operator、predicate、formula、
logical shape、result semantics、source topology、frontend API 或 per-kernel executable policy。

```text
MaskC(C) = erase(concrete values, private provenance)
```

Replay 只读取 `MaskC(C)`；Harness 将同一 C valuation 绑定给 Source 与 Replay。C 不拆成
`C_source`、binding map、portable hints 或其他顶层对象。

## π：依赖类型实现见证

```text
Gamma_(J,C) |- pi : RealizationWitness(J,C)
```

π 可以表达 region、instance、route、operand slot、order、sharing、placement、materialization 与
受约束 representation refinement。其锚定关系为：

```text
mu : Instances -> Ids(J) union Ids(C)
```

不存在 `Ids(interface)`。需要接口信息时读取 `InterfaceView(J)`，但该投影不增加命名空间、
不序列化成独立 artifact，也不作为 Replay 的额外输入。

π 禁止携带：

- operator/function/reducer、predicate/fill/index formula 或 semantic constant；
- logical shape、result relation、destination/conflict semantics；
- Source text/name/path/span、Kernel/rule/family ID；
- frontend API、C concrete value 或可执行自由文本；
- 任何能在删除 J/C 后获得独立 denotation 的结构。

所有 observable realization instance 必须锚定 J-owned node/access/effect。representation-only
scaffolding 折叠后，π 不得增加、删除或改写 J carrier 的逻辑边。

## Authorization 与 parametricity

```text
RealSig(J) = (Carrier(J), Capabilities(J,B))
```

`Capabilities(J,B)` 只包含 realization authorization，不复制 operator、predicate、fill、reducer
等 semantic labels。在相同 `RealSig(J)` 下，替换 semantic label 不得改变 π 的 TypeCheck 结果。

对 integer width、numeric cast、accumulation dtype、reassociation、atomic form 等 sensitive
refinement：

```text
Sensitive(r)
  => auth_ref(r) in Auth(J,B)
     or kind(r) in GlobalSafeRefinement
```

`GlobalSafeRefinement` 是系统统一维护的封闭安全集合，不由 Lift 或单个 Kernel 修改。两项均不
满足时 fail closed；TypeCheck 不现场运行 SMT、range proof 或一般语义等价证明。

## 最小 TypeCheck

`TypeCheck(J,C,π)` 至少检查 refs、route slot/type、observable J anchor、语义/API/Source/C-value
隔离以及 sensitive authorization。自闭合 `Parse(π)` 只能作为语法解析器，不能冒充 dependent
TypeCheck。

TypeCheck 的结构化错误可以返回 source-aware Lift；检查结果和 sidecar 不得进入 Replay 作为
新的信息通道。
