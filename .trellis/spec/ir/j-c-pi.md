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
任意可执行 comparator；外部验证政策 Q 根据 B 的封闭类别构造实际比较关系。

J access 采用唯一的 canonical fill 合同：

```text
fill_value != null => kind = load and predicate != null
```

无条件 access 的 `predicate` 与 `fill_value` 均为空。parser/schema 可以提前报告该错误，但
`TypeCheck(J,C,pi)` 必须权威拒绝 orphan fill；任何消费者都不得通过静默删除或猜测 predicate
来归一化候选 J。

## J 的构造与标准化

模型先返回尚未接受的 response JSON；固定 decoder/elaborator 成功后才构造出 J：

```text
response JSON -> Decode/Elaborate -> J | deterministic diagnostic
```

因此不存在 `J_wire` 与 `CanonicalJ` 两个对象。接受后的 typed archive 就是唯一 J；解析得到的
ContractExpr AST/axes 只是 J 的私有派生缓存，不能单独序列化或拥有身份。该阶段不读
Source、Harness concrete value、C、π、Gate diff/output 或模型历史，只能完成 closed grammar 中
唯一、语义不变的推导：

- `k0`/`[k0]` 与 `k1,k0`/`[k1,k0]` 分别归一为相同的 structured ordered axes；
- domain/predicate/index formula 解析为 ContractExpr AST，consumer 不再私自解析字符串；
- closed op signature 与 J value graph 唯一确定的 result dtype/shape 被物化；显式类型必须一致；
- 可机械证明的 direct same-handle output identity mapping 归一为唯一省略形式。

canonicalizer 不得猜 operator、缺失 predicate/effect/alias/authorization，不得从 Source 修复 J，也
不得把 general affine expression 偷换成 direct axes。语法无唯一解释、类型冲突或需要语义选择时
fail closed。响应 schema 目前仍以 `index_expression` 文本保存，但 J 的 canonical dump 只写 bracketed vector；
内部消费者读取 structured AST/axes。

因此接受条件准确写成：

```text
Decode/Elaborate(response) = (J,C,pi)
and TypeCheck(J,C,pi)
```

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

`region.placement=program` 只说明 instance 位于单个 program 的 region 中，不表示 Source 读取了
program/grid coordinate。若 Source 的确存在 `program_id`、grid offset 等 compiler-visible stage，
Lift 必须用一个独立 `realize` instance 及 routes 记录；无需新增 mechanism enum。没有该独立
instance 时，Replay 不得因 placement、常见前端写法或 C 自行发明 program coordinate、乘加或
affine address stage。

该 generic relation 的机械区分是：program region 中锚定相应 J domain 的 `realize` instance，其
scalar dependency path 先进入 `address_base` 再进入 transfer，表示 program-coordinate stage；进入
`address_lane` 的 domain/C dependency path 表示 lane-varying stage。Replay 必须保持 base/lane
区分，`U_f` 再把它映射到公共前端 API。该规则不在 π 中保存 `program_id`、axis 公式或前端文本，
也不增加 mechanism enum。

`address_base/address_lane` 标记 address contribution 角色，不等价于两次 pointer addition。Source
若先形成可分离的 program-uniform integer base 与 lane offset，再合并后执行一次 pointer add，
program path 仍进入 `address_base` 并 route 到 `address_lane`；Replay 保留两个 integer stage 和
唯一 pointer add，不能因当前 grid extent 为 1 将 program path 常量化。只有不存在可分离
program-uniform contribution 时才允许 lone `address_lane`。

单个直接锚定 J signature handle 或 C slot 的 `realize` instance，可以由对应参数 binding 及其直接
routed use 满足，不要求为了“显式 materialize”增加恒等赋值。尤其 `c*` 为 constexpr 时，前端
合同可以要求直接 use 以保留编译期类型。若同一 handle 有多个 instance，它们仍必须对应不同的
routed use-site/stage，不能据此全部折叠。

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

`TypeCheck(J,C,π)` 接收已经构造完成的唯一 J，并至少检查 J access 的 canonical fill 合同、refs、route slot/type、observable
J anchor、语义/API/Source/C-value 隔离以及 sensitive authorization。自闭合 `Parse(π)` 只能作为
语法解析器，不能冒充 dependent TypeCheck。

TypeCheck 的结构化错误可以返回 source-aware Lift；检查结果和 sidecar 不得进入 Replay 作为
新的信息通道。schema 为 closed-world：未知字段、未知 relation 与自由文本扩展均 fail closed。

准确的 TypeCheck claim 是：

```text
TypeCheck(J,C,pi) -> FreshLogicalSemDecl(C,pi | J) = empty
```

在合法引用解析回 J 后，可简写为 `SemDecl(J,C,pi)=SemDecl(J)`；这不等于 π 没有统计语义
线索，也不证明 J 或 realization 的行为正确。
