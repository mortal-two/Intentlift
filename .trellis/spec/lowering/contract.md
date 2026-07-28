# RefEval 与 J-Grounded、Q/D-Certified Lowering-State Search 合同

> 状态：production backend 当前派生规范。上位设计见
> [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与
> [后端最终设计](../../../doc/IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)。

## 1. 两个消费者，共享一个 J

```text
RefEval      : J x logical_inputs -> observable_outputs
CoreLower    : J x optional_C x target x compiler_state -> executable
SearchClose  : J x optional_C x target x Q x D_search x D_cert x opaque_runners
               -> executable + closure_result
```

`RefEval` 是 J-only 的简单语义执行路径。`CoreLower` 是真实 compiler 路径，只读取 accepted J、可选 C
和 target capability。`SearchClose` 是所有 kernel 共用的外层 compile/execute/compare/search 状态机。

以下内容不能进入 core lowering decision：Source/Replay 文本、π、Gate diff、hidden input/output、kernel
name、source path、provider label、历史 benchmark 或 per-kernel recipe。

## 2. 唯一 typed contract semantics

Decode/Elaborate 先把 response 中的 index、predicate/domain、access、value-dimension 和 result type 变为
typed canonical J。RefEval 与 production materializer 都 lower 同一套 `ContractExpr` AST，不各自维护
字符串 parser/matcher 作为语义真值：

```text
response JSON -> Decode/Elaborate -> canonical J
                                      |
                         typed ContractExpr semantics
                           |                       |
                           v                       v
                     RefEval(J,x)              Emit(J,nu)
```

production projection 是 value-free analysis；RefEval projection 可以在一次 invocation 内使用 J-owned
rank-0 logical inputs和完整 dimension valuation证明 access predicate 在 logical domain 恒真。该证明只在
本次调用中存在，不回写 J，也不能被 production core 读取。

投影必须完整消费影响 logical execution 的 domain/access/effect/output/reduction/value-dimension/
access-guard relation，否则在 codegen 前 fail closed。π 专用 realization authorization 不充当 target
permission。

## 3. RefEval 合同

RefEval 必须：

- 逐项消费 J logical signature、A semantics 与 B result relation；
- 对 unsupported op/relation 在执行前返回具体 capability reason；
- 不使用 C、Source/Replay output、case-specific tolerance 或 target result决定语义；
- 只把 observable output 交给外部固定 Q 比较；Q 不进入 RefEval；
- 与 production materializer 共用 typed boundary/index lowering，而不是并存旧 matcher。

当前保守规则包括：

- direct ordered dense coordinates必须与 tensor rank 一致；
- load/store、effect、output mapping必须完整且一致；
- predicate/fill、broadcast、reduction padding 和 consumer extension 必须来自 J；
- value-dimension relation 在 invocation 时检查 J rank-0 value 与 symbolic dimension valuation；
- access guard 必须被显式 materialize/执行或 unsupported；
- alias/inout/update、affine/gathered access、padded mean 等未实现关系 fail closed；
- structured reference 只接受 closed typed op table，不允许图中途 fallback。

runtime predicate proof record 只含 access id、logical input refs 和 domain axes；不包含 raw runtime value，
也不计算 predicate或 value 的内容标识。

## 4. Deterministic materialization

```text
M0 = Emit_v(J; nu)
```

`Emit_v`：

- 不读取 C、Source、π、Gate 或 Q；
- 完整编码 A 的 typed graph；
- 将 B 投影为 structured dataflow、effect、contract op/attr、verifier fact 或 invocation-local analysis；
- 对影响 denotation、safety 或 ABI 且尚未实现的 relation fail closed；
- 不选择高性能实现，不产生第二个 J。

materialization 完成后才执行：

```text
S_C = Attach_C(M0,C,target)
BaseIR(S_C) = M0
Erase_C(S_C) = M0
```

C-derived prior 只影响初始 schedule和访问顺序，不改变 pass applicability、hard boundary 或 target ABI。

## 5. Hard implementation judgement

定义 invocation-local contract environment：

```text
Gamma_(J,t) = ContractView(J,t)
Gamma_(J,t) |- M : HardImpl_t(J)
```

`HardImpl` 覆盖不能由数值 tolerance 放宽的事实：

- typed value/state graph 与 observable carrier；
- domain、bounds 与 memory safety；
- alias/effect；
- update/conflict；
- predicate/output mapping；
- target ABI。

每个 transformation 声明 typed A/IR match、required B facts、current-state precondition、target capability、
transformation result、schedule domain 与 `exact-preserving|observation-changing` class，并满足：

```text
Gamma |- M : HardImpl(J)
and HardPre_rho(J,M,target)
=> Gamma |- rho(M) : HardImpl(J)
```

缺少 hard precondition 时该 transition 不适用；不得从 Q、运行结果、Source 惯例或空 alias list 猜测。

## 6. Concrete lowering-state search

compiler 不发射 implementation family。候选是具体 transformation trace 及其结果 MLIR：

```text
p = (rho_1, ..., rho_k)
M_p = rho_k(...rho_1(M0))
```

候选只按本次显式枚举中的可读 label、trace、schedule 和 actual structure 组织。禁止内容寻址、基于代码
内容的去重、候选性能缓存或以内部身份链做准入。每个显式候选占一个预算单位；compile failure 也占
一个预算单位。

允许的通用 transformations 包括：

- producer/reduction/epilogue fusion与 intermediate materialization elimination；
- sequential/subgroup/workgroup/hierarchical reduction；
- partial/finalize multi-stage decomposition；
- prefix/segmented scan；
- contraction/conv/recurrence 等具有独立 contract 的 rewrite；
- scalar capture/load motion/predicate specialization等 local transformations。

搜索空间定义为：

```text
ReachHard_t(J) = all explicitly generated M whose transitions preserve HardImpl_t(J)
Feasible_(t,Q,D)(J) = {M in ReachHard_t(J) | QDAdmissible(M,J,t,Q,D)}
```

C 只给 `Order_C(ReachHard_t(J))`。C/no-C 必须访问同一显式候选序列；C 不改变 hard universe 或 Q/D。

## 7. 数值观察边界

以下候选可能改变浮点观察：

- reduction tree/order；
- FMA/contraction；
- internal accumulator dtype；
- target native approximation；
- typed contraction 上的 IEEE/TF32/MMA；
- floating update order。

只要它们保持 HardImpl，可以标记为 `observation-changing` 并进入统一 Q/D evaluator。它们不修改 J，
也不能从 result tolerance 反向授权 memory/effect/ABI transformation。

搜索期：

```text
Close_Q(Target(M,x), RefEval(J,x)) for x in D_search
```

固定最终 winner 后，在独立 D_cert 上逐边直接检查：

```text
Close_Q(Target, Source)
Close_Q(Target, Replay)
Close_Q(Target, RefEval(J))
```

三边不能依靠 tolerance 传递。D_cert 结果不得回流候选生成、winner retry 或 comparator 选择。

## 8. 实验臂

```text
C-Fixed:
  canonical structural path + deterministic target projection of C
  one candidate, no search

Schedule-Only:
  same canonical structural path
  schedule search only

IntentIR:
  same J
  complete Schedule-Only spine
  additional structural/numeric states
  independent schedule search per state

Erase-C:
  same IntentIR candidate sequence, Q/D and budget
  only traversal order changes
```

主指标：

```text
G_schedule = latency(C-Fixed) / latency(Schedule-Only)
G_lowering = latency(Schedule-Only) / latency(IntentIR)
```

二者从同一次 candidate timing run 的 fresh finalist samples 计算。C-prior 另报告 fixed-budget best 与
candidates-to-95%。

## 9. Backend result schema

search/result 只记录直接实验事实：

```text
kernel/case label
J-declared operator, shape, dtype, layout
concrete extent valuation
target/device/compiler configuration
transformation trace and actual structure
requested and actual schedule
launch plan and code size
hard admission/rejection reason
D_search direct comparison
D_cert three direct comparisons
latency samples and median
```

后端不计算、校验或输出 J/C、input、case、MLIR、toolchain、object、binary、spec 或 report 的 hash/digest，
不产生 artifact freeze/receipt/lineage，不保留兼容入口。accepted artifact 历史字段不参与 backend logic。

## 10. J-facing target execution

```text
L = LowerTarget(LoweringRequest(J,C,target), concrete_extents)
O_target = ExecuteTarget(L, logical_inputs)
```

`L` 直接持有 materialized J、Intent MLIR 与编译后的 executable。execution 使用 J logical handle 的精确
input set，shape/dtype/device 来自同一 J 发射的 ABI。已编译 executable 在同一次 benchmark 中由 prepared
launch closure 反复执行；运行阶段不得重新生成 J、替换 MLIR 或调用 compiler。

该 execution edge 是通用 D_search filter 的一部分，不是新的 SCRC Gate，也不修改 SCRC verdict。
