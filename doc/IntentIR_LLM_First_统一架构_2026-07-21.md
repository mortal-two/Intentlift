# IntentIR：LLM-first、Verifier-closed 的统一架构与验证设计

> 状态：2026-07-21 起的规范性设计主文档
> 目标：统一系统哲学、`J/C/π`、Gate1、Gate2、开放前端、在线成本与迁移边界
> 口号：**不把碎片化生态编进系统，只把可验证性编进系统。**

## 1. 最终判断

IntentIR 不是“用 LLM 辅助传统编译器开发”，也不是依靠不断增加规则来覆盖 Kernel
生态。它是一个：

```text
LLM-driven, verifier-closed system
```

系统把开放世界中的理解和生成交给 LLM，把不可变的裁决交给一个极小、通用、可审计
的验证核心：

```text
LLM absorbs ecosystem entropy;
the verifier bounds model uncertainty.
```

固定代码可以拒绝错误，但不替 LLM 构造正确实现。

## 2. 设计来源与冲突消解

本设计整合了 [source-notes](source-notes/README.md) 中的五份材料。它们存在两次明确
的设计推进：

1. `回答1` 正确识别了 composer/RuleCard/C.strategy 的工程膨胀，但其“放松 Gate1”
   建议已被推翻；当前 Triton 强观察路径仍要求规范化后的严格 TTIR realization 闭合。
2. `回答2` 正确恢复了 π 作为 Intent 外的 source-realization witness，但仍保留了
   deterministic finalizer/RuleCard；该部分又被 `回答3` 与“高级 AI 哲学”推翻。

最终结论是：

```text
无 deterministic composer
无 RuleCard/RuleRegistry
无 C.strategy
无 per-kernel deterministic policy
Lift LLM 直接产出 J、纯 C、π
Replay LLM 直接产出完整前端源码
```

## 3. 唯一顶层对象

### 3.1 Intent

```text
Intent = J(A+B) + C
```

- `J=A+B` 是唯一语义对象；
- `A` 描述逻辑值和可观察状态关系；
- `B` 描述会改变可观察结果的最小正确性边界；
- `C` 是纯调度与性能提示，不改变 `J`。

### 3.2 π 不属于 Intent

```text
π ∉ Intent
```

π 是 Gate1 的 source-realization witness，回答“源码选择了怎样实现 J”。它可以保存
实现拓扑、共享、placement、region 和顺序，但没有独立 denotation，也不参与 Gate2
的 J-only 语义腿。

### 3.3 HarnessContext 不是 IR

函数真实名字、参数 spelling、具体实参、设备、grid、源前端调用方法等放在极薄的
`HarnessContext`。J 从 Lift 输出开始使用按声明顺序建立的匿名 handle；生成目标程序
无需恢复原始参数名。

不建立独立 `LogicalABI` 语义对象。运行时的 positional binding 是 harness 行为，不是
第二份语义真值。Gate1 比较前也按声明位置统一源码与 Replay 的参数 spelling；参数名字
可以不同，但 arity、kind、rank、dtype、constexpr 位置与 C 值不能不同。

这里的 `constexpr` 只是前端绑定方式，不等于 Layer C。匿名 interface 保留 non-C constexpr
的位置/类型，冻结值留在 HarnessContext：改变 logical extent、predicate、shape 或允许结果的
consequence 由 J 拥有；stride/layout 等纯表示 use 可由 pi 匿名引用。只有 tile、block、stage、
num_warps 等在合法变化下不改变 J denotation 的 constexpr 才进入 C。Gate1 将 non-C
constexpr 的同一调用值按位置交给 source 与 Replay，不建立 `LogicalABI`，也不允许 pi 承担
遗漏语义。

## 4. 系统主流程

```text
Source S
   │
   ▼
Lift LLM(S, optional retrieved context)
   ├── J = A+B
   ├── C = pure schedule
   └── π = anonymous source-realization witness
   │
   ▼
Replay LLM(J, C, π, frontend context)
   │
   └── complete replay source R_f
          │
          ├── Gate1: frontend-local compiler observation closure
          └── Gate2: Source / Replay / J-only semantic execution
```

LLM 的开放世界职责包括：

- 理解任意来源 Kernel；
- 从 SPMD realization 提升成 whole-tensor J；
- 直接产生 π；
- 自行选择当前前端 API；
- 根据 `J+C+π` 写出完整 Replay；
- 在失败后由 source-aware Lift 重新提出完整 artifact。

固定代码仅负责：

- schema、基础类型和引用合法性；
- opaque handle 与源码隔离；
- artifact commitment/hash；
- build、run、observe、optional dump；
- compiler observer 和确定性比较；
- Gate2 challenge、comparator、receipt 和审计。

## 5. J：小型、可组合、健壮的语义核心（待下一阶段冻结）

本节的小型构造仍是开放研究基座，但 2026-07-22 已冻结首版 typed indexed bounds、
logical update 与 allowed-result relation。schema 可表达不等于 executor 或 Kernel family
已经受支持；能力结论仍必须来自真实 consumer、Gate1/Gate2 和 backend conformance。

2026-07-21 的第一轮真实源码审计已经完成，详见
[IntentOp_健壮性审计_2026-07-21.md](IntentOp_健壮性审计_2026-07-21.md)。审计冻结的是
结构禁令与 fail-closed 能力边界。后续 Typed Relations 03 又把 bounds、update 和
result relation 放入唯一 canonical B；per-op `layer_b`、`out`、J 内 `trip.tile_hint`
以及 J 中的 `atomic/synchronize` spelling 仍被硬拒绝。

### 5.1 Elementwise expression

```text
map(inputs, scalar_expression)
```

标量表达式由小型表达式 AST 描述；不为每个 `add/exp/sigmoid` 建立前端施工规则。

### 5.2 Logical index/map

```text
output[index] = input[index_map(index)]
```

统一表达 copy、reverse、transpose、reshape、slice、broadcast、gather 等逻辑映射。
改变输出 denotation 的 index 在 A；源码 pointer arithmetic 不在 A。

### 5.3 Aggregate

统一表达 reduce、scan、contraction 和 segment reduction：

```text
reduce(input, axes, combiner, identity)
contract(inputs, logical_index_relation)
```

### 5.4 Semantic recurrence

只承载算法本身不可消去的状态递推，例如 online softmax、scan state 和数据依赖迭代。
K-tile loop、unroll loop、pipeline loop 不因此进入 J。

### 5.5 Logical update

```text
update(base, key(index), value(index), combine, conflict/order policy)
```

统一表达 scatter、histogram、in-place update、duplicate-index accumulation，以及可能由
atomic 实现的逻辑更新。J 不保存 `atomicAdd`、CAS spelling 或线程私有化方案。

### 5.6 B 的最小边界

B 只保留真正影响可观察结果的内容：

- logical domain、bounds、predicate、fill；
- output/destination、alias、in-place、partial write coverage；
- duplicate/conflict policy；
- 必需的数值顺序、允许结果或 tolerance；
- observable state dependency。

不进入 B：warp、barrier spelling、`cp.async` token、shared-memory bank、pipeline stage、
double buffer、pointer association、source fusion placement。

### 5.7 复杂机制的分解

| 源码机制 | J | C | π |
|---|---|---|---|
| copy | identity logical map | tile/vector | load/store topology |
| reverse | reverse index map | tile/vector | address construction |
| elementwise fusion | compositional value DAG | target grouping | fused producer/reuse |
| reduction fusion | value DAG + aggregate | fusion/tile | shared load/loop placement |
| atomic add | logical update + observable conflict/order | privatization/tile | atomic realization |
| async pipeline | final logical computation | stage/prefetch/buffer | copy/wait/swap topology |

普通 race-free pipeline 不需要在 J/B 中复制 async token 状态机。只有源码顺序本身改变
可观察状态时，J 才保存抽象逻辑依赖，而不是具体 barrier/API。

## 6. π：由 Lift LLM 直接产生的匿名 realization witness

π 不是 RuleCard 列表，也不是 composer work item。建议的最小结构只包含：

```json
{
  "regions": [
    {"id": "r0", "parent": null, "placement": "root"}
  ],
  "instances": [
    {"id": "i0", "semantic_ref": "a0", "region": "r0", "mechanism": "realize"},
    {"id": "i1", "semantic_ref": "n2", "region": "r0", "mechanism": "realize"}
  ],
  "routes": [
    {"source": "i0", "target": "i1", "slots": [0, 1]}
  ],
  "order": [{"before": "i0", "after": "i1"}],
  "sharing": [
    {"source": "i0", "consumers": ["i1"]}
  ]
}
```

π 可以引用 J node/access/value，以及 realization region、edge、slot、order、share、placement
和必要的 source-neutral mechanism tag。例如它可以区分 synchronous/async transfer、
atomic/privatized update 或 staged/direct access，但不能在这些 tag 中携带逻辑 combiner、
predicate 或值语义。π 不得声明 operator、constant、predicate、fill、reducer、logical
index map、output mapping、conflict combiner、具体前端 API 或源码自由文本。

确定性代码只检查 ID 唯一、引用存在和 schema 合法，不推导、不重排、不展开规则。

这里的 instance 是 compiler-observable realization event。源码若让同一匿名参数/access 经历
多级实现，可以用多个 instance 锚定同一 handle；route 保存匿名 def-use，sharing 保存单一
producer 的 fanout。所有实际参与 address/transfer/synchronization/placement/share 的匿名
interface/C handle 都必须进入 pi。Replay 不得丢弃这些依赖、压平多级 chain 或复制共享
producer。pi 仍不声明这些 stage 的逻辑 operator/formula；LLM 根据 J/C/interface 写前端，
Gate1 用最终 compiler topology 判断是否真的消费正确。

“handle 进入 pi”只有一种合法写法：raw J/C/interface handle 只能出现在 instance 的
`semantic_ref` anchor 中；route/order/sharing 的所有 graph endpoint 必须是已声明的 `i*`
instance ID。edge 不隐式声明 instance，也不能直接写 `p0 -> i1`。若 p0 参与一个 access，
应声明锚定 p0 的 i0 与锚定 access 的 i1，再写 `i0 -> i1`。确定性 validator 只检查这一
foreign-key closure，不按 anchor 猜测或补写 graph。

当 strict compiler topology 需要区分寻址阶段时，π 只使用两个 source-neutral mechanism tag：
`address_base` 表示 program 内 lane-uniform 的独立 pointer/base realization，`address_lane` 表示
lane-varying final address realization。它们不能保存 offset/index 公式、cast/width、dtype/shape、
mask 或 API；logical index 仍只在 J。address stage 后再连接 direct/async transfer 或 atomic/
privatized update，因此这不是一个新的可执行 address plan。

## 7. C：纯调度

C 只包含替换后不改变 J 的调度与性能提示，例如 tile extent、vector width、线程映射、
warps、unroll、pipeline depth、prefetch distance、buffer count 和 target retuning seed。

canonical C 只有 `{slots, values}`。slot 名是匿名 handle；`kind` 区分 constexpr 与
compiler option；`role` 是 `tile_extent/num_warps/vector_width` 一类 source-neutral 调度名。
compiler role 到公共 frontend option 的映射可以是固定 adapter 逻辑，不能按 Kernel 选择。

源码 realization 选择不得进入名为 `C.strategy` 的兜底字段。Gate1 source/replay 使用同一
冻结 C；目标 backend 可以消费、改写、忽略或重新调优 C，但不能违反 J。

## 8. Gate1：source-blind 构造闭合

设：

- `S` 为源码；
- `R_f = ReplayLLM(J,π,C,Context_f)`；
- `F_f^(k)` 为前端 f 的第 k 级 observer。

Gate1 定义为：

```text
R_f 在源码隔离下生成
且 F_f^(k)(S;C) = F_f^(k)(R_f;C)
```

Observer 可分级：

- Level 0：可构建、接口与 specialization 闭合；
- Level 1：AST/intrinsic/mask/atomic/barrier 等必要结构；
- Level 2：TTIR/TIR 的规范化数据流、predicate/fill/effect/realization topology。

当前 Triton 强路径以 Level 2 strict closure 为 hard pass；新前端可以先以 Level 0/1
接入，结论必须显式带 observer 等级。增强 observer 不改变 J、Replay 或 Gate2。

这里的 strict 是规范化后的严格一致：函数名、参数名、SSA 名等 spelling 通过声明顺序
alpha-normalize；op multiset、数据流边、reduction、predicate/fill/effect topology 仍逐项
hard-close。源码调用与 artifact 的 arity、kind、rank、dtype、constexpr 位置和 C 值在
编译 Replay 前先闭合，不能靠改名掩盖接口差异。

Gate1 证明 `J+π+C` 足以构造性重建源码 realization；它不负责直接证明 `S≈J`。

## 9. Gate2：Source-anchored 三角验证

定义：

```text
O_src = Exec(S, x)
O_rep = Exec(R_f, x)
O_J   = Exec(E_J, x)
```

`E_J` 是只读取 J 的简单、无优化、backend-neutral 标量语义执行路径。令 `R_J(x)` 为 J
声明的 exact、tolerance 或 one-of-updates 允许结果关系，正式裁决推广为：

```text
O_src ∈ R_J(x)
O_rep ∈ R_J(x)
O_J   ∈ R_J(x)
```

exact/tolerance 下仍是原来的两条 source-anchored comparison；one-of-updates 下 source
与 replay 可选择不同 contender，但必须分别通过同一个 J membership checker。容差不再由
Gate2 case/harness 注入。

Gate2 直接验证 `S≈J`，并且完全不读取 π。因此即使 π 泄漏部分语义，也无法修复一个
错误 J 的 Source–J-only 比较。

## 10. 联合证明对象与概率声度

对冻结测试 envelope `T`，定义：

```text
S ≈SCRC^(f,k,T,R_J) J
```

当且仅当存在 `π,C,R_f`，使 Gate1 observer 闭合，并且对所有 `x∈T`：

```text
Obs(S,x) ∈ R_J(x)
Obs(R_f,x) ∈ R_J(x)
Obs(E_J,x) ∈ R_J(x)
```

若最终 artifact 在随机 challenge 前冻结，输入独立采样自声明分布 `D`，而一个错误 J
的错误质量满足：

```text
d_D,R_J(S,J) > ε
```

则通过 m 个隐藏 Source–J challenge 的概率至多为：

```text
(1-ε)^m ≤ exp(-mε)
```

该上界对任意 π、Lift/Replay 模型和 prompt 都成立，因为 Gate2 的 J-only 边不读取 π。
这称为 π-robust semantic soundness。

协议要求：先 commitment `Hash(S,J,π,C,R,observer/compiler versions)`，再生成隐藏 seed；
失败后 relift 必须提交新 artifact 并使用新 seed。确定性 hard cases 与随机 challenge 可以
组合，但不能对同一隐藏集反复修补。

## 11. masked-J 的准确位置与在线成本

masked-J 不证明 `S≈J`；它审计 J/π 的 representation factorization：删除 J 后，π+C+
其他辅助通道是否足以独立完成 Gate1 closure。

正式在线流程最多保留一次附加检测：

```text
1 × Lift LLM
1 × normal Replay LLM
source/replay compile + Gate1
若 normal closure：最多 1 × masked-J Replay LLM
Gate2（不新增 LLM 调用）
```

J mutation、π swap、J-only、π dropout、C swap、多 seed/multi-model 等全部属于论文离线
消融，不进入每个 artifact 的正式在线系统，也不遍历在线 fixed-C family。

masked-J packet 不得包含 selected RuleCard、semantic rule ID、C.strategy 或按当前 Kernel
选择的 API mapping。

## 12. 失败和重新提出

Replay 失败后，不能把源码、旧 Replay 或 source/replay diff 交给 Replay LLM 局部修补。
正确循环是：

```text
failure evidence
  -> source-aware Lift
  -> fresh (J',π',C')
  -> fresh source-blind Replay
```

旧候选只作审计证据。这样每次 closure 都是新的完整构造 witness，而不是看过答案后的
patch accumulation。

## 13. 开放前端合同

新前端最低只需要：

```text
Build(source, specialization, C)
Run(artifact, logical_inputs)
Observe(outputs, inout state, effects)
OptionalDump(artifact)
```

不要求编写 J→frontend lowerer、API mapping、source-pattern parser、RuleCard 或 composer。
前端文档和仓库示例可以由 LLM 动态检索，不成为核心系统中的永久语义规则。

## 14. Lowering 与 backend conformance

J 可以确定性转写为 Intent MLIR；该转写是 schema 到 dialect 的直接编码，不进行源码
realization 推导。目标 lowering 必须接收完整 J，C 作为独立可替换 hint。

`E_J` 是 Gate2 的可信语义基座；CUDA/RVV 等 production lowering 是独立 backend
conformance 命题，不能代替 Gate2 的任何一腿。删除重型 `IntentProgram/LogicalABI`
时，必须避免重新退回 A-only lowering。Gate2 scalar 与 production backend 直接接收同一
canonical J；尚未实现的 B relation 必须在执行/codegen 前 fail closed。

## 15. 明确禁止的系统形态

活动代码和 spec 不得重新引入：

- deterministic composer 或 work-item 枚举；
- RuleCard/RuleRegistry、selected card 或 per-op API mapping；
- `C.strategy` 或 source-pattern family；
- 以兼容为名保留的旧 artifact 双轨；
- 重型 `IntentProgram` 第二 wire IR；
- 独立 `LogicalABI` 语义对象；
- kernel-name recognizer 或 per-kernel deterministic construction；
- 在线 mutation/swap/dropout 矩阵；
- 看过源码/diff/旧候选后的 Replay repair。

## 16. 当前状态与下一步

Architecture Reset 01 已完成：活动主链已切换到 LLM-direct `J+C+π`，旧控制面和第二
semantic wire 已迁出 importable tree；新 epoch 对旧 artifact/cache 硬失败。实现证据见
[归档任务](../.trellis/tasks/archive/2026-07/07-21-architecture-reset-01-direct-pi/task.md)。

IntentOp Robustness 02 与 Typed Relations 03 也已完成：错误的第二 B/C/atomic 通道已回收，
typed bounds/update/result relation 和 J-owned Gate2 comparator 已闭合；CPU scatter 与一般
recurrence 仍按真实 executor 能力 fail closed。

下一阶段是 28 个真实高性能 Kernel 的正式 LLM Lift/Replay/Gate1/Gate2/backend
conformance/performance 批次。批次必须先冻结 source/symbol/digest 和调用预算；不得用
hand-written J、Replay 或 MLIR 替代失败样本，也不得为了凑 28/28 绕过 executor 缺口。
