# SCRC：J 语义完备与确定性 π 重构方案（v8 已实现，归档）

> 归档状态：核心实施规划已完成；未完成的复杂特性真实验证继续由当前 Gate1 spec、
> held-out 与规模验证主线跟踪。

> 状态：**主线已实现；进入真实 Kernel v8 复验阶段。**
>
> 日期：2026-07-16
>
> 当前正式协议：`j_complete_deterministic_pi_relift_v8`。
> 已修改 Lift schema/prompt、artifact、composer、Gate1 因果口径、cache/report identity
> 与 fixed-C multi-specialization 基础路径。全程未使用 Trellis task/workflow。

## 1. 结论

本候选赞同以下核心方向：

> J 承载全部算法语义、可移植依赖与影响正确性的状态；C 承载实现策略与调度；
> 规则库承载 frontend 领域知识；interface 提供冻结 ABI/grid 合同；π 不再由 LLM
> 决策，而是 `DeterministicCompose(Canon(J), Interface, C, RuleRegistry)` 的
> 无歧义组装结果。

这样 π 从“可能成为第二语义通道的 Lift 产物”变成“可重算、可逐字节验证的编译
中间物”。它仍然是 writer 可直接执行的严格施工协议，但不再拥有任何智能决策。

目标不是让 π 在信息论上“看不出 J”。可直接执行的施工协议必然反映部分 J 拓扑；
真正要求是它不携带任何独立于权威输入的附加信息：

```text
H(π | Canon(J), Interface, C, RuleRegistry) = 0
```

必须同时修正一个现有边界：当前 C 只表示 schedule slot/value。若实现策略也归 C，
C 必须显式拆成两个子域，不能把调度真值和可 relift 的策略选择混在一个对象里：

```text
C.schedule   # compiler/launch/constexpr，来自可信 compiler view，冻结
C.strategy   # source implementation choices，有限枚举，可由 Lift/relift 修正
```

## 2. 当前实现与目标之间的真实差距

### 2.1 loop carry：有表示，但 SCRC schema 不闭合

项目已有：

- `IntentOp.carry/trip/scan/body/yield`；
- `intent.loop + intent.yield` dialect；
- `scf.for iter_args` lowering；
- Gate1 TTIR 对 `scf.for`、region block args、yielded values 的指纹提取；
- `realize_loop_carry` 规则卡。

但当前 SCRC wire validation 对 `carry/trip/scan` 仅检查“是任意 JSON”，没有验证：

- carry 名称、初值、body block arg、next value、yield、loop result 一一对应；
- carry 数量和类型在 init/body/yield/result 间一致；
- trip 的 semantic axis 与 C tile slot 分离；
- body 使用的值是否在正确 region/scope 定义；
- loop 外是否只引用合法 result，而不是内部临时值。

因此 loop carry 是“底层能力已有、SCRC J 合同尚未闭合”，不是需要发明一个新 op。

### 2.2 buffer lifetime：普通 SSA 可推导，语义 storage lifetime 缺一等表达

当前已有 tensor/value、destination-passing `out`、alias、access 和 effect，但没有
统一表达下列情况：

- 一个逻辑 state/buffer 在哪个 scope 创建；
- 是否跨 loop iteration 存活；
- 初始化与第一次可读之间的约束；
- 最后一次使用和可观察写回；
- 多个名字是否代表同一逻辑 storage；
- 临时对象是否 escape；
- 复用是否会改变 alias/effect 语义。

不是所有物理 lifetime 都应进入 J。寄存器/shared memory 选择、double buffer、
prefetch、software pipeline stage 和实际 alloc/dealloc 点属于 C.strategy/规则库。
只有影响可观察值、alias、跨迭代状态或 effect ordering 的**语义生命周期**进入 J.B。

### 2.3 frontend scope：需要 portable scope，不应复制 Python/Triton AST

当前 J 有嵌套 body，但没有一等 scope tree 来说明：

- value/access/effect 属于 function、program instance、loop body 或 branch region；
- 哪些值被 region capture；
- 定义、使用、yield、escape 的合法边界；
- 临时 state 是否跨 iteration；
- effect 与 barrier/atomic 的 region 归属。

J 不应保存 Python 变量名、缩进、源码 block 或 Triton API scope。目标是 portable
semantic scope；具体怎样生成合法 Triton 词法结构属于规则库和 writer。

### 2.4 π 当前承担了本应由 J/C 拥有的决策

当前 Lift prompt 要求 LLM 同时决定完整、有序 π，包括：

- native/wide program coordinate；
- integer-offset/pointer-staged address association；
- direct/staged mask materialization；
- explicit-axis/flattened reduction call；
- 施工顺序与 context refs。

其中前四类已经对应 RuleCard 的 override family，本质是 source implementation
strategy，应进入 C.strategy；数据依赖和 scope 顺序则应由 J/B 推出。LLM 生成 π
导致以下问题：

- J 缺口可能被 π 顺序和 refs 补偿；
- π 可能通过规则序列/重复次数/引用拓扑编码算法；
- relift 容易只修 π，而没有修正 J；
- 同一 J/C/registry 可能得到不同 π；
- masked-J 无法区分合法派生拓扑和越权泄漏。

## 3. 新的信息所有权

| 信息 | 唯一所有者 |
|---|---|
| 算子、操作数、结果、shape/dtype、semantic DAG | J.A |
| loop init/carry/next/yield/result | J.A |
| mask、bounds、access、alias、effect、output mapping | J.B |
| portable scope、semantic lifetime、跨迭代 state | J.B |
| BLOCK/warps/stages/constexpr/compiler values | C.schedule |
| program coordinate width、address association、mask staging、reduction call shape 等实现策略 | C.strategy |
| frontend API、合法构造、坑点、phase/dependency 模板 | RuleRegistry |
| 完整施工步骤和顺序 | π，纯代码派生、无所有权 |
| Python/Triton source text | 仅 Lift 可见 |

## 4. J schema 收紧计划

### 4.1 闭合 `LoopRegion`

逻辑模型：

```text
LoopRegion
  loop_id
  induction / trip
  scan_inputs[]
  state_bindings[]
  body[]
  yields[]
  results[]
  scope_id
```

每个 `StateBinding`：

```text
state_id
init_value
body_argument
next_value
result_value
type
```

纯代码必须验证：

- 数量、顺序和类型一致；
- init 在外层 scope 可见；
- body_argument 只在 loop body 可见；
- next_value 在本 body 定义；
- yield 与 next_value 精确对应；
- result 只在 loop 后可见；
- semantic trip axis 与 C.schedule tile slot 分离；
- schedule-only tiling loop 仍 re-roll，不错误进入 J。

### 4.2 新增 semantic storage/lifetime

建议逻辑对象：

```text
StorageObject
  storage_id
  logical_value/tensor_ref
  scope_id
  alias_group
  mutability
  init_event
  last_use_event
  escapes
  crosses_iteration
```

普通纯 SSA value 的 lifetime 直接由 def-use 推导，不重复序列化。只有 destination-
passing、in-place、alias、跨 iteration state、多输出共享 storage 或可观察 effect
需要显式 `StorageObject`。

### 4.3 新增 portable scope tree

```text
Scope
  scope_id
  kind: function | program_instance | loop_region | branch_region
  parent_scope
  captures[]
  definitions[]
  effects[]
```

所有 id 使用 opaque handle。scope 只表达 dominance/capture/lifetime/effect 语义，
不保存源码行、变量名或 frontend API。

## 5. C 从 schedule 扩展为 schedule + strategy

### 5.1 `C.schedule`：继续冻结

沿用当前 `ScheduleSpec`：

```text
slots
values
```

值来自可信 compiler view；normal relift 不可修改。Writer 只看到 symbolic slot，
具体值仍由 injector 注入。

### 5.2 `C.strategy`：有限、可审计、可 relift

建议以 target + strategy family + enum choice 表达：

```json
{
  "bindings": [
    {"target_ref":"d0", "family":"program_coordinate_width", "choice":"wide"},
    {"target_ref":"a0", "family":"address_association", "choice":"pointer_staged"},
    {"target_ref":"a1", "family":"mask_materialization", "choice":"staged_reuse"},
    {"target_ref":"n3", "family":"reduction_call_shape", "choice":"flattened"}
  ]
}
```

这只是候选示意。strategy 必须满足：

- family/choice 来自闭合词汇表；
- 只引用 J/B/interface/C slot opaque id；
- 禁止源码文本、表达式、算子语义值和 API 名；
- 同一 family/target 最多一个 choice；
- 不需要 override 的 canonical target 不必重复存储；
- relift 可以修改 C.strategy，但不能修改 C.schedule。

已落地的 `address_association=pointer_staged` 不再是模糊的“多做几次
pointer add”。RuleCard v3 把 J domain coordinate 当作原子，并且只按 access
`index_expression` 的顶层加法项从左到右逐项形成 pointer association；禁止预合并
顶层项，也禁止继续拆分乘法项或 coordinate 内部的 program-base/lane。这样施工规则
保持跨 Kernel 通用且唯一，不需要把源码拓扑复制进 π。

当前四个 override family 可直接成为第一批 strategy family：

- `program_coordinate_width`；
- `address_association`；
- `mask_materialization`；
- `reduction_call_shape`。

## 6. RuleCard 为确定性 composer 增加机器可执行元数据

当前 `applies_to/reads/api_mapping` 主要是 writer 说明，strict coverage 已经有部分
硬编码的 op→rule 要求。确定性生成 π 需要把规则选择条件变成机器可执行合同：

```text
strategy_family / strategy_choice（可选）
target_kinds
applicability predicate
required J/B/C fields
provides capability
requires capability
construction phase
frontend mapping
```

选择纪律：

1. J/B/C 对一个 work item 必须解析出恰好一个规则；
2. 多于一个是 ambiguity，fail-closed；
3. 零个是 registry/J/C gap，fail-closed；
4. registry priority 不能静默替代 C.strategy 的智能选择；
5. rule 不能自动补一个没有 J/B target 的语义 step。

## 7. `DeterministicCompose(Canon(J), Interface, C, Registry)`

### 7.1 输入与输出

```text
输入：Canon(validated J)、interface、C.schedule、C.strategy、RuleRegistry
输出：π_materialized + ComposeProvenance
```

provenance 至少保存：

```text
composer_version
J digest
interface digest
C.schedule digest
C.strategy digest
registry digest
π digest
```

除整体 digest 外，每个 π JSON leaf 都必须记录来源：

```text
π JSON pointer
  → J JSON pointer
  | Interface JSON pointer
  | C JSON pointer
  | RuleCard JSON pointer
  | 固定 protocol constant
```

provenance 覆盖必须为 100%，orphan leaf 必须为 0。

### 7.2 算法

```text
1. 先规范化 J graph/region/scope，并消除 JSON object order、opaque id 和独立节点
   任意声明顺序的影响
2. 从 J.A 节点、J.B access/effect/output、interface 参数、C slot 枚举 work items
3. 对每个 work item 读取 J/B 语义和 C.strategy choice
4. RuleRegistry 必须解析出唯一 RuleCard
5. 从 J def-use、loop carry、scope、lifetime、effect 构造依赖边
6. 加入 RuleCard 预注册的 frontend construction dependency
7. 检查缺 step、冲突、环、非法跨 scope 引用
8. 用固定 phase + canonical target key 做稳定拓扑排序
9. 在规范化后生成 opaque refs/reference_bindings
10. 生成完整 ReplayPlan、逐 leaf provenance、canonical bytes 和 hash
```

composer 不能：

- 推断缺失 J 语义；
- 猜 C.strategy；
- 根据 kernel 名或源码选择规则；
- 引入未注册默认 step；
- 用 score 解决 ambiguity；
- 调用 LLM。

`n0/a0/v0` 等 opaque id 必须在 `Canon(J)` 之后分配。若两个 J 只差 Lift 自选名字、
JSON object 顺序或互不依赖节点的任意排列，规范化结果和 π bytes 必须相同。

## 8. Artifact 与 Lift/relift 改造

### 8.1 区分 Lift 产物和完整 Gate1 artifact

候选结构：

```text
LiftArtifact
  J
  interface
  C.schedule
  C.strategy
  location_map
  source/provenance

Gate1Artifact
  LiftArtifact
  π_materialized        # composer 生成
  compose_provenance
```

Lift response schema 不再允许 `plan`。若 provider 输出 plan，closed schema 直接拒绝。

### 8.2 relift 可修改字段

```text
允许：J、C.strategy、location_map
冻结：interface、C.schedule
派生：π
```

每轮 relift 后先重新 compose π，再进行静态扫描、source compile、fresh replay 和
Gate1。失败 replay 仍不回给 writer 修补。

### 8.3 compose failure 责任归属

| 失败 | 动作 |
|---|---|
| J schema/依赖不完整 | 回流 Lift 修 J |
| C.strategy 缺失/冲突 | 回流 Lift 修 C.strategy |
| 合法 J/C 没有唯一规则 | registry gap，记录工程缺口，不能无限 relift |
| composer 环/跨 scope 非法 | 优先判断 J dependency 或 rule metadata bug |

## 9. π 验证与反泄漏必须重新定义

π 变成确定性派生产物后，“π 是否偷藏了 J”不再由随机 masked replay 判断，而由
更强的结构事实证明：

```text
π == DeterministicCompose(J, C, Registry)
```

任何额外字段、step、edge、literal 或不同顺序都无法通过 digest/equality 检查。

### 9.1 保留并调整 paired J/B intervention

```text
修改一个 J/B 坐标
→ 重新 compose π
→ fresh replay
→ 检查精确 TTIR delta
```

这里不能冻结旧 π，因为 π 是 J 的编译结果。应冻结 C/registry/interface，并验证
π 的变化恰好是 composer 对 J 变化的确定性结果。

### 9.2 mechanism substitution 改为 C.strategy intervention

不再直接篡改 π step：

```text
只修改一个 C.strategy choice
→ 重新 compose π
→ replay
→ J.A/J.B 语义硬项保持，预期实现结构发生变化
```

这直接验证“实现策略住在 C”，比修改派生 π 更符合所有权。

### 9.3 J-only / step-drop 的含义改变

- J-only 不再表示“writer 不看 π 猜代码”；正式路径始终先 compose π。
- π step-drop 不再证明 Lift π 是否必要，而是 composer/rule completeness 的负向测试；
  删除派生 step 后必须被 `π != compose(...)` 立即拒绝，不需要调用 LLM。
- 直接 mechanism substitution 同样应被 equality 检查拒绝；合法替换必须从
  C.strategy 发起。

### 9.4 masked-J 不能冻结 full-J 派生 π

若把 full J 生成的 π 与 masked J 配对，π 当然可能反映 J 拓扑；这不再是一个可识别
的信息分区实验。新的 masked 探针只能：

```text
mask J
→ 尝试重新 compose
```

预期结果是缺少语义/依赖而 compose fail-closed，或者产生可明确审计的不完整 work
items。不得把 full-J π 当成独立输入继续测试。

### 9.5 新的 π 证据

- determinism：同一 J/C/registry 多次 compose 字节相同；
- derivability：保存的 π 与重算结果精确相等；
- completeness：每个 work item 恰好一个 rule instance；
- ambiguity-free：零/多规则都拒绝；
- causal J：J intervention 后 π/replay 出现精确派生 delta；
- causal C：C.strategy intervention 改实现但不改 J 语义；
- registry isolation：规则版本/digest 变化使 cache 失效；
- leaf provenance：每个 π 字段都有且只有合法来源，零 orphan 字段；
- source non-interference：源码变量、helper、注释、路径、kernel 名或等价无关声明
  改变，但 Canon(J)/interface/C/registry 相同时，π bytes 必须不变；
- alpha/order invariance：opaque id 改名、JSON key 顺序与独立 J node 排列不改变 π。

## 10. Gate1 与 Gate2 的位置

Gate1 仍保持：

```text
Lift J/C
→ deterministic compose π
→ source-blind writer
→ compile-only
→ deterministic TTIR hard comparison
→ causal checks
```

没有新增 verifier 或第三道 Gate。Gate2 仍执行 source/replay/Lower2C 三方行为；
确定性 π 不改变 Gate1/Gate2 非包含关系。

## 11. 分阶段实施状态

### P0：冻结 v7 基线与协议报告（完成）

- 保存当前五核和 held-out v7 结果；
- 明确旧 Lift-generated π 只作迁移对照；
- 新旧 cache/protocol 不互认。

### P1：先补 J loop carry 闭合 schema（完成）

- typed loop/state dataclass；
- wire schema；
- init/body/yield/result/scope/type 验证；
- JSON↔IntentIR↔MLIR round-trip；
- TTIR loop fingerprint 正负测试；
- 真实 structured-loop/scan Kernel extraction。

### P2：补 portable scope 与 semantic lifetime（完成）

- scope tree、capture/dominance/escape；
- storage identity、alias、effect、cross-iteration lifetime；
- 明确普通 SSA liveness 不重复序列化；
- destination-passing/in-place/多输出正负测试。

### P3：引入 `C.schedule/C.strategy`（完成）

- 迁移当前四个 override family；
- schedule 冻结、strategy 可 relift；
- static leakage 和 closed vocabulary；
- 不改变现有 rule cards 的 frontend API 所有权。

### P4：composer shadow mode（完成并已转正式判定）

- deterministic composer 已接管 π；
- 保存逐叶 provenance 与 canonical input/plan digest；
- opaque alpha-renaming、JSON key order、独立 node declaration order 有正向测试；
- drop/add/ref/order tampering 在 writer 前精确拒绝；
- `run_composer_shadow` 只保留为报告兼容开关，compose equality 始终参与 verdict。

### P5：切换 Lift 协议（完成）

- Lift 不再输出 π；
- Gate1 artifact 自动 compose；
- protocol/report/cache schema 升级；
- relift 允许修改 J/C.strategy；
- 删除旧 π repair/ablation 语义的残余入口。

### P6：重做反作弊与真实验证（机制完成，真实五核/held-out v8 重跑待执行）

- J intervention→recompose；
- C.strategy intervention→recompose；
- compose equality/determinism；
- fixed-C 多 specialization；
- Gate2 三方矩阵；
- 重新执行 add、row-sum、softmax、RMSNorm、matmul 和 held-out。

## 12. 已落实的边界决定

1. C 已拆为冻结的 `C.schedule` 与可 relift 的闭合 `C.strategy`；
2. Lift/relift wire schema 已禁止 π，composer 独占 π 生成权；
3. storage 只保存 semantic lifetime，物理 allocation 继续归 C/规则库；
4. scope 是 portable semantic scope，不保存 Python/Triton AST；
5. 旧 full-J π + masked-J writer 组合已废止，masked 探针改为 mask→recompose；
6. Gate1/Gate2 仍为独立证据，不新增第三道 Gate；
7. fixed-C multi-specialization 已有 production runner，跨 C family 仍不在本阶段。

## 13. 验证状态与未完成实验

实现侧聚焦回归已覆盖：closed loop、scope/storage、strategy、composer canonicality、
Lift/relift v8、paired J recompose、Gate1 正式 equality、Lower2C artifact path 和
fixed-C family all-pass。

全量 `python/tests` 本轮结果为 595 passed、3 skipped、3 failed。3 个失败不在本次
改动路径：两个是已安装 xFormers/Triton varargs 对 `BLOCK_SIZE` 的重复绑定；一个是
旧 universe `count_nonzero` manifest 从 `ok` 漂移到 `lowering_fail`。

仍需作为真实实验执行，而不能由单元测试替代：

- 用 v8 schema 重新经真实 LLM Lift 跑 add、row-sum、softmax、RMSNorm、matmul；
- 至少两个未参与规则设计的 held-out Kernel；
- 对可动态化 Kernel 跑 fixed-C 多 specialization family；
- Gate2 source/replay/Lower2C 三方矩阵；
- 根据真实失败决定补 J schema、strategy family 或 RuleCard，不向 π 塞回决策。
