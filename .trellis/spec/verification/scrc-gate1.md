# SCRC Gate1：J/C 驱动的确定性重放闭合

## 1. 定位

Gate1 只回答一个问题：源码隔离 writer 能否依据完整语义 `J`、冻结接口、
`C.schedule + C.strategy`、共享 RuleRegistry，以及由这些权威输入**确定性派生**的
施工协议 `π`，生成可编译重放，并在同一前端、同一 C 下与源 Kernel 的 TTIR
硬结构闭合。

Gate1 不执行 Kernel，也不解释 JSON 得到数值结果；它同时拥有 LiftCore 的语义准入
与 source/replay hard closure。活动 Gate2 是
Gate1 硬闭合后、绑定同一最终 J 的 replay/scalar 动态差分；Gate2 PASS 不补偿任何
Gate1 hard mismatch。Gate1 拒绝样本可以另做离线诊断，但该结果不是 Gate2 evidence。

当前正式协议：

```text
gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11
```

全程不依赖 Trellis task/workflow；本文件仅作为项目规范。

## 2. 信息所有权

| 信息 | 唯一所有者 |
|---|---|
| 算子、operands/results、shape/dtype、semantic DAG | J.A |
| loop carry、mask、bounds、access、effect、alias、output mapping | J.B |
| portable scope、semantic storage/lifetime、跨 iteration state | J.B |
| constexpr/compiler slot 与具体编译值 | C.schedule |
| 有限、审计过的来源实现选择 | C.strategy |
| frontend API、合法构造与 construction dependency | RuleRegistry |
| 完整施工步骤及稳定顺序 | π，纯代码派生，无独立所有权 |
| ABI、grid、参数顺序与 `do_not_specialize` | interface |
| source span 到 J/B handle 的映射 | location_map，仅 Lift/诊断侧 |

J.A 必须对 program/tile decomposition 保持不变。`iota(n)` 表示全局逻辑 ordinal，
runtime `n` 必须保留为输出 shape 使用的符号，不得烘焙当前 specialization 的样本值。
源码中的 `program_id * BLOCK + lane`、tile base、warp/stage 与为了匹配 TTIR 结合顺序
而进行的 block/lane 算术分配均属于 C/RuleRegistry，不得写入 J.A。TTIR op-count 或
dataflow association 差异本身不能证明 J 有语义缺口；现有 C.strategy 无法表达时，
保持正确 J 并允许预算耗尽返回 `NO_CLOSURE`。

核心约束：

```text
π = DeterministicCompose(Canon(J), interface, C, RuleRegistry)
H(π | Canon(J), interface, C, RuleRegistry) = 0
```

Lift/relift response schema 不允许 `plan` 或 `compose_provenance`。Extraction 只把
JSON-constructable 的原始 `LiftCoreProposal` 连同空 plan 占位和 provenance 交给 Gate1，
不提前解析 Layer A/B、opaque identity、interface contract 或 π。Gate1 每轮按以下命名
类别 fail closed，并把失败作为 bounded source-aware Relift evidence：

```text
artifact_constructability -> layer_a_semantics -> layer_b_semantics
-> opaque_identity -> interface_contract -> pi_capability -> pi_completeness
```

全部准入通过后，Gate1 才由 composer 生成／核对 canonical `plan` 与逐叶 provenance。
不存在语义 Gate0，也不允许 extraction 通过 alias normalization、字段推断、默认 predicate
或 kernel-name 分支把坏 proposal 改造成可接受 artifact。

## 3. J 的闭合合同

### 3.1 loop carry

每个 semantic loop 必须有 `scope_id`、semantic `trip.axis`、可选 scan，以及闭合
state binding：

```text
state_id / init_value / body_argument / next_value / result_value / type
```

纯代码检查 init/body/yield/result 数量、顺序、类型、scope 可见性和一一映射。
schedule-only tiled loop 仍 re-roll，不伪装成 semantic loop。当前 MLIR prototype 对
不支持的 carry dtype 明确 fail，而不是静默转成 f32。

### 3.2 portable scope

新 Lift 必须输出非空 scope tree：恰好一个 parentless function scope，可选唯一的
direct-child program instance，以及与 semantic loop 一一对应的 loop scope。
branch scope 在一等 branch-region op 落地前 fail-closed。captures、definitions、
effects 必须与真实 dominance/def-use/effect ownership 精确相等。

### 3.3 semantic storage/lifetime

`StorageObject` 只表示影响可观察语义的 storage identity/lifetime：destination-
passing、alias/in-place、跨 iteration state、多输出共享 storage 或 observable effect。
普通 SSA liveness 不重复序列化；register/shared/global placement、double buffer、
prefetch 和物理 alloc/dealloc 属于 C/规则库。

storage 使用 typed semantic event：scope entry/exit、node、access、state、effect。
每个 access 与一个带稳定 `effect_id + access_ref` 的 effect 精确绑定；must_alias 在
同一 storage，no_alias 在不同 storage。read-before-init、use-after-last、非法 escape、
错误 mutability 和跨 scope endpoint 都拒绝。

### 3.4 portable domain/access predicate

`boundary.domains` 与 access predicate 的字符串 wire 只是一种 Lift 传输形式，不是
自由文本语义。它们在 Gate1 Layer-B admission 时必须由 `ir.contracts` 的唯一 canonical
parser 无损转换为 boolean-rooted `ContractExpr v1`；同一检查在 typed IntentProgram
handoff 保留为 defense in depth。assignment、tuple、`where/with`、区间 prose、任意
attribute/subscript、unknown call 与 `program_id/grid_axis` 等 source-realization call
全部 fail-closed。纯算术根和未证明类型的 bare symbol 也不是 predicate。

parser 不做 regex 修复、后缀提取、默认 `True`、predicate 删除或 Kernel-name 分支。
非法 provider 输出形成命名的 Gate1 admission failure，并进入 bounded source-aware
Relift；预算耗尽即 `NO_CLOSURE`，不属于
paired semantic intervention，更不可能产生 `wrong_fill_delta` acceptance evidence。
Runner 每个 relift round 在 writer/compiler 前对当前 proposal/artifact 做完整 strict
round-trip；构造后篡改同样只能形成 admission failure／`NO_CLOSURE`，不得记录
`REPLAY_CLOSED`。

## 4. C.schedule 与 C.strategy

`C.schedule` 来自可信 compiler view，在 normal relift 中逐字节冻结。

`C.strategy` 是可 relift 的有限枚举，只保存非 canonical override：

```text
program_coordinate_width = wide
address_association      = pointer_staged
mask_materialization     = staged_reuse
reduction_call_shape     = flattened
reduction_tiling         = multi_tile_accumulate
reduction_preheader_staging = stage_invariant_access_scaffolding
reduction_tile_fusion    = shared_producer_outputs
```

absence 表示 canonical choice。同一 target/family 最多一个 binding；target kind 必须
匹配 family。strategy 禁止源码文本、表达式、常量、算法语义、API、rule id 和施工
顺序。

`reduction_tiling=multi_tile_accumulate` 只绑定到一个 J reduce node，并额外携带
`tile_slot` 与 `accumulator_dtype`。它表示源码以 runtime `scf.for`/Triton range 遍历
同一 whole-tensor reduction axis 的多个局部 tile，loop carry 仅保存该 reducer 的
associative partial accumulator。J 仍只有一个全局 reduce，不增加 loop/scan/carry；
RuleCard 使用 J 的 semantic extent 作为上界、`tile_slot` 指向的 constexpr C slot 作为
步长、BoundaryReduction.tile_axis 作为 tile-local reduction axis，并按
canonical IntentIR `accumulator_dtype`（f16/bf16/f32/f64/i1/i8/i16/i32/i64）构造
init→partial→combine→yield→final result 数据流。mean 只在循环
结束后按全局 semantic extent 除一次。若 source carry 含额外递推状态、无法证明完整
domain coverage、lower 不是 0、upper 不等于 semantic extent、step 不是单一 constexpr
slot、seed 不是 reducer identity、tile slot 不是 kernel constexpr，或 accumulator dtype
不明确，则该 choice fail-closed，不能回退 direct reduction，也不能向 J 添加伪循环。
Welford tuple、arg/index、计数器及其他有序 recurrence 都不属于该 choice。当前 composer
也不允许同一 target 同时绑定 `reduction_call_shape` 与 `reduction_tiling`；需要二者组合的
source 必须 fail-closed，不能静默丢弃任一机制。
该 composite RuleCard 将 π 中先出现、且绑定到该 reduction 的 address/mask/access steps
解释为每轮 tile body 的施工模板；pointer progression、tail mask 与 load 必须生成在 runtime
loop 内，不能在循环前只加载一次再跨迭代复用。

`reduction_preheader_staging=stage_invariant_access_scaffolding` 与
`reduction_tile_fusion=shared_producer_outputs` 都是依附于同一 target
`reduction_tiling` 的正交 C choice。前者只把 source 已在 loop 前构造、且可证明与
tile induction 无关的 outer-index base 与共享 lane basis 放入 preheader；最终 tile
address、mask、load 与 carry update 仍在 loop 内。后者只在 source 的同一规范化 tile
loop 用共享 per-tile producer 同时计算 non-reduced output 和 reduction partial 时选择：
共享 address/load/producer 每轮只实现一次，side-output store 与 partial/carry update
同处一轮，final reduced-output consumer 留在 loop 后。两者都不改变 J DAG、Boundary
access/effect/output mapping，也不是通用 LICM/fusion；source 使用分离 loop、side output
不共享 producer 或存在额外 semantic carry 时必须省略/fail closed。

`address_association=pointer_staged` 的共享规则语义是确定的：J domain coordinate
视为原子，只把 access `index_expression` 的顶层加法项按从左到右顺序逐项加到
pointer；不得把多个顶层项预合并成一个整数 offset，也不得把乘法项或 coordinate
内部的 program-base/lane 构造继续拆成额外 pointer additions。该施工知识属于
RuleRegistry，不进入 π，也不要求 π 复制来源拓扑。

## 5. RuleRegistry 与 deterministic composer

RuleCard 具有机器可执行元数据：target kinds、strategy family/choice、required
fields、phase、provides/requires。每个 work item 必须匹配恰好一张卡：零匹配是
J/C/registry gap，多匹配是 ambiguity；两者都 fail-closed，priority 不得替代选择。

RuleRegistry 的 semantic identity 对 card 输入顺序不敏感，按 card id canonicalize 后
计算 SHA-256。registry 注册时复制 RuleCard，读取时也返回副本；一个 Lift/Gate1/batch
开始前必须创建 detached frozen snapshot。运行中注册、替换或按当前 Kernel 热加载卡片
一律拒绝。composer、Lift prompt、artifact/cache identity 与 replay code policy 使用同一
snapshot digest；显式空 registry 不得静默回退 default registry。

### 5.1 受控动态扩充（`scrc-rule-release-v1`）

“动态扩充”只表示在**下一次** run/batch 开始前显式装载已审计 release，不表示当前
Kernel 失败后自动发明规则并立即重试。release 控制面与 writer 可见 RuleCard 分离；
trigger/regression/evidence/rationale 不进入 source-blind replay packet。

release 必须包含：

- `proposed | approved | revoked` 状态；只有 `approved` 可 apply；
- exact `parent_registry_digest`；
- `add | replace` 变更；replace 还绑定旧 card canonical digest；
- 非空 trigger、独立 regression、evidence 与 rationale；
- 覆盖上述全部内容的 `approval_content_digest`。

loader 使用严格 JSON，拒绝未知字段、重复 key、NaN/Infinity、错误 parent/digest、
重复 card id 与 revoked/proposed release。apply 返回新 registry，不修改 parent。
add v1 只允许具有显式 `target_kinds + construction_phase` 的 derived card，不能新增
frontend 或 override family；replace 必须提升 card version，且不得改变 frontend、
target/strategy/phase/required-fields/provides/requires 等 composer selection metadata。
新 IntentIR 语义、branch/region、strategy family 或 frontend 仍必须先走代码/schema/spec
release，不能借 data-only RuleCard 绕过。

每次 Gate1 缺口先分类：`existing-card-too-narrow` 用 replace release；只有“现有 schema
op 缺少 derived mechanism”才可用 add release；错误 J/B、错误 C、composer work-item
错误、TTIR parser/normalization 缺口以及新 strategy/op 均不得伪装成 RuleCard 扩充。
zero/ambiguous match 只产生候选证据并结束当前 run，同一 run 不因候选而自动重试。

live、held-out 与 saved-observation recheck 仅通过显式 `--rule-release PATH` 装载；不扫描
目录、不读取环境变量、不从网络下载。正式 batch 中切换 release 必须结束旧 batch，
以新 registry digest 全量重跑；历史 snapshot 缺失时 fail-closed，不得用 current default
替代。

### 5.2 RuleRegistry 驱动的 replay API policy

replay AST validator 不再维护与 RuleCard 分离的 libdevice 函数白名单。允许的
`libdevice.<fn>` 精确来自当前 frozen registry 的 `api_mapping` 调用拼写，并随
`KernelSource` 携带 registry digest/允许调用集合进入 compiler 二次校验。允许一张新卡
不等于放开整个 namespace；未在 active registry 映射中出现的 libdevice 调用仍拒绝。
当前真实 `liger_geglu` 证据只批准 `map_unary(tanh) -> libdevice.tanh`，不据此泛化开放
任意数学函数。

deterministic composer 是 **Lift 内部的 π finalizer**，不是 Lift 与 Gate1 之间的
第三个独立生产阶段。source-aware LLM 子响应先形成 J/interface/C/location core，
同一次 Lift 随后生成 canonical π 与 provenance，并把完整 LiftArtifact 交给 Gate1。
Gate1 只允许重新 compose 做完整性复核，不得为不完整 artifact 补造 π。

Composer：

1. Canon(J)，消除 JSON key order、opaque id spelling 和独立节点声明顺序影响；
2. 枚举 J node、domain、access/effect、output、interface scalar、schedule slot；
3. 从 def-use、loop carry、scope、storage、effect 与 RuleCard capability 构造依赖；
4. 固定 phase + structural key 稳定拓扑排序；
5. 生成完整 ReplayPlan、canonical bytes/hash 和 100% leaf provenance；
6. 保存 artifact 后，Gate1 每轮重新 compose 并精确比较。

当前版本为 `deterministic_composer_v5`。strict coverage 与 card selection 均按
`target_kinds` 驱动，并按 `artifact.interface.frontend` 过滤；新增已存在于 IntentIR schema
中的 derived mechanism card 不再需要修改 hard-coded rule-id coverage map。不同 frontend
的同 target card 不得造成当前 frontend ambiguity。

`AccessKind.SYNCHRONIZE` 是 effect construction，不是地址访问。composer 自 v4 起不为它生成
`boundary.access.address` work item，而只选择 `realize_synchronize -> tl.debug_barrier`；
因此不得构造 arange、pointer、predicate 或 load/store mask。该修正来自真实
`high_performance_liger_cross_entropy` 的 zero-rule-match 证据。

`Boundary.broadcasts` 是 operand/axes 级 J
事实，而 `realize_broadcast(node_ref)` 是节点级 π action；同一 `node_ref` 的多个不同
BroadcastSpec 必须合成一个 work item。合并后的 `facts.broadcasts[]` 保留完整配对，
成员 required fields 逐项校验，operand producer dependencies 取并集，`j_digest` 与
rule/node_ref leaf provenance 绑定全部声明。声明换序不影响 canonical identity；完全
相同的重复声明 fail-closed。该规则不增加 π 信息，也不允许 π 替代 J 的 broadcast
语义。`Boundary.broadcasts.axes` 是全局语义轴；`realize_broadcast` v4 必须先投影到
program-domain/local-tile rank：被 program selection 消去的全局轴直接使用 operand，
只有仍在本地 tile 中存活的 rank insertion 才发出 consumer-local `tl.expand_dims`。

`realize_broadcast` v5 进一步把该投影收紧为确定的施工顺序。writer 必须先建立有序的
operand-local axes、consumer-local axes，以及被 program-domain 固定或消去的全局轴；
全局 axis 编号不得直接作为 `tl.expand_dims` 的 local axis。rank-0 operand 或 projected
missing-local-axis 为空时直接使用 operand。只有 consumer tile 中仍物化、但 operand
tile 尚未表示的轴才允许扩维；插入位置按最终 consumer-local axis 的升序解释，显式
插入次数必须严格等于 local rank 差，插入后的有序 axes 与 shape 必须在 elementwise
使用前精确匹配 consumer。相同 `(operand, missing-local-axes, consumer-tile-shape)` 的
projected view 最多构造一次并复用，不能按每条 BroadcastSpec 或每个 consumer 盲目重复
扩维。若局部 axes、projection 或 shape 不能从 J/B/C 与先前 π construction 确定，必须
fail-closed 并回到 source-aware Lift；不得靠补维度直到 rank 相同、`broadcast_to` fallback
或掩盖非 broadcast extent/dataflow mismatch。

典型反例是 LayerNorm `[M,N]`：一个 program 选择 M 后，Mean/RSTD `[M]` 和 eps 在局部
都是 scalar，W/B `[N]` 与 X/output 都是 `[BLOCK_N]`。J 中 axes `[1]`/`[0]` 仍保留全局
语义，但局部施工全部为 direct use；把全局 M=0 直接翻译成 `expand_dims(...,0)` 会错误
制造 `[1,BLOCK_N]`，并与 `[BLOCK_N]` pointer/mask 或 value dataflow 错位。相反，若真实
consumer tile 同时物化 `[M,N]` 而 operand tile 仅物化 `[N]`，M 投影到 local position 0，
此时才精确发出一次 `tl.expand_dims(operand,0)`。

opaque handle alpha-renaming、JSON object order 和独立 J node 声明换序不得改变
canonical π bytes 或 canonical input digest。删除、增加、改 ref、换 step 顺序或直接
替换 mechanism，均在 writer 前由 compose equality 拒绝。

## 6. Lift、normal replay 与 relift

```text
source
  -> Lift {
       source-aware LiftCore(J, interface, C.schedule, C.strategy, location_map)
       strict J/C validation
       deterministic finalize π + provenance
       return complete LiftArtifact(J, C, π, interface, location_map)
     }
  -> source-blind writer
  -> source/replay compile-only with same frontend and C
  -> deterministic TTIR hard comparison
```

normal 失败时只回流 source-aware Lift；不修 replay code：

```text
允许修改：J、C.strategy、location_map
冻结：interface、C.schedule
派生：π、compose_provenance
```

回流证据包括 source fingerprint、净化后的失败类别和确定性 hard diff，不含 replay
源码和 Gate2 结果。每轮 relift 后从 schema、compose、静态扫描、source compile、
fresh replay 全部重启。预算耗尽或差异不再改善时返回 NO_CLOSURE，并保留最佳失败
证据；不得用 fuzzy score 覆盖硬差异。

“差异不再改善”采用 `exact_artifact_and_hard_diff_v1`，不得凭 artifact 相同就在
writer 前停止。至少先完成初始 replay、一次 Relift 和第二次 fresh replay；只有连续
两轮 canonical J/interface/C.schedule/C.strategy/π/registry digest 与全部 hard
component 的结构化 difference signature 同时完全相同，才记录 deterministic
stagnation 并停止。signature 排除 score、自然语言诊断、raw TTIR、计时与 replay
source；generation/compile/parser 不可比较失败会打断连续链。停滞只产生
`NO_CLOSURE`，不新增 Gate verdict；该 artifact 不具备活动 Gate2 资格。

Fresh-writer 不变量：任意一次 writer 调用的输入都不得包含任何更早的
`ReplayGeneration`、replay source 或 replay-local compiler/comparator feedback。
normal、masked-J、paired J/B 与 C.strategy intervention 没有例外；它们只能共享模型
配置、interface/C、规则库、specialization 和确定性 compose 协议。

活动配置和报告同样不得保留 `replay_repair` budget、attempt counter 或
`replay_code_repair=disabled` 一类否定开关。该子系统的缺席由接口与架构回归保证，
不是一个可重新开启的运行选项。每个 replay 都只是当前 LiftArtifact 的不可变观测；
只有最终 `REPLAY_CLOSED` candidate 可由 Gate2 原样消费；Gate2 不能修改它、把它送回
writer，或从被拒绝轮次挑选旧 replay。

Gate1 内部审计可以保存完整 `CandidateRecord`，但跨向 Relifter 的接口只能传
`ReliftEvidence`：轮次、是否生成/编译、确定性 comparison、净化后的 diagnosis 以及
adopted/regressed 状态。`ReplayGeneration`、replay code、`CompileResult` 和原始编译
输出在类型边界上均不可见；Relifter 收到原始 candidate 对象时必须在 provider 调用前
拒绝。

完成 source/replay hard comparison 后，`ReliftRequest.semantic_guidance` 必须按
`deterministic_gate1_hard_diff_guidance_v2` 生成。它包含 exact hard-diff digest、失败与
已确认 hard category、去内部 node hash 后按语义 edge role 分组的 delta、应复查的 J
surface、允许考虑的既有 C.strategy family 和当前 J 摘要。该 worklist 的目标不是提高
初始 Lift 一次响应通过率，而是让**已观察 Gate1 差分后的下一次 Lift**作出更有依据的
J gap / strategy gap / no-expressible-change 判断。没有语义依据时必须允许返回 J 不变，
随后由 deterministic stagnation 停止；不得为重采样 writer 而制造 artifact 改动。

provider timeout、配额或 model cooldown 在尚未形成 replay candidate/hard diff 时不具备
语义指导价值，不调用 Relift；它们单独记录为 replay provider unavailable。合法 replay
被 AST policy 拒绝、编译失败或 hard mismatch 仍按现有 source-free 证据回流 Lift。
初始 Lift 的 provider request failure 遵循相同原则：live harness 必须保存结构化
`lift.failure`（无 LiftCore response、无 candidate、不可 semantic Relift）后终止该 case，
不得以未捕获异常丢失报告，也不得把传输重试解释成 J repair。

## 7. TTIR 判定

硬项：

1. interface；
2. A：op multiset、dataflow、reduction；
3. B：predicate/mask/fill/store guard 与 observable effect。

C 只报告。双方都为空的 category 视为硬相等，但 `applicable=false`、score=null，
不贡献聚合 score。score 使用 active-evidence weighting，只作诊断/候选排序；Gate1
不使用模糊相似度阈值。

## 8. 反泄漏与离线 mutation 诊断边界

### 8.1 π derivability

正式主证据是 `saved π == fresh compose(...)`、registry digest isolation、零 orphan
leaf 和 canonical invariance，而不是 masked replay 的分数。

### 8.2 paired J/B intervention（离线诊断，不是 Gate）

以下机制只用于历史报告复核或显式 mutation self-check。它不属于 canonical
Gate1/Gate2 live 拓扑，不是 normal J 的准入条件，也不能形成 Gate0/Gate1.5。其结果
必须标记为 diagnostic，不能计入官方 acceptance evidence。

修改一个注册的合法 J/B 坐标后，冻结 interface/C/registry，**重新 compose π**，
再由 source-blind writer 从 mutant artifact 全新生成 replay。normal replay code 不得
进入 mutant prompt。要求出现精确、注册过的 TTIR delta；
忽略 J、方向错误或 off-target drift 均拒绝。禁止冻结 full-J π 配 mutant J。

### 8.3 可执行 π 必要性与 C.strategy intervention

formal Gate1 在 canonical normal replay 硬闭合后，必须执行两类正式反事实：

```text
J-only:              writer(J, empty π, no RuleCard metadata)
causal step dropout: writer(J, π minus exactly one causal step,
                            remaining selected RuleCards only)
                     -> compile -> same score-free hard comparator
```

每个 RuleCard 以 `ablation_role=causal|administrative` 明确分类；J-only 与每个 causal
step 都必须各自调用一次 fresh source-blind writer。被删 step 的 selection/target metadata
不得出现在 packet；仅因其他剩余实例仍需要同一通用 RuleCard 时，通用卡可保留，但不能
携带被删 target。裁决为：compile nonclosure 或 hard mismatch 是 causal kill；hard closure
是 `LEAKAGE_REJECTED`；provider/generation/evaluation failure 是
`NO_CLOSURE + pi_evidence_incomplete`。在 writer 前调用 compose equality 拒绝 tampered π
只能是 canonicality self-check，不能计作 kill，也不能令 `attempts=0` 的记录通过。

另行运行的实现机制替换只属于离线 C.strategy diagnostic，不是 formal Gate 条件。它
只能修改一个 C.strategy binding，再重新 compose/replay。J 与 interface/
C.schedule 必须不变；J 语义硬项应保持，预期实现结构可以变化。直接篡改 π rule
不是合法的 C.strategy intervention；它只能通过上述明确标记的 post-admission
counterfactual 进入 writer。

### 8.4 masked-J + Gate1 canonical π 终检

normal replay 的 opaque rename 只关闭命名和显式文本通道，不能验证 π 的 rule
序列、步骤数量、引用复用和依赖拓扑是否足以替代 J。因此 canonical Gate1 在 normal
硬闭合后直接执行一次有意构造的反事实攻击，不依赖 paired intervention：

```text
normal:   writer(full J, canonical π, interface, C, rules) -> 必须硬闭合
masked:   writer(masked J, same canonical π, interface, C, rules)
          -> 合法候选在编译期拒绝，或编译后硬指纹不闭合
```

masked packet 不是合法 LiftArtifact，不对 masked J 重新 compose π；它先校验原始
完整 artifact，再仅在 writer view 中遮蔽 J/A/B 内容与全部 J/ABI/C 命名，同时保留
Gate1 admission 后保存的 canonical π 规则序列和引用拓扑。活动 writer API 根本不提供 prior-replay
candidate 参数；masked writer 只调用一次，不接收 normal replay、diff 或 repair
feedback。生成代码恢复 opaque ABI 后，
使用同一 frontend、同一 C、同一 compile case 和同一 TTIR hard comparator。

裁决只认确定性硬结果：

- masked 编译且 hard fingerprint 全部闭合：`LEAKAGE_REJECTED`；
- masked 合法候选被 compiler 拒绝，或编译后至少一个 hard category 不同：反泄漏
  攻击未闭合，允许保留 normal
  `REPLAY_CLOSED`；
- masked 没有形成可审计 generation、ABI 恢复失败或指纹工具不可用：实验
  inconclusive，Gate1 返回
  `NO_CLOSURE`，不能以 normal opaque 或 paired intervention 替代；
- 不使用 score ratio、向量阈值或空集合满分裁决。

masked 硬闭合证明“J 被遮蔽后剩余 packet 仍足以恢复源指纹”，但不能单次归因为
π 独占责任；interface/C/规则卡、低熵 Kernel 和 writer 先验也可能参与。若要归因，
可另做 π-neutralized 对照，但该对照不覆盖上述保守 Gate1 verdict。

### 8.5 官方 fresh live 的零干预边界

高性能 Kernel 的官方 fresh live profile 固定：

```text
semantic_interventions = 0
pi_ablations            = true (required inside formal Gate1)
gate1.interventions     = []
```

`test/scrc_live_e2e.py` 的 formal profile 固定该边界；不存在关闭 π 必要性证明、缩小
causal-step 集合或把 mechanism substitution 代替 dropout 的 CLI。normal/轻量诊断 profile
可以不运行昂贵反事实，但其产物不能冒充 formal live acceptance evidence。

该 profile 不运行 paired J/B/C intervention，因此 `wrong_fill_delta` 不可能出现在其
正式调用链或 acceptance evidence 中。若旧枚举仍用于 mutation/self-check，它只描述
人为 mutant 的诊断结果，绝不是“修正 normal J”的因果干预，更不能帮助 normal
artifact 通过 Gate1。

官方 profile 仍要求 actual-π masked-J 终检。masked-J 是 normal hard closure 后的
单向反泄漏攻击：它不修改 normal J、不把结果反馈给 Lift/Relift、不能提高 normal
score，也不能把失败候选变成通过。fresh contract 必须同时检查干预预算为 0、
`interventions=[]`、masked generation 为新的真实 provider 调用且 masked 结果没有
hard-close。任何条件不满足都不是正式 live PASS。

## 9. 固定 C 多 specialization

`Gate1SpecializationFamily` 第一阶段只允许 fixed-C：共享一个 artifact、一份 J/π、
一份 fresh replay source；runtime tensor shape/scalar 与 grid 可按显式 profile 改变。

每个 profile 纯代码检查 ABI kind/order/dtype/rank、动态 shape symbol binding、静态
extent、grid rank 和冻结 schedule。source/replay 对每个 profile 分别编译比较，family
采用严格 all-pass：任一 profile 失败即 NO_CLOSURE，禁止平均 score、多数票和
profile-local replay repair。

schema-30 使用 `fixed_c_family_full_sweep_relift_v1` 将 normal family sweep 接入每个
artifact round 的 writer 之后、Relift 之前：同一 replay 必须完成全部声明 profile，
任一失败都阻止 normal Gate1 closure；完整 sweep 的有序差分按
`exact_ordered_fixed_c_family_hard_diff_v1` 聚合，只调用一次 source-aware Relift。已闭合
profile 是下一轮不可回归约束，下一 artifact round 必须重新验证全部 profiles。

family stagnation 仍使用 canonical replay-bearing artifact digest，但第二项改为完整
family hard-diff digest；只有连续两轮完整 fresh family sweep 完全相同才停止。source/
replay compile、interface、parser 或 setup failure 不产生 family digest。Relift history
只暴露 profile id/stage/pass/digest；去内部 node hash 的逐 profile 差分由
`deterministic_fixed_c_family_hard_diff_guidance_v2` 提供，不传 replay code/raw TTIR。

`fg_ones` 的保存真实 LLM artifact/replay 曾用 schema-30 stack 复验：6/6 normal family
hard-close，历史三方 Gate2 6 shape × 3 seed 共 18/18 零误差 PASS。该结果不自动升级为
当前 replay/scalar v2 execution protocol。family-wide paired
intervention 与 actual-π masked-J 尚未接通，因此报告必须标记
`anti_leakage_scope=primary_profile_pending_family_wide`，不得宣称完整 family
anti-leakage 已完成。

历史 schema-31 曾将 paired mutation 诊断也接入同一 fixed-C family。每个注册
paired J/B intervention
只生成一份 mutant replay，并在所有 profiles 上分别与该 profile 的 normal replay
fingerprint 检查注册 exact delta；任一确定性因果差分失败即拒绝，writer/compile/parser
不完整则返回结构化 `intervention_evidence_incomplete`，不得伪装成 leakage 结论。

actual-π masked-J 同样只生成一份 masked replay。源码 ABI 恢复后，所有 profiles 都必须
编译并形成可比较 fingerprint，且逐项 hard mismatch；任一 profile hard-close 即
`LEAKAGE_REJECTED`，任一 profile 缺少可比较证据即 `masked_evidence_incomplete`。masked
replay 的 pointer access/memory effect 变化本身属于 hard interface/effect 差异，不能被
前置 artifact-interface 校验吞掉。复核该历史协议时使用：

- `fixed_c_family_exact_delta_all_profiles_v1`；
- `actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1`；
- `normal_paired_and_masked_fixed_c_family_all_required_v1`。

π schema、实际 π 拓扑、C 和 RuleRegistry 均未因此改变；扩展的是同一实验臂的 profile
覆盖。报告中的 `anti_leakage_scope` 只有在该能力接通后才可写为
`full_fixed_c_family`。

schema-31 的 family evidence digest 必须绑定完整对象身份，不能只哈希 verdict 文本：
paired arm 绑定 baseline/intervention J digest、normal/mutant code SHA-256、canonical
family contract，以及逐 profile 的 score-free exact delta；masked arm 绑定 actual π、
masked/restored code、family contract、逐 profile source/replay fingerprint 与 hard-diff
digest。score、rank、diagnostic prose 均不得进入身份。语义失败与 compile/parser 等
不完整同时出现时，family outcome 必须是 `semantic_failure_and_incomplete`，语义失败
优先决定 leakage reject，但全部 profile 证据仍完整保留。

## 10. Gate2

活动 Gate2 协议为 `final_j_replay_scalar_differential_v2`。它只在 Gate1
`REPLAY_CLOSED` 后，将 final accepted J 严格交接为 typed `IntentProgram`，并对该 J
的 replay executable 与 scalar MLIR lowering 使用同一输入矩阵实跑。两腿均成功时按
`exact_integer_tolerant_float_v2` 比较：整数/布尔精确比较，浮点应用逐输出容差。
任一腿失败或 scalar unsupported 只形成 execution coverage gap；只有真实
`replay_scalar` mismatch 是完成的语义 non-PASS。

Gate2 使用 backend-neutral Intent→Linalg/loops→LLVM CPU scalar reference，但不调用
CUDA/ROCm/RISC-V production target lowering、归档 JSON 解释器或 source runtime。
旧三方 Gate2 规范与 Lower2C/CUDA adapter 已归档，见
[归档说明](back/gate2_production_lowering_v1/scrc-gate2-three-way.md)；中间存在的
`source_replay_dynamic_equivalence_v1` 也只是 legacy report protocol。

Gate2 使用的 replay 必须与最终 Gate1 artifact canonical digest 精确对齐。每个
`CandidateRecord` 在形成时绑定当前 artifact digest；最终 artifact 改变但未形成新
candidate 时不得运行 Gate2，禁止选用旧轮 replay。typed `IntentProgram`、
`ScalarLoweringSpec` 与 replay 必须同时绑定 final artifact digest、Gate1 J digest 和
各自 canonical digest；fixed-C specialization evidence 遵循同一对齐规则。

suite 指标只在所有 cases 都是完整合法 replay/scalar 终态时完成；若出现 `case_id`，
所有 cases 必须有非空唯一 ID；声明的 `status_counts` 必须等于实际计数；顶层
`passed` 必须等于逐 case 推导值。input digest、两腿成功状态或实际 comparison evidence
缺失，以及任何聚合矛盾，都只进入 attempted-but-incomplete coverage，不得计作
Gate2 non-pass，更不得计作 PASS。

schema-26 绑定 `scrc_case_performance_v1`，记录 initial Lift、Gate1 loop、family、
π ablation、Gate2 与 total 墙钟以及可审计逻辑调用数。性能字段只作 observation，
不进入 prompt、ReliftEvidence、J/C/π、fingerprint 或 verdict；未执行阶段为 null，
空统计集合不被写成 0 耗时或满分。

schema-27 进一步绑定 `exact_artifact_and_hard_diff_v1` 与
`triton-ttir-3.6-gate1-v2`。后者使用 lexical region scope 解析 SSA，允许 sibling
region 合法复用同名 SSA，但仍拒绝同一 region 的 duplicate definition。旧 schema-26
报告不得静默恢复为当前正式结果。

schema-28 绑定 `deterministic_gate1_hard_diff_guidance_v1` 和
`gate1_diff_guided_deterministic_pi_relift_v9`。它同时规定没有 replay candidate/hard
diff 的 provider transport/cooldown 失败不得调用 Relift。旧 schema-27 只验证了
parser/stagnation identity，不能静默恢复为当前 diff-guided Relift 正式结果。

schema-29 绑定 `structured_no_closure_termination_v1`。活动 Runner 的每个
`NO_CLOSURE` 必须结构化区分 deterministic stagnation、replay provider unavailable、
Relift budget exhausted、Relifter unavailable、Relift failed 和 masked evidence
incomplete；schema-31 同一 termination policy 又增加 intervention evidence incomplete。
禁止 live/batch 从诊断 prose 猜测。termination 不改变 verdict；保存的 replay 可以
另做离线诊断，但 Gate1 未闭合时不得产生活动 Gate2 evidence 或 PASS。

schema-30 绑定 `gate1_fixed_c_family_diff_guided_pi_relift_v10`、完整 fixed-C normal
family sweep/Relift/stagnation，以及历史
`scrc_gate1_gate2_conditional_v2` 三方四象限汇总。该旧报告不得混入当前统计；
schema-29 cache/report 也不得静默升级为 schema-30 family 证据。

schema-31 绑定
`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`、held-out identity 31 和
完整 fixed-C paired/masked 终检。schema-30 保存报告仍是 normal-family 历史证据，不能
静默升级为 schema-31 anti-leakage evidence。schema-31 后续证据收紧仍保持同一 verdict
协议，但要求 family contract/code/fingerprint/artifact digest 的完整身份和 Gate2
artifact-replay 对齐；旧报告缺少直接 candidate digest 时，只允许通过同轮、match 状态、
完整 compose provenance 恢复身份，否则按 coverage gap 处理。

held-out identity 36 绑定当前 `scrc_gate1_gate2_conditional_v4`、
`final_j_replay_scalar_differential_v2` 与 `intent_program_scalar_cpu_mlir_v1`。只有
Gate1 `REPLAY_CLOSED`，且 evidence 显式携带当前 execution protocol、两腿均成功、
input digest 合法并完成 replay/scalar comparison，才进入当前统计；
scalar unsupported、setup/execution incomplete、协议不匹配和未运行样本单独进入
coverage。旧四象限中的 `gate1_not_closed + gate2_*` 在 v4 中必须保持为 0。

schema-31 fresh `fg_log_softmax` 后，Relift prompt 使用
`deterministic_fingerprint_structure_guidance_v1`：精确记录 loop carry/yield、reduction、
memory effect、mask 和 dot 结构，并按 fixed-C profile group 去重。低层 family difference
改为 consensus/profile-specific exact groups；完整 comparison 仍保留在 Gate1 audit，
prompt history 只发送 digest/category/count 摘要，禁止用 score 或模糊比例替代。只有
observed Gate1 diff 后的 Relift 路径会据此要求区分 semantic recurrence 与 schedule-only
loop；generic initial Lift 目标不因此改变。

当前 scope 补强版本为 `triton-ttir-3.6-gate1-v3`。v2 的 lexical binding stack 保留，
但 `x + 0`、`1 / sqrt(x) -> rsqrt(x)` 两条 bounded normalization 已从全函数文本阶段
移动到 scope-resolved SSA graph，确保 sibling/nested region 是否复用局部 SSA spelling
不会改变 fingerprint。未注册的 region-bearing op 必须 fail-closed，禁止静默 flatten
未知 SSA scope。

当前结构指导版本为
`deterministic_scoped_fingerprint_structure_guidance_v2`。loop witness 除 bounds/effects
外，还必须携带 block-argument 数量、induction/carried argument use counts、exact
init-to-yield pairs、外部 captures 和 body op multiset。它只在第一次真实 Gate1 hard
diff 后进入下一次 source-aware Lift，不改变 generic initial Lift 目标，也不授权把 TTIR
拓扑逐项复制到 J。scope-v3 对 `fg_log_softmax` 的重检显示两个 source `scf.for` 都只有
induction block argument，没有 init/yield carry pair；因此现有证据更支持 schedule/output
traversal loop，而不是缺失 semantic loop carry。

## 11. 当前明确限制

以下涉及 `lowering_diverged`、`lift_diverged` 或 Lower2C 的 schema-26～32 结果均是
旧三方／source-replay Gate2 的历史记录，只能按 legacy 协议解释；当前 Gate2 只产生
replay/scalar v2 状态，也不把 production target lowering 纳入裁决。

- branch-region 尚无一等 IntentIR op，因此 branch scope fail-closed；
- loop MLIR prototype 当前只实现有限 carry dtype；
- fixed-C family runner 已实现；跨 C family 尚未定义；
- schema-26 的 17-Kernel fresh 批次中，`fg_log_softmax` 暴露 sibling region SSA
  scope 缺口；parser v3 已将解析与 bounded normalization 一并 scope 化。schema-31
  fresh 已形成 2 个真实、可编译 replay，
  四个 fixed-C profiles 均在 `op_multiset + dataflow + predicate_effect` 不闭合；legacy
  source/replay Gate2 对 single/exact tile 6/6 PASS，对两个 multi-N profiles 6/6
  `replay_diverged`。source
  有 2 个 runtime loops、replay 为 0，当前 J 也为 0 loop/0 carry；该证据已驱动
  deterministic family consensus diff、结构语义 guidance 和 bounded Relift prompt。
  scope-v3 witness 进一步确认两个 source loop 均无 carry pair；不能再把 loop count gap
  本身解释为 J semantic recurrence 缺口；
- schema-26 的 `fg_arange` 在三轮 hard-diff signature 不变时仍消耗完整 Relift 预算；
  deterministic stagnation early-stop 已实现，schema-31 fresh 已验证第二次真实 replay
  后以同一 artifact/exact hard-diff digest 提前停止；其 9/9 PASS 只属于 legacy
  source/replay diagnostic，不具备 v2 Gate2 资格；
- scope-v3 fresh `fg_index_select` 覆盖二维间接寻址和三个 fixed-C shape profiles：两轮
  replay 均可编译，Relift 返回相同 artifact，exact family hard diff 不变后触发
  stagnation；legacy Gate2 为 6 PASS + 12 `lift_diverged`。动态复现发现安装的 FlagGems source
  使用已弃用的 tensor `and`，多行输出不完整，因此该样本违反 trusted-source 正确性
  前提，不能用于归因 J/replay 准确率；
- schema-32 fresh `fg_eye` 已验证 composer-v3：首次 Lift 与 normal replay 均一轮成功，
  四个 fixed-C profiles 全部 hard-close，masked-J family 四个 profile 均为可比较 hard
  mismatch，Gate1 为 `replay_closed`。π 顺序满足 iota producer → node-level broadcast
  → comparison/select，且每个 broadcast step 的 provenance 绑定两条 J 声明。独立
  schema-32 legacy 三方 Gate2 最初为 6 `lowering_diverged` + 6 `execution_failed`；Lower2C v4
  接入已有 Boundary axes 后，同一保存真实 artifact/replay 无模型复验为 12/12
  `gate2_pass`，Gate1/family/masked 结论不变；
- schema-32 fresh `fg_trace` source preflight 通过，但三次 Lift 分别因未注册 `triu`、
  非 compiler scalar shape symbol `M`、二维输入 tensor_ref rank 错误而失败；尚未进入
  Gate1，不能记为 Gate1 false reject。其 fixed-C profile 改为保持静态输入 shape，
  仅变化 `num_diag={1,1024,1025}`，三者 source compile 均通过；
- schema-32 fresh `fg_embedding` normal replay 已编译并 hard-close，但实验性
  `K={1,17,33}` family 因 J 的静态 batch extent 17 无法绑定 launch grid extent 而
  fail-closed。非法 profiles 已撤回。当前 `PortableScope(program_instance)` 不提供
  grid-derived tensor extent 绑定；若新增该 schema 能力，必须先提交计划外必要性报告；
- scope-v3 fresh `fg_skip_layer_norm` 在一次 hard-diff-guided Relift 后 normal replay
  全硬项闭合；两个 masked-fill paired interventions exact-delta 通过，但 final
  `add→sub` mutant 同时删除两个 `/N` 和一个整数转浮点，因 off-target drift 返回
  `LEAKAGE_REJECTED`。这不是 masked-J/full-π topology closure，后者因 paired 阶段已拒绝
  而未运行。独立 legacy Gate2 原 contract 3/3 PASS；
- 同一 `fg_skip_layer_norm` normal replay 已在固定 `BLOCK_SIZE=512`、
  `N={1,257,511,512}` 上 Gate1 family 4/4 hard-close。36-case legacy 三方 Gate2 matrix 中
  normal/large-dynamic 27 PASS，near-constant 在三个非单元素宽度上 9 个
  `lowering_diverged`；source↔replay 精确一致，当前保留 production Lower2C 数值稳定性
  缺口，不以统一放宽 tolerance 覆盖；
- actual-π masked-J 已在 `fg_addr` 上 hard-close，证明 opaque naming 不能消除全部
  topology leakage；Gate1 保守拒绝是当前正确行为。当前不修改 π 拓扑模型，只保留
  批次监测；后续跨 family 出现成批 masked hard closure 时再报警并提交必要性报告；
- 17-Kernel 批次五次 Relift 均未改变 J/C.strategy/C.schedule/canonical π，当前只证明
  通道常开和完整重启，不证明 Relift 已能稳定修正语义核心；
- RuleRegistry 新特性只从真实 Kernel/Lift/source-replay diff 增加，不按想象枚举；
- 当前实现不宣称已覆盖所有 Kernel。schema-26 的 17-Kernel 结果为 12 closed、
  1 leakage-rejected、1 no-closure、2 lift-failed、1 parser-blocked；完整数据见
  `doc/SCRC_Gate1_17Kernel规模性能实验_2026-07-17.md`。当前 schema-33 manifest 为
  27 candidates / 20 ready；新增 `fg_index_select` two-axis indirect indexed gather、
  `fg_diag_extract` strided-gather 和 `fg_skip_layer_norm` fused norm 均已有 fresh terminal
  report，但分别仍受 trusted-source 异常、Lower2C gather contract 与 paired-writer
  causal drift 限制；
- schema-26 保存的 14 份真实 Lift/replay observation 曾完成 legacy 诊断迁移：
  Gate1 normal hard-close 13/14、source/replay Gate2 v1 14/14 PASS。`fg_arange` 的
  `Gate1 no-closure + legacy Gate2 pass` 只作历史观测；v2 不运行 rejected artifact，
  该结果不得合并为 overall pass，也不替代 schema-31 fresh SCRC；
- schema-31 已复用保存真实 `fg_ones` artifact、normal replay、paired mutant replay 与
  actual-π masked replay 完成诊断迁移：normal 6/6 hard-close、paired exact delta 6/6、
  masked-J 6/6 可比较 hard mismatch、legacy Gate2 18/18 PASS。该结果不替代 schema-31 fresh
  Lift/replay，也不能外推到其他 family。
