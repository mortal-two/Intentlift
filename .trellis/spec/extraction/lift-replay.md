# Lift 与 Replay 合同

> 状态：当前派生规范。上位设计见 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。

## Lift

```text
Lift_H(Source) -> (J,C,pi)
```

Lift 直接完成 whole-tensor semantic lifting、C 分离与 source realization π 的提出。proof-relevant
输出只有 J、C、π。diagnostic、location、provider receipt 等可以作为 proof-irrelevant sidecar
保存，但不得进入 Replay、Gate verdict 或 production lowering。

J 必须包含 logical signature；不允许 root sibling `interface` 成为独立对象。实现若需要接口，
只能按需计算 `InterfaceView(J)`。H 可以向 Lift 提供匿名、未分类的 bindable coordinates，但
不得预先替 Lift 做 J/C/π 语义分类。

匿名 handle 是机械 wire contract：J tensor 必须按 `intent.inputs` 后接 `intent.outputs` 独立、
无间隙地编号为 `t0,t1,...`，不得把 `p*` 参数位置直接复制为 `t*` 后缀。例如 input pointer `p0`、
output pointer `p1`、runtime extent scalar `p2` 对应 input `t0`、output `t1` 与 dimension `x0`。
`harness_compiler_view` 中 runtime scalar 的当前 value 只是本次 Harness binding；若 Source 用它控制
logical extent、index、predicate 或 result，J 必须保留 symbolic `x*` 并在相关 tensor shape 中引用，
不得把当前 value 特化成 J 常量。`x*` 是 signature/type handle，不是 SSA value，不能进入 portable
scope definitions/captures 或 storage。

`J.signature.logical_ref` 只允许指向声明的 J input/output tensor 或 symbolic dimension，不能指向
body 内部 `v*` SSA value。logical runtime scalar 或 non-C semantic constexpr 若作为 predicate、
operator 或 result 的逻辑数据，必须声明 dtype 相容、shape `[]` 的 rank-0 J input，并让该输入实际
进入 J dataflow；只有 integral extent-only handle 才绑定 `x*`。value 参数不能为了复用现成 ID 而
绑定方向或 dtype 不相容的 output tensor。

runtime scalar 若只用于 representation/address realization，则不是 logical data：Lift 声明
`kind=scalar,mode=value,rank=0,logical_ref=null`，不创建 J input，π 可引用该 p* carrier。分类只能
来自 Source use，不来自参数名/位置。任何以 `logical_ref=t*` 创建的 rank-0 input 必须进入
observable output 的反向依赖，或由显式 boundary/shape semantic relation 使用；identity/direct-output
与合法 boundary-only use 保留，dead node/π-only route 不算。`typecheck.v7` 只执行这一窄规则，
拒绝但不自动解绑/删 input/重编号，也不臆测一般 tensor liveness。

`logical_ref=null` 的 representation-only 形态只允许 runtime scalar。Harness 中的
`bound_constexpr_values` 必须已有 J logical denotation：logical data/predicate/result 绑定
dtype-compatible rank-0 J input，integral extent-only 绑定 declared `x*`；只影响 layout/tile/stage
等 implementation selection 的 constexpr 必须归入 C。无法归入 J 或 C 的 constexpr 使 Harness/case
invalid，Lift 不得把它留作 π-only carrier。

同一 v7 只在无 logical guard 的 predicated direct load、完整零起点矩形 index/domain、predicate
bound 与同轴 typed tensor extent 相等，且 masked axis 能经已建模 closed pointwise/broadcast 位置
传播并被 `reduce(mean)` 消去时，以 `padded_mean_count_unproven` 拒绝：中性 fill 只闭合分子，当前
J/B 没有 logical valid-count/denominator contract。dense/unpredicated mean、guard-bearing、
cross-axis/general predicate、domain/extent 不闭合、无关非消去轴 mask 与 unknown-op 传播不触发
TypeCheck；guard/cross-axis/unknown-op 由 consumer 明确 unsupported。Lift 必须按 Source 明示的阶段
生成例如 `reduce(sum)` 后 explicit division；CandidateCheck/TypeCheck 不读取 Source/C/π，也不自动
改写候选。

对 ordinary dense logical access，Lift 只需使用 closed ContractExpr surface grammar。`k0`/`[k0]`
与 `k1,k0`/`[k1,k0]` 分别由固定 elaborator 归一为相同 structured axes；这类标点差异不触发
relift。predicated pure functional boundary 的 observable store producer 类型若能由 closed op
signature 与 J graph 唯一确定，可以省略；显式类型必须一致。direct same-handle identity mapping 可
省略或显式写出，canonicalizer 统一为省略形式；非恒等 `returns`/mapping 仍必须明确且一致。
`fill_value` 只允许出现在带非空 predicate 的 load 上；无条件 access 的
predicate/fill 均为空。static schema 可以提前报错，TypeCheck 仍必须权威拒绝 orphan fill，消费者
不得静默删除。这些是全局 J wire/consumer 合同，不是 per-kernel Replay hint。

对可识别的零起点矩形 functional-domain profile，`typecheck.v6` 进一步要求 direct access index、
padding predicate、`boundary.domains` 与 typed tensor extent 显式闭合。形如 `k0 < x0` 的 predicate
必须直接索引 `k0`，并有匹配的 `0 <= k0 and k0 < x0` domain；Source、C、π、shape、runtime/hidden
input 都不能补缺失 relation。该检查只覆盖这个可机械决定的 profile，不把 general affine/交叉轴
logical predicate 因当前 RefEval 未实现而判成非法，也不删除合法 predicate/fill/effect。

provider 已返回、但因 schema/匿名合同不合法而被拒绝的响应不进入 TypeCheck，也不计正式 Gate1
round；Runner 对这类 pre-Replay rejection 使用有限调用上限并如实报告。拒绝响应、receipt、耗时
和 token 作为 proof-irrelevant sidecar 保存，不能只留下一个错误字符串，也不得进入 Replay 或
认证谓词。上限耗尽可以结束为 `NO_CLOSURE`，但报告必须显示正式 Gate1 round 为 0，而不是伪造
一次 Gate1 失败。worker 连 candidate 都没有提交时同样属于 pre-TypeCheck Lift rejection；报告必须
显式显示 submitted candidate、canonical decode、TypeCheck、Replay 与 Gate 执行数均为 0，不能
把 aggregate `NO_CLOSURE` 解读成一个已经进入 Gate1 的 artifact 未闭合。

### 有界 agent 实现

Lift 可以由仓库内的有界 mini-swe agent 实现，但其 agent loop 只属于一次生成动作：

```text
一次 LiftAdapter 调用
  = 一个全新 agent session
  = 一个独立受限 workspace
  = 最多提交一个完整 (J,C,pi)
```

该 session 内可以多次调用模型，按需读取匿名 Source、最小 Harness compiler view、闭合 JSON
Schema、source-neutral Lift 合同与短检查表，在私有草稿目录中修改 candidate，并运行 JSON
语法/静态 Schema 检查。这是普通文件上的轻量 retrieval；不需要 vector database，也不能按
当前 Kernel、历史失败或 sibling candidate 动态选择知识库。

agent 还可以对自己已经写出的完整 candidate 调用有界、只读、非认证的 `CandidateCheck`。为避免
已完整序列化的候选绕过这一机械诊断边界，若当前完整 candidate bytes 在本 session 内还没有
`PASS`，提交动作必须先自动调用同一 `CandidateCheck`。该工具通过窄 host RPC 或同等隔离边界，
共享正式 Runner 的同一 source-free strict parser、Decode/Elaborate、anonymous-view consistency 与
`TypeCheck(J,C,pi)` callable core；正式 Runner 另执行 source-identifier privacy、receipt/provenance，
并在正式提交后重新运行该 core。禁止维护第二套近似 checker：

```text
ctx = {
  anonymous_compiler_view,
  schema_version,
  decode_version,
  typecheck_version
}
CandidateCheck_v(candidate_bytes, ctx) -> PASS | structured diagnostics
```

工具不得：

- 修改 candidate 或写 `/output/candidate.json`；
- 返回 patch、corrected/canonical artifact 或 recommended field value；
- 创建 logical binding、operator、authorization、result relation 或 π topology；
- 读取 Replay、compiler observer、Gate1、Gate2、RefEval、历史/sibling evidence；
- 执行行为测试或签发正式 TypeCheck/证书。

`ctx` 精确等于 agent 原本已经可见的匿名机械 Harness compiler view 与协议版本；RPC 禁止读取
未声明的 ambient host state。工具可以在私有内存中构造 ephemeral parsed/elaborated validation view，
但不能将其作为 artifact 返回。diagnostics 只包含 closed
`protocol/status/rule_code/path/observed_refs_or_types`。agent 根据 diagnostics 自行修改完整 object，
最终仍最多提交一个完整
`(J,C,pi)`。提交前置检查失败只把同样的 closed diagnostics 返回当前 worker，不会提交、
修补、canonicalize 或派生任何 artifact 字段；worker 必须自行重写完整 object。只有当前 candidate
bytes 与本 session 内某次 `PASS` 对应的精确 bytes 完全相同时才能复用；任何不同 bytes 都必须
重新检查。提交后 session 结束，Runner 必须重新运行全部权威 Decode/TypeCheck；工具 `PASS`
不进入 Replay packet、Gate verdict 或 certificate，且 CandidateCheck/Runner 必须使用相同
`typecheck_version`。激活的 vNext SCRC adapter 必须安装 CandidateCheck；checker 缺失或不可用属于
提交前 infrastructure failure。generic environment 可以在非认证测试中不安装 checker，但不得把
该路径报告为 vNext SCRC run。

agent 内部多步和 CandidateCheck 不增加正式 Gate round，也不削弱门禁。报告必须分别记录外部
Lift invocation、agent model turns、实际 provider calls、CandidateCheck 调用次数与诊断类别。
Replay、compiler observer、Gate1、Gate2 与 RefEval 在 session 内仍然绝对不可调用。

当前 fresh worker 最多 24 个 model turns。CandidateCheck 总 evaluation 上限为 8：7 次
`shared_budget` 可由显式调用或提前 submission preflight 消费，另有 1 次
`submission_reserve` 仅允许 submission 使用；任何顺序都不能超过 8。receipt 按
`explicit`/`submission` trigger 分别记录 calls/evaluations，并分别报告 shared allowance 与 reserve。
submission `PASS` 仍非认证，外部 Runner 必须重跑 TypeCheck。

当前 fresh worker 使用 `artifact.v3 / decode.v3 / typecheck.v7 / candidate-check.v1` 实现上述
CandidateCheck、C attachment、numeric-cast dependent typing 与 result-relation compatibility。v7
继承 v6，v6 继承 v5；v5 继承 v4 的
rank-0 closed signature、Layer-A graph value typing total 与显式 NumPy broadcast 合同；它只消费 Lift
已在 `J.boundary.broadcasts` 中声明的 non-trailing relation，不推导或补写轴。v5 新增
已知 `reduce` 与 binary `contraction` 的 canonical fail-early，拒绝别名、缺失/非法 canonical
attrs 与可机械判定的 axis/rank/shape 矛盾；它还拒绝 absent
`opaque(mode=value,dtype=none)` 成为 π instance semantic ref、realization authorization subject 或
refinement carrier。v6 新增上述 functional-domain closure fail-early；v7 新增上述 rank-0 liveness/
representation carrier 合同及 `padded_mean_count_unproven` necessary condition。checker 不翻译/
补齐 attrs、生成 dummy carrier、将 mean 改写成 sum/div 或修复 artifact。既有
typecheck.v3/v4/v5/v6 receipt 保留原 judgement，不由 v7
静默重判。
`artifact.v3/decode.v3` 只新增 required `pi.iterations` closed relation。在 attachment 闭合的
C-tiled structured J loop/slot pair 中，只有 role 为 `tile_extent|reduction_tile_extent` 的 pair
在 relation 完全遗漏时触发 mechanical missing-totality；`extent_step` 表示 `[0,Trip)` 以 C tile 为
step，`tile_ordinal` 表示 `[0,ceil_div(Trip,Tile))` 的 unit-step ordinal。显式 relation 可引用任意
role 的 required positive-integer slot，只要 attachment 指向同一 loop/trip axis；axis ordinal `0` 与
对应 `trip.axis=x*` 都合法。
Lift 仍完整选择并生成该 relation；Decode/TypeCheck 只验证，不从 Source、op family、名字、concrete
value 或 frontend 惯例补造。不同 physical loop region 可分别引用同一 pair；checker 不能发现被 Lift
完全遗漏且没有 handle 的 Source region，也不能发现 custom/misclassified role 下完全遗漏的 relation；
两者必须由 Gate1 topology 暴露。

Lift receipt 分别记录 artifact/decode/typecheck version，以及 CandidateCheck 调用次数与 closed
diagnostic rule codes。正式提交后 Runner 仍独立重跑 Decode/TypeCheck；`artifact.v1/v2` 历史对象按
原 wire 读取并禁止 `pi.iterations`，不能被静默升级到 v3。DYT 与 SwiGLU 的既有闭合 receipt 保留，
不因 wire 升级重跑，也不冒充 fresh v3 evidence。

若 worker 在预算内没有提交 candidate，只允许把其最后一次**实际执行**的 source-free checker
诊断中最多 20 条 `rule_code/path` 摘要带入同一 branch 的下一 fresh Lift；没有 evaluation 就没有
摘要。observed value、候选修复、Source/Gate/hidden input、conversation 与 sibling 信息不得跨越。
新 worker 仍从完整对象构造开始，固定 runner 不把该摘要应用成 patch。

若完整 JSON object 已提交但被外部 canonical parser 拒绝，Runner 可以把该完整 rejected candidate
和与它精确对应的机械 diagnostic 交给同 branch 的紧邻下一次 fresh Lift invocation。新 worker 应
在完整对象上做聚焦修正并重新提交完整 `(J,C,pi)`；它不能恢复旧 trajectory，也不能只交 patch。
rejected candidate 不是第四 artifact，不进入 Replay 或 certificate。没有提交对象、对象不是有限
JSON、feedback 不匹配或中间隔着其他 invocation 时不得复用更早候选。

agent shell 必须 fail closed 地隔离：`/input` 只读、`/output` 只允许唯一 candidate、私有 `/work`
可写、网络 namespace 隔离、宿主环境清空，且不挂载仓库、历史 evidence、Gate executable 或
provider secret。隔离设施不可用时报告 infrastructure failure，不得降级到宿主 LocalEnvironment。

> 实现状态（2026-07-23，不是认证证据）：第一次 DYT canary 执行了 3 个 fresh MiniSWE Lift
> invocation，每个达到 8 model turn，共 24 次物理 provider 调用；三次均在浏览 schema 时耗尽预算，
> 未提交 candidate，因此 canonical decode、TypeCheck、Replay、Gate1 和 Gate2 均未执行。该记录
> 只用于改进 Lift worker，不改变任何认证谓词；详细数据见 Task04 的 run-001。
>
> 第二次 DYT canary（run-002）使用 knowledge-v2：3 个 fresh worker 均在第 9 turn 提交完整
> candidate，证明“只浏览、不生成”已解决；但三份 candidate 都在 canonical signature binding
> 检查被拒，TypeCheck/Replay/Gate 仍为 0。该结果定位了 schema 中允许 `logical_ref` 指向任意
> “value”的旧文字与 canonical parser 只接受 tensor/dimension 的冲突，详见 Task04。
>
> 第三次 DYT canary（run-003）使用 knowledge-v3：3 份 candidate 均通过 canonical decode 与
> TypeCheck，第 3 个 formal round 获得 normal Replay closure，masked-J mismatch 证明 J necessity。
> Gate2 的 Source--Replay 腿通过且误差为 0，但当时 RefEval 因二维 dense access 使用未括号化
> `k1,k0` 而 `REFERENCE_UNSUPPORTED`。Task06 随后把它确认为 response decoding/elaboration 缺口：
> 原 artifact 不经 relift即可规范为 `[k1,k0]`、物化唯一 result type、消去冗余 identity mapping，
> 并通过同一 RefEval boundary。该后续系统修改不回写 run-003 的历史 verdict。

## Replay 前检查

```text
response JSON -> Decode/Elaborate -> (J,C,pi)
TypeCheck(J,C,pi)
```

Decode/Elaborate 是 Replay 前固定代码的一部分，不是 Agent/Gate，也不产生第二份 J。它不读
Source/C concrete values/Gate output，只处理唯一可判定的
surface normalization、handle/type elaboration 和恒等表示消除；无唯一解时 fail closed。

最小 TypeCheck 必须机械检查：

1. J access 满足 `fill != null => load and predicate != null`；
2. 所有 J/C 引用存在；
3. route operand slot 与类型兼容；
4. observable instance 锚定 J-owned node/access/effect；
5. π 不含语义原子、Source/API、C value 或可执行自由文本；
6. sensitive refinement 引用 J/B authorization，或属于系统级安全白名单；
7. 未知字段、未知 relation 与自由文本扩展 fail closed。

TypeCheck 不做 SMT、range proof 或一般语义等价证明。检查失败时当前 artifact fail closed，
不进入 Replay。它只支撑 `FreshLogicalSemDecl(C,pi | J)=empty` 的显式所有权结论，不证明 π
没有统计语义线索，也不证明 realization 行为正确。

## Replay

```text
ReplayLLM(J,pi,MaskC(C);U_f) -> complete source R_f
```

当前 packet protocol 为 `intentir.scrc.replay-input.v1`，Triton 公共上下文为
`public_frontend_context_v7`；source-aware MiniSWE Lift 的当前 knowledge receipt 为
`intentir.lift-knowledge.v17`。三个版本必须随 fresh run 原样记录；旧 packet/context/knowledge
evidence 不因实现升级而被重判。

`U_f` 是版本化、source-neutral 的公共前端文档与工具上下文，不按当前 Kernel/op/pattern/失败
样本选择。Replay 看不到 Source、Harness、C values、source parameter name/position、真实
BindingMap、独立 interface sidecar、source/compiler fingerprint、旧候选、旧 Replay、Gate1 diff、
旧模型 conversation context 或 per-kernel RuleCard。`MaskC(C)` 只暴露 C 的匿名 schema/slot
视图，不暴露 valuation 与私有 provenance。

Replay 必须生成完整、可独立构建的当前前端源码；固定代码只做绑定和检查，不根据 π 或 compiler
diff 施工。

Replay instruction 的共享前缀只能陈述 frontend-neutral 的 J/C/π ownership、拓扑保持、拒绝与输出
合同。`tl.dot`、`tl.sum`、Triton program-axis 以及 base/lane 到具体 pointer operation 的拼写，只能在
`frontend=triton` 时由 Triton `U_f`/instruction clause 给出；CUDA、CuTe、TileLang prompt 不得收到
这些 Triton 实现话术。π 的 source-neutral mechanism grammar 仍对所有前端共享，它只定义 artifact
tag 的意义，不指定某个前端 API。

π region placement 不是隐式代码生成指令：`placement=program` 不等于存在 `program_id`。只有 π
中另有一个表示 compiler-visible program/grid coordinate 的独立 `realize` instance 与 routes 时，
Replay 才能生成对应 stage；不得按常见 kernel 模板从 placement、grid 惯例或 C 猜出额外 grid
offset、整数乘加或 affine address。

现有关系的机械解释是：锚定 J domain、位于 program region 且 scalar dependency path 进入
`address_base` 的 `realize` stage 是 program coordinate；进入 `address_lane` 的 domain/C path 是
lane coordinate。Triton `U_f` 按 distinct J domain `semantic_ref` 的首次声明顺序分配
`tl.program_id(axis)`；锚定同一 `d*` 的多个 instance 仍分别 materialize，但共享同一 program-axis
identity，instance multiplicity 绝不自动产生下一条 axis。只有 distinct domain anchor 才能分配
下一条 axis。lane path 才使用 `tl.arange` 等 lane realization；placement 单独出现时两者都不能
推断。Triton 的 `tl.arange` 等
compile-time-only API 必须直接读取仍带 `tl.constexpr` 注解的匿名 `c*` 参数，不得先复制到普通
Python local 再传入。单个锚定 `c*` 的 `realize` instance 由参数 binding 和直接 routed use 即可
满足，不要求恒等 alias。`IntentFunction.reduce.axes` 是 logical axes；匹配的
`J.boundary.reductions.tile_axis` 才是 Triton reduction call 使用的物理 tile axis。这里都是固定、
source-neutral 的前端合同，不是 per-kernel hint。

`pi.iterations` 是必须消费的 source-realization topology，不是可选 schedule hint。对
`extent_step`，Replay 使用 J trip extent 为 upper、匹配 C tile 为 step；对 `tile_ordinal`，Replay
使用 `ceil_div(J trip,C tile)` 为 upper 且 step 为 1。即使 induction value 未被 body 使用，也不得
插入/删除 ceiling division 或互换 basis。所有 frontend 接收同一闭合 denotation，只有具体公开语法
由各自 `U_f` 决定；Replay 不得从 Kernel 类型或习惯模板猜 basis。

当前 packet 不携带任何 pad-discharge derived hint。若 J 写出 symbolic input extent、
literal pad 与 static output extent，却没有闭合它们的 dependent equality，U_f 不能读取 Harness
valuation 猜出等式，也不能授权 Replay 省略 pad。应由下一次 Source-aware Lift 给出可检查的 J
relation（或不生成该冗余 pad）；固定 Replay consumer 不在 packet 内建立临时 shape/slot theorem
prover。

对 canonical binary J contraction，若 per-program 两个 operand 与 result 都是 rank-2 floating
tile、恰有一个 contracted label，且声明的 operand/result/accumulation dtype 均由 public
`tl.dot` 支持，则一个锚定该 contraction node 的 `realize` instance 在当前 Triton `U_f v6`（继承
v5 direct-contraction 合同）中使用一个
direct `tl.dot` realization。它不能被展开成 broadcasted outer-product multiply + `tl.sum`，因为
后者制造了 artifact 未声明的 compiler-visible stages。该规则不适用于 outer product、batched 或
其他 unsupported contraction/tile/dtype，也不覆盖 π 已显式声明多个 contraction-anchored stage 的
情况；Replay 不得为了强用 `tl.dot` 修改 J 或删除 π stage。若 Source 本来就是多个显式 stage，下一次
Lift 必须用完整 π multiplicity/order 表达；若 Source 是一次 direct primitive，则只能声明对应的一次
stage。

若 Source 中一个 SSA coordinate/address intermediate 只定义一次却被多个 access 使用，Lift 必须
声明一个 producer、把 consumers/routes 与 one-producer sharing 闭合；多个语法 use 不等于多个
producer instance。只有 Source 中确有不同 compiler-visible definition/stage 时，才可重复同一
semantic ref。Replay 不得通过 `base + (coordinate - base)` 一类代数恒等式伪造额外
materialization 来满足错误 multiplicity。

base/lane 是 address contribution 角色而不是 pointer-add 计数。Source 若形成可分离的
program-uniform integer base 与 lane offset，并在一次 pointer add 前合并，π 仍保留 program path
进入 `address_base`、再 route 到 `address_lane` 的拓扑；Replay 保留两个 integer stage、一个
pointer add，且即使当前 grid extent 为 1 也不能用常量替换 program coordinate。只有 Source 没有
可分离 program-uniform contribution 时才允许 lone `address_lane`。

## Gate1 失败与 relift

Gate1 的结构化 diff 只返回 source-aware Lift。下一轮 Lift 可以读取 Source、上一轮完整
`(J,C,π)` 与累计 Gate1 diff，但必须重新输出完整新三元组。随后使用新三元组发起全新的盲
Replay。

runner 必须完整保存 observer comparison 作为审计 evidence；返回 Lift 的可以是该 comparison
的确定性、有限、可行动诊断投影，例如 hard category、op-count 差异、关键 topology 差异与每类
少量代表边。该投影不得改变 Gate1 verdict、隐藏失败类别或替 Lift 修改 artifact；低层节点
fingerprint、重复 components、score/rank 等非行动信息不应占满 relift 上下文。完整 evidence 与
relift diagnostic view 必须明确区分。

禁止把 diff 交给 Replay、patch 旧 Replay、只输出 artifact 增量，或跨候选复用 Replay。

协议允许独立的 `intentir.scrc.gate1-diagnostic-recheck.v1` 对一份已落盘 artifact 做只读诊断：
保持 artifact bytes/digest 不变，执行当前 canonical Decode、`typecheck.v7`、一个 fresh normal Replay
与 Gate1，并写入独立 `recheck-NNN` evidence。它不调用 Lift、不运行 masked-J 或 Gate2、不参与
candidate/winner 选择，也不改变 parent run 的 verdict 或 certificate；因此只用于定位固定 consumer/
observer/context 改动，不能冒充 fresh SCRC rerun。provider 调用前必须把 selected case/current prepared
request、source projection 与 source bytes 对上 parent run receipts，并要求 parent Gate1 记录的
artifact digest 精确等于输入 artifact；错 case、错 artifact、错 Source 或缺少旧 lineage 字段均
fail closed。

当前同一 case 使用 `1 <= B <= 5` 个独立并行 branch；每 branch 最多 `K=3` 个 TypeCheck-valid 正式
Gate1 round，`K` 包含首次候选。每个 branch 的 previous artifact、Gate1 diff、Lift conversation、
Replay writer 与调用预算完全隔离；sibling candidate 不能进入 Lift/Replay 输入，也不能拼接为一个
artifact。Runner 等待当前 search wave 的所有 branch 到达 pre-Gate2 barrier，再仅依据完整
TypeCheck/Gate1 evidence，按固定 `(branch_id, formal_round)` 顺序选择第一个 Gate1-closed 候选。
未选但已闭合的 branch 记录为 `GATE1_CLOSED_NOT_SELECTED`；只有固定 winner 随后获得隐藏 Gate2
输入并执行一次 Gate2，Gate2 失败不得 fallback 到 sibling。没有 Gate1-closed 候选时才报告
`NO_CLOSURE_(B,3)`；infrastructure-incomplete branch 必须单独报告。wall-clock 完成先后不进入
winner 函数。
