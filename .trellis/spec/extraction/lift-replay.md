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
round；同一 branch-local Lift owner 在累计 active wall-time 与 sandbox/provider 资源合同内继续修正，
默认不以 model/tool/check/pre-Replay rejection 次数作为能力截止。拒绝响应、receipt、耗时和 token
作为 proof-irrelevant sidecar 保存，不能只留下一个错误字符串，也不得进入 Replay 或认证谓词。
owner 的 terminal resource signal 可以结束为 `NO_CLOSURE`，但报告必须显示正式 Gate1 round 为 0，
而不是伪造一次 Gate1 失败。worker 连 candidate 都没有提交时同样属于 pre-TypeCheck Lift rejection；报告必须
显式显示 submitted candidate、canonical decode、TypeCheck、Replay 与 Gate 执行数均为 0，不能
把 aggregate `NO_CLOSURE` 解读成一个已经进入 Gate1 的 artifact 未闭合。

### 时间治理的持久 agent 实现

Lift 可以由仓库内的 mini-swe agent 实现。一个 branch 只拥有一个 source-aware conversation 与
一个受限 persistent workspace：

```text
一个 branch
  = 一个持久 source-aware Lift owner conversation
  = 同一 Source 的 Gate1 continuation 复用同一受限 workspace
  = 每次暂停点最多提交一个完整 (J,C,pi)
  = Gate1 FAIL 后在原 conversation 中继续
```

活动运行时只允许这一 `single-owner` workflow；旧 planner/builder/reviewer staged 路径只保留历史
evidence，不得从 CLI 或配置重新启用。MiniSWE 与 Codex CLI 是同合同的两个执行面，正式 branch
必须在创建前选择其一，不能中途交接 owner/candidate/feedback。当前 Task14 批量闭合以 Codex 为
主路径，MiniSWE 用于共享工具链核验与备用执行；后续任务可以按吞吐重新选择。两者必须共享同一
artifact/schema/checker/privacy/Gate 合同；选择执行引擎不改变 Lift、Replay 或证书谓词。

该 session 内可以多次调用模型，按需读取匿名 Source、最小 Harness compiler view、闭合 JSON
Schema、source-neutral Lift 合同与短检查表，在私有草稿目录中修改 candidate，并运行 JSON
语法/静态 Schema 检查。这是普通文件上的轻量 retrieval；不需要 vector database，也不能按
当前 Kernel、历史失败或 sibling candidate 动态选择知识库。

agent 还可以对自己已经写出的完整 candidate 调用只读、非认证的 `CandidateCheck`。为避免
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
每次交付仍只提交一个完整 `(J,C,pi)`。提交前置检查失败只把同样的 closed diagnostics 返回当前
owner，不会提交、
修补、canonicalize 或派生任何 artifact 字段；worker 必须自行修改并提交完整 object。generic
`apply_patch` 只把 LLM 明示的唯一 `old_text` 原子替换成其 `new_text`，不解析 JSON、不返回建议值、
不分配 handle，也不构造字段。只有当前 candidate
bytes 与本 session 内某次 `PASS` 对应的精确 bytes 完全相同时才能复用；任何不同 bytes 都必须
重新检查。提交后 owner conversation 暂停而非销毁，Runner 必须重新运行全部权威
Decode/TypeCheck；工具 `PASS`
不进入 Replay packet、Gate verdict 或 certificate，且 CandidateCheck/Runner 必须使用相同
`typecheck_version`。激活的 vNext SCRC adapter 必须安装 CandidateCheck；checker 缺失或不可用属于
提交前 infrastructure failure。generic environment 可以在非认证测试中不安装 checker，但不得把
该路径报告为 vNext SCRC run。

agent 内部多步和 CandidateCheck 不增加正式 Gate round，也不削弱门禁。报告必须分别记录外部
Lift invocation、owner session id/累计 active wall time、agent model turns、实际 provider calls、
CandidateCheck 调用次数与诊断类别。
Replay、Gate1、Gate2与hidden input在session内仍然绝对不可调用。

source-aware owner可以在CandidateCheck/SourceReconcile通过后，独立调用一次非认证的public
`SemanticProbe(Source,RefEval(J))`。它使用与hidden Gate2不同的预先固定seed；Source腿使用当前候选C，
RefEval腿只接收J与logical inputs，π和Replay均不进入。工具只返回逐输出差异与外部Q的当前诊断，
不得生成、修改、接受或拒绝artifact。大差异由LLM自行回读Source/J并修改；unsupported或小幅浮点
归约差异不能成为membership verdict。该probe调用与结果作为proof-irrelevant telemetry保存，正式
TypeCheck、fresh Replay、Gate1和hidden Gate2仍由外部Runner独立执行。

当前 active single-owner 路径默认每 branch 累计 3600 秒 active Lift wall time；`step_limit=0` 与
CandidateCheck count limit 0 都表示不按次数截断，而不是无资源约束。model/tool/check 调用全部作为
telemetry；每个响应最多四个独立只读 query 只是原子工具 envelope。只有 TypeCheck-valid 候选进入
fresh Replay/Gate1 后才消耗本次显式声明的正式 `K`。submission `PASS` 仍非认证，外部 Runner 必须重跑
TypeCheck。

MiniSWE 的无状态 provider transport 可以把旧 response/tool 大字段替换成可重读 exact bytes 的
hash/length receipt；正式 continuation 不再重复嵌入同一 workspace 中已经存在的完整 candidate，
owner 按路径读取 exact bytes，同时保留最新 closed feedback、完整本地 trajectory 与
persistent workspace。Codex 可以通过隔离的 allowlisted host bridge 调用同一
CandidateCheck/ReplayCheck/public synthetic compile。bridge 不得写 artifact 或选择/修复语义；
Replay episode 的 syntax-only composer 可以把该 agent 自己写入的有序 source fragments 原样拼成
`replay.source`，这不是 J/π 到目标代码的映射。两条路径均保持 configured reasoning effort，不把 repair/continuation 从 `xhigh`
自动降级。

每个持久 Lift owner 在创建时固定自己的 Source/Harness/schema/knowledge/tool 私有输入快照；后续
invocation 只刷新该 branch 的 closed feedback（Codex 另刷新 CandidateCheck 结果）。仓库中并行发生的
工具升级不得替换或使既有 owner 失效；同一 owner 的 Source/Harness bytes 真正变化则仍 fail closed。

默认 exact stagnation detector 只在连续三步的工具请求（忽略 provider 分配的 call id）、当前
candidate/Replay bytes digest 和完整 closed output 三者全部相同时终止为
`OrchestrationStall`。任一请求、bytes 或诊断变化立即清零连续计数；这不是 model/tool/check 次数预算，
不评价语义质量，也不选择或回滚 candidate。

首稿可由标准 `submit_candidate(candidate_text)` function tool 在一个 action 中原样写入完整文本并
调用同一 submission preflight；现有文件的 exact patch 可设置 `submit=true`，在唯一替换后调用
同一 preflight。两者的输入内容均完全由 LLM author，runtime 不构造字段、不选版本、不返回修复；
FAIL 仍是同一 closed diagnostics，PASS 后 Runner 仍独立重跑。

单次 provider request 与 owner 总时间分离：MiniSWE 的普通 effort 单次 request 当前上限为 300 秒，
`xhigh` 为 900 秒；Codex CLI 单 turn 上限为 900 秒。owner 累计 active Lift 上限仍为 3600 秒；
timeout 不再次重烧，
transient transport failure 仍按有界 retry 恢复。正常路径是完整 candidate 直接 submit，由同一
action 自动 preflight，PASS 立即交付、FAIL 返回短 diagnostics。模型 API packet 可把旧的大型 tool
arguments/outputs 替换为 deterministic hash/length receipt；原始审计 trajectory 与 persistent
workspace 文件不被修改，agent 需要精确内容时重新读取。该 transport compaction 不解释 Source 或
J/C/π，也不进入 correctness predicate。

若 `source.py + harness_view.json + feedback.json + lift_contract.md + preflight.md + minimal_example.json`
总量不超过 64 KiB，初始 user packet 直接携带六者的 exact raw text；否则只携带逐文件
bytes/hash/read command。Gate1 continuation 的 closed diagnostics 采用相同 64 KiB 规则。fresh Codex
owner 还直接获得 compact `source-outline.v2`：它只列出 AST 已有的 control 位置、descendant-return
presence、literal memory-call 顺序，以及“loop 前已有 lexical binding 且 loop 内被写”的名称；这些
只是导航/checklist，不把名称分类成 carrier，不决定 effect 或 frontend tuple order。该 prefetch 不生成
semantic outline、摘要、operator、binding、π 或修改建议；目的是让首个高 effort 回合直接做语义
工作，而不是决定下一条读取命令。

`intentir.scrc.triton-workload-adapter.v6` 把 tensor shape 的**参数恒等关系**与本次 concrete
valuation 分开：`shape` 仍是本次可执行值；`shape_contract` 每个轴只能是正整数或匿名 Harness-only
`h*`，相同 `h*` 表示跨 buffer 的同一 source-grounded logical extent；可选 scalar
`shape_coordinate=h*` 或 explicit `bound_constexpr_shape_coordinates` 表示 runtime scalar/semantic
constexpr 与 extent 相等；`grid_shape_coordinates` 只记录 launcher 如何 realization 该 extent。Lift
必须把每个 `h*` 显式、双射地作者化为自己的 declared `x*`，并在 scalar/constexpr 同时是 logical
data 时用 rank-0 `t* + value_dimension_relations(eq)` 保留相等关系；fixed
CandidateCheck/Runner 只核对完整映射，不能从 concrete `1/2/33/16/64`、grid 或参数名生成 `x*`、
relation 或 J shape。整数 contract 仍必须保持静态。`h*`、grid 与本次 valuation 都不进入 J/C/π、
Replay 或 production；production 只消费最终已作者化的 `x*`。不同 `h*`（例如 valid sequence 与
rounded capacity）保持独立，除非 Source-aware Lift 在现有 J/B vocabulary 中另行作者化其语义关系；
Harness 不按数值猜 `ceil_div`、容量或有效域公式。v6的`input_profile`只进入Harness/evidence receipt，
用于区分capture、model/upstream与direct-tile valuation；它不进入J/C/π、Replay或Gate，也不向Lift
提供operator/shape公式。

持久 Lift owner 还可使用四个等合同的只读投影：`harness-abi` 将匿名 Harness view 中已经声明的
`p*/c*` 顺序、bound non-C constexpr membership、compiler role 与 `h*` shape-coordinate ledger
并置；`source-outline` 只分页上述 syntax-only navigation facts；`candidate-control` 只分页当前 LLM
已写出的 J control/reduction nodes、branch arms、boundary
output/effect/guard、以及 π regions/instances/iterations/carriers/terminal/routes/order；
`feedback-category` 只投影 closed frontier 已命名的一个 Gate1 类别。它们是 attention/transport
工具，不是第三份 IR：不选择 `logical_ref`、`x*`、operator、predicate、effect、output mapping 或
π relation，也不读取 Source/Gate/Replay。MiniSWE 的 `context_query.py` 与 Codex bridge 提供同一
字段投影，任何缺失仍由 Lift 自己从 Source 作者化，任何冲突仍由 CandidateCheck/TypeCheck
fail closed。

Codex Lift v12另在owner创建时，以同一匿名Source和冻结匿名compiler view执行一次
`intentir.lift-source-ttir.v1`。成功结果只在该branch的immutable input中保存exact TTIR与现有Gate1
observer fingerprint；owner可分页读取TTIR、按token或单个`%SSA`查看局部上下文、读取syntax
outline或一个fingerprint path。该工具不解释Source、不决定J/C/π所有权、不生成relation或patch；
raw TTIR/fingerprint不写入模型trajectory，更不得进入Replay、证书或production。

Codex Lift v18 在 source-free `CandidateCheck` 之后增加独立的
`intentir.lift-source-reconcile.v1`。它只把模型当前完整候选可确定投影出的 compiler obligations
（direct transfer multiplicity、structured/physical iteration 与 carrier arity、已声明 reduction
linearization、闭合时的 pointer-address operation、program-coordinate occurrence，以及无歧义的
ordered transfer）同该 owner 已冻结的 Source TTIR observer 比较。FAIL 只返回匿名 handle、rule 与
expected/actual count/order；模型仍须自行修改完整 J/C/π。该工具不生成字段、关系、源码或 patch，
不读取 Replay/Gate feedback，不形成 TypeCheck/Gate/证书结论；投影不能唯一确定时跳过该项，最终
一致性仍由 fresh Replay + Gate1 裁决。v18 的 host submission preflight 要求当前 exact bytes 依次
通过 CandidateCheck 与 SourceReconcile，并分别记录调用与诊断。

Codex Lift v19 不再把整份 relift `feedback.json` 暴露给持久 owner。前一 artifact 已存在于同一
workspace 的 `output/candidate.json`；closed Gate1 diff 只按既有 protocol label 分成固定的
`feedback_category_NAME.json`，frontier 仍是首读入口，非 observer 诊断另存一个小 detail 文件。
这是 exact bytes 的机械分区，不选择类别、不生成修复。首次 contract 也只要求读取 top-level shape
与当前需要的 leaf section，禁止把整份 schema/op catalog pretty-print 进线程；最终 artifact 与所有
字段仍完全由同一 LLM owner 作者化。
Codex Lift v20 同时删除与普通文件读取重复的 `feedback_frontier` host tool；workspace 中 exact frontier
文件是唯一入口。当前特殊 bridge 只保留无法由本地文件操作替代的CandidateCheck、SourceReconcile，
以及上述显式调用、非认证的public SemanticProbe。

Codex Lift v11 / knowledge v68 将同一 exact Harness ABI ledger 与 syntax-only
`source-outline.v2` 直接放入首次 owner prompt，并将
后续 closed feedback frontier 中已有的 interface、reduction、predicate/effect 与 op-multiset 直接
类别一次并列展开到同一持久 owner continuation；recursive dataflow 保持按一个命名类别分页。
representation-only op 差异只要求 owner 对照 Source 重新审查已有 π refinement grammar，fixed code
不选择 refinement、不写 π，也不把 compiler op 复制成 J semantics；
`lift_harness_abi`、`lift_candidate_control` 与 `lift_feedback_category` 保留为可选的局部自读，不再是
提交前的强制串行协议。MiniSWE knowledge v64 暴露同一 outline/checklist 合同。该 transport 不生成
`logical_ref`、J/C/π 字段或修复。

当前持久 owner 使用 `artifact.v12 / decode.v17 / typecheck.v30 / candidate-check.v1` 实现上述
CandidateCheck、C attachment、numeric-cast dependent typing 与 result-relation compatibility。v30
只接受同一authored `effect_if`两臂各唯一完整store定义的conditional post-state output；不补
`select`、mapping或新字段。v7
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

`artifact.v8` 历史上对 J 中没有 structured loop、但 Source 用物理 tile loop实现已有 J 事实的情形
增加 `pi.physical_iterations` 与 `pi.physical_loop_carriers`。body instance 可以锚定具有 closed axis
的 reduce/contraction，也可以锚定同 subject/axis 的 boundary access/domain，从而表达输出 epilogue、
transfer traversal 等不改变 `[[J]]` 的物理时间；tile instance 仍必须锚定同 subject/axis 的 required
positive-integer C slot。后者只绑定 typed init/state/next/result 时间拓扑。两者都不携带
bound/formula/value/API，也不让 fixed code创建 J loop、选择锚点或推导 traversal。真实性和遗漏继续
由 fresh Replay/Gate1 检查。
每条 physical carrier 是一个 tuple slot：slot 内四个 stage 类型一致；不同 slot 可异构、无需匹配
body instance 类型，因此 pointer/address recurrence 与 accumulator 都由 Lift 分别作者化。
`artifact.v9` 再加入 `pi.terminal_controls`，只在 J 已有 `effect_if` 的 sibling arm scopes 上指定
terminal/continuation topology；不复制 predicate、return、Source 或 frontend spelling。
source-blind construction compile 对每条 relation 只要求至少一个 compiled terminal conditional，拒绝
用普通 structured value/effect `if` 冒充 early-return CFG；它不生成 predicate、arm 或 return。
`artifact.v10` 再加入 `pi.route_extent_transforms`：只允许 attached required positive-int C extent
在一条已存在 route 的一个 target slot 上按 literal divisor>1 做 floor/ceil partition。Lift 必须显式
作者化 source/target/slot/divisor/rounding 与 C attachment；checker 不从 Source、J domain 或 valuation
推导。它只约束 Source realization，不能创建 J shape/domain/predicate，production 仍不读取 π。

同一 v9 wire 的 closed refinement vocabulary 现包含 `reduction_linearization`。它只表示 Source 已有的
全轴线性化表示阶段：`auth_ref=null`，refined instance 必须是 `realize`，且恰有一条 slot-0 route
进入一个 J `reduce` 的 `realize` instance。逻辑 axes 和 reducer 仍只在 J；π 不携带 axis、API 或
Source text。Lift 必须从 Source 显式作者化，TypeCheck v25 只验证 topology，不能因 Gate1 缺
`tt.reshape` 而自动补造。

Lift receipt 分别记录 artifact/decode/typecheck version，以及 CandidateCheck 调用次数与 closed
diagnostic rule codes。正式提交后 Runner 仍独立重跑 Decode/TypeCheck；`artifact.v1/v2` 历史对象按
原 wire 读取并禁止 `pi.iterations`，不能被静默升级到 v3。DYT 与 SwiGLU 的既有闭合 receipt 保留，
不因 wire 升级重跑，也不冒充 fresh v3 evidence。

若 owner 在一次 active episode 中没有提交 candidate，Runner 可以把其最后一次**实际执行**
的 source-free checker 诊断中最多 20 条 `rule_code/path` 写入同一 branch workspace 的 closed
feedback view；没有 evaluation 就没有摘要。该 view 与真实 user continuation 一起追加到原 conversation。observed value、
候选修复、hidden input 与 sibling 信息不得跨越；Source 只因它本来就在该 source-aware owner 的
同一 conversation 中而保留。固定 runner 不把摘要应用成 patch。

若完整 JSON object 已提交但被外部 canonical parser 拒绝，Runner 可以把该完整 rejected candidate
和与它精确对应的机械 diagnostic 交回同 branch 的持久 Lift owner。owner 在原受限 workspace 中
继续修改当前完整对象，再提交完整 `(J,C,pi)`；feedback 中的历史对象只作 receipt/fallback，不得
覆盖更新的 working bytes，也不能只交 patch-only artifact。
rejected candidate 不是第四 artifact，不进入 Replay 或 certificate。没有提交对象、对象不是有限
JSON、feedback 不匹配或跨 branch 时不得把其他候选混入当前 owner。

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

因此，Lift 对 closed ContractExpr grammar 已能表示的 Source index/domain/predicate，必须作者化保留
原始整数算子树（如 `//`、`%`、`ceil_div`、`+`、`-`、`*`），不得擅自改写为代数等价式。
这是对 compiler-observed realization topology 的保真要求，不是 TypeCheck 增加等价证明，也不由
固定代码替模型补造表达式。

## Replay

```text
ReplayLLM(J,pi,MaskC(C);U_f) -> complete source R_f
```

当前外部 packet protocol 仍为 `intentir.scrc.replay-input.v2`。Codex Replay backend
`openai.codex-cli.intentir-replay.v51` 只把 packet 中 canonical π 的固定 relation list 编为
`intentir.pi-min.relational-column-wire.v1`：columns 明示字段顺序，发送前必须精确 round-trip 回原 π；
它不修改 artifact π、TypeCheck judgement 或证书身份，旧 evidence 也不因 codec 升级而重判。

每个 formal round 是一个 fresh、single-writer、source-blind coding episode，以 active wall time而非
极小 tool-call 数治理。默认由 agent 直接维护一份完整 `replay.source` 并做局部 patch/compile；不得
为了 host 协议把几 KB 源码强制拆成每 occurrence 一个文件。只有 agent 自己判断有利时，才把源码拆为
本轮 `work/composer/*.source` fragments；plain mode只按词法文件名拼接 exact UTF-8 bytes，或另写
episode-local
`work/replay.compose.json` 与 `NNN-iK.source`：fragment 中的 `{{out}}`、显式 operand
`{{slot:N}}` 和按同 target empty-slot route 声明序编号的 `{{dep:N}}` 都是 agent 作者化目标语法的
一部分，host 只按 accepted π 中已经显式、唯一的 route 替换 stable local ID，并
检查 closed source/target、slot、sharing 与 order。host 不选择 target op/expression/control/API，
不把 π relation 翻译成 code fragment；歧义、缺 occurrence、重复 slot 或错误顺序立即拒绝。direct
完整 `replay.source` 是首选 coding 路径。agent 可反复 render、ReplayCheck、public synthetic compile 与
查看自己 Replay 的 TTIR；任何 manifest、fragment、checkpoint 或 helper 都不得跨 formal round 保留。
\nv50/v8 只升级 proof-irrelevant Agent execution transport：活动 staged writer 被移除，MiniSWE 使用
一个持久 Lift owner 和每轮一个 exact Replay packet；首轮 exact allowlisted workspace mirror 在
模型收到一次后可于后续无状态 provider packet 中按路径重读 exact 文件；旧 response/tool group
只保留最近窗口，完整本地 trajectory 与 workspace 文件不变。Lift 可把多个显式 schema lookup 合并为
一个只读 bundle；Replay 可把完整 LLM-authored source 直接原样提交，或对自己的当前 source 做唯一
exact-text replacement 后在同一 action 触发既有 ReplayCheck preflight。固定代码不得据此解释、生成、
重排或修复 J/C/π/Replay 语义。每个 formal round 的 Replay conversation/workspace 仍全新；transport
receipt、startup packet view 与 transaction audit 都不是 Replay 输入 artifact 或 acceptance oracle。
v51/v9 只增加上述 exact stagnation fail-fast 和分离的 Replay/Gate1 progress telemetry。v52/v10
再只增加 first-winner cooperative cancellation：同 case 首个已完成 Gate1 closure 被记录后，正在
Lift、MiniSWE Replay 或 Codex CLI provider turn 中的 sibling 可被终止并保存
`candidate_superseded + winner_branch_id` receipt；取消检查位于 transport retry 之外，不能把取消
伪装成 transient retry，也不丢弃已经完成的 winner evidence。这些版本都不改变 artifact、packet、
TypeCheck、ReplayCheck、Gate1 或 Gate2 接受集合。

MiniSWE 的 `intentir.replay-authored-requirements.v2` 只是对 immutable packet 中既有事实的
有界 exact 重排：除 π time/control/route 外，它可列出 J/B 的 accesses、aliases、
output mappings、physical reduction tile axes、index bounds、updates、consumer extensions、
access guards 与 result relations；超过上限的行保留精确 section query。Codex 启动只收到 packet/
contract/tool 的路径和字节数，自己按需读文件，不再重复注入该 projection。两者都不读取 Source、
Harness value 或 Gate，也不把 J op/relation 映射成 target code。普通 effort 的 fresh Replay episode
默认可用三个 provider wall-time window，xhigh 可用两个长 window；到最终截止点时，只有当前
LLM-authored exact bytes 已重新通过 Replay ingress 与 synthetic compile 两项固定检查，host 才可在
缺少最后一次 submit 调用的情况下保留它进入正式 Gate1。任一检查未通过仍以 infrastructure failure
结束，host 不补写或修复源码。

该 requirements view 还按 declaration order 原样列出每个 J node 的
`id/op/operands/attrs/result`，并把 empty-slot program-domain route 与其 branch region 作 exact join。
这些行只是在同一 packet 内重索引，防止 coding agent 漏读或改写已有 J；它们不映射 target API、
不生成源码，也不能成为 Gate1 membership oracle。

`replay-check.v10` 在 v9 的 transfer、iteration、carrier 与 pointer-address guard 之外，按 π 中
`reduction_linearization` 的作者化数量要求 synthetic TTIR 至少出现相同数量的 distinct
`tt.reshape -> tt.reduce(axis=0)` dataflow；额外
表示 stage 留给正式 Gate1 精确比较，避免这个非认证 probe 误当 membership oracle。它不从 J reduce、
shape 或 Gate diff生成 relation。v9 地址规则只按
J.signature 中已声明的 pointer/tensor carrier kind 和 π 已作者化的 address routes 传播
pointer reachability，并核对 synthetic TTIR 的 `tt.addptr` occurrence。address instance 可以锚定
`p*/d*/a*` 等不同 authored ref；检查不按 `semantic_ref` 字符串配对，而要求每个 transfer 从绑定同一
boundary tensor 的 pointer/tensor signature carrier 可达，并要求每个 pointer-producing address stage
最终流向至少一个 transfer。独立 integer base/lane contribution 可以具有自身 semantic ref，并不因此令闭包失效。
无法满足该闭包时不猜 final address。它不读取 Source/Gate，不构造 offset、
program id、lane、dtype/cast 或 frontend spelling，也不具有 Gate/认证权威。
v26/v22 还把 J.signature 的每个 `p*` 固定解释为一个已存在的 callable ABI carrier：同一 `p*` 的
多个 π instance 只定位或路由该 carrier 的多处 use，不生成新值或 program coordinate。source-blind
construction compile 从位于 program region、沿作者化 route 到达 `address_base` 的 distinct J domain
精确投影合法 program-coordinate 数，并拒绝额外 compiled `tt.get_program_id` occurrence；该检查不
分配 axis、不生成 coordinate，也没有 Gate 或认证权限。

v55 的 continuation 默认读取 deterministic bounded `feedback_frontier`：interface、reduction、
predicate/effect、op 与 dataflow 只保留计数、少量 direct examples/local frontier/effect-order witness；
完整 closed evidence 按 category/path 查询。projection 不解释 Source、不推荐或应用 patch。Replay
v12/v4 同时明确 logical J axis ordinal 不是 frontend active-tile rank axis；一维 tile reduction 使用
tile-local axis 0，不得照抄原 logical axis number。

`U_f` 是版本化、source-neutral 的公共前端文档与工具上下文，不按当前 Kernel/op/pattern/失败
样本选择。Replay 看不到 Source、Harness、C values、source parameter name/position、真实
BindingMap、独立 interface sidecar、source/compiler fingerprint、旧候选、旧 Replay、Gate1 diff、
旧模型 conversation context 或 per-kernel RuleCard。`MaskC(C)` 只暴露 C 的匿名 schema/slot
视图，不暴露 valuation 与私有 provenance。

`MaskC(C)` 中只有 `kind=constexpr` 的 slot 进入 Replay callable ABI 并保持 frontend constexpr
annotation；`kind=compiler` 的 slot 只声明由外部 compiler invocation 消费的匿名 role，不得出现在
Replay 函数签名、body 或 π route 中。正式 Gate1/Gate2 在 Replay 提交后以与 Source 相同的 C value
绑定该 role；source-blind construction compile 不读取其 value，也不能把它伪装成公开代表值。

Replay 必须生成完整、可独立构建的当前前端源码；固定代码只做 syntax-byte composition、绑定和
reject-only 检查，不根据 J、π 或 compiler diff 选择目标语义、修复或施工。

Replay instruction 的共享前缀只能陈述 frontend-neutral 的 J/C/π ownership、拓扑保持、拒绝与输出
合同。`U_f` 只声明 callable/import ABI 与公共文档入口，不再列举 `J/π relation -> tl.dot/tl.sum/
program-axis/address spelling` 的 RuleCard。具体 public frontend API 与源码组织由 coding agent 根据
J/π 自行选择；construction compile 与 Gate 只拒绝。π 的版本化 schema定义 relation 的意义，但不
指定某个前端 API。

π region placement 不是隐式代码生成指令：`placement=program` 不等于存在 `program_id`。只有 π
中另有一个表示 compiler-visible program/grid coordinate 的独立 `realize` instance 与 routes 时，
Replay 才能生成对应 stage；不得按常见 kernel 模板从 placement、grid 惯例或 C 猜出额外 grid
offset、整数乘加或 affine address。
J.signature `p*` 同样不会因位于 program region 或拥有多个 π instance 而变成 program coordinate；
所有同名 instance 共享该唯一 ABI carrier。只有下述 distinct J-domain relation 可以授权 coordinate。

现有关系的机械解释是：锚定 J domain、位于 program region 且 scalar dependency path 进入
`address_base` 的 `realize` stage 是 program coordinate；进入 `address_lane` 的 domain/C path 是
lane coordinate。`pi.program_axes` 为每个 distinct coordinate domain 作者化精确 axis `0..2`；
domain_ref 唯一，但多个 logical consumer domain 可按 Source realization 共享一个 physical axis，
compiled coordinate 按 unique axis 只 materialize 一次。锚定同一 `d*` 的多个 instance 仍分别
realize但共享该 coordinate；instance multiplicity、声明顺序、grid extent 与 C 均不能分配或重编号
axis。若 empty-slot route 从 program-region d* instance 进入 `placement=branch` region，该 region
只在 J domain 内 realization；J 保存 domain 公式，π 不复制 predicate。具体 frontend spelling 由
agent 选择并由 construction compile/Gate 拒绝，`U_f` 不再提供 relation-to-API 映射。

若该 coordinate 在 Source realization 中还执行 row-major mixed-radix decode，
`pi.program_linearizations{domain_ref,dimension_refs}` 以 outer-to-inner J `x*` 顺序给出唯一 closed
projection。它只引用已存在的 J dimensions，不携带 concrete extent、公式文本、predicate、grid、
Source/API 或 target recipe。Replay 必须消费其整数 dataflow，fixed code只检查引用与闭合，不生成
实现。若 coordinate 还参与 logical block interval/mask，相关 mul/add/compare/select 必须显式在 J；
π 不得复制一条 partition relation。

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

一个锚定 J contraction 的 `realize` instance 只授权一个 compiler-visible contraction occurrence；
若 Source 本来有多个显式 stage，Lift 必须用 π multiplicity/order 表达。Replay 自己选择该 frontend
的公开构造，但不得把一次 occurrence 展开成额外 compiler-visible stages，也不得为了某个 API 修改
J 或删除 π stage。这个约束来自 occurrence identity，不是 op-to-API RuleCard。

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

Gate1 的结构化 diff 只返回同一 branch 的持久 source-aware Lift owner。owner 保留自己的 Source
上下文，并读取刚提交的完整 `(J,C,π)` 与本轮闭合 Gate1 diff，继续修改后重新输出完整新三元组。
随后使用新三元组发起全新的盲 Replay；Replay conversation 不续接。

runner 必须完整保存 observer comparison 作为审计 evidence；返回 Lift 的可以是该 comparison
的确定性、有限、可行动诊断投影，例如 hard category、op-count 差异、关键 topology 差异与每类
少量代表边。该投影不得改变 Gate1 verdict、隐藏失败类别或替 Lift 修改 artifact；低层节点
fingerprint、重复 components、score/rank 等非行动信息不应占满 relift 上下文。完整 evidence 与
relift diagnostic view 必须明确区分。

禁止把 diff 交给 Replay、patch 旧 Replay、只输出 artifact 增量，或跨候选复用 Replay。

协议允许独立的 `intentir.scrc.gate1-diagnostic-recheck.v1` 对一份已落盘 artifact 做只读诊断：
保持 artifact bytes/digest 不变，执行当前 canonical Decode、`typecheck.v30`、一个 fresh normal Replay
与 Gate1，并写入独立 `recheck-NNN` evidence。它不调用 Lift、不运行 masked-J 或 Gate2、不参与
candidate/winner 选择，也不改变 parent run 的 verdict 或 certificate；因此只用于定位固定 consumer/
observer/context 改动，不能冒充 fresh SCRC rerun。provider 调用前必须把 selected case/current prepared
request、source projection 与 source bytes 对上 parent run receipts，并要求 parent Gate1 记录的
artifact digest 精确等于输入 artifact；错 case、错 artifact、错 Source 或缺少旧 lineage 字段均
fail closed。

当前同一 case 使用 `1 <= B <= 5` 个独立并行 branch；每次运行声明 `1<=K<=12`，每 branch 最多 `K`
个 TypeCheck-valid 正式 Gate1 round，且 `K` 包含首次候选。Task 14默认上限为5；每个 branch只有
一个持久 Lift conversation；其 previous artifact、
Gate1 diff 与活动时间状态同 sibling 完全隔离。Replay writer 则每个正式轮次全新，不能接收 Lift
conversation、旧 Replay 或 Gate1 diff；sibling candidate 不能进入 Lift/Replay 输入，也不能拼接为一个
artifact。Runner 采用版本化的 `first_observed_gate1_closure_cooperative_stop.v4`：第一个向 fan-out
coordinator 交付完整 Gate1-closed evidence 的 branch 原子成为 winner，随后只收集 sibling 的
cooperative-stop receipt，而不再等待它们继续搜索后续 formal round。已经同时闭合但稍后被观察到的
branch 记录为 `GATE1_CLOSED_NOT_SELECTED`；被取消的 branch 记录为 `SUPERSEDED`，不能伪装成
`NO_CLOSURE` 或 infrastructure failure。只有 winner 随后获得隐藏 Gate2 输入并执行一次 Gate2，
Gate2 失败不得 fallback 到 sibling。该调度依赖 proof-irrelevant 的完成次序，所以 report 必须记录
selection rule、winner branch/formal round 和各 sibling stop receipt；它不能读取 Gate2 输入/结果，
不改变 winner 自身的 TypeCheck/Gate1/Gate2 认证谓词。没有 winner 时，仍只有所有 branch 完整耗尽
并各自为 `NO_CLOSURE` 才能报告 `NO_CLOSURE_(B,K)`；infrastructure-incomplete branch 必须单独报告。
