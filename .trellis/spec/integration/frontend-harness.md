# Frontend Harness 合同

> 状态：当前派生规范。上位设计见 [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)。

Harness 是预先存在的运行与调用环境，不是 IR、模型产物或第四个 proof artifact。

## 最小职责

```text
EnumerateBindings(source invocation) -> anonymous coordinates
BindSource(Source,C,inputs)           -> source invocation
BindReplay(Replay,C,inputs)           -> replay invocation
Build(invocation)                     -> executable
Run(executable)                       -> observable result
Observe(executable/result)            -> Gate evidence
```

Harness 可以保存：

- concrete logical inputs 与 C values；
- source/replay 的机械调用坐标；
- grid、device、toolchain 与 compiler options；
- build/run/observation diagnostics。

Harness 不得：

- 根据参数名、`constexpr` 或 Kernel family 判断 J/C/π 归属；
- 生成 operator、predicate、formula、schedule strategy 或 Replay code；
- 把真实 Source binding、compiler diff 或 H 整体交给 Replay；
- 建立独立 `LogicalABI`、`InterfaceArtifact` 或带语义的 BindingMap。

Lift session 的只读 `CandidateCheck_v(candidate_bytes,ctx)` 只能通过显式 `ctx` 接收当前 agent 已经
获准看到的匿名机械 compiler view 与 schema/decode/typecheck version，以检查 candidate 是否篡改
参数 kind/mode/dtype/rank、C valuation 或 anonymous handle 合同。工具禁止读取 ambient host state，
也不得把 hidden inputs、真实 Source binding、Gate evidence 或额外 Harness 状态返回 agent，不得替
Lift 生成 `logical_ref`、C `subject_ref/axis_ref`、J/C 分类或任何 artifact 字段。它共享正式 Runner
的同一 source-free parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行
source-identifier privacy、receipt/provenance，并在提交后重新运行该 core。不得维护
Harness-private 近似规则。

同一权威 Decode/TypeCheck 调用若已经机械判定出多个互不依赖的错误，CandidateCheck 可以按
声明/字段的稳定顺序一次返回多个 closed diagnostics；当前统一上限为 20 项，每个 path/observed
文本上限为 2048 字符。它只能展开权威 checker 已经得到的错误，不得为了继续检查而给缺失字段
填默认值、改别名、删除冲突项或构造 repaired artifact。若 JSON、父对象或某个参数本身尚不能
合法解析，依赖该前提的后续检查必须留到修正后的下一次 evaluation，不能猜测修复结果后提前报错。

当前 single-owner Lift 默认按每 branch 3600 秒累计 active wall time 治理；model turns、tool calls 与
CandidateCheck evaluations 不设低次数型 capability cutoff，全部按 `explicit`/`submission` trigger
记录为 telemetry。owner conversation 与受限 workspace 均在同一 branch/Gate1 continuation 中保留；
Harness 在 owner 创建时固定 Source/Harness/schema/knowledge/tool 私有快照，续轮只原子刷新 closed
feedback，不复制、重排或覆盖当前 candidate，也不因 host 仓库并发升级而替换既有工具。若一次 active episode 没有
提交 candidate，同一 branch 的持久 owner 可以在下一真实 user continuation 中接收最后一次实际 source-free evaluation 的至多 20 条
`rule_code/path`；不得包含 observed value、patch、hidden/sibling 信息。Source 仅作为该 owner 原有
source-aware conversation 的上下文保留。Harness 不应用摘要修补 artifact，submission `PASS` 也不能
替代正式 Runner 的 `typecheck.v23`。

generic exact-text patch 不是 Harness semantic construction：LLM 必须同时提供唯一 old/new 字符串，
runtime 只原子替换并使旧 PASS 失效；Harness 不解析字段、不推荐修改、不生成 corrected artifact。
旧 tool 回包的 deterministic hash receipt 只用于 model transport compaction，完整文件与审计记录仍
留在 owner workspace/trajectory 中。

Harness 可以把 Lift 已获准读取的 exact Source/compiler view 和 deterministic bounded
`feedback_frontier` 放入初始/continuation packet；完整 closed feedback 只给 bytes/hash/path 与命名
category/path 查询。frontier 只投影 checker 已有类别、计数与代表 witness，不解释 Source、推荐字段、
生成 binding 或应用 patch，也不得把额外 hidden state 混入 packet。

Harness runtime 可以接收 LLM 在一个 action 中给出的完整 exact candidate text，或 LLM 明示的唯一
old/new text 与 `submit=true`，原样写入后调用既有 submission preflight。该事务不赋予 Harness
field-level construction/repair 权限，且不替代正式 Runner 重检。

## Logical signature 与绑定

input、output、runtime scalar、destination、dtype、rank、shape、alias/inout/effect 属于 J logical
signature。adapter 若需要函数视图，只能使用 `InterfaceView(J)`，且该视图必须由 J 确定性产生。

C 是一个对象。Source 与 Replay 在同一候选中使用同一 C valuation；`MaskC(C)` 只给 Replay
提供匿名 slot/schema。non-C compile-time value 作为调用事实绑定，但其 semantic consequence
必须已经在 J 中表达，其 realization use 才可由 π 引用。

该 valuation 记为 `c0 = Bind_H(C)`。Gate1/Gate2 的主证书只覆盖实际 c0；测试多个配置时只能
报告实际覆盖的有限集合，不能据此声称所有 admissible C 都已认证。

## 三方执行权限

```text
Source  <- H binds C and inputs
Replay  <- H binds the same C and inputs
RefEval <- logical inputs only
```

`RefEval` 的函数接口不能接收 C 或 π。实验记录可以关联当前候选，但关联 metadata 不得变成
RefEval 的功能输入。

新增前端首先实现上述薄 adapter；不要求 J-to-frontend lowerer、RuleCard、source-to-π parser
或 Kernel-specific mapping table。observer 强度可以逐步增强，并必须记录 frontend、observer
profile/version、compiler/dump stage 与 canonicalization version。

## 高性能工作负载投影与有界隔离批量入口

SCRC 的高性能工作负载清单只能是现行 lowering capability registry 的确定性 Harness-side
投影。direct kernel 的 ID、ordinal 与总数继续由
`spec/lowering/high-performance-kernels.json` 及其 typed loader 唯一决定；SCRC 层不得复制第二份
28 项 truth，也不得把 scenario 29 计作 direct kernel。

投影可以附加 source module/file、entry symbol、frontend、observer/Q 版本、现有 lowering evidence、
adapter readiness 与代表性 tier。这些字段只用于定位真实 Source、选择 Harness adapter 和组织稳定
运行，不是 `(J,C,π)`、TypeCheck 或 Gate predicate 的输入。特别是，source path、Harness factory、
concrete C valuation、BindingMap、历史 diff 和其他 case 的会话内容都不得进入 Replay packet。

离线 manifest 校验必须做到：

- 从现行 registry 逐项构造投影，并机械检查 `1..28` ordinal 与 ID 唯一性；
- source file 可解析且目标函数定义恰好出现一次；
- optional dependency 或 Dataset 缓存缺失记为 `CASE_UNAVAILABLE`；
- source/schema 不一致记为 `MANIFEST_INVALID`，不得伪装成 Gate 失败；
- list、validate 与 dry-run 不调用 Lift/Replay provider。

live 批量入口使用 `intentir.scrc.batch.v3` outer batch protocol。设同时运行的独立 case lifecycle
数为 `W`，每个 case 的候选 branch 数为 `B`，全局 branch/provider/Gate1 in-flight 上限为 `G`，
hidden Gate2 设备并发上限为 `G2`；必须满足 `W >= 1`、`1 <= B <= 5`、`G >= 1`、`G2 >= 1` 与
`W*B <= G`。`W/B/G/G2` 只作为 proof-irrelevant audit metadata 报告，不进入 TypeCheck、Gate1、
Gate2、winner 或 certificate predicate。

v3 相对 v2 只新增 `observer_unsupported` batch classification。它只表示 Source 在当前 observer
profile 下无法产生可比较 observation；不是候选非法、模型失败、Gate1 hard mismatch 或
`NO_CLOSURE`。单生命周期只在 Source observer closed failure 时使用该类；fan-out 只有所有完整
branch 都因同一 Source-observer capability boundary 停止才聚合为该类，混合分支仍走既有分类。
Replay observation/compile failure 不能冒充 Source observer 缺口。历史 `intentir.scrc.batch.v2`
report 保持原 protocol、分类与 verdict，不由 v3 重判。

live CLI 必须由重复的显式 case 集或 `--tier` 选择运行范围，不能无选择地隐式启动 28 项；默认
evidence root 是仓库外的 `/tmp/intentir-scrc-task11`，也允许调用方显式覆盖。批量入口的逻辑顺序
由 registry ordinal 唯一决定；实际执行可以串行，也可以在上述资源上限内对多个 case 做有界并行。
case evidence 目录必须以原子、无冲突方式分配，不能使用跨 worker 的 `max(existing)+1` 读改写
竞态。单个 case 的准备、编译或认证异常必须收敛为该 case 的结构化结果，不得取消、污染或改写
其他 case 的结果。汇总层最终按 registry ordinal 稳定排序，只统计既有 TypeCheck、Gate1、Gate2、
main status、J-necessity status 与真实调用预算，不能重新定义认证条件。

live run 默认在 stderr 输出 `intentir.scrc.progress.v1` JSONL 旁路，用于 proof-irrelevant
observability；stdout 的最终 batch JSON 与 evidence 不依赖该旁路。除固定 protocol/event envelope
外，closed payload 只能包含 case identity、branch、stage、attempt、formal round、elapsed 与 closed
status（实现字段为 `case_id/ordinal/branch_id/stage/attempt/formal_round/elapsed/status`），并约每 60 秒
为仍活跃的 case/branch 输出 heartbeat。它绝不能包含 Source、artifact、diff、diagnostic、provider
response/conversation 或 hidden Gate2 input；写入、flush、heartbeat 或 stderr 异常必须被隔离，不能
改变任何 case verdict。调用方可用 `--no-progress` 完全关闭该旁路。

跨 case 并行的最小隔离单位是完整 case lifecycle：每个 case/branch 分别新建 Harness 可变执行状态、
provider conversation/session、Lift/Replay runner、Replay writer、TypeCheck/Gate runner、relift history、
输入输出 buffer 与 evidence state/目录。只读 `ProviderConfig`、registry、source file 和不含 case 状态
的 adapter 定义可以共享；candidate artifact、provider response/conversation、Replay、diff、winner、
C valuation 和任何可变 Harness 状态不得跨 case 或 branch 共享。资源争用或 provider 限流必须记为
相应 case/branch 的执行证据，不得通过交换候选或压缩隔离边界规避。

同一 case 内的候选 fan-out 仍按同一 Source、同一 Harness 合同与同一 `c0` 建立 `B` 个互不
重叠的 branch evidence 目录。`B=1` 主要用于横向诊断 canary：若该 branch 通过全部门禁，其
`SCRC_CERTIFIED` 仍然有效；只有失败结论不能声称标准五分支 `NO_CLOSURE_(5,K)`。最终未闭合项的
搜索可以在全局 `G` 内升到 `B=5`。就 artifact 构造而言，每个 branch
都由 LLM 完整生成 `(J,C,pi)`；固定代码只做 closed Decode/Elaborate、dependent checking、拒绝与
diagnostics，绝不从 Source、Gate、其他 branch 或历史 evidence 推断、合成或自动补写 artifact。

Harness 不能因此升级为 `HarnessAgent`：它不选择 operator、修 J/π 或组合候选。fan-out v4 由受信任
runner 只依据已完成 TypeCheck/Gate1 evidence，原子选择第一个被 coordinator 观察到的完整
Gate1-closed winner；sibling cooperative stop并保存 `SUPERSEDED` receipt。随后 Harness 才为 winner
生成隐藏 Gate2 输入，并在独立 `G2` semaphore 下只执行一次 Gate2；选择过程不能读取这些输入，
Gate2 失败也不 fallback 到 sibling。完成次序是 proof-irrelevant 调度，必须与 selection rule、winner
branch/formal round 和 stop receipts 一起审计；它不改变被选 artifact 的认证谓词。若没有 winner，
只有全部 `B` 个 branch 都带完整 evidence 终止为 `NO_CLOSURE` 时，case aggregate 才能报告
`NO_CLOSURE_(B,K)`；infrastructure-failed 或 incomplete branch 必须保留并使 aggregate fail closed。

对已经落盘的 parent attempt，Harness 还可运行独立的
`intentir.scrc.gate1-diagnostic-recheck.v1`：先确认 artifact 位于 parent evidence 内并保持其
bytes/digest 不变，再使用当前 TypeCheck、一个 fresh normal Replay 与 Gate1 写入独立原子
`recheck-NNN` 目录。该路径明确 `diagnostic_only=true`、`enters_gate2=false`、
`changes_prior_verdict=false`；它不调用 Lift、masked-J、Gate2 或 winner logic，不能将旧 candidate
升级为当前协议结果。调用 provider 和分配新 evidence 前，必须核对 parent request 的 case/frontend/
c0/grid/anonymous binding、parent source projection 与 source bytes，并要求 parent Gate1
`artifact_digest` 等于输入 artifact；任何缺失或错配都 fail closed。协议/context/knowledge 改变后的
正式结论仍必须 fresh rerun。

Task 11 的旧 `public_frontend_context_v2`、`W=4/B=1` xhigh 横向波中，Softmax、RMSNorm、attention
row reduction 与 linear contraction 均终止为 `NO_CLOSURE_(1,K)`；所有正式 candidate 均已通过
TypeCheck。RMSNorm、Softmax 与 attention row reduction 的共同证据进一步确认：pointer carrier 已在
π 中直接进入 `address_base`，但 v2 Replay 错把 base/lane 先合并为整数 offset，破坏了后续 pointer
carrier dataflow。这是通用 Replay consumer/U_f 问题，已通过 `public_frontend_context_v3` 与
`intentir.lift-knowledge.v11` 修正并要求 fresh rerun；旧 verdict 保持原样，不被新协议静默追认。

## 真实 Triton Source 投影与声明式 adapter

真实工作负载进入统一 SCRC 生命周期前，Harness 可以从已解析的 source file 机械构造 fixed-C
Source 投影。该投影只允许：

- 选择 registry 指定且唯一的顶层 Triton entry；
- 保留其必要 import、模块常量与同模块/可解析 imported helper 闭包；
- 去除只负责搜索 C valuation 的 autotune/heuristics decorator；
- 对 source AST 的参数顺序、runtime/constexpr 分类和 adapter formals 做 exact ABI 校验。

它不得改写 Kernel body、预先解释 operator/predicate/formula，也不得把搜索 decorator 的配置变成
新的 artifact。若 helper、decorator 或 ABI 不能机械收敛，case 必须报告 `HARNESS_INVALID` 或
`CASE_UNAVAILABLE`，不能带着猜测进入 Lift/Gate。

`intentir.triton.fixed-c-source-slice.v3` 对跨模块 plain-helper closure 使用 closed audit boundary：
只从调用方显式给出的 source roots 解析实际被投影引用的 import；目标必须唯一解析为模块顶层、
无 decorator 的同步 source helper，并只递归纳入其实际依赖闭包。这里的 closed 只表示词法依赖可
封闭审计，不额外宣称 helper 无副作用。闭包只允许 stdlib/Triton absolute
imports、同样受审计的 undecorated helper 与 static literal；第三方依赖、dynamic initializer、
`global/nonlocal`、helper 内 local import、decorator、`async/await`、`yield`、多重/冲突/未解析 binding
均 fail closed。实现直接读取显式 root 下的 source file，不 import package 或执行 `__init__`，也没有
xFormers、`libdevice_find`、`rsqrt` 或其他 symbol/family 特判。

v3 还支持 Source 以精确字符串 annotation `VAR_ARGS_ARRAY` 声明的编译前 fixed-arity ABI 展开。arity
必须作为 Harness 调用事实显式给出且限制在 `[1,64]`；投影只展开普通 positional 参数、bare local
array 声明与精确 `for i in range(len(array))`，按 Source 声明次序形成 `name0,name1,...`。marker 位于
非 entry helper 或 selected top-level entry 的任意 nested lexical scope、动态下标/循环、残留
marker/reference、缺失或无 marker 的 arity 均 fail closed；expander 不进入 nested function/class/lambda
改写同名或捕获变量。被展开的 positional marker 参数自身若带 default 也必须拒绝，因为复制后无法
无歧义保持 Python ABI；未标 marker 的合法 trailing defaults 原样保留。
该变换不调用外部 `unroll_varargs`，不执行 `exec`、linecache monkey patch 或 package initializer，也
不读取 Kernel 名称、参数位置或语义公式。

若被移除的 decorator 带有改变调用初态的机械 `pre_hook`（例如把一个 `inout` buffer 清零），fixed-C
Harness 必须用 closed、通用的 adapter initializer 显式保留该调用事实；不能按 Kernel 名称在 runner
中补行为。当前 `intentir.scrc.triton-workload-adapter.v5` 只允许 `random|zeros` 输入初始化，纯 output
没有 initializer；另可声明真实 runtime `None` 与上述 bounded vararg arity。`None` 必须贯穿 compile、
Source/Replay launch 与 Gate2 为真正的 absent carrier，在 J.signature 中只能是
`opaque(mode=value,dtype=none)`，无 logical_ref/J tensor/node/effect/π instance，绝不能换成 dummy
pointer 或 scalar。TypeCheck 还必须分别拒绝它成为 π instance anchor、authorization subject 或
refinement carrier；sandbox CandidateCheck 复用同一个 source-free checker，只给出拒绝诊断，不得
自动删除这些字段或补造 artifact。一次生成的 fixture 必须由 Source、Replay 与 RefEval 三腿共享，
不能分别重采样。
由于 `none -> opaque(dtype=none)` 与 canonical contraction/stage-multiplicity 是新增的模型
可见构造合同；v14 加入 generic functional logical-domain closure 与上述有界诊断反馈合同。fresh
Codex/MiniSWE Lift knowledge 当前使用 `intentir.lift-knowledge.v61/v59`，后者加入 runtime scalar 的
logical-data/representation-only source-use 分类：只有显式 rank-0、`mode=value`、public dtype、
`logical_ref=null` 的 scalar 可不创建 J input并只供 π realization 使用；Harness 不按参数名/位置
替 Lift 分类；该 null form 只允许 runtime scalar。`bound_constexpr_values` 必须表示逻辑语义 constexpr：
logical data/predicate/result 绑定 rank-0 J input，integral extent-only 绑定 `x*`。纯 schedule constexpr
进入 C；representation-only bound constexpr 不得留成 π-only carrier，无法归入 J/C 时 Harness
invalid。v56 另允许 adapter 对 integral semantic constexpr 显式声明 valuation-matched
`bound_constexpr_shape_coordinates`；Lift 必须直接绑定对应 `x*`，或自行作者化 rank-0 `t*=x*`，Harness
只拒绝遗漏/不一致，不按名称、相等数值或 grid 推导 relation。v15 同时要求：若 Source 明示 masked sum 后再除 logical count，保持
`reduce(sum) + explicit div` 两阶段；当前 boundary 不能用中性 fill 为 padded `mean` 补出 denominator。
TypeCheck 只在无 guard、完整零起点矩形 direct access/domain、同轴 bound/typed extent 闭合且 masked
axis 被 mean 消去时拒绝；guard/cross-axis/unknown-op 是 consumer unsupported。旧 v11/v12/v13/v14 receipt 不被
静默改写。

Source 投影不能因当前 `c0` 恰好让某个 grid axis 的 extent 为 1，就删除该 axis 的
`program_id`、address-base 或其他 compiler-visible realization topology；Gate1 比较的是本次真实
编译实例，但 fixed-C 只消除调度搜索外壳，不授权语义化简 Kernel body。

真实 adapter 使用声明式 buffer/scalar/constexpr/C/grid binding。compiler view 可以为 runtime
argument 附带 `input`、`output`、`inout`、`value` 等机械调用模式；这些模式只来自实际 Harness
buffer role，用于检查 J logical signature 是否覆盖真实调用 ABI，不表示 operator 语义，也不能
进入 Replay packet。adapter 不能拥有 per-kernel Replay、Python semantic oracle 或第二份 J。

adapter v5 继承 v4，把 source/launcher 已经给出的 shape identity 暴露为匿名 Harness-only contract：tensor
的 `shape` 是本次 concrete valuation，逐轴 `shape_contract` 的正整数是静态轴，`h*` 是跨 tensor
共享的 logical extent identity；runtime scalar 的 `shape_coordinate=h*` 或 semantic constexpr 的
explicit `bound_constexpr_shape_coordinates` 声明 value/extent equality，
`grid_shape_coordinates` 只记录 launcher realization。Lift 必须把每个 `h*` 显式、双射地映射为 J 的
declared `x*`，logical-data scalar/constexpr 还须作者化 rank-0 `t*` 与
`value_dimension_relations(eq)`；固定
checker只拒绝遗漏、literal specialization、coordinate 合并或 valuation 冲突，不生成 J shape/relation。
`h*`、grid 与 valuation 都不进入 J/C/π/Replay/production，也不从不同 coordinate 猜 derived formula。
当前 RoPE/MRoPE adapter只提升 Source ABI 与 tensor axes 唯一确定的 batch、sequence、head-count、
head-dim identity；相等 valuation 的 tensor batch 与 broadcast batch使用不同 `h*`。Llama4 的
`full_head_dim=2*head_dim_half`、SplitK 的产品/derived capacity/variant family在当前 closed relation
vocabulary中保持 selected-variant literal，不能伪造独立 `h*` 或从 grid/string contract反推公式。

scalar/constexpr 的归属按以下机械边界处理：

| 调用事实 | 归属/锚点 | 三方执行 |
|---|---|---|
| logical runtime scalar 或 bound semantic constexpr data | `J.signature` 的 rank-0 handle | Source/Replay 按 ABI 接收 scalar/constexpr；RefEval 接收同值 rank-0 input |
| logical dimension/extent | J 的 dimension handle | 只绑定 shape/domain，不伪装成 rank-0 data tensor |
| representation/address-only runtime scalar | `J.signature` 中 rank-0 `logical_ref=null` carrier | Source/Replay 按 ABI 接收；RefEval 无 logical input，π 可引用 p* |
| schedule constexpr | C slot | Harness 绑定实际 `c0`；Replay 只读取 `MaskC(C)` |
| absent runtime `None` | `J.signature` 的 `opaque(dtype=none)` handle | Source/Replay 原样接收 `None`；RefEval 无输入且 J/π 无语义节点 |

不存在 representation-only bound constexpr 这一第五类：如果 constexpr 只有 implementation-selection
后果，它就是 schedule constexpr/C；否则必须有 J logical denotation。两者都不满足时 manifest/Harness
invalid，不能借 π-only `logical_ref=null` 形态延后决定。

当前 `intentir.scrc.triton-workload-adapter.v5` 继承 v4 shape identity，并新增显式
bound-constexpr/shape-coordinate relation 与真实 store-then-load scratch `inout` role；它在现行 28 项
registry 上声明全部 28 项 Harness 配置视图：
原六项 DYT、SwiGLU、Softmax、RMSNorm、attention row reduction、linear contraction；full
FlashAttention forward/backward、fused-add RMSNorm、Sparsemax、one-column-block backward；以及
Qwen2-VL MRoPE、GroupNorm、JSD、CrossEntropy、GEGLU、PolyNorm、Llama4 RoPE、FlashAttention
LayerNorm、RoPE、LayerNorm、KLDiv、TVD、xFormers RMSNorm/RMSNorm-add，以及三个 SplitK entry。
ordinal 24/25 由上述通用 plain-helper closure 形成无 package import 的投影；ordinal 26 forward 使用
8 个真实 `None` carrier、arity 1 local-array expansion 与非退化 split grid，ordinal 27/28 使用 arity 2
并分别形成 38/51 参数的 expanded ABI。28 项 static prepare 与真实 Triton CUDA frontend compile
均通过；`declared` 只表示 Harness/source ABI 已有闭合配置，不是 SCRC 终态。adapter 集不得演化成
第二份 Kernel 清单；每次准备仍须重新解析 registry source、检查 entry/ABI，并在真实 CUDA/Triton
frontend 上使用同一份 Source 投影与实际 `c0`。
