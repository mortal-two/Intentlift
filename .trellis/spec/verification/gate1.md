# Gate1：Source/Replay realization closure

> 状态：当前派生规范。上位设计见 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## 前置条件

Gate1 只接收已经通过 `TypeCheck(J,C,π)`、由 `J+MaskC(C)` 生成的全新盲 Replay；π 不进入
Replay request、workspace 或工具。Source 与 Replay
使用同一个 C 对象、同一份实际 valuation `c0 = Bind_H(C)` 与相同 specialization/runtime
conditions。该 Gate1 claim 只针对这个 c0。

当前 fresh-run artifact/decode/typecheck 组合为
`intentir.scrc.artifact.v12 + intentir.pi.normalization-witness.v1 /
intentir.scrc.decode.v18 / intentir.scrc.typecheck.v31`。Lift LLM 是完整
artifact 的唯一作者：value/dimension equality、closed dimension definition、consumer-extension、access-guard、typed control/inout、
output binding 都必须在 J/C 中显式提交；active π 只含 program-axis rows。Decode、
CandidateCheck 与 TypeCheck 只能做 closed
解析、检查、拒绝和诊断，不会补 relation、选择 axes/polarity、插入 `select`、创建 pre-state，或把
相同 concrete extents 当作 axis identity；v28验证dimension relation的closed grammar、引用与acyclicity，
并拒绝同tensor多个不同却都无consumer锚点的node-less load views，但不从Source
TTIR/Harness/grid/C/π生成relation或猜consumer；v27 另检查 route extent transform 只能限定一条已存在
route slot，且 source 是 attached required positive-int C、target 是匹配 J domain realization；它不
生成 divisor/attachment/relation。其余 v26 合同继承 v25；v25 继承 v24，并检查作者化
`reduction_linearization` 只能经唯一 slot-0 route 进入一个 J reduce；v24 继承 v23 对 canonical output binding 的完整
dtype/shape equality，并在 Replay 前拒绝没有从匹配 boundary tensor 的 pointer/tensor signature
carrier 到达 transfer 的 authored route；浮点 typed constant 的字符串值只允许 closed
`inf/-inf/nan`。axis permutation 和 special constant 都必须由 J 自己显式表达。它也不会把
output 改成 inout。CandidateCheck 的 PASS 也不是 Gate1 前置证书；外部 Runner 仍对同一完整 bytes
重跑权威 TypeCheck。

## 裁决关系

对前端 `f`、声明的 observer profile `p`、同一个 `c0`：

```text
Phi(P,c)    = Canon_(f,p)(Observe_(f,p)(Compile_f(P,c)))
Phi_pi(P,c) = Normalize^pi_(f,p)(Phi(P,c))
G1 = defined(Phi_pi(Source,c0))
     and defined(Phi_pi(Replay,c0))
     and Phi_pi(Source,c0) == Phi_pi(Replay,c0)
```

raw `Compile/Observe/Canon` 不读取 π。`Normalize^pi` 是 profile 预先冻结的、可能未定义的单臂
normalizer；它对 Source 与 Replay 独立调用，不得读取另一臂、当前 diff、Kernel identity、Source
文本或历史失败。Gate1 回答 J-only Replay 与 Source 是否在 J/C-owned compiler-visible structure 上
闭合。它不执行数值语义、不读取 Gate2，也不通过 host composer/RuleCard 构造 Replay。

Replay normal view 必须保留当前 J-owned typed relations；它从 J 读取 value/dimension equality、
closed dimension definitions、consumer-specific outside-domain contract 与 access guard，但不读取 π。
logical J axis ordinal 不是 frontend active tile 的 rank-local reduction axis；Replay 只能根据 J、
MaskC schema 与公共 frontend knowledge 独立选择实现，不能机械复制 logical axis。特别地，
v5 guarded store 的 merge 已经是 Layer A 中显式 `select`，且精确 inout/pre-state、polarity 与 axes 已
通过 TypeCheck；Replay 不得把 relation 解读成允许它自行发明 branch、merge 或 mutation。Gate1
比较的是 J-only Replay 与 Source 的 normalized observations，不为缺失 relation建议或生成答案。

### certification-only π normalization

π relation 的 grammar、字段和 observer denotation 由冻结 schema 与 `(f,p,ν)` 预先定义。Lift只能
提交 typed relation instances，不能提交 wildcard、任意 `allowed_diff`、normalization rule 或 free-text
equivalence。单臂 normalizer 必须 deterministic、idempotent、hard-preserving 且 fail closed；任何匹配
歧义、缺少 typed anchor 或 profile 未支持的 relation 都不得 normalize。

新 profile 中所有 π relation 默认不放宽 Gate1。只有 profile-wide 实现了独立于 peer observation/diff
的局部 canonical form，并固定 typed anchor、前置条件、scope 与 hard-preservation 的 exact relation
kind 才可使用；candidate 不能选择 rule、改 denotation 或提交 `normalize_to`。规则可读取通用 typed
rank/dtype、J-authored relation 与 C attachment，不得读取 Kernel identity、Source 路径或仅凭当前
concrete extents 相等发明关系。

每个可 normalize relation `q` 都实现一个冻结的 partial unary normalizer：

```text
Normalize_(f,p,nu)^q(O; J,C,pi_q) -> O' | undefined
```

只有 `q` 唯一锚定已有 J/C occurrence、当前 observation 的 exact scope 满足固定局部前提、作用域
不歧义且固定 J/C hard projection 不变时才应用。`undefined` 不会变成 PASS 权限。TypeCheck 只证明
`q` 是 closed typed instance；normalizer 不搜索或补造 π，也不是完整 Source→π extractor。

J-owned operator/operand、logical domain、predicate/fill、reducer、value recurrence、observable
effect/control/post-state 与 output binding 始终保持 hard。当前 active π 可携带 typed
`program_axes`，但 `jc-normalizer.v2` 不因此折叠物理 axis label：在观察 occurrence
无法唯一回指 J domain 前，axis identity 保持 hard。mixed-radix、address、placement、
materialization、physical iteration、order、sharing、carrier、terminal control 与 sensitive
numeric/update residual 均保持 hard；它们没有可由候选选择的 `witness_only` 通道。

完整 observation 默认全 hard。固定 `H^(J,C)_(f,p)` 投影由两臂共同的匿名 profile vocabulary 与
J/C typed contracts决定，候选 π 无权选择它。H 必须覆盖该 profile 可观察的 interface、
operator/operand/dataflow、logical domain/access/index、predicate/fill、reducer/recurrence、result
relation、observable effect/control/post-state/output 与 C identity/type/valuation/attachment；未知 atom
默认 hard。无法建立唯一 anchor 时 `observer_unsupported`，不能形成空/弱投影。每次 defined
normalization 必须满足 `H(Normalize^pi(O)) = H(O)`；未知 alias、memory-order、numeric、
address-provenance atom 默认保留。多个 relation 不得通过重叠 scope 或候选指定顺序扩大可擦除范围。

正式 profile 必须在候选外满足 `ProfileOK(f,p,nu,J,C)`：rule allowlist、recognizer、scope 与 canonical
replacement 全部冻结，每条启用 rule 都有机械 hard-preservation check 或独立 rule audit。TypeCheck
只验证 compact π row well typed，不证明 `ProfileOK`；无法满足的 relation 不进入 active schema。
`ProfileOK` 只是 TCB release contract 的简写，不是新 Gate、artifact、receipt 或候选字段。

实现可以保存 applied/unsupported relation 作为 proof-irrelevant debug telemetry；它不进入正式谓词，
不形成 `Diff subset pi` acceptance，也不认证 π 的 completeness、minimality 或 Source 真值。

因此只在固定 `(f,p,ν,c0)` 下有：

```text
ProfileOK(f,p,nu,J,C) and TypeCheck(J,C,pi) and G1
  => H^(J,C)(Phi(Source,c0)) == H^(J,C)(Phi(Replay,c0))
```

π 从结论中消失。该结论不等于 raw observation equality、全输入域语义等价或完整 Source→J verifier。

以下 `replay-check.v14` 条款记录 Task 20 迁移前的 π-consuming Replay runtime；新 J-only Replay
路径不得调用这些 π packet checks。它们在 Task 20 完成 runtime/spec 同步后归档，不得作为新路径
已经实现的证据。迁移前 `replay-check.v14` 的 synthetic compile 只消费匿名 packet：核对作者化 `transfer_direct`
occurrence、structured J loop 加 `pi.physical_iterations` 的 loop occurrence、对应 carrier arity；并且
只在 signature pointer/tensor kind 与 π address routes 完全闭合时核对 pointer-address operation
occurrence。完整闭包允许独立 integer base/lane contribution，且不要求 address instance 与 transfer
使用相同 `semantic_ref` spelling；它要求每个 transfer 从匹配 boundary tensor 的 pointer/tensor
signature carrier 经作者化 routes 可达，并要求每个 pointer-producing address stage 最终流向某个
transfer；并要求 compiled distinct `tt.reshape -> tt.reduce(axis=0)` dataflow 不少于已作者化
`reduction_linearization` 数量，并拒绝 authored target-local route extent transform 完全缺失 closed
floor/ceil-div spelling。launch-grid relation 不属于该字段；精确 target-local topology 仍由 Gate1，
而不是 ReplayCheck 裁决。无法闭合的
address topology 不猜。v13 还只在同一 π region 的 ordered `transfer_direct` 可分别经作者化
routes/carrier 唯一回溯到一个 signature pointer，且 Replay 自身 TTIR 中对应 load/store occurrence
也唯一并处于同一词法 region 时，核对其词法顺序是否遵守 `pi.order`；任一映射有歧义就不猜，留给
正式 Gate1。该 check 只拒绝错误 Replay bytes，不生成 source、
address formula、loop/carrier 或 relation，也没有 Gate1/认证权威。

presentation/alpha canonicalization 只能消除名称、SSA 编号、无关打印顺序等表示噪声。它不得
删除会改变必要 realization closure 的 op、def-use、predicate/fill、effect、region、order、
sharing、placement 或 representation topology。

自 `triton-ttir-3.6-gate1-v4` 起有一条全局、窄而 closed 的例外：对 `tt.load` 只归一
`cacheModifier in {ca,cg,cv}`，对 `tt.store` 只归一 `{wb,cg,cs,wt}`。它们是不由
J/C/π 拥有的 target cache-policy presentation hint，不改变 J denotation、memory effect、
predicate/fill、address/dataflow 或 transfer topology。未知值、load/store 方向不合法的值、eviction
以及其他 memory attributes 仍参与 hard comparison。

当前 active profile 是 `triton-ttir-3.6-gate1-v14`。v14 保留 v13 raw fingerprint，但撤销
不 sound 的 program-axis label quotient。v13 只增加一条全局 closed value-tree
canonicalization：一个 direct typed `math.fma(a,b,c)` 与使用相同 typed operands 的 direct
`a*b+c` 等价；若 `c` 是同 region、direct `0-d`/`neg(d)`，则只与 direct `a*b-d` 等价。它不移动
cast、不改变 operand/type、不重关联更大的树，也不授权 production backend 进行 FMA contraction；
J 的 typed graph 与 result relation、Gate2/Q 仍各自保持原所有权。
v13 完整继承 v12。v12 在既有 canonicalization 后作一项全局、
source-neutral 的 ownership quotient：从 public return 与 store/update value 反向保留完整值流；从
load/store mask、structured condition 与 terminal condition 保留 comparison/boolean 结构、polarity 和
直接常量；effect、region、loop/carrier、reduction、`tt.get_program_id`、`tt.addptr` 与 transfer anchor
始终保留。只有剩余 known-pure、仅服务 program scheduling/address/rank/bound 的子图被压成带结果类型
与 ABI/region origin 的不透明 representation edge。该规则不读取 J、Source 文本或 kernel family，
也不把错误 mask/fill、reducer、effect、pointer provenance、program-axis 数量或 transfer topology
NormalizeAway。v13 保留既有对已建模 op 与 interprocedural
observation 的 comparison rules：从 public entry 可达、callee 静态已知且 call graph 无环的
structured `tt.call` 仍按
formal-to-actual 与 helper return 展开，op/def-use、effect 与已支持 region 继续参与 hard comparison；
public entry ABI 仍是边界，private helper 名称本身不进入 J/C/π。

Replay construction compile 不读取 π，因此不执行任何 π-derived occurrence、branch 或 terminal
obligation。Source-aware Lift 可以用 exact anonymous Source TTIR 对模型已提交的 program-axis row 做
reject-only 诊断，但该诊断不生成 row，也不是 certification。正式 Gate1 的 raw observation始终保留
program-coordinate operation 数量、axis identity、所有 downstream dataflow、predicate/control/effect；
当前 `N^π` 对 `program_axes` 不作变换。mixed-radix 与旧 empty-slot route、branch、
terminal-control construction checks 仅解释迁移前 evidence，不属于当前 J-only Replay。

v6 新增 Triton 3.6 quoted-generic `tt.scan` 的 typed、single-block、encodingless observer contract：
输入/结果必须是同 shape 的 ranked tensor，axis/reverse、`2N` 个有序 combine block arguments、唯一
`tt.scan.return` 的 `N` 个 typed values，以及 recurrence/region topology 都是 hard evidence。当前
observer 不解释 module-level tensor encoding alias，因此任何非空 scan tensor encoding 都明确
unsupported，不能只比较 alias 名称。

v6 同时把 operation/effect classification 收成全局 closed profile：只有显式登记的 pure op 才能按
pure dataflow 处理，只有显式登记的 effect op 才能进入 lexical effect trace；未知 operation 即使位于
function root 或 `scf.for` 内也不能默认 pure。畸形 quoted-generic、非空但无法解析的 operation line、
未知/动态 callee、递归/cyclic call、signature/return 不一致、reachable arbitrary multi-block CFG、
多 block scan/reduce 与未支持 control/region 一律 fail closed。Source 侧能力缺口仍由 batch v3 报
`observer_unsupported`；历史 v5 evidence 保持原 profile/verdict，不由 v6 追认。上述范围只是当前
compiler contract 与测试边界，不是 observer 或 TTIR 语义完备性的形式证明，也不声称 v5→v6
acceptance 单调。后续窄升在同一 closed 原则下增加 function metadata、value/effect-only structured
`scf.if` 与 exact operand-free terminal binary `cf.cond_br` profile；terminal fork 的 predicate、true/false
角色、arm member/dataflow、effect order、terminator 与 structured/interprocedural parent context 都是 hard
evidence，physical block printing order 只是 presentation。任何带 block argument、额外 block、非 void
function、arm 间 SSA 越界、嵌套未支持 control 或超出该 exact profile 的 CFG 仍 fail closed。v11 另把
closed cache modifier 与 eviction spelling 一并视为不由 J/C/π 拥有的 frontend presentation noise；
transfer instance 的数量、route、数据流与 effect order 仍不可消除。

## Observer 能力

frontend adapter 必须声明其实际 observer 能力，例如：

- Level 0：build、signature 与 specialization；
- Level 1：AST/intrinsic/mask/atomic/barrier 等结构；
- Level 2：规范化中层 def-use、predicate/fill/effect 与 realization topology。

较弱 observer 的通过只能形成相应强度的 Gate1 claim，不能冒充 Level 2。observer 未实现或
证据不完整时 fail closed 或明确 inconclusive，不能自动 PASS。Source 在当前 profile 下无法形成
observation 时，outer batch v3 使用 `observer_unsupported`：它只标记 Source observer 能力缺口，
不是 TypeCheck/Gate1 失败、候选非法或 `NO_CLOSURE`。Replay 侧失败不能改写为该类；历史 batch v2
报告也不由 v3 重判。

## 失败与 relift

Gate1 失败产生完整结构化 diff，只返回产生它的 source-aware Lift owner。runtime 可以同时提供该 diff
的 deterministic bounded frontier：它只按 closed category priority 保留计数和有限原始 witness，不
生成解释、修复或推荐；完整 diff 仍按 exact bytes 落盘并可由 owner 按需读取：

```text
Source + previous complete (J,C,pi) + accumulated Gate1 diff
  -> complete new (J,C,pi)
  -> TypeCheck
  -> J/C changed: fresh J-only Replay
  -> pi-only changed: exact-R witness recheck is permitted
  -> Gate1
```

diff/frontier 不进入 Replay。禁止 patch 旧 Replay 或把局部 patch 当成新的 proof artifact；owner
可以用 exact-text patch 编辑自己的 working file，但每轮必须提交完整新三元组。只有 J、MaskC、
Replay request policy 与 exact Replay bytes 均不变的 π-only witness 修订可以复用 R，并必须重跑
TypeCheck/Gate1；任何 J/C 改动都要求 fresh Replay。当前
同一 Source/c0 可启动 `1<=B<=5` 个独立 branch，并显式声明正有限 `K`（CLI为`1<=K<=12`，
Task14默认5）；每 branch最多 `K` 个正式候选，feedback只返回
产生它的 branch。首个被观察到并已完整落盘的 Gate1 closure 原子成为 winner，sibling cooperative
stop 并保存 receipt；该调度不读取 hidden Gate2，只有 winner 进入 Gate2。

Gate1 的适应性 relift 是固定 normalizer contract 下的候选搜索，不是统计检验；正式报告必须保留
first-pass/closure@k 与搜索成本。Gate1 不认证 π clause 的 Source 真值、完整性或跨 profile
有效性，只裁决本次 normalized J/C structural closure。

若 Gate1 已闭合而失败只发生在 Gate2 的 J reference consumer，保留这份精确 artifact 与 Replay；
修复通用 RefEval 能力后走 Gate2-only exact-artifact recheck，不重新 Lift 或重新 Gate1。该复查的
结果属于新的独立 evidence，不能覆盖或改写 parent Gate1/Gate2 verdict。

全部完整 branch 都失败才是 `NO_CLOSURE_(B,K)`；该状态只说明本次有界搜索未找到 closure，不能
解释为合法 witness 不存在。provider/infrastructure 未完成的 branch 不得计作 Gate1 失败。

## masked-J 的位置

masked-J 是独立的 J-necessity audit，不是主认证 Gate：

- hard mismatch：支持 `J_NECESSARY=PROVED`；
- hard close 且 TypeCheck 通过：`J_NECESSARY=NOT_PROVED`，不自动否定主认证；
- TypeCheck 失败：当前 artifact FAIL；
- generation/compiler/tool failure：`INCONCLUSIVE`。

`PROVED` 只表示在记录的模型、解码配置、Replay packet 与 masked-J 干预下观察到 J 的操作性
必要性，不是全局不可替代性证明。J-authority challenge 只能作为另行启用的离线消融；不能把
topology prediction 本身等同于显式语义泄漏，也不能把该挑战恢复成主在线 Gate。

## 记录要求

实验记录保存 frontend、observer profile/version、compiler/dump stage、canonicalization version、
toolchain、C valuation、两侧 observation、结构化 diff 与裁决理由。记录标识用于追溯，不是候选
合法性的 hash/commitment 门。

批量报告至少包含 `closure@1/2/3`、Lift/Replay/总 LLM 调用与 `NO_CLOSURE` 分类；`closure@k`
是截至第 k 个正式轮次的累计闭合率。mutation test 可评估 observer 的经验敏感性，但不进入主
Gate，也不证明 observer 完备。
