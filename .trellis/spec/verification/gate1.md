# Gate1：Source/Replay realization closure

> 状态：当前派生规范。上位设计见 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## 前置条件

Gate1 只接收已经通过 `TypeCheck(J,C,π)`、由当前三元组生成的全新盲 Replay。Source 与 Replay
使用同一个 C 对象、同一份实际 valuation `c0 = Bind_H(C)` 与相同 specialization/runtime
conditions。该 Gate1 claim 只针对这个 c0。

当前 fresh-run artifact/decode/typecheck 组合为
`intentir.scrc.artifact.v13 / intentir.scrc.decode.v18 / intentir.scrc.typecheck.v30`。Lift LLM 是完整
artifact 的唯一作者：value/dimension equality、closed dimension definition、consumer-extension、access-guard、typed control/inout、
output binding、physical iteration/carrier、terminal-control、route-local extent transform与participant
projection都必须由模型显式提交。Decode、
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

对前端 f 和声明的 observer profile p：

```text
G1 = [F_(f,p)(Compile_H(Source,c0)) == F_(f,p)(Compile_H(Replay,c0))]
```

Gate1 回答 Replay 是否重建了 Source 对 J 的 realization。它不执行数值语义、不读取 Gate2
结果，也不通过 semantic composer/RuleCard 构造 Replay。Replay agent可在自己的fresh episode内用
syntax-only composer编辑agent-authored源码；该state不进入Gate1，Gate1仍只重编译和比较最终exact
Replay source。

Replay normal view 必须保留当前 J-owned typed relations；它从 J 读取 value/dimension equality、
closed dimension definitions、
consumer-specific outside-domain contract 与 access guard，从 π 读取 realization topology，以及 J
reduce/contraction 之外已声明的 physical iteration/carrier。logical J axis ordinal 不是 Triton tile
tensor 的 rank-local reduction axis；Replay 必须从 J/C/π 共同确定 active physical tile，并对一维 tile
使用 local axis 0，不能机械复制 logical axis。特别地，
v5 guarded store 的 merge 已经是 Layer A 中显式 `select`，且精确 inout/pre-state、polarity 与 axes 已
通过 TypeCheck；Replay 不得把 relation 解读成允许它自行发明 branch、merge 或 mutation。Gate1
比较的是 Replay 对这些已声明事实的 source realization，不为缺失 relation 建议或生成答案。

当前 `replay-check.v16` 的 synthetic compile 只消费匿名 packet：核对作者化 `transfer_direct`
occurrence、structured J loop 加 `pi.physical_iterations` 的 loop occurrence、对应 carrier arity；并且
只在 signature pointer/tensor kind 与 π address routes 完全闭合时核对 pointer-address operation
occurrence。完整闭包允许独立 integer base/lane contribution，且不要求 address instance 与 transfer
使用相同 `semantic_ref` spelling；它要求每个 transfer 从匹配 boundary tensor 的 pointer/tensor
signature carrier 经作者化 routes 可达，并要求每个 pointer-producing address stage 最终流向某个
transfer；并要求 compiled distinct `tt.reshape -> tt.reduce(axis=0)` dataflow 不少于已作者化
`reduction_linearization` 数量，并拒绝 authored route extent transform 完全缺失 closed
floor/ceil-div spelling。精确 target-local topology 仍由 Gate1，而不是 ReplayCheck 裁决。无法闭合的
address topology 不猜。v13 还只在同一 π region 的 ordered `transfer_direct` 可分别经作者化
routes/carrier 唯一回溯到一个 signature pointer，且 Replay 自身 TTIR 中对应 load/store occurrence
也唯一并处于同一词法 region 时，核对其词法顺序是否遵守 `pi.order`；任一映射有歧义就不猜，留给
正式 Gate1。该 check 只拒绝错误 Replay bytes，不生成 source、
address formula、loop/carrier 或 relation，也没有 Gate1/认证权威。

CUDA hard observer `cuda-ptx-semantic-effects-v4`继续放松register/label/cache与ordinary affine address
spelling，但沿每个memory address的reaching definitions保留information-losing participant fold：closed
power-of-two low-bit `and` 与positive `rem`归一为cyclic projection，其他bitmask保持bitmask类别。
Source有fold而Replay直接使用raw participant coordinate时，`address_coordinate_folds`必须hard fail。
该观察器不把PTX、fold tag或差异交给source-blind Replay；它只在Gate1比较最终exact Source/Replay，
也不替Lift生成`participant_mappings`。TileLang的hard closure仍是结构TIR，PTX仅作diagnostic。

presentation/alpha canonicalization 只能消除名称、SSA 编号、无关打印顺序等表示噪声。它不得
删除会改变必要 realization closure 的 op、def-use、predicate/fill、effect、region、order、
sharing、placement 或 representation topology。

自 `triton-ttir-3.6-gate1-v4` 起有一条全局、窄而 closed 的例外：对 `tt.load` 只归一
`cacheModifier in {ca,cg,cv}`，对 `tt.store` 只归一 `{wb,cg,cs,wt}`。它们是不由
J/C/π 拥有的 target cache-policy presentation hint，不改变 J denotation、memory effect、
predicate/fill、address/dataflow 或 transfer topology。未知值、load/store 方向不合法的值、eviction
以及其他 memory attributes 仍参与 hard comparison。

当前 active profile 是 `triton-ttir-3.6-gate1-v13`。v13 只增加一条全局 closed value-tree
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

Replay/Lift construction compile 对 artifact v12 的π-owned `program_axes{domain_ref,axis}`执行
reject-only核对：compiled `tt.get_program_id` 的数量与literal axis集合都必须匹配；旧v11 packet仅保留
原count合同，不被静默升级。该检查不生成program coordinate，也不从grid、C或声明顺序选择axis。

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
  -> new blind Replay
  -> Gate1
```

diff/frontier 不进入 Replay。禁止 patch 旧 Replay、复用旧 Replay 或把局部 patch 当成新的 proof
artifact；owner 可以用 exact-text patch 编辑自己的 working file，但每轮必须提交完整新三元组。当前
同一 Source/c0 可启动 `1<=B<=5` 个独立 branch，并显式声明正有限 `K`（CLI为`1<=K<=12`，
Task14默认5）；每 branch最多 `K` 个正式候选，feedback只返回
产生它的 branch。首个被观察到并已完整落盘的 Gate1 closure 原子成为 winner，sibling cooperative
stop 并保存 receipt；该调度不读取 hidden Gate2，只有 winner 进入 Gate2。

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
