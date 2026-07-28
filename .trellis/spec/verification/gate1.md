# Gate1：Source/Replay realization closure

> 状态：当前派生规范。上位设计见 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## 前置条件

Gate1 只接收已经通过 `TypeCheck(J,C,π)`、由当前三元组生成的全新盲 Replay。Source 与 Replay
使用同一个 C 对象、同一份实际 valuation `c0 = Bind_H(C)` 与相同 specialization/runtime
conditions。该 Gate1 claim 只针对这个 c0。

当前 fresh-run artifact/decode/typecheck 组合为
`intentir.scrc.artifact.v8 / intentir.scrc.decode.v12 / intentir.scrc.typecheck.v20`。Lift LLM 是完整
artifact 的唯一作者：value/dimension、consumer-extension、access-guard、typed control/inout、
output binding，以及 physical iteration/carrier relation 都必须由模型显式提交。Decode、
CandidateCheck 与 TypeCheck 只能做 closed
解析、检查、拒绝和诊断，不会补 relation、选择 axes/polarity、插入 `select`、创建 pre-state，或把
output 改成 inout。CandidateCheck 的 PASS 也不是 Gate1 前置证书；外部 Runner 仍对同一完整 bytes
重跑权威 TypeCheck。

## 裁决关系

对前端 f 和声明的 observer profile p：

```text
G1 = [F_(f,p)(Compile_H(Source,c0)) == F_(f,p)(Compile_H(Replay,c0))]
```

Gate1 回答 Replay 是否重建了 Source 对 J 的 realization。它不执行数值语义、不读取 Gate2
结果，也不通过 composer/RuleCard 构造 Replay。

Replay normal view 必须保留当前 J-owned typed relations；它从 J 读取 value/dimension equality、
consumer-specific outside-domain contract 与 access guard，从 π 读取 realization topology，以及 J
reduce/contraction 之外已声明的 physical iteration/carrier。logical J axis ordinal 不是 Triton tile
tensor 的 rank-local reduction axis；Replay 必须从 J/C/π 共同确定 active physical tile，并对一维 tile
使用 local axis 0，不能机械复制 logical axis。特别地，
v5 guarded store 的 merge 已经是 Layer A 中显式 `select`，且精确 inout/pre-state、polarity 与 axes 已
通过 TypeCheck；Replay 不得把 relation 解读成允许它自行发明 branch、merge 或 mutation。Gate1
比较的是 Replay 对这些已声明事实的 source realization，不为缺失 relation 建议或生成答案。

presentation/alpha canonicalization 只能消除名称、SSA 编号、无关打印顺序等表示噪声。它不得
删除会改变必要 realization closure 的 op、def-use、predicate/fill、effect、region、order、
sharing、placement 或 representation topology。

自 `triton-ttir-3.6-gate1-v4` 起有一条全局、窄而 closed 的例外：对 `tt.load` 只归一
`cacheModifier in {ca,cg,cv}`，对 `tt.store` 只归一 `{wb,cg,cs,wt}`。它们是不由
J/C/π 拥有的 target cache-policy presentation hint，不改变 J denotation、memory effect、
predicate/fill、address/dataflow 或 transfer topology。未知值、load/store 方向不合法的值、eviction
以及其他 memory attributes 仍参与 hard comparison。

当前 active profile 是 `triton-ttir-3.6-gate1-v11`。v11 保留既有对已建模 op 与 interprocedural
observation 的 comparison rules：从 public entry 可达、callee 静态已知且 call graph 无环的
structured `tt.call` 仍按
formal-to-actual 与 helper return 展开，op/def-use、effect 与已支持 region 继续参与 hard comparison；
public entry ABI 仍是边界，private helper 名称本身不进入 J/C/π。

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
