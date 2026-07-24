# Gate1：Source/Replay realization closure

> 状态：当前派生规范。上位设计见 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## 前置条件

Gate1 只接收已经通过 `TypeCheck(J,C,π)`、由当前三元组生成的全新盲 Replay。Source 与 Replay
使用同一个 C 对象、同一份实际 valuation `c0 = Bind_H(C)` 与相同 specialization/runtime
conditions。该 Gate1 claim 只针对这个 c0。

## 裁决关系

对前端 f 和声明的 observer profile p：

```text
G1 = [F_(f,p)(Compile_H(Source,c0)) == F_(f,p)(Compile_H(Replay,c0))]
```

Gate1 回答 Replay 是否重建了 Source 对 J 的 realization。它不执行数值语义、不读取 Gate2
结果，也不通过 composer/RuleCard 构造 Replay。

presentation/alpha canonicalization 只能消除名称、SSA 编号、无关打印顺序等表示噪声。它不得
删除会改变必要 realization closure 的 op、def-use、predicate/fill、effect、region、order、
sharing、placement 或 representation topology。

自 `triton-ttir-3.6-gate1-v4` 起有一条全局、窄而 closed 的例外：对 `tt.load` 只归一
`cacheModifier in {ca,cg,cv}`，对 `tt.store` 只归一 `{wb,cg,cs,wt}`。它们是不由
J/C/π 拥有的 target cache-policy presentation hint，不改变 J denotation、memory effect、
predicate/fill、address/dataflow 或 transfer topology。未知值、load/store 方向不合法的值、eviction
以及其他 memory attributes 仍参与 hard comparison。

当前 active profile 是 `triton-ttir-3.6-gate1-v5`。v5 继承上述 canonicalization，并将从 public
entry 可达、callee 静态已知且 call graph 无环的 structured `tt.call` 展开到同一 SSA observation：
formal-to-actual、helper returns、op/def-use、effect 与当前支持的 structured region 都参与 hard
comparison；public entry ABI 仍是边界，private helper 名称本身不进入 J/C/π。未知或动态 callee、
递归/cyclic call、signature/return 不一致、reachable arbitrary multi-block CFG 与未支持的
control/region 形态 fail closed。该范围是当前 compiler contract 与测试边界，不是 observer 或 TTIR
语义完备性的形式证明。

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

Gate1 失败产生结构化 diff，只返回 source-aware Lift：

```text
Source + previous complete (J,C,pi) + accumulated Gate1 diff
  -> complete new (J,C,pi)
  -> TypeCheck
  -> new blind Replay
  -> Gate1
```

diff 不进入 Replay。禁止 patch 旧 Replay、复用旧 Replay 或只修改 artifact 的局部字段。当前
同一 Source/c0 启动 `B=5` 个独立 branch，每 branch 最多 `K=3` 个正式候选；diff 只返回产生它的
branch。并行 wave 结束后，仅根据 TypeCheck/Gate1 evidence 按固定 branch/round 顺序选择一个
Gate1-closed winner；未选 closed branch 不进入 Gate2。

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
