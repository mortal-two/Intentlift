# Gate1：Source/Replay realization closure

> 状态：当前派生规范。上位设计见 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)。

## 前置条件

Gate1 只接收已经通过 `TypeCheck(J,C,π)`、由当前三元组生成的全新盲 Replay。Source 与 Replay
使用同一个 C 对象、同一份 valuation 与相同 specialization/runtime conditions。

## 裁决关系

对前端 f 的声明 observer `F_f`：

```text
G1 = [F_f(Compile_H(Source,C)) == F_f(Compile_H(Replay,C))]
```

Gate1 回答 Replay 是否重建了 Source 对 J 的 realization。它不执行数值语义、不读取 Gate2
结果，也不通过 composer/RuleCard 构造 Replay。

presentation/alpha canonicalization 只能消除名称、SSA 编号、无关打印顺序等表示噪声。它不得
删除会改变必要 realization closure 的 op、def-use、predicate/fill、effect、region、order、
sharing、placement 或 representation topology。

## Observer 能力

frontend adapter 必须声明其实际 observer 能力，例如：

- Level 0：build、signature 与 specialization；
- Level 1：AST/intrinsic/mask/atomic/barrier 等结构；
- Level 2：规范化中层 def-use、predicate/fill/effect 与 realization topology。

较弱 observer 的通过只能形成相应强度的 Gate1 claim，不能冒充 Level 2。observer 未实现或
证据不完整时 fail closed 或明确 inconclusive，不能自动 PASS。

## 失败与 relift

Gate1 失败产生结构化 diff，只返回 source-aware Lift：

```text
Source + previous complete (J,C,pi) + accumulated Gate1 diff
  -> complete new (J,C,pi)
  -> TypeCheck
  -> new blind Replay
  -> Gate1
```

diff 不进入 Replay。禁止 patch 旧 Replay、复用旧 Replay 或只修改 artifact 的局部字段。最多三轮
正式候选，仍失败则 `NO_CLOSURE`。

## masked-J 的位置

masked-J 是独立的 J-necessity audit，不是主认证 Gate：

- hard mismatch：支持 `J_NECESSARY=PROVED`；
- hard close 且 TypeCheck 通过：`J_NECESSARY=NOT_PROVED`，不自动否定主认证；
- TypeCheck 失败：当前 artifact FAIL；
- generation/compiler/tool failure：`INCONCLUSIVE`。

只有额外 J-authority challenge 证明非 J 通道覆盖当前 J 语义时，才可报告 authority leakage；
不能把 topology prediction 本身等同于显式语义泄漏。

## 记录要求

实验记录保存 frontend、observer 能力、toolchain、C valuation、两侧 observation、结构化 diff 与
裁决理由。记录标识用于追溯，不是候选合法性的 hash/commitment 门。
