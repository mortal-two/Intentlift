# Gate2：Source / Replay / J 三方语义验证

> 状态：当前派生规范。上位设计见 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。
> 数学与 claim 边界见 [SCRC 最终数学总述](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。

## 三腿与权限

对 logical input x 和实际绑定 `c0 = Bind_H(C)`：

```text
O_S = Exec(Source, c0, x)
O_R = Exec(Replay, c0, x)
O_J = RefEval(J, x)
```

Source 与 Replay 使用同一个 C 对象和实际 valuation `c0`。该证书是 instance-wise 的，不自动
覆盖所有 admissible C。`RefEval` 的接口中不存在 C 或 π，也不读取
Source/Replay。所有 output、inout、alias/writeback 与声明的 observable effect 都必须裁决。

## 比较关系

```text
G2 = Close_Q(O_S,O_R)
     and Close_Q(O_S,O_J)
     and Close_Q(O_R,O_J)
```

Q 是实验协议或 benchmark correctness policy 预先确定的可信比较政策；B 只声明封闭 comparison
class 与必要语义合同。Q 根据经验证的 class、dtype、reduction extent、reassociation 权限等
构造 comparator，并验证 tolerance 上限、NaN/Inf 等全部参数。B 不提供任意可执行 comparator；
不支持的类别、非法参数、缺腿或执行不完整均 fail closed。Q 不是 Lift artifact，也不进入 Replay
或 RefEval。

`Close_Q` 必须是确定性、对称的谓词；若底层 `isclose` 有方向性，Q 必须显式对称化或执行双向
检查。第三条 Replay–J 边不增加执行，只比较已经产生的三个输出；两条 Source-centered 近似边
不能依靠传递性推出它。

当前主认证域只覆盖 deterministic / contract-deterministic Kernel，即内部调度可变化，但 J/B
在协议可观察层面给出唯一合同结果。真正集合值的 Kernel 留给未来 `Accept_{J,Q}(x,o)` 的
per-observation conformance；有限次 membership 不能被宣称为完整的行为集合包含证明，两个不同
但都合法的集合值结果也不应继续强制三条 `Close_Q`。

## Gate2 输入模式

多候选搜索先完成 pre-Gate2 wave，并仅依据 TypeCheck/Gate1 evidence 按固定
`(branch_id, formal_round)` 顺序选择一个 `(J,C,π,Replay)` winner。选择函数看不到 Gate2 输入、
输出或 verdict；winner 此后不再替换或修改，随后才产生 Gate2 输入。未选 sibling 不执行 Gate2，
winner 失败后也不能 fallback。两种输入口径必须明确区分。

### 隐藏随机输入

从预先声明的分布 D 独立同分布抽取 m 个对 Lift/Replay 隐藏的输入，Gate2 结果不回流本次实验。
令完整候选为 `a*=(J,C,pi,Replay)`，并令 `g2_Q(a*,x)` 表示三条边在 x 上全部通过：

```text
epsilon_a = Pr_{x ~ D}[not g2_Q(a*,x)]
```

若 `epsilon_a >= epsilon > 0`，该固定坏候选通过全部 challenge 的概率至多为：

```text
(1-epsilon)^m <= exp(-m*epsilon)
```

Source–J disagreement mass 是充分特例；“错误 J”本身不保证错误落在 D 的支持内或能被 Q 区分。
三条边不会把 m 变成 3m。

### 有限测试集合

若只使用预先给定的有限集合 T，只能声明：

```text
for every x in T:
  Close_Q(Source(x),Replay(x))
  and Close_Q(Source(x),RefEval(J,x))
  and Close_Q(Replay(x),RefEval(J,x))
```

此时不得引用隐藏随机 challenge 的概率上界。

每份 Gate2 记录必须包含 Q/protocol version、测试模式、D 或有限 T 的标识、实际 c0、三条边各自
结果与失败原因；这些是审计 metadata，不是新的 proof artifact 或 Replay 输入。

## 当前 RefEval 的标量与 logical-domain 支持边界

runtime scalar 或 non-C semantic constexpr 若代表 J 的逻辑数据，可以锚定一个 rank-0 J input。
Harness 给 Source/Replay 绑定前端标量，同时给 `RefEval(J,x)` 提供相同值的 rank-0 logical input；
它不能因此被折叠成 C。只有表示 shape/domain extent 的 integral handle 才进入 dimension binding，
不得同时冒充 J 的数据输入。

当前 pure functional pointwise profile 可以机械消去物理 logical-domain tail mask，但只在以下条件
同时成立时成立：

- observable output 是直接、无 update/alias/inout 的 functional value；
- memory access 与 J 的 logical tensor/domain 直接锚定，各轴 logical bound 与实际 shape 一致；
- 额外 guard 是 rank-0 `i1` logical input，并真实出现在 J value dataflow 中；
- direct singleton index `[0]` 只允许对应 logical extent 确为 1。

masked reduction 继续走独立的窄 profile，不从 pointwise mask erasure 推导。update、alias/inout、
不一致 bound、未锚定 guard 或其他 topology 必须明确返回 `REFERENCE_UNSUPPORTED`/fail closed，
不能由 Harness adapter 写 per-kernel 数值公式补齐。

## 失败与认证

Gate2 失败结束本次实验，不在同一实验中把结果反馈给 Lift/Replay 做局部修补。实验结束后可以
聚合失败、修改系统，再从新实验开始。

```text
SCRC_CERTIFIED
iff TypeCheck(J,C,pi) and Gate1 and Gate2
```

masked-J/J-authority 状态单独报告，不进入上述合取。Production CUDA/RVV 等 backend conformance
也独立于 Gate2，不能替代 Source–Replay、Source–J 或 Replay–J 任一腿。
