# Gate2：Source / Replay / J 三方语义验证

> 状态：当前派生规范。上位设计见 [C](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。

## 三腿与权限

对 logical input x：

```text
O_S = Exec(Source, rho_H^S(C), x)
O_R = Exec(Replay, rho_H^R(C), x)
O_J = RefEval(J, x)
```

Source 与 Replay 使用同一个 C 对象和 valuation。`RefEval` 的接口中不存在 C 或 π，也不读取
Source/Replay。所有 output、inout、alias/writeback 与声明的 observable effect 都必须裁决。

## 比较关系

```text
G2 = (O_S equivalent_B O_R)
     and
     (O_S equivalent_B O_J)
```

`equivalent_B` 是系统比较协议根据 B 所声明的封闭 comparison class 和必要语义合同构造的关系。
B 不提供任意可执行 comparator。系统必须验证 dtype、reduction extent、reassociation 权限、
tolerance 上限、NaN/Inf 与 allowed-result/atomic 等全部参数；不支持的类别、非法参数、缺腿或
执行不完整均 fail closed。

对 exact/tolerance relation，关系按系统模板比较。对允许多个合法结果的 relation，
`equivalent_B` 检查各腿是否满足同一个 J-owned allowed-result contract，不能把两个不同但合法的
winner 误报为失败，也不能接受 B 未授权的任意结果。

## Gate2 输入模式

Gate1 首次通过的 `(J,C,π,Replay)` 在本次 Gate2 期间不再修改，随后才产生 Gate2 输入。两种
实验口径必须明确区分。

### 隐藏随机输入

从预先声明的分布 D 独立同分布抽取 m 个对 Lift/Replay 隐藏的输入，Gate2 结果不回流本次实验。
在系统 `equivalent_B` 下定义：

```text
epsilon_J = Pr_{x ~ D}[Exec(Source,C,x) not equivalent_B RefEval(J,x)]
```

若 `epsilon_J >= epsilon > 0`，错误 J 通过全部 Source–J challenge 的概率至多为：

```text
(1-epsilon)^m <= exp(-m*epsilon)
```

### 有限测试集合

若只使用预先给定的有限集合 T，只能声明：

```text
for every x in T:
  Source(x) equivalent_B Replay(x)
  and Source(x) equivalent_B RefEval(J,x)
```

此时不得引用隐藏随机 challenge 的概率上界。

## 失败与认证

Gate2 失败结束本次实验，不在同一实验中把结果反馈给 Lift/Replay 做局部修补。实验结束后可以
聚合失败、修改系统，再从新实验开始。

```text
SCRC_CERTIFIED
iff TypeCheck(J,C,pi) and Gate1 and Gate2
```

masked-J/J-authority 状态单独报告，不进入上述合取。Production CUDA/RVV 等 backend conformance
也独立于 Gate2，不能替代 Source–Replay 或 Source–J 任一腿。
