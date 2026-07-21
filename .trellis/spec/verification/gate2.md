# Gate2：Source-anchored 三角语义验证

## 三腿与一般结果关系

对同一逻辑输入 x：

```text
O_src = Exec(Source, x)
O_rep = Exec(Replay, x)
O_J   = Exec(E_J, x)
```

`E_J` 是简单、无优化、只读取完整 J 的 backend-neutral scalar semantic executor。对单值
输出，J 的 relation 是 exact 或 tolerance；对允许多个结果的 logical update，记完整允许
结果集合为 `R_J(x)`。`E_J` 产生一个独立合法 witness，Gate2 comparator 负责 membership。
relation、`rtol/atol/equal_nan` 均来自同一个 canonical J，不属于 case harness。

## 正式裁决

Gate2 PASS 对每个冻结 case 都要求三腿均属于同一个 relation：

```text
O_src ∈ R_J(x)
O_rep ∈ R_J(x)
O_J   ∈ R_J(x)
```

对 exact/tolerance，这退化为原来的 `O_src≈O_rep` 与 `O_src≈O_J` 两条 source-anchored
边。对 `one_of_updates`，source 与 replay 可以选择不同的合法 contender；Gate2 检查二者
各自 membership，不能错误要求 pairwise equality。

所有 output、inout、alias/writeback 和声明的可观察 effect 都必须逐项裁决。unsupported、
缺腿、执行不完整或 comparator 未冻结都不能 PASS。`E_J` 必须接收 Gate1 使用的同一个
canonical J 对象，不得先转换成第二份 semantic wrapper。尚未实现的 B relation 必须在
执行前 fail closed；不能只执行 A 后把 boundary digest 记作“已消费”。

每一腿在执行前获得 committed input pre-state 的独立副本。typed bounds 与
`duplicate_policy=forbid` 先作为 J 输入域检查；不满足前置条件的 case 是无效 challenge，
不能进入执行或产生 PASS。

## 诊断边界

- exact/tolerance 下 `Source=Replay≠E_J`：J/scalar-semantic 路径有问题；
- exact/tolerance 下 `Source=E_J≠Replay`：Replay/π/调用路径有问题；
- one-of 下报告各腿的 membership，不把不同合法 winner 误报为 divergence；
- 任一腿不属于 `R_J(x)`：只报告具体 relation failure，不由 comparator 武断归因。

## 有限关系和随机声度

冻结 envelope T 上，Gate1 与 source-anchored Gate2 共同定义
`Source≈SCRC^(f,k,T,R_J)J`。若最终 artifact 在 challenge 前 commitment，随机 case 独立
采样自 D，则任何错误质量大于 ε 的固定 artifact 通过 m 个 challenge 的概率至多：

```text
(1-ε)^m <= exp(-mε)
```

该界对任意 π 和任意 LLM 都成立。失败后提交新 artifact 必须使用新 seed。确定性 hard
envelope 可与随机 challenge 并存。

## TCB 与 backend 边界

`E_J` 与 relation comparator 是可执行语义的可信基座，必须简单、版本化并由
primitive-level test 验证。CUDA/RVV production lowering 属独立 backend conformance，
不能代替 Gate2 任一腿。
