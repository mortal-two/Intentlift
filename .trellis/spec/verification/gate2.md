# Gate2：Source-anchored 三角语义验证

## 三腿

对同一逻辑输入 x：

```text
O_src = Exec(Source, x)
O_rep = Exec(Replay, x)
O_J   = Exec(E_J, x)
```

`E_J` 是简单、无优化、只读取完整 J 的 backend-neutral scalar semantic executor。

## 正式裁决

Gate2 PASS 对每个冻结 case 都要求：

```text
O_src ≈τ O_rep
O_src ≈τ O_J
```

`O_rep≈τO_J` 只作派生诊断。所有 output、inout、alias/writeback 和声明的可观察 effect
都必须逐项比较。unsupported、缺腿、执行不完整或 comparator 未冻结都不能 PASS。

`E_J` 必须接收 Gate1 使用的同一个 canonical J 对象，不得先转换成第二份 semantic
wrapper。尚未实现的 B relation（当前包括显式 access/domain/effect/alias 等）必须在执行前
fail closed；不能只执行 A 后把 boundary digest 记作“已消费”。

## 诊断边界

- `Source=Replay≠E_J`：J/scalar-semantic 路径有问题，不能只怪 lowerer；
- `Source=E_J≠Replay`：Replay/π/调用路径有问题；
- `Replay=E_J≠Source`：J 与 Replay 自洽但不对应 source，或 source binding 错；
- 三者均不同：多重或基础设施问题。

## 有限关系和随机声度

冻结 envelope T 上，Gate1 与两条 source-anchored Gate2 边共同定义
`Source≈SCRC^(f,k,T,τ)J`。

若最终 artifact 在 challenge 前 commitment，随机 case 独立采样自 D，则任何错误质量
`d_D,τ(Source,J)>ε` 的固定 artifact 通过 m 个 challenge 的概率至多：

```text
(1-ε)^m <= exp(-mε)
```

该界对任意 π 和任意 LLM 都成立。失败后提交新 artifact 必须使用新 seed。确定性 hard
envelope 可与随机 challenge并存。

## TCB 与 backend 边界

`E_J` 是可执行语义的可信基座，必须简单、版本化并由 primitive-level test 验证。
CUDA/RVV production lowering 属独立 backend conformance，不能代替 Gate2 任一腿。
