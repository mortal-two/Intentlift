# IntentIR 项目宪法（LLM-first v1）

## 目标

让 LLM 负责理解任意 Kernel、抽取 `J/C/π`、并在源码隔离下生成完整 Replay；让极小的
固定验证核心只负责隔离、构建、执行、比较和审计。

```text
不把碎片化生态编进系统，只把可验证性编进系统。
```

## 必须坚持

1. `Intent = J(A+B)+C`，且 `π∉Intent`。
2. J 是唯一语义对象；π 是 source-realization witness；C 是纯调度。
3. Lift LLM 直接输出匿名 J、纯 C 和 π；Replay LLM 直接输出完整前端源码。
4. 固定代码可以拒绝错误，不得理解 Kernel family 或替 LLM 构造实现。
5. Gate1 是 source-blind、frontend-observer-parameterized 的构造闭合。
6. Gate2 的正式边是 `Source↔Replay` 与 `Source↔E_J`；`E_J` 只读取 J。
7. 新前端只需最小 build/run/observe harness；强 observer 是可插拔证明增强器。
8. 在线最多一次 normal Replay 和一次 gated masked-J Replay；其他消融离线运行。
9. 失败只能回到 source-aware Lift，随后 fresh blind Replay；禁止 patch Replay。
10. production lowering 直接消费完整 J，C 独立传入；backend conformance 不属于 Gate2。

## 禁止

- deterministic composer、work-item 枚举或 source-pattern dispatcher；
- RuleCard/RuleRegistry、selected card、per-op frontend API mapping；
- `C.strategy` 或任何 realization 兜底字段；
- 重型 `IntentProgram` 第二 wire IR和独立 `LogicalABI` 语义对象；
- kernel-name recognizer、per-kernel construction policy；
- 在线 semantic mutation、π swap、C swap、dropout/family sweep；
- 使用源码、旧 Replay 或差分对 Replay 做局部修补；
- 让 Gate verdict、provider identity、source path 或 dataset identity进入 lowering；
- 把当前 IntentOp 候选设计伪装成已经完成的通用支持 claim。

## 审查问题

每个新增组件都必须回答：

1. 它是在理解/生成，还是在检查已生成对象？前者默认交给 LLM。
2. 维护成本是否随 `|frontend|×|operator|×|realization|` 增长？若是则拒绝。
3. 它增强的是外部闭合证据，还是复制了第二份语义真值？后者拒绝。
