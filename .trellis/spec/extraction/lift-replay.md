# LLM Lift 与 Replay 合同

## Lift

```text
LiftLLM(source, retrieved_context?) -> {J, C, π, diagnostics?}
```

Lift 直接完成 whole-tensor semantic lifting、J 原子化/组合化、纯 C 分离和 source
realization π 提取。固定代码不得禁止 LLM 输出 π 后再由 composer 重建，也不得用
source-pattern recognizer修复结果。

模型可见 Lift response 是一个封闭 root：

```text
root = {schema, j, interface, schedule, pi, location_map}
j    = {kernel_id, intent, node_ids, boundary, scopes, storage}
```

`j` 必须在 `storage` 后关闭，`interface/schedule/pi/location_map` 始终是 root sibling。内部
`source/provenance` 不属于模型输出。固定 parser 对错位、缺项或 wrapper 只拒绝，不 hoist、补项
或从旧 artifact 复制。

Lift 必须按 denotation 而不是按 Python/Triton binding kind 分类参数：semantic consequence
进入 J，representation-only use 进入 pi，schedule constexpr 才进入 C。`tl.constexpr`、
模板参数或编译期常量均不能被固定代码一律投影成 C；名字匿名化也不能改变该分类。

J/C/π 在 Lift 边界已经使用 opaque handle。真实源码命名只留在不可见 harness sidecar。

J 的每个 load/store/update access 都显式带非空 ordered logical `index_expression`，与 tensor rank
无关。源码存在的 logical domain、predicate 与 masked-load fill 必须保留在 J，即使 frozen
invocation 令 mask 恒真；源码不存在时也不得凭模板发明。单个 JSON parser 只能验证表示完整性，
不能自行证明这些内容与源码一致，该义务由 source-aware Lift 提议并由 Gate1 strict topology 检验。

## Replay

```text
ReplayLLM(J, C, π, Context_f) -> complete source R_f
```

Replay 看不到源码、source span、真实名字、旧候选或 compiler diff。`Context_f` 可以来自
LLM 动态检索的公共前端文档、仓库示例和工具说明，检索 trace 进入 artifact receipt；
核心仓库不维护或预选按当前 Kernel/op/pattern 组织的 RuleCard/API mapping。

Replay 的 harness 可以给 non-C constexpr 提供与 source 相同的冻结调用值，但 Replay
不能从 C 获得该值的语义权威；逻辑 extent/predicate/result 只能来自 J，纯表示 use 只能
来自 pi 的匿名 topology。

## 失败

Replay/compile/Gate1 失败只能反馈给 source-aware Lift。新 Lift 产生完整新 artifact，随后
重新执行一次 blind Replay。禁止 base-candidate repair、diff-guided Replay patch 和同隐藏
challenge 上的自适应修补。
