# 能力矩阵

每行回答 **可表示 / LLM 可提取 / Gate 可验证 / backend 必须如何处理 / v1 是否纳入**。
这五个维度彼此独立：IR 能表示不等于某个 target 已 lower；Gate 通过也不授权 backend；
某个 specialized pass 可执行也不证明 Lift 已闭合。

具体 Kernel／target 的已执行 lowering 状态不写死在本稳定矩阵中，统一以机器可读
[`spec/lowering/high-performance-kernels.json`](../../../spec/lowering/high-performance-kernels.json)
及其 loader 为证据来源。

| Kernel 模式 | 可表示 | LLM 可提取 | Gate 验证 | Backend 合同 | v1 |
|---|---|---|---|---|---|
| elementwise / map | ✅ | ✅ 强 | 结构 + 动态 | generic 或 specialized；保留 dtype/broadcast | ✅ |
| reduce | ✅ | ✅ 强 | 结构 + 动态 + 数值合同 | 消费 axes/identity/mask/order；否则 unsupported | ✅ |
| contraction / matmul | ✅ | ✅ 强 | 结构 + 动态 | generic contraction 或 target MMA；tail/acc dtype 绑定 | ✅ |
| scan / recurrence | ✅ | ⚠️ 中 | carry/loop 结构 + 动态 | semantic carry 必须保留；tile 可重调 | ✅ |
| logical layout/view | ✅ | ✅ 强 | logical map 必须 hard-visible | transpose/reverse/reshape 等 value-driving map 必须消费 | ✅ |
| gather / scatter | ✅ | ✅ 中 | logical index + 动态 | bounds/conflict/atomic obligation 必须检查或拒绝 | ✅ |
| select / where | ✅ | ✅ 强 | 结构 + 动态 | predicate/value broadcast 必须消费 | ✅ |
| segment-reduce | ✅ | ⚠️ 中 | 动态为主 | segment/conflict/identity 明确后再 lower | ✅ |
| atomic-reduce | ✅ | ✅ 中 | effect + tolerance | atomic/order/reassociation obligation 不得静默忽略 | ✅ |
| sync / barrier | ✅（B） | ✅ | effect/ordering 结构 | target 无等价机制则 unsupported | ✅（识别） |
| data-dependent control flow | ⚠️ 门控 loop/branch | ⚠️ 难 | scoped 结构 + 动态 | 未有一等 region contract 时 fail closed | 🔶 门控 |
| alias / in-place / 多输出 | ✅（typed destination） | ✅ | mapping/effect + 动态 | ABI、alias、write coverage、preserve-unwritten 必须消费 | ✅ |

## 五元组要求

每新增一种 Kernel 模式，必须同时回答：

- **represent**:IR 里用什么构造表示?
- **extract**:LLM 能从混淆源码里恢复出来吗?难度?
- **verify**:Gate1/Gate2 对什么命题给出何种有限保证?
- **lower**:generic/specialized backend 如何 consumed/checked/unsupported B?
- **evidence**:哪份机器可读 registry、lit/device test 或 conformance record 支撑声明?

禁止用“有一个 pass”“解释器能跑”或“历史 Gate2 三方曾通过”代替当前 backend
closure 证据。

## FlashAttention 自检

FlashAttention 落在 `contraction × 2` + `scan/recurrence` + `softmax(map/reduce)` 这几行的组合上,其 K 循环按 [../ir/layers-abc.md](../ir/layers-abc.md) 因式分解。它是 v1 必须支持的旗舰用例,不得破功。
