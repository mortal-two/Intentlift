# SCRC Gate2：源／重放／Lowering 三方实跑

## 定位

只有 Gate1 `REPLAY_CLOSED` 的正式 artifact 进入 Gate2。Gate2 对同一批输入实际
执行三条独立腿：

```text
O_src = 原 Kernel 实际执行
O_rep = Gate1 重放 Kernel 实际执行
O_low = J 经项目 production lowering 实际执行
```

Gate2 不调用归档 JSON 解释器充当第三条腿。

## 当前 lower2c 原型

仓库已经存在真实链路：

```text
IntentIR JSON → intent MLIR → linalg/loops → LLVM → mlir-runner
```

新入口 `python/intentir/verify/gate2/lower2c.py` 直接复用这条链。当前名字沿用
“lower2c 原型”的任务称呼，但实际产物是 CPU LLVM JIT，不伪称已经输出 C 文本。
若未来确需字面意义的 C source，应另行论证并获批；当前 SCRC 只要求一条真实
production lowering 执行腿。

## 拆分执行

- `TritonRunner` 对 source 与 replay 分别加载独立模块、分别 launch；
- 两腿从同一份 canonical NumPy 输入各自构造 GPU tensor；
- `Lower2CRunner` 从同一输入绑定到 IntentIR 参数并执行 lowering；
- 三路输出统一回到 NumPy，再按 dtype/case 的 tolerance 比较。

机械归属：

| 两两一致关系 | Gate2 状态 | 含义 |
|---|---|---|
| 三方一致 | `GATE2_PASS` | 本测试域闭合 |
| 源≈重放，lower 偏离 | `LOWERING_DIVERGED` | lowering 问题 |
| 源≈lower，重放偏离 | `REPLAY_DIVERGED` | Gate1/replay 漏检 |
| 重放≈lower，源偏离 | `LIFT_DIVERGED` | J 与重放自洽地错 |
| 三对均不一致 | `ALL_DIVERGED` | 需按三腿错误继续定位 |

有限 tolerance 可能不满足传递性；出现“两对通过、一对失败”时返回
`INCONCLUSIVE`，不强行归属。

## 输入

沿用主方案的三个维度：

- shape：1、block−1、block、block+1、非整除、多 tile；
- value：随机、零/负值、大动态范围、需要时的 inf/NaN；
- seed：每组至少 3 个。

这些输入属于 Gate2，不参与 Gate1 TTIR 裁决。

## 当前实现

代码入口：`python/intentir/verify/gate2/`。

- source/replay 实际拆分执行已实现；
- production lowering 执行已实现；
- 三方 tolerance 比较与机械归属已实现；
- `python/tests/test_gate2.py` 已真实跑通 add 的 source/replay/lowering 三方；
- `test/scrc_live_add_e2e.py` 已进一步跑通真实 LLM lift + 真实 source-blind
  replay 后的完整三方 add：`N=1003, BLOCK=256`，source↔replay 最大误差 0，
  source/replay↔lowering 最大误差约 `5.01e-6`，状态 `GATE2_PASS`；
- 其余 op 能否进入 Gate2，由现有 A/B/C 表达和 production lowering 的真实
  支持情况决定；不引入“verify-ready op”概念。

若 lower 当前不支持某个真实 LLM 提取 artifact，Gate2 返回
`LOWERING_UNSUPPORTED`。这是一条需要补 A/B/C 或 lowering 的具体失败证据，
不能退回解释器后伪报通过。
