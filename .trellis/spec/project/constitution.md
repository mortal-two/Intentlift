# IntentIR 项目边界

> 状态：当前派生规范。上位权威见 [spec 入口](../index.md)。

## 目标

IntentIR 让 Lift 负责从 Source 提出完整 `(J,C,π)`，让 source-blind Replay 根据受限输入生成
完整前端源码，让很薄的固定核心只做机械检查、绑定、构建、执行和比较。

```text
Lift_H(S) -> (J,C,pi)
Replay = W_f(J,pi,MaskC(C);U_f)
```

## 不可破坏的边界

1. 证明相关 artifact 严格只有 J、C、π。
2. J 是唯一 denotational semantic object，并拥有 logical signature。
3. C 是不改变 J denotation 的 implementation-selection 对象；一个实验中的 Source 与 Replay
   使用同一个 C 对象和 valuation。
4. π 是依赖 J/C 才有意义的 source-realization witness，没有独立语义权威。
5. 工程接口只能是 `InterfaceView(J)` 的确定性投影，不是第四对象、额外命名空间或 Replay 输入。
6. Replay 只接收 `J + π + MaskC(C) + U_f`；U_f 是与当前源码无关的公共前端上下文。
7. Harness 只保存和绑定调用事实，不构造 J、C、π 或 Replay。
8. `RefEval` 只读取 J；production lowering 读取 J/C，不读取 π。
9. 主认证严格是 `TypeCheck(J,C,π) ∧ Gate1 ∧ Gate2`。
10. masked-J 只报告 J 的操作性必要性，不自动推翻主认证。

## 固定代码可以做什么

- closed schema、引用、类型、授权和隔离检查；
- `MaskC(C)`、`InterfaceView(J)` 等确定性投影；
- build、bind、run、observe、compare 和结构化诊断；
- 按系统比较协议从 B 的封闭类别构造 Gate2 comparator；
- 对不支持或不合法的输入 fail closed。

固定代码不得识别 Kernel family、从规则库构造实现、补写遗漏语义、从 compiler diff 修补 Replay，
或把历史候选转换成新的证明输入。

## 明确禁止

- 独立 interface/ABI 语义对象、BindingMap 或 source-conditioned sidecar 进入 Replay；
- deterministic composer、RuleCard/RuleRegistry、`C.strategy` 或 per-kernel construction policy；
- 以 hash、digest、commitment、epoch 或所谓冻结门决定候选是否合法；
- 让 Lift 自定义任意 comparator、任意 tolerance 或“所有输出均可接受”；
- Gate1 失败后 patch Replay，或 Gate2 失败后在同一次实验中继续修补候选；
- 将 masked-J hard close 直接映射为 `LEAKAGE_REJECTED`；
- 让 production backend 从 π、源码身份、provider 或 Gate verdict 补语义。

## 实验与系统改进

实验期间规则、prompt、模型配置、TypeCheck、observer 和 comparator protocol 保持不变，只修改
候选 artifact。Gate1 失败的 diff 只反馈给 source-aware Lift；relift 必须产生完整新三元组，
随后生成全新的盲 Replay。三轮仍未闭合则报告 `NO_CLOSURE`。

Gate2 失败结束本次实验。此后才允许根据跨样本失败修改系统，再从新的实验开始运行。
