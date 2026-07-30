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
   使用同一个 C 对象和实际 valuation `c0`。主证书只覆盖该 c0，不自动覆盖所有 admissible C。
4. π 是依赖 J/C 才有意义的 source-realization witness，没有独立语义权威。
5. 工程接口只能是 `InterfaceView(J)` 的确定性投影，不是第四对象、额外命名空间或 Replay 输入。
6. Replay 只接收 `J + π + MaskC(C) + U_f`；U_f 是与当前源码无关的公共前端上下文。
7. Harness 只保存和绑定调用事实，不构造 J、C、π 或 Replay。
8. `RefEval` 只读取 J；production lowering 读取 J/C，不读取 π。
9. 主认证严格是 `TypeCheck(J,C,π) ∧ Gate1 ∧ Gate2`。
10. masked-J 只报告 J 的操作性必要性，不自动推翻主认证。
11. 当前 Gate2 主线覆盖 deterministic / contract-deterministic Kernel，并直接检查 Source–Replay、
    Source–J 与 Replay–J 三条对称 `Close_Q` 边。
12. production core 对 accepted J 做 deterministic Intent MLIR materialization，并只生成 generic
    pass contracts 保持 hard boundary 的 concrete states；C 只排序。post-SCRC 通用 controller 用冻结
    Q/D 裁决 observation-changing candidate，性能只在通过者中选择。

## 固定代码可以做什么

- closed schema、引用、类型、授权和隔离检查；
- `MaskC(C)`、`InterfaceView(J)` 等确定性投影；
- build、bind、run、observe、compare 和结构化诊断；
- 将正式 Runner 的同一 source-free parser/Decode/anonymous-view/TypeCheck callable core 以时间治理、
  只读、非认证的 CandidateCheck 暴露给持久 source-aware Lift owner；Runner 另做 privacy/provenance
  并重跑 core；
- 在同一 owner 的受限 workspace 中保存文件、原子刷新 closed feedback、对 LLM 明示的唯一 old/new
  字符串执行 exact-text replacement，以及用 hash/长度 receipt 压缩旧 tool transport；这些机制不得
  解析、选择、补写或推荐任何 J/C/π 语义；
- 将 Lift 本来获准读取、且在固定大小阈值内的 exact Source/Harness/closed feedback 与 source-neutral
  contract/preflight/example 原文直接搬入 user packet；超限只能给 receipt/path，不得由 fixed code
  生成语义摘要；
- 对 exact Gate1 diff 作版本化、确定性的 bounded frontier 投影：只保留 closed category count 与有限
  原始 witness，完整 diff 继续原样落盘并可按需读取；不得解释 mismatch、排序候选、建议语义或产生
  corrected patch；
- 将 LLM 在一个 action 中给出的完整 candidate text 原样落盘并调用既有 preflight，或在 LLM 明示
  exact old/new text 后立即 preflight；不得由 runtime 生成、选择或修复字段；
- 按外部比较政策 Q 从 B 的封闭类别构造确定性、对称的 Gate2 comparator；
- 对不支持或不合法的输入 fail closed。

在 Lift/Replay/SCRC 路径中，固定代码不得识别 Kernel family、从规则库构造实现、补写遗漏语义、
从 compiler diff 修补 Replay，或把历史候选转换成新的证明输入。production backend 可以根据 A
的通用 op/region/graph pattern 应用 registered source-neutral transformations 并扩展具体 MLIR
lowering derivations，但不得读取 kernel name、Source、π 或使用 per-kernel RuleCard。

## 明确禁止

- 独立 interface/ABI 语义对象、BindingMap 或 source-conditioned sidecar 进入 Replay；
- Lift/Replay/SCRC 中从 J/π pattern 选择 target algorithm/API 的 deterministic semantic composer、
  per-kernel RuleCard/RuleRegistry、`C.strategy` 或 per-kernel construction policy；允许 source-blind
  Replay owner 在本次 fresh episode 内使用 syntax-only editing composer维护自己作者化的源码，但其
  固定实现不得读取J/π语义、展开recipe、跨episode复用或进入artifact/Gate/backend；production只允许
  以 typed A pattern、B/current-state precondition 与 target capability 为 key 的 generic compiler
  transformation registration；
- 以 hash、digest、commitment、epoch 或所谓冻结门决定候选是否合法；
- 让 Lift 自定义任意 comparator、任意 tolerance 或“所有输出均可接受”；
- Gate1 失败后 patch Replay，或 Gate2 失败后在同一次实验中继续修补候选；
- 将 masked-J hard close 直接映射为 `LEAKAGE_REJECTED`；
- 让 production backend 从 π、源码身份、provider 或 Gate verdict 补语义。

## 实验与系统改进

实验期间规则、prompt、模型配置、TypeCheck、observer profile 和比较政策 Q 保持不变，只修改
候选 artifact。Gate1 失败的 bounded frontier 与 exact diff 只反馈给同一 branch 的持久 source-aware
Lift owner；owner 继续修改
自己刚提交的完整 artifact，relift 必须产生完整新三元组，
随后生成全新的盲 Replay。三轮仍未闭合则报告 `NO_CLOSURE`。

Gate2 失败结束本次实验。此后才允许根据跨样本失败修改系统，再从新的实验开始运行。
