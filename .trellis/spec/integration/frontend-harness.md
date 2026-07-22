# Frontend Harness 合同

> 状态：当前派生规范。上位设计见 [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)。

Harness 是预先存在的运行与调用环境，不是 IR、模型产物或第四个 proof artifact。

## 最小职责

```text
EnumerateBindings(source invocation) -> anonymous coordinates
BindSource(Source,C,inputs)           -> source invocation
BindReplay(Replay,C,inputs)           -> replay invocation
Build(invocation)                     -> executable
Run(executable)                       -> observable result
Observe(executable/result)            -> Gate evidence
```

Harness 可以保存：

- concrete logical inputs 与 C values；
- source/replay 的机械调用坐标；
- grid、device、toolchain 与 compiler options；
- build/run/observation diagnostics。

Harness 不得：

- 根据参数名、`constexpr` 或 Kernel family 判断 J/C/π 归属；
- 生成 operator、predicate、formula、schedule strategy 或 Replay code；
- 把真实 Source binding、compiler diff 或 H 整体交给 Replay；
- 建立独立 `LogicalABI`、`InterfaceArtifact` 或带语义的 BindingMap。

## Logical signature 与绑定

input、output、runtime scalar、destination、dtype、rank、shape、alias/inout/effect 属于 J logical
signature。adapter 若需要函数视图，只能使用 `InterfaceView(J)`，且该视图必须由 J 确定性产生。

C 是一个对象。Source 与 Replay 在同一候选中使用同一 C valuation；`MaskC(C)` 只给 Replay
提供匿名 slot/schema。non-C compile-time value 作为调用事实绑定，但其 semantic consequence
必须已经在 J 中表达，其 realization use 才可由 π 引用。

该 valuation 记为 `c0 = Bind_H(C)`。Gate1/Gate2 的主证书只覆盖实际 c0；测试多个配置时只能
报告实际覆盖的有限集合，不能据此声称所有 admissible C 都已认证。

## 三方执行权限

```text
Source  <- H binds C and inputs
Replay  <- H binds the same C and inputs
RefEval <- logical inputs only
```

`RefEval` 的函数接口不能接收 C 或 π。实验记录可以关联当前候选，但关联 metadata 不得变成
RefEval 的功能输入。

新增前端首先实现上述薄 adapter；不要求 J-to-frontend lowerer、RuleCard、source-to-π parser
或 Kernel-specific mapping table。observer 强度可以逐步增强，并必须记录 frontend、observer
profile/version、compiler/dump stage 与 canonicalization version。
