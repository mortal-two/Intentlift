# IntentIR 文档入口

## 当前权威设计

当前设计由三份相互约束、缺一不可的核心文档共同组成：

1. [C：所有权、绑定、消费与三方验证](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)：确定信息归属、组件可见范围、Harness/RefEval/lowering 边界与三对象因子化。
2. [π：依赖类型实现见证与 Replay 安全](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)：确定 realization witness、`TypeCheck(J,C,π)`、授权规则与 Replay 安全边界。
3. [Lift、Replay、Gate1、Gate2 最终执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)：确定实验轮次、relift、失败停止、Gate2 与实验结束后的系统改进流程。

三份文档共同规定：

```text
Lift_H(S) -> (J,C,pi)
Replay = W_f(J,pi,MaskC(C);U_f)
SCRC_CERTIFIED iff TypeCheck(J,C,pi) and Gate1 and Gate2
```

任何实现文档或派生 spec 若与这三份文档冲突，应修订派生材料，而不是增加第四个设计对象、
额外认证 Gate 或隐藏信息通道。

## 派生规范

当前工程规范位于 [`.trellis/spec`](../.trellis/spec/index.md)。它只把上述设计投影为 schema、
API、runner 与测试要求，不具有独立于三份核心文档的设计权威。本项目只借用 `.trellis` 的
目录布局，不使用 Trellis workflow、task CLI 或 session 流程。

## 来源与历史

- [source-notes](source-notes/README.md) 保存设计推理、外部回答、纠错与实现审计；这些材料不是当前规范。
- [archive](archive/README.md) 保存被取代的设计和历史实验文档；这些材料只用于追溯与迁移取证。
