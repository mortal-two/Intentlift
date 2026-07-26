# IntentIR 文档入口

## 当前权威设计

当前规范性设计由五份相互约束的权威文档共同组成。前四份定义 SCRC，
第五份定义 SCRC 之后的 production backend：

1. [C：所有权、绑定、消费与三方验证](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)：确定信息归属、组件可见范围、Harness/RefEval/lowering 边界与三对象因子化。
2. [π：依赖类型实现见证与 Replay 安全](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)：确定 realization witness、`TypeCheck(J,C,π)`、授权规则与 Replay 安全边界。
3. [Lift、Replay、Gate1、Gate2 最终执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)：确定实验轮次、relift、失败停止、Gate2 与实验结束后的系统改进流程。
4. [IntentIR / SCRC 最终数学、执行流程与论文表述](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)：是上述三份 SCRC 设计的数学与 claim 总述，统一 `TypeCheck` 所有权结论、observer-relative Gate1、外部比较政策 Q、Gate2 三条直接比较、instance-wise `c0` 与概率声明边界。它不增加第四个设计对象或新的运行阶段；若对象/schema/API 细节发生歧义，仍回到对应的三份核心设计修订。
5. [J-Grounded、Q/D-Certified Lowering-State Search 后端最终设计](IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)：定义 accepted J+C 的 pure core lowerer、Intent MLIR materialization、hard verifier、统一 Q/D search-and-closure controller、concrete lowering-state search 与实验设计；它不修改 SCRC 接受谓词或 J。文件名为保持既有链接稳定而保留旧名称。

对应的 production 实现与实测记录：

- [Accepted J+C production backend](production-backend-jc-search.md)：已冻结 18-J catalog 的真实 C++ MLIR lowering、J/B 驱动的图级/规约/数值 state search、统一 Q/D controller 与证据边界。
- [18 个闭合 J 的 production backend 结果](results/production-backend-closed-18/README.md)：RTX 5090D 正式 E2/E3 与 L0/L1 closure、numeric/C 消融、H100/RVV portable 结果、压缩原始报告和 SHA-256。
- [16 项 backend implementation smoke](results/production-backend-closed-16/backend-implementation-smoke-2026-07-26.md)与 [12 个闭合 Kernel 历史报告](results/production-backend-closed-12/production-backend-report.md)：只用于追溯，不参与当前 18-J aggregate。

前四份 SCRC 文档共同规定：

```text
Lift_H(S) -> (J,C,pi)
Replay = W_f(J,pi,MaskC(C);U_f)
SCRC_CERTIFIED iff TypeCheck(J,C,pi) and Gate1 and Gate2
```

其中 Gate2 在当前 deterministic / contract-deterministic 主线中直接检查
Source–Replay、Source–J 与 Replay–J 三条 `Close_Q` 边；证书相对于 observer profile、Q、实际
绑定 `c0` 和测试域成立。

第五份文档在此之后规定：

```text
accepted J
  -> M0 = deterministic Intent MLIR
optional C
  -> S_C = Attach_C(M0,C,t) after M0
  -> erasable search seed / candidate ordering
  -> hard-safe structural/numeric lowering-state search
  -> generic Q/D candidate filter
  -> performance selection among passing states
  -> held-out final-winner backend closure
```

任何实现文档或派生 spec 若与这五份文档冲突，应修订派生材料，而不是增加额外证明对象、
认证 Gate、隐藏信息通道或 per-kernel construction policy。

## 派生规范

当前工程规范位于 [`.trellis/spec`](../.trellis/spec/index.md)。它只把上述设计投影为 schema、
API、runner、lowering 与测试要求，不具有独立于五份权威文档的设计权威。本项目只借用 `.trellis` 的
目录布局，不使用 Trellis workflow、task CLI 或 session 流程。

## 来源与历史

- [source-notes](source-notes/README.md) 保存设计推理、外部回答、纠错与实现审计；这些材料不是当前规范。
- [archive](archive/README.md) 保存被取代的设计和历史实验文档；这些材料只用于追溯与迁移取证。
