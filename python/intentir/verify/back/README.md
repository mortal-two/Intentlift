# 旧验证实现归档

这里保存 SCRC 之前的验证/解释执行代码。它们继续可执行，是为了不破坏历史
测试与已有调用方；但不再参与新的正确性主线。

新的验证只有两个过程：

1. `intentir.verify.gate1`：源码隔离重放、双 TTIR 指纹闭合；
2. `intentir.verify.gate2`：源、重放、IntentIR production lowering 三方实跑。

约束：`gate1/`、`gate2/` 和新的 extraction 主路径不得导入本目录。旧路径
`intentir.verify.interpret` 只是兼容转发层。
