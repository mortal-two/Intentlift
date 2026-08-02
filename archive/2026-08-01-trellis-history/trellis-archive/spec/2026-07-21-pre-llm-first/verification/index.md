# Verification：SCRC Gate1 + Gate2

项目当前的 `verify` 仅指两个正式过程：

| 文件 | 内容 |
|---|---|
| [scrc-gate1.md](scrc-gate1.md) | J/π/C 驱动的源码隔离重放、双 TTIR 结构闭合、规则库与 source-aware Lift/relift 外循环。 |
| [scrc-gate2.md](scrc-gate2.md) | source、final-J replay 与 typed scalar MLIR lowering 三腿实际执行、两份 source-anchored 逐输出比较与严格矩阵聚合。 |
| [architecture.md](architecture.md) | 当前两门架构入口及旧方案归档位置。 |

原则：

- LLM 负责理解与生成，纯代码负责 Gate1 裁决；
- Gate1 做编译期结构闭合，Gate2 做三腿 runtime 执行；活动 Gate2 只接收 Gate1 硬闭合的最终 J；
- Gate2 必须同时完成 `source↔replay` 与 `source↔scalar`；handoff/scalar 不可用不能
  阻止第一份证据执行，但整体不得 PASS；
- Gate2 PASS 不覆盖或补偿 Gate1 拒绝，rejected sample 的离线执行只算普通诊断；
- 两门都不依赖旧 JSON 解释器；
- Gate2 的 scalar MLIR 路径是 verification/reference lower；production target lowering
  是独立 backend 能力，不参与 Gate2 verdict；
- 正式样本必须从真实 Kernel 经 LLM extraction 产生 J/π/C；
- 手写 Intent MLIR/固定 PK 只证明 lowering 局部能力，不证明 SCRC 端到端。
