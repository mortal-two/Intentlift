# Verification：SCRC Gate1 + Gate2

项目当前的 `verify` 仅指两个正式过程：

| 文件 | 内容 |
|---|---|
| [scrc-gate1.md](scrc-gate1.md) | Gate1-owned A/B/interface/π admission、可执行 π 必要性、源码隔离重放、双 TTIR 结构闭合与 source-aware Relift 外循环。 |
| [scrc-gate2.md](scrc-gate2.md) | final-J replay 与 typed scalar MLIR lowering 双腿实际执行、逐输出比较与严格矩阵聚合。 |
| [architecture.md](architecture.md) | 当前两门架构入口及旧方案归档位置。 |

原则：

- LLM 负责理解与生成；extraction 只交付原始 proposal，纯代码 Gate1 独占语义准入与裁决；
- formal Gate1 的 J-only 与全部 causal-step π dropout 必须真实到达 writer/compile/hard
  compare；writer 前 canonicality rejection 和 `wrong_fill_delta` 都不是 acceptance evidence；
- Gate1 不执行，Gate2 才执行；活动 Gate2 只接收 Gate1 硬闭合的最终 J；
- Gate2 PASS 不覆盖或补偿 Gate1 拒绝，rejected sample 的离线执行只算普通诊断；
- 两门都不依赖旧 JSON 解释器；
- Gate2 的 scalar MLIR 路径是 verification/reference lower；production target lowering
  是独立 backend 能力，不参与 Gate2 verdict；
- 正式样本必须从真实 Kernel 经 LLM extraction 产生 J/π/C；
- 手写 Intent MLIR/固定 PK 只证明 lowering 局部能力，不证明 SCRC 端到端。
