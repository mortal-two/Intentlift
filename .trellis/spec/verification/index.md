# Verification：SCRC Gate1 + Gate2

项目当前的 `verify` 仅指两个连续过程：

| 文件 | 内容 |
|---|---|
| [scrc-gate1.md](scrc-gate1.md) | J/π/C 驱动的源码隔离重放、双 TTIR 结构闭合、规则库与有限修复。 |
| [scrc-gate2.md](scrc-gate2.md) | source/replay/lowering 三方实际执行与机械归属。 |
| [architecture.md](architecture.md) | 当前两门架构入口及旧方案归档位置。 |

原则：

- LLM 负责理解与生成，纯代码负责 Gate1 裁决；
- Gate1 不执行，Gate2 才执行；
- 两门都不依赖旧 JSON 解释器；
- 正式样本必须从真实 Kernel 经 LLM extraction 产生 J/π/C；
- 手写 Intent MLIR/固定 PK 只证明 lowering 局部能力，不证明 SCRC 端到端。
