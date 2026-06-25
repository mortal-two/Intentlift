# Verification 层:验证与契约

定义"如何裁决一个 LLM 提案的 IntentIR 是否可接受":三道检查、契约等级、以及动态执行一致性如何反哺契约。

## 何时读

- 改动验证栈、obligation、契约判定、tolerance 之前。

## 文件

| 文件 | 内容 |
|---|---|
| [validation.md](validation.md) | 三道检查(schema/type、静态 obligation、动态执行一致性)与契约等级。 |
| [dynamic-feedback.md](dynamic-feedback.md) | 动态执行一致性反哺契约;atomic / 浮点非结合的 tolerance 验证。 |
| [architecture.md](architecture.md) | 分层验证架构(L0–L9):lift vs lowering 两个正确性问题、decidable vs fp-reassociating 轴、逐层诚实保证标签与当前状态。 |

## 核心原则

- **LLM 提案,验证裁决**:正确性不由 LLM 自证。
- **动态可救静态**:静态证不出但动态执行一致的,应被接受,而不是被拒或烂在 PARTIAL。
- **证书不是全域证明**:certificate = "测试输入域执行一致 + 静态义务"级别的保证。
- **FULL/PARTIAL/OOS 沿用论文术语**,但**不背旧实现的任何包袱**(旧的 partial/oos 判定逻辑一律不参考)。
