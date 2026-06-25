# Project 层

项目身份、科研定位、不可偏离论文的约束、能力契约理念与 anti-drift 规则。

## 何时读

- 任何架构决策、对外定位、paper 对齐、或"要不要支持某个方向"的判断之前。
- 怀疑某个改动是否偏离了 IntentIR 科研构思时。

## 文件

| 文件 | 内容 |
|---|---|
| [constitution.md](constitution.md) | 项目宪法:Purpose、核心科研问题、三层脊柱、Required / Forbidden / Check Questions。 |
| [positioning.md](positioning.md) | 科研定位:与 TVM/Halide/MLIR/反编译的区别,以及"LLM 提取规范"这条护城河。 |
| [code-layout.md](code-layout.md) | 代码物理结构草案与待定决策(dialect 实现方式、包结构)。 |

## Pre-Development Checklist

- [ ] 这个改动锚定论文科研构思了吗?(论文是锚,实现可全新)
- [ ] 它守住了 A/B 绑定、C 非绑定的分层不变量吗?
- [ ] 它把原子化分解留在 LLM 提取里,而不是塞进 pass 吗?
- [ ] 它对能力边界是诚实的(标注而非假装)吗?

## Quality Check

- 改动后,项目仍是"LLM 提取 + 验证裁决"的语义提升系统,不是"LLM 直接生成 kernel 代码",也不是传统编译器。
- 没有引入 org / OptGuide / provider29 思路,没有继承旧仓库实现。
