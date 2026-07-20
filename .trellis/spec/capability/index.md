# Capability 层:能力契约

成熟工具卖的是**清晰、可审计的能力边界**，不是“号称万能”。本层分别定义
represent、extract、Gate verify 与 target lowering 的边界；这些维度可以独立演进，
但任何对外声明都必须绑定证据。

## 何时读

- 决定"要不要支持某类 kernel 模式"之前。
- 对外声明覆盖范围、写 paper claim 之前。

## 文件

| 文件 | 内容 |
|---|---|
| [capability-matrix.md](capability-matrix.md) | 能力矩阵：每种 Kernel 模式的 represent / extract / Gate verify / backend contract / v1。 |
| [boundaries.md](boundaries.md) | 四种“不能”、固定图 vs 数据依赖结构、诚实天花板。 |

## 核心原则

- **能力边界是设计选择,不是天生定律**:绝大多数"不能"是"还没加构造",不是"做不到"。
- **四种不能要分清**：不能表示 / 不能可靠提取 / 不能在当前 Gate 命题下充分验证 /
  某 target backend 尚不能正确消费。
- **诚实优先**:覆盖不了就标注,绝不假装能做。`v1 是否纳入`列就是规范对外承诺的能力契约。
- Gate PASS、device execution 与 performance PASS 是不同证据，禁止互相升级。
