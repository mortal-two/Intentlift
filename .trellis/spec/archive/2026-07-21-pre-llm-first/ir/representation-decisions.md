# 表示决策

这些是地基级决策。实现可以迁移，但长期语义所有权不能含糊。

## 决策一:控制流 —— 进(in)

- **默认**:Layer A 是**无显式循环的张量代数**,循环都折进算子的轴(reduce/contraction 的 reduction 轴)或 `scan`。
- **逃生舱**:提供一个**门控的结构化 `loop` 构造**(scf 风格,带 carry),用于:
  - 递推(loop-carried,如 FlashAttention online-softmax)——次数静态;
  - 真·数据依赖迭代(while / 收敛)——次数由数据决定。
- **理由**:支持 FlashAttention 这类算子**强制要求**有循环构造(见 [structured-loop.md](structured-loop.md));90% 的 kernel 走无循环张量代数保持干净,前沿情况用门控逃生舱给出原则性答案,而不是"以后再说"。

## 决策二：逻辑结果与 destination-passing 分层

- **A 先定义逻辑结果**：每个 output 映射到一个 logical SSA value；
- **B 再定义 destination**：写入哪个 storage、must/may/no alias、full/partial/
  conditional coverage、是否 preserve unwritten；
- dialect lowering 可以使用 linalg-style `outs`，但 `outs` 是 B／bufferization 的
  realization，不能成为逻辑返回值的唯一真值来源；
- 多输出、in-place、共享 storage 和 optimizer update 必须使用 typed destination
  bindings，不再依赖单个 `out` 字符串表达全部情况。

这样既保留 destination-passing 对真实 Kernel 和 bufferization 的价值，又避免把
“算出什么”与“写到哪里”混成一个字段。

## 决策三:named op(库级命名算子)—— 起步不做

- **起步阶段纯原子**,一个库级 named op 都不留。
- **理由**:张量级原子粒度已消除"长图爆炸"问题(FlashAttention 也只需有限个原子),named op 在起步阶段只带来"何时用 named vs 原子"的判定模糊和"语义≡展开"的维护负担。
- **未来**:若某库级 pattern 高频且原子展开确实笨重,再作为"钦定语义的标准词"加进规范(语义 ≡ 固定原子展开,LLM 可直接产出)。它**绝不是** pass 分解——语义由规范定死,不由 pass 猜。

## 决策四:算子集 —— 受控开放

- 核心算子集封闭明确(见 [atomic-ops.md](atomic-ops.md));新增算子必须走规范(语义 + shape/dtype 规则 + 层归属 + 可执行 reference),不得随手加、不得留空语义。
