# JSON ↔ Intent MLIR Dialect

## 链路

```
kernel 源码
  → [LLM 提取]  IntentIR JSON   ← LLM 的产出物,人可读
  → [1:1 转换]  Intent MLIR     ← 我们自己的 dialect,确定性转写
  → [中立合同]  IntentProgram(A+B+ABI, optional C)
  → [lowering]  标准 MLIR / target executable
```

## 为什么 JSON 在前

- **LLM 的产出物**:LLM 输出 JSON,不直接吐 MLIR 文本(JSON 对 LLM 更稳、更好约束、更好自检)。
- **人可读 / 可审计**:JSON 是人能直接读懂的三层意图,便于 review 与作为证书载体。
- **自带 witness**:JSON 里 LLM 可附上它声称的 witness/claim(access pattern / mask / bound),供验证核对(见 [../extraction/evidence.md](../extraction/evidence.md))。

## JSON → dialect 是 1:1 转写,不是分解

- JSON 已经是**张量级原子粒度**,转成 Intent MLIR dialect 是确定性的 1:1 transliteration,**粒度不变**。
- 这意味着:**原子化分解的决策在 LLM 产出 JSON 时就完成了**,MLIR 这一层不做"高层 → 原子"的语义分解。
- dialect 复用 MLIR 基础设施:SSA values、region(给 `loop` / 带 region 的算子)、type system、attribute(给 Layer B 约束与 Layer C hint)。

## 三层在表示里的组织

- **Layer A**:dialect 里的算子 + region 结构(可执行语义对象)。
- **Layer B**:统一挂在 program boundary 上的 typed contract(mask、bound、
  destination、alias、effect、reduction、scope/storage 等)。
- **Layer C**:独立的非绑定 schedule hint(tile、向量宽度、unroll 因子);可被后端覆盖。

完整语义对象是 `J=A+B`。如果某个 correctness-relevant B 只能存在于外层 archive，
则 backend hand-off 必须携带该 archive；不能因为 `IntentFunction` 能单独发射 MLIR，
就把外层 B 静默丢掉。

## 中立 Lowering 合同

规范目标是一个 verification-neutral 对象，例如：

```text
IntentProgram {
  function          // A 与 op-local typed contract
  boundary          // function/program-level B
  interface         // ABI 与 tensor/scalar binding
  schedule_hint?    // optional C
}
```

该对象不得包含 Gate verdict、source/replay code、provider response、dataset identity
或 paper receipt。现有 `IntentFunction`、Gate1 `SemanticArchive` 与历史 Gate2 adapter
之间的桥接属于迁移状态；新 backend 能力不得继续把完整 B 合同放在 `verify/` 私有对象
中。

## Lowering 纪律

- JSON → Intent MLIR 不做高层算子分解；
- Intent → standard/target dialect 可以做合法 lowering、融合、特化与 schedule 选择，
  但不能重新猜测或改变 A/B；
- unknown correctness-relevant B 必须 fail closed；
- backend conformance 独立于 Gate1/Gate2 verdict，并绑定其实际消费的 program digest；
- generic 与 specialized path 必须消费同一 J；specialized 未完整命中时回退或返回
  unsupported，不能部分消费后伪装成功。
