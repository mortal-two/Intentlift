# JSON ↔ Intent MLIR Dialect

## 链路

```
kernel 源码
  → [LLM 提取]  IntentIR JSON   ← LLM 的产出物,人可读
  → [1:1 转换]  Intent MLIR     ← 我们自己的 dialect,确定性转写
  → [后续]      lower 到标准 MLIR
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
- **Layer B**:挂在算子 / 模块上的约束 attribute(mask、bound、结合性、同步)。
- **Layer C**:挂在算子上的非绑定 schedule hint attribute(tile、向量宽度、unroll 因子);可被后端覆盖。

## 当前不做

- 不在当前阶段设计"lower 到 linalg/vector/llvm"的 pass——那是 [../lowering/index.md](../lowering/index.md) 的后续阶段。
- 当前焦点只有两样:**JSON 规范** 与 **Intent dialect 本身的形态**。
