# 代码物理结构

> **目标:真实、成熟的工程实现,不是 prototype。** dialect 走真 C++ MLIR + TableGen,不走 Python binding / 纯 Python 捷径。

## 已定决策

1. **Intent dialect 实现 = 真 C++ MLIR + TableGen(`.td`)。**
   - 工具链:系统已装 **MLIR 20.1.8**(`/usr/lib/llvm-20/`,含 `mlir-opt` / `mlir-tablegen` via cmake / `clang` / `llvm-config`)。
   - 走标准 out-of-tree MLIR dialect 工程:`find_package(MLIR REQUIRED CONFIG)` + 自有 TableGen dialect + 自有 `intent-opt` 工具。
2. **语言分工**:Intent dialect / lowering = C++ MLIR + TableGen;LLM 提取 / 解释器 / 验证 = Python。
3. **build**:CMake + Ninja(C++ 侧)+ Python 包管理(Python 侧)。
4. **LLM provider**:`/home/kingdom/phdworks/TianchenIR/tianchenir/llm_providers.local.json`(当前 `gpt-5.3-codex`,responses API;在 gitignore 内)。

## 工程目录结构

```
tianchenir/
  CMakeLists.txt              # 顶层:find_package(MLIR), 加子目录
  include/Intent/             # dialect 头 + TableGen(.td)
    IntentDialect.td/.h
    IntentTypes.td/.h
    IntentOps.td/.h
    CMakeLists.txt            # mlir_tablegen 规则
  lib/Intent/                 # dialect / ops / types 的 C++ 实现
    CMakeLists.txt
  tools/intent-opt/           # 自有 opt 驱动(注册 Intent dialect + 标准 dialect)
    CMakeLists.txt
  python/intentir/            # Python:提取 + 验证
    ir/                       # JSON schema + 三层模型 + JSON↔dialect 桥
    extraction/               # LLM 提取(prompt、re-roll、evidence、provider 客户端)
    verify/                   # 解释器(reference 语义)+ 验证栈 + 契约
  test/                       # lit + FileCheck:*.mlir 往返 / 验证
  pyproject.toml
```

- `Dataset/` 在仓库外 `/home/kingdom/phdworks/Dataset`(见 [../extraction/frontend-inputs.md](../extraction/frontend-inputs.md))。

## 工程纪律(成熟工程,非玩具)

- C++ 侧每个 op 在 TableGen 里声明 verifier、assembly format、类型约束,不留"注册了不可执行"的算子(见 [../ir/atomic-ops.md](../ir/atomic-ops.md))。
- 每步要能 build、能被 `intent-opt` 解析、能过 lit 测试,才算真进展。
- 不为赶进度写 Python 模拟 dialect 充数——dialect 就是 C++ MLIR。
