# 代码物理结构

> **目标:真实、成熟的工程实现,不是 prototype。** dialect 走真 C++ MLIR + TableGen,不走 Python binding / 纯 Python 捷径。

## 已定决策

1. **Intent dialect 实现 = 真 C++ MLIR + TableGen(`.td`)。**
   - 工具链:系统已装 **MLIR 20.1.8**(`/usr/lib/llvm-20/`,含 `mlir-opt` / `mlir-tablegen` via cmake / `clang` / `llvm-config`)。
   - 走标准 out-of-tree MLIR dialect 工程:`find_package(MLIR REQUIRED CONFIG)` + 自有 TableGen dialect + 自有 `intent-opt` 工具。
2. **语言分工**：Intent dialect、结构 verifier、通用与目标特化 lowering = C++ MLIR + TableGen；LLM 提取、JSON 模型、前端发射、验证编排、backend 选择／编译／运行时 = Python。
3. **build**:CMake + Ninja(C++ 侧)+ Python 包管理(Python 侧)。
4. **LLM provider**：凭据与本地 provider 配置必须 gitignore，日志与 artifact 不得泄露 key；
   具体 provider/model 是运行配置，不写死为项目语义。

## 工程目录结构

```
TianchenIR/                   # 项目根(2026-06-26:原 tianchenir/ 内容上移至此,不再嵌套子文件夹)
  CMakeLists.txt              # 顶层:find_package(MLIR), 加子目录
  include/Intent/             # Intent dialect、op 与 conversion pass 声明
    IntentOps.td/.h
    Conversion/Passes.h       # 通用／CUDA lowering pass API
  include/Target/             # target 侧 pass API；未实现 target 不得伪装支持
  lib/Intent/
    IntentOps.cpp             # dialect op verifier／语义约束
    Conversion/               # Intent → linalg / GPU / NVVM 等 pass
  tools/intent-opt/           # 自有 opt 驱动(注册 Intent dialect + 标准 dialect)
  python/intentir/
    ir/                       # IntentFunction 与可移植语义模型
    extraction/               # LLM 提取(prompt、re-roll、evidence、provider 客户端)
    frontend/                 # JSON／IntentFunction → Intent MLIR
    backend/
      common/                 # 跨 target 能力与中立记录
      cuda/                   # CUDA selection / compile / launch / dispatch
      riscv/                  # 能力边界；无实现时不得导出伪 backend
    verify/
      gate1/                  # Lift artifact、RuleRegistry、结构闭合
      gate2/                  # source/replay 动态比较
      back/                   # 历史验证／lowering adapter 归档
    benchmarks/               # 公平、可复现的性能评估入口
  spec/lowering/              # 机器可读 lowering 能力与证据 registry
  corpus/                     # 提取语料、覆盖与报告；不是 spec 真值
  test/                       # lit / FileCheck 与 live harness
  python/tests/               # Python unit、backend correctness 与 campaign wiring
  pyproject.toml
```

- `Dataset/` 在仓库外 `/home/kingdom/phdworks/Dataset`(见 [../extraction/frontend-inputs.md](../extraction/frontend-inputs.md))。
- 参考仓库(只读,借鉴用):`/home/kingdom/phdworks/refs/{intentir,IntentIR,TianchenForge}`(2026-06-26 从 `TianchenIR/` 下移出,工作区只留本项目)。

## 工程纪律(成熟工程,非玩具)

- C++ 侧每个 op 在 TableGen 里声明 verifier、assembly format、类型约束,不留"注册了不可执行"的算子(见 [../ir/atomic-ops.md](../ir/atomic-ops.md))。
- 每步要能 build、能被 `intent-opt` 解析、能过 lit 测试,才算真进展。
- 不为赶进度写 Python 模拟 dialect 充数——dialect 就是 C++ MLIR。
- 新 backend 必须放入 target-specific namespace；共享层不得偷偷绑定 CUDA。
- `verify/` 不拥有 production lowering。若完整 A/B/ABI 合同需要被 backend 消费，
  应迁入 `ir/` 或 `backend/common/` 的中立对象，不能长期依赖 Gate2 adapter。
- selector、dispatch 与 capability registry 不得读取 Gate verdict、provider、Kernel 名、
  source path 或 paper receipt 作为 lowering 授权条件。
- 实验进度与性能数字写入 task/report/corpus；本文件只保存长期目录职责和工程纪律。
