# IntentIR

IntentIR 是一个把高性能 Kernel 的源程序提升为可验证语义、再从该语义生成 CPU/CUDA/RISC-V 实现的研究系统。仓库只维护一条权威主线：前端提升、SCRC 闭包和生产后端共享同一套 `J/C/π` 合同，不再由多个 worktree 分别定义系统状态。

系统的核心边界是：

```text
Lift(Source) -> (J, C, π)
Replay(J, MaskC(C); U_f) -> R
Gate1(Source, R, π)
Gate2(Source, R, RefJ) = Source–Replay ∧ Source–J ∧ Replay–J
```

- `J` 是与源语言无关的语义归档，描述张量、访问、谓词、归约、效果和输出语义。
- `C` 是实现选择，包括 tile、launch 和编译器调度参数；它不改变 `J`。
- `π` 是 Gate1 使用的冻结、带类型对应关系。它可以表达程序轴、访问、guard、布尔和归约关系，但不能修改 Source 或 Replay 的观测结果。
- Gate1 必须先闭合；Gate2 随后在同一 Harness 估值上执行 Source、Replay 和 `J` 的参考语义，三条比较边必须同时通过。

## 仓库结构

- `include/`、`lib/`：IntentIR 方言、转换与 CPU/CUDA/RISC-V lowering。
- `python/intentir/ir/`：`J` 的规范化、物化和生产快照边界。
- `python/intentir/extraction/`：Source 到 `(J,C,π)` 的 Lift 入口。
- `python/intentir/verify/`：TypeCheck、Gate1、Gate2、真实 workload 与证据生命周期。
- `python/intentir/backend/`：CUDA 与 RISC-V 生产后端，以及 Gate2 使用的 CPU 标量参考路径。
- `tools/`：SCRC、闭包复核、后端导出、搜索和基准入口。
- `corpus/`、`spec/`：生产 Source、目标规格与机器可读清单。
- `docs/`：当前公开架构、构建、流程与唯一最终任务。
- `local/`：被 Git 忽略的内部理解、真实实验结果和历史归档；它不是公共合同的一部分。

## 快速开始

```bash
python -m pip install -e '.[verify]'

cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build build --target intent-opt

PYTHONPATH=python python tools/scrc_batch.py --validate-manifest --json
PYTHONPATH=python python tools/scrc_frontend.py --list --json
```

完整依赖、真实闭包命令和后端工作流见：

- [系统架构](docs/architecture.md)
- [构建与运行](docs/build-and-run.md)
- [工作流](docs/workflows.md)
- [唯一最终任务](docs/final-task.md)

## 验证原则

活动树不包含测试目录，也不以 mock、smoke、协议测试或 TypeCheck 数量替代研究结论。TypeCheck 和构建只负责机械前提；进展只由真实 Kernel 的 Gate1 闭包、Gate2 三条比较边、闭包 Kernel 数量和最终交付物衡量。历史测试、旧文档和 Trellis 状态只保存在本地归档中。
