# IntentIR 后端目标分层（2026-07-19）

## 结论

后端必须区分目标无关契约、CUDA 实现和未来 RISC-V 实现。Gate、论文工具、
CLI 和其他调用方只能显式选择目标后端，不能让验证 verdict 参与 lowering 选择。

当前实现已经完成无兼容层的 target 分区：

```text
python/intentir/
├── ir/
│   ├── abi.py           # target-neutral logical ABI
│   └── layer_c.py       # target-neutral schedule-hint parsing
├── backend/
│   ├── toolchain/       # host MLIR/LLVM tool discovery
│   ├── cpu/             # CPU pipeline/runtime
│   ├── cuda/            # CUDA lowering/selection/dispatch/runtime
│   └── riscv/           # future target boundary; no claimed implementation
└── benchmarks/
    └── flaggems_shim.py # benchmark-only third-party compatibility
```

`intentir.backend.cuda.pipeline` 是唯一 CUDA lowering 入口。原先平铺在
`backend/` 的 CUDA module、CPU 单文件、benchmark kernel driver 和 re-export
facade 均已迁移或删除；仓内调用方必须显式选择所属 target/module。

## 模块归属

最终中立合同：

- canonical `IntentProgram(A+B+ABI, optional C)`；
- target-independent error/result types 与 backend protocol；
- target-independent capability vocabulary；具体 target 的 capability value 由各
  backend 实现。

当前已经中立化并落在 `ir/`／`backend/toolchain/` 的组件：

- `ir.model` 的 `IntentFunction`；
- `ir.abi` 的 logical ABI；
- `ir.layer_c` 的 Layer-C 查询与冲突检查；
- `backend.toolchain.tools` 的 host MLIR/LLVM 工具发现。

完整 `IntentProgram` 和统一 backend protocol 尚待 A/B 合同迁移时落地；不得用
`backend/common` 空壳或根包 re-export 伪装已经完成。

CUDA 专属：

- `intent-opt` 的 CUDA fast-path 选择与 NVVM/PTX/cubin 编译；
- CUDA Driver launch、CUDA tensor ABI、设备执行；
- `cuda.*` fast-path ID 和 SM/warp 约束；
- CUDA runtime predicate、variant manifest/artifact provenance 与 dispatch。

RISC-V 专属（尚未实现）：

- LLVM RISC-V/RVV target 配置；
- RISC-V ABI、runtime 和执行器；
- RISC-V-specific schedule/feature selection。

## C++ 侧边界

实现与 pass API 按 target 独立构建：

```text
include/Intent/Conversion/Common/  Intent -> Linalg pass API
include/Intent/Conversion/CUDA/    CUDA fast-path pass API
include/Target/CUDA/               NVGPU/NVVM target pass API
lib/Intent/Conversion/             target-neutral Intent -> Linalg
lib/Intent/Conversion/CUDA/        CUDA/NVGPU fast paths
lib/Target/CUDA/                   NVVM target legalization
```

CLI pass argument 保持稳定；旧 C++ umbrella header 与聚合 CMake alias 已删除。
调用方必须显式 include/link Common、CUDA conversion 与 CUDA target 组件。

## 迁移约束

1. 调用方显式使用 target namespace，例如 `intentir.backend.cuda.pipeline`。
2. 不保留 flat import、`sys.modules` alias、批量 re-export 或 C++ umbrella。
3. `ir`、`toolchain`、`cpu` 和 `riscv` 不得导入 CUDA，也不得初始化 CUDA context。
4. RISC-V 包存在不等于 RISC-V 已支持；没有真实编译/执行证据前不导出伪 lowering。
5. Gate1/Gate2 适配只把 canonical IntentIR/ABC 映射到目标后端；verdict 不进入后端 API。
6. 架构测试必须阻止 `backend -> benchmarks`、`cpu -> cuda` 和旧路径复活；
   行为测试覆盖 MLIR lowering 与相应 target 执行。
