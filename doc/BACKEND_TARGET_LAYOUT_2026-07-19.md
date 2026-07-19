# IntentIR 后端目标分层（2026-07-19）

## 结论

后端必须区分目标无关契约、CUDA 实现和未来 RISC-V 实现。Gate、论文工具、
CLI 和其他调用方只能显式选择目标后端，不能让验证 verdict 参与 lowering 选择。

本轮先完成兼容式第一步：

```text
python/intentir/backend/
├── common/              # 目标无关 ABI 与 runtime contract 入口
├── cuda/                # CUDA lowering/selection/dispatch 命名空间
├── riscv/               # 未来 RISC-V 边界；当前不声明实现能力
└── pipeline.py          # 旧 CUDA import 的兼容转发层
```

`intentir.backend.cuda.pipeline` 是新的 CUDA lowering 规范入口；旧的
`intentir.backend.pipeline` 继续导出相同对象，不删除、不改变调用行为。

## 模块归属

目标无关：

- canonical IntentFunction 与 A/B/C 数据；
- logical ABI、runtime predicate/contract；
- variant manifest、artifact provenance；
- target-independent error/result types。

CUDA 专属：

- `intent-opt` 的 CUDA fast-path 选择与 NVVM/PTX/cubin 编译；
- CUDA Driver launch、CUDA tensor ABI、设备执行；
- `cuda.*` fast-path ID 和 SM/warp 约束；
- CUDA variant dispatch。

RISC-V 专属（尚未实现）：

- LLVM RISC-V/RVV target 配置；
- RISC-V ABI、runtime 和执行器；
- RISC-V-specific schedule/feature selection。

## C++ 侧边界

后续按独立提交逐步整理，避免一次大规模移动掩盖 lowering 回归：

```text
lib/Intent/Conversion/Common/   Intent -> Linalg/SCF/Arith/MemRef
lib/Intent/Conversion/GPU/      只依赖 MLIR GPU dialect 的通用变换
lib/Intent/Conversion/CUDA/     warp=32、NVVM、PTX、SM 专属变换
lib/Intent/Conversion/RISCV/    未来 RVV/RISC-V 变换
```

当前 C++ 文件名和 pass 名暂时保留，保证旧 CLI、测试和 Gate reader 可继续使用。
只有在相应 lit/CUDA execution 回归就位后，才移动一个实现族并留下兼容注册。

## 迁移约束

1. 新代码使用 target namespace，例如 `intentir.backend.cuda.pipeline`。
2. 旧 import 作为薄兼容层保留，直到所有外部调用迁移完成。
3. `common` 和 `riscv` 的导入不得初始化 CUDA context。
4. RISC-V 包存在不等于 RISC-V 已支持；没有真实编译/执行证据前不导出伪 lowering。
5. Gate1/Gate2 适配只把 canonical IntentIR/ABC 映射到目标后端；verdict 不进入后端 API。
6. 每次目标目录迁移都必须至少通过 import compatibility、MLIR lowering 和真实设备执行测试。
