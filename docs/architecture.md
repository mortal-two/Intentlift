# 系统架构

## 1. 系统边界

IntentIR 把“理解一个源 Kernel”和“为目标硬件实现它”拆成两个可核验问题。Lift 读取 Source 与编译器可见 Harness，输出完整的 `(J,C,π)`；Replay 在看不到 Source 的条件下消费 `J` 与遮蔽后的 `C`，产生目标前端程序；Gate1 和 Gate2 分别检查结构闭包与执行语义闭包。

```text
Source + Harness
      │
      ▼
     Lift ───────────────► J + C + π
                            │   │
                 J + MaskC(C)  │ π
                            │   │
                            ▼   ▼
                         Replay Gate1 ◄── Source observer
                            │
             Gate1 closed ─┴────────────► Gate2
                                           ├─ Source–Replay
                                           ├─ Source–J
                                           └─ Replay–J
```

任何单边通过都不足以认证；只有 Gate1 闭合且 Gate2 三边全部通过，Kernel 才进入闭包计数。

## 2. `(J,C,π)` 的职责

| 对象 | 拥有的事实 | 不得承担的职责 |
|---|---|---|
| `J` | 张量签名、shape、访问、谓词、fill、归约、效果、输出语义 | 目标 tile、launch 参数、源文件路径、算法名称驱动的后端选择 |
| `C` | 实现槽位及其具体估值，例如 tile、warp、stage、编译选项 | 修改 `J` 的数学语义，或为缺失语义提供补丁 |
| `π` | Source 与 Replay 观测之间冻结、带类型的对应关系 | 读取后改写观测、放宽 Gate、携带新的语义或目标实现配方 |
| Harness | 具体输入、shape 估值、grid 与初始 `C` | 充当 `J` 候选集合或证明语义正确 |

当前 `π` 不只覆盖程序轴；它也可表达访问、guard、布尔和归约关系。所有新关系仍必须是冻结、可类型检查、可由两侧独立观测验证的对应关系。

## 3. 可见性与所有权

- Lift 可以读取 Source 和编译器视图，并负责产出整个 `(J,C,π)`。
- Replay 只接收规范化 `J`、`MaskC(C)` 和固定工具环境 `U_f`。它不能读取 Source、`π`、未遮蔽的 `C` 值、Gate 差异或以往失败历史。
- Source observer 和 Replay observer 独立产生原始结构观测；observer 不读取 `π`。
- normalizer 依据同一个冻结 `π` 对两侧观测做单边规范化，不能通过联合查看两侧来制造相等。
- TypeCheck 在 Gate 前失败关闭，检查 artifact 的结构与类型一致性；它不证明 Source/J 等价，也不计作闭包。

## 4. Gate1 与 Gate2

Gate1 比较 Source 与 Replay 的规范化编译器观测。正常臂必须闭合；可选的 masked-J 臂只用于诊断 `J` 是否提供必要信息。Gate1 先于任何隐藏 Gate2 输入执行。

Gate2 在冻结 Harness 上运行三个执行腿：

1. 原始 Source；
2. Gate1 已接受的精确 Replay；
3. 从同一 `J` 构造的参考执行 `RefJ`。

比较策略由 workload 的外部 `Q` 策略决定，不写入 `J`。Source–Replay、Source–J、Replay–J 必须全部通过；运行失败、reference unsupported 或任一数值比较失败都不能认证。

## 5. 前端

活动 SCRC 清单支持三类真实 Source：

- Triton：从已安装包或外部 Dataset 解析精确源码切片；
- TileLang：从多前端 Dataset 读取 Source、Harness 与生成 CUDA 所需信息；
- CUDA：对 Dataset 中的原生 CUDA Kernel 使用对应 observer/compiler 路径。

仓库中可能保留其他 observer 实现，但它们不自动等于活动、可运行的闭包清单。公开能力以 `tools/scrc_frontend.py --list --json` 的实际输出为准。

## 6. 后端

生产后端从规范化 `J` 开始确定性物化 Intent MLIR；只有物化完成后才附加 `C` 并进入目标 lowering、候选搜索、编译和执行。候选选择依据 `J` 的结构与目标能力，不依据源算法名称。

- CUDA 后端：Intent MLIR 到 NVIDIA GPU/PTX/cubin 的 lowering、Driver 执行、结构化候选搜索和基准导出。
- RISC-V 后端：Intent MLIR 到 LLVM/RVV 的 lowering、目标对象导出和目标机搜索/计时。
- CPU 路径：标量 MLIR/LLVM 参考执行，主要服务 Gate2 的 `RefJ`；它不是 CUDA/RVV 生产性能的替代实现。

研究消融可以显式擦除 `C` 或限制候选空间，但这属于外部实验臂；后端不得从 Source 名称或历史手工标签恢复被擦除的信息。

## 7. 可移植性边界

Git 跟踪的仓库包含构建系统、Python 包、生产 Source、机器可读规格和公共文档。Dataset、大模型 provider 配置、GPU/RVV 运行证据、内部研究笔记和历史归档均位于仓库之外或 `local/`，通过显式路径注入。任何新机器都应能在没有这些私有文件时构建核心系统并列出静态能力；真实闭包和目标基准则需要相应 Dataset、工具链和硬件。
