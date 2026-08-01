# RVV 外部基线数据（2026-08-02 修订）

当前正式后端集合是 24 个 Triton 来源 J。此前误混入的 TileLang Gather2D 与 RowMask Where2D 已从正式输入、
baseline registry 和结果中删除；原六项 `figure-table.{json,csv}` 因包含这两个无效实验格也一并删除，不能继续用于
作图。通用 gather 与 row-mask compiler pass 不受影响。

现有 RVV4 单线程、固定 affinity、code-warm/data-cold、交替配对结果中，可以继续使用的五项如下。定义
`G_schedule = C-Fixed / Schedule-Only`，`G_full = Schedule-Only / Full`；外部比值大于 1 表示 IntentIR 更快。

| Kernel | Natural shape / dtype | `G_schedule` | `G_full` | LLVM-RVV C / Full | Best Native / Full |
|---|---|---:|---:|---:|---:|
| TVD | two `[4096,4096]` f32 inputs, two f32 outputs | 1.638× | 6.522× | 0.971× | 0.954× Expert |
| Softmax | `[2048,4096]` f32 | 1.581× | 9.852× | 7.599× | 1.617× ggml |
| xFormers RMSNorm | `[4096,12288]` f32 | 4.730× | 4.888× | 1.647× | 1.109× Expert |
| Flash backward preprocess | two `[4,16,8192,128]` f16 inputs, `[4,16,8192]` f32 output | 1.487× | 28.771× | 2.490× | 1.634× Expert |
| Llama4 RoPE | Q `[8,1024,8,128]`, K `[8,1024,4,128]`, f16 | 1.017× | 1.000× | 0.734× | N/A |

五项几何平均为 `G_schedule=1.793×`、`G_full=6.183×`、`LLVM-RVV C / Full=1.859×`。Best Native 只在前
四项适用，其适用子集几何平均为 `1.293×`；不能把它写成 24 项全覆盖结果。

同目录的 Triton-CPU 与 ggml 配对文件没有包含被删除的两个 TileLang 输入，继续有效：

- Triton-CPU/RVV：8 项 correctness，7 项统一 formal latency，GeGLU 另有 natural-shape quick；
- ggml CPU/RVV：4 项统一 formal latency，GeGLU 另有 natural-shape quick。

新的 24-J RVV 作图表必须在其余正式 J 完成相同三柱和外部配对后重新生成；当前不使用删行后的不完整 JSON
冒充全量矩阵。
