# RVV 外部基线作图数据（2026-08-01）

这是当前完成正式 code-warm/data-cold、单线程、固定 affinity、7 批交替测量的六项 RVV4 子集。RVV4 为
RV64GCV、VLEN=128，当前浮点工具链为 Clang/LLVM 17。CSV 一行对应一根可画柱；`Best Native RVV` 是从同项
已测 Expert/ggml 中机械选择的最小 latency，并记录被选实现，不产生新的测量值。JSON 同时保留内部三柱、
外部实现覆盖和 paired Full 数字。不存在或未执行的实现不伪造 latency。

| Kernel | Natural shape / dtype | G_schedule | G_full | LLVM-RVV C / Full | Best Native / Full |
|---|---|---:|---:|---:|---:|
| Gather2D | data `[64,64]` f32, indices `[256]` i32 | 1.103× | 1.608× | 1.387× | 0.972× Expert |
| RowMask Where2D | `[65536,64]` f32 + mask `[65536]` i1 | 1.009× | 1.000× | 0.983× | 0.982× Expert |
| TVD | two `[4096,4096]` f32 inputs, two f32 outputs | 1.638× | 6.522× | 0.971× | 0.954× Expert |
| Softmax | `[2048,4096]` f32 | 1.581× | 9.852× | 7.599× | 1.617× ggml |
| xFormers RMSNorm | `[4096,12288]` f32 | 4.730× | 4.888× | 1.647× | 1.109× Expert |
| Flash backward preprocess | two `[4,16,8192,128]` f16 inputs, `[4,16,8192]` f32 output | 1.487× | 28.771× | 2.490× | 1.634× Expert |

定义：`G_schedule = C-Fixed / Schedule-Only`，`G_full = Schedule-Only / Full`；外部比值来自与 Full
交替配对后的 `external / Full`，大于 1 表示 IntentIR 更快。六项几何平均分别为 `G_schedule=1.651×`、
`G_full=4.940×`、`LLVM-RVV C / Full=1.859×`、`Best Native / Full=1.178×`。

同目录的 `triton-cpu-rvv-portability.json` 是迁移覆盖结果；`triton-cpu-rvv-pairs.{json,csv}` 是七项可直接
作图的正式配对数据。Triton-CPU/RVV 当前 correctness 覆盖 8/24，formal latency 覆盖 7/24：SwiGLU、
Softmax、KLDiv、Flash backward preprocess、DYT、Liger RMSNorm 与 LayerNorm；七项比值分别为
`2.008× / 11.647× / 1.369× / 86.736× / 1.242× / 4.907× / 1.740×`，几何平均 `4.349×`。
GeGLU 是第八个通过同源 Q 的算子，natural-shape 1×1 数据单列于 `triton-cpu-rvv-geglu-quick.{json,csv}`，
不进入 formal 聚合。

`ggml-rvv-pairs.{json,csv}` 保存四项同协议 formal 数据：SwiGLU、DYT、Softmax、xFormers RMSNorm，
`ggml / Full` 分别为 `0.236× / 1.050× / 1.617× / 2.610×`，几何平均 `1.011×`。GeGLU 的精确公共
primitive graph 已通过 Q，natural-shape 1×1 数据单列于 `ggml-rvv-geglu-quick.{json,csv}`。所有比值均为
`external / Full`；大于 1 表示 IntentIR 更快，小于 1 表示外部实现更快。
