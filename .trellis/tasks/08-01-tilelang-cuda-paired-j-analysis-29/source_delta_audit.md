# Task29 source-grounded audit of five non-exact semantic cores

> 性质：对 Task28 `results_current.json` 所选 exact accepted run 的只读 Source/J 审计。  
> 作用：解释 Task29 deterministic comparator 找到的五个 non-exact core；不修改 artifact，不改变
> closure denominator，也不把结构相似度当 Gate。

| Stem | Exact Source evidence | J delta | Classification |
|---|---|---|---|
| `any_kernel_dim` | TileLang `run-002/source.py:6-7,20` 使用 bool output；CUDA `run-006/source.py:12-20` 使用 `signed char*` 并显式 cast store | CUDA 多一个 i1→i8 cast；output buffer i1→i8；extra scalar `M` | source-grounded typed output carrier |
| `rowmask_where2d` | TileLang `run-002/source.py:6-20` 使用 bool mask 并 cast 到 f32；CUDA `run-002/source.py:11-15` 使用 `signed char*`，先 cast bool 再 cast f32 | CUDA 多一个 i8→i1 cast；mask buffer i1→i8；extra scalar `M` | source-grounded typed input carrier |
| `masked_softmax2d` | TileLang `run-002/source.py:6-8,27-28` 使用 bool mask；CUDA `run-002/source.py:12-26` 使用 `signed char* mask` 并 materialize bool→f32 | CUDA 多一个 carrier-normalization cast；mask buffer i1→i8；extra scalar `M` | source-grounded typed mask carrier |
| `matmul_fused_epilogue2d` | TileLang `run-002/source.py:9-12,26-40` 使用两个 bool masks；CUDA `run-002/source.py:12-18,89-111` 使用两个 signed-char masks、显式 bool casts，且 A/B 为 TensorMap | CUDA 多两个 carrier casts；两个 i1 buffers→i8；extra `K/M/N` | source-grounded typed masks；TensorMap/TMA 是 physical Source fact |
| `gather2d` | TileLang `run-001/source.py:5-20` 直接 gather；CUDA `run-002/source.py:11-21` 携带 `M`，检查 row/col bounds，越界写 0 | CUDA 增加 bounds constants/comparisons/select、predicated access、zero fill 与 value-dimension relation | genuine Source boundary difference |

五对的两侧 artifact 都独立满足 TypeCheck、frontend-specific Gate1 与三边 Gate2。四个 typed-carrier
case 的共同计算保持，但 exact typed J 必须保留 frontend Source 中的 i1/i8 与显式 cast。`gather2d`
在越界输入上的行为确实不同，不能归一成同一个 boundary J。

因此 full-31 的五个 non-exact semantic core 均已有 Source/ABI 归因；当前没有 unexplained delta。该结论
只适用于本次 exact accepted artifacts，不外推任意 TileLang/CUDA translation。

