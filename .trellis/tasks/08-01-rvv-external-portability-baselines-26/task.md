# Task 26：RVV 可迁移性与成熟外部基线

> 状态：进行中
>
> 目标：为当前 accepted-J 后端建立能进入论文 Figure 8 的 RVV 外部比较，而不是继续把一份手写 C 实现统称为 Native

## 1. 第一性问题

RVV 外部实验必须同时回答两个不同问题：

1. **可迁移性**：一个不使用 IntentIR 的通用编译路径，把同一算子迁到 RVV 后能达到什么性能？
2. **目标竞争力**：当前存在严格匹配的成熟库或专家 RVV 实现时，IntentIR 与其中最快者相比如何？

因此外部 panel 使用：

```text
LLVM-RVV C       全量可执行对照：独立 C 语义实现 + Clang/LLVM RVV 自动向量化
Triton-CPU/RVV   仅在上游 CPU backend 经真实 RISC-V 移植后可执行的子集
Best Native RVV  每项严格匹配且通过相同 Q/D 的 ggml/llama.cpp 或 Expert RVV 中最快者
IntentIR         同一 J 的 Full winner
```

`LLVM-RVV C` 当前由旧 `Native` 源码提供，但必须重新命名：源码按算子人工表达，编译器负责目标迁移；它不是
ggml、不是显式 intrinsic，也不是直接 Triton Source port。`Triton-CPU/RVV` 若尚不能在真实 RVV4 执行，就记为
`unsupported`，不得画虚构 latency。`Best Native RVV` 只在 operator、shape、dtype、layout、所有 observable
outputs、cache policy 与 timed boundary 严格匹配时出现。

## 2. 实施

### A. 多外部基线 runner

- 一个 kernel 可携带多个相互独立的 external baseline；
- 每个 baseline 先在相同输入上检查所有输出，再与 IntentIR Full 做交替配对计时；
- report 保留每条 baseline latency、实现类别与 correctness，并机械选择最快的严格匹配项；
- 保留旧 single-native sidecar 兼容入口，但新图只读取明确命名的 external baseline；
- 不让 external source、kernel 名或性能结果进入 J materializer、transformation admission 或搜索。

### B. Expert RVV

先实现跨算子复用的四类动态 VL primitive：

- contiguous FP32/FP16 packet；
- ordered row reduction 与 normalization；
- paired FP16 RoPE；
- indexed gather。

它们必须显式使用 RVV intrinsic，VLEN-agnostic，tail-safe；汇编中应出现对应 `vle/vse/vfred/vluxei`，不能只
因为包含 `riscv_vector.h` 就称为 intrinsic baseline。

### C. ggml/llama.cpp

使用官方 CPU/RVV backend，只接入严格匹配子集。优先验证 RMSNorm、LayerNorm、Softmax 与可匹配的
RoPE/GEMM；graph/plan 在计时外创建，主数字只计重复 compute。若输出 ABI、epsilon、dtype 或 fused boundary
不一致，则该 cell 记为不适用，不用分解实现冒充严格外部基线。

### D. Triton-CPU/RVV

从上游实验 CPU backend 的最小 vector-add/elementwise 路径开始，处理非 x86 nanokernel、unroll、vector math
library 与 host-target 代码生成阻塞。在真实 RVV4 上成功构建、执行并通过 Q/D 后，才扩到正式 J 中可直接表达的
elementwise/reduction 子集。构建失败本身只说明当前工具链不支持，不能作为性能柱。

## 3. 测量口径

- RVV4，VLEN=128，单线程，固定 affinity；
- code warm、data cold，驱逐在计时外；
- 相同 workload、输入 payload、输出 comparator、warmup、iterations 与 alternating batches；
- 主内部三柱继续是 `C-Fixed / Schedule-Only / Full`，外部基线不改变三柱；
- 图表同时给出全量 portability 对照和 Best Native 适用子集，缺失项明确为 N/A；
- 不计算或维护 artifact/hash/freeze/receipt，不用流程材料替代可执行代码与结果。

## 4. 完成条件

1. runner 能在一次执行中测量多个外部基线并产生 `Best Native RVV`；
2. 当前 accepted-J RVV 矩阵至少有全量 `LLVM-RVV C`，并补齐实际可执行的 ggml/Expert RVV 子集；
3. Triton-CPU/RVV 有真实可执行结果，或有可复现的具体 upstream blocker 与已完成移植边界；
4. 强基线与 IntentIR 使用相同 natural shape/dtype、所有 observable output 与 cold-data protocol；
5. 生成可直接作图的逐 kernel 表与聚合，不隐藏负例或 N/A；
6. 代码、测试、结果和文档提交，工作区干净。

## 5. 当前进展

- [x] runner 支持一个 kernel 的多条 external baseline、逐输出检查、交替配对计时与 Best Native；
- [x] 23 项 LLVM-RVV C、4 项 Expert RVV、5 项 ggml CPU/RVV adapter 已导出；FlashAttention forward 当前 N/A；
- [x] ggml 单线程 graph/plan adapter 与 RVV4 浮点构建已完成；
- [x] plot-ready JSON/CSV exporter 已完成；
- [x] TVD 新增通用 multi-output sibling fusion，RVV4 正式结果为
  `C-Fixed 494062.57 / Schedule-Only 301694.86 / Full 46255.39 us`，Full/Schedule = 6.522×，达到 Expert RVV 的 95.4%；
- [x] Softmax 新增 fused vector + execution-domain checked approximate-exp MLIR path，正式结果为
  `C-Fixed 493062.53 / Schedule-Only 311769.56 / Full 31646.82 us`，Full/Schedule = 9.852×，相对 ggml CPU/RVV 为 1.617×；
- [x] Flash backward preprocess 新增 mixed-precision widening-dot RVV MLIR path，natural shape 正式结果为
  `C-Fixed 1913189.67 / Schedule-Only 1286453.22 / Full 44713.90 us`，Full/Schedule = 28.771×，相对 Expert RVV 为 1.634×；
- [x] xFormers RMSNorm 正式结果 Full/Schedule = 4.888×，相对 Expert RVV 为 1.109×、相对 ggml 为 2.465×；
- [x] 删除误混入正式分母的两个 TileLang 输入；其结果不再进入任何 aggregate，通用 gather/row-mask pass 保留；
- [x] Triton-CPU/RVV 锁定版本 LLVM/MLIR 已在 RVV4 构建；vector-add 与 accepted SwiGLU 原始 Source 均真实生成 RVV shared object 并通过 Q；
- [x] runner 新增 strict prebuilt Triton object + ABI bridge 路径，并增加只重测既有 Full winner 与指定 external 的 paired runner；
- [x] accepted SwiGLU 自然 shape 的 Triton-CPU/RVV 正式结果为 `10.849 s`，配对 Full 为 `5.403 s`，IntentIR 快 `2.008×`；
- [x] Triton-CPU/RVV 横向扩到 Softmax、KLDiv、Flash preprocess；正式 `Triton/Full` 为 `11.647× / 1.369× / 86.736×`，四项几何平均 `7.259×`；
- [ ] 用新 external runner 继续补齐其余 accepted-J 主表并提交。
