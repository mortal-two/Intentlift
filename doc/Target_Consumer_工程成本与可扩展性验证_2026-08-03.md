# Target Consumer 工程成本与可扩展性验证

日期：2026-08-03
范围：当前正式 24 个已接受 Triton J，CUDA 与 RVV production consumer

## 0. 结论

当前代码能够支持下面两个结论：

1. **24 个 Kernel 并不对应 24 套手写后端模板。** CUDA 上它们落入 6 个由 Intent/typed-contract 结构决定的 lowering 类别，只触达 7 个 CUDA conversion translation units；其中 20/24 只复用 `Elementwise` 和 `RowReduction` 两个 pass。
2. **新增一个使用已有 Intent operation、已有合同关系和已有 lowering 结构的 J，通常不需要新增 target C++ pass。** 变化主要是 artifact intake、workload binding 和测量，不是 consumer 计算实现。

但当前代码**不支持**更强的说法：

> consumer 成本只随 target 数增长，与 operation 或执行结构完全无关。

完整 CUDA consumer 目前有 23 个结构特定 conversion passes、2 个 NVIDIA legalization passes，以及完整的搜索与运行时控制层。真实的增量关系更接近：

```text
总工程成本
  = 一次性的 J / Intent / typed-contract 共享基座
  + 每个 target 的 ABI、编译、搜索与运行时成本
  + 每个 target × 新执行结构的 matcher/lowering 成本
```

因此论文中最稳健的表述应当是：

> IntentIR 消除的是 frontend realization × target 的重复实现，并把 target lowering 成本摊销到可复用的 Intent 图结构上；它没有消除 target 对新执行结构的实现成本。当前 24 个 J 显示的是 24→6 的多对一复用，而不是 24→24 的模板复制。

这比“成本严格只按 target 增长”更准确，也足以回答 reviewer 提出的“只是把 source×target 工作转移进复杂 consumer”这一反驳。

## 1. 要验证的四个问题

本审计回答四个可量化问题：

1. CUDA、RVV consumer 分别有多少 production LoC？
2. 有多少通用 operation rewrite、结构 matcher 和 target legalization？
3. 当前 24 个 J 是否逐 Kernel 对应手写实现？
4. 新增 J、新增 Intent operation、新增高性能执行结构分别需要多少工程改动？

这里不使用模糊的 “implementation family”。本文只区分：

- **operation-level lowering**：例如 `intent.reduce -> linalg.reduce`；
- **结构 matcher/lowering**：例如完整 row-reduction 图、padded short-dot 图、Split-K merge 图；
- **schedule**：tile、threads、vector width、pipeline 等不改变主要计算图结构的参数；
- **target legalization**：例如 NVGPU/NVVM 兼容与 native math lowering。

## 2. LoC 统计边界

### 2.1 计数方法

表中同时报告：

- `physical LoC`：受 Git 跟踪文件的换行数，等价于 `wc -l`；
- `nonblank LoC`：去除空行后的物理行数，但保留注释。

之所以不报告经过启发式注释剥离的 “logical LoC”，是因为 Python、C++、TableGen 的注释语法不同，容易引入不可复核的计数偏差。physical/nonblank 两列可由标准工具直接重算。

### 2.2 纳入 production consumer 的内容

- J 到 Intent MLIR 的确定性 materialization、typed contract projection；
- Intent dialect 定义与 verifier；
- target-neutral Intent→Linalg/loop/vector passes；
- CUDA 的 C++/MLIR lowering、NVVM legalization、搜索、编译、launch、dispatch；
- RVV 的合同检查、搜索和真实 MLIR/LLVM 编译控制。

### 2.3 明确排除的内容

- `corpus/` 中的 J JSON；
- workload shape、输入生成与 Source baseline；
- benchmark、报表、bundle 导出和论文画图代码；
- GGML、LLVM-native、Triton-CPU 等外部 baseline 实现；
- 测试、文档、CMake 和 build 生成文件；
- 只被 Lift、TypeCheck、Gate1/Gate2/RefEval 使用而未进入 production lowering 的验证模块；
- 已退出 production 路径的 Python kernel emitter。

这些排除项不应被伪装成 consumer 成本，同时也不能拿它们来缩小真实 lowering 成本。排除清单的当前规模为：

| 排除项 | 文件数 | physical LoC | 原因 |
|---|---:|---:|---|
| CUDA workload/benchmark/capability 辅助层 | 4 | 1,362 | 实验与清单，不负责计算 lowering |
| CUDA 旧 Python emitter/旧 graph helper | 5 | 2,253 | production imports 不依赖该集合；architecture test 额外禁止旧 emitter 回流 |
| RVV external baseline 实现 | 3 | 1,509 | 对照实现，不属于 IntentIR consumer |

## 3. 当前 consumer inventory

### 3.1 一次性共享基座

| 共享部分 | 文件数 | physical LoC | nonblank LoC | 内容 |
|---|---:|---:|---:|---|
| Intent dialect | 6 | 1,412 | 1,310 | 28 个 Intent ops、verifier、dialect registration |
| target-neutral C++/MLIR conversion | 14 | 5,536 | 5,117 | 12 个 common passes |
| J materialization/contract/search protocol Python | 16 | 10,476 | 9,526 | J→Intent MLIR、typed B projection、C attachment、通用搜索闭环 |
| **共享基座合计** | **36** | **17,424** | **15,953** | 每个 target 不重复计算 |

共享 Python 数字是保守的 production 口径：它包含完整的 typed boundary、ContractExpr、canonical J elaboration 与 paired-rotation view，而不只统计 `materialize.py` 的入口代码；但不计只被 SCRC/RefEval 使用的旧 `functional_boundary.py`、dimension checker 和 schema helper。production materializer 使用的是 typed ContractExpr projection，并未把旧 matcher 当作 membership gate。

### 3.2 CUDA 增量

| CUDA target-local 部分 | 文件数 | physical LoC | nonblank LoC |
|---|---:|---:|---:|
| CUDA Intent conversion + NVVM target C++/TableGen | 32 | 27,577 | 25,995 |
| CUDA production Python controller/runtime | 13 | 10,876 | 10,057 |
| **CUDA target-local 合计** | **45** | **38,453** | **36,052** |

CUDA Python 并不发射 Python kernel。其 10,876 行可进一步分为：

| Python 职责 | physical LoC | 主要文件 |
|---|---:|---|
| J admission、pass probe、schedule/state search | 6,469 | `program.py`、`selection.py`、`schedules.py`、`search.py` |
| MLIR→cubin 编译与真实 launch/orchestration | 2,934 | `compile.py`、`pipeline.py`、`launch.py`、`launch_config.py`、`orchestrate.py` |
| Q/D、runtime contract、offline variant dispatch | 1,473 | `qd.py`、`runtime_contract.py`、`variant_manifest.py`、`dispatch.py` |

真实计算实现位于 C++/MLIR passes；Python 负责候选编排、编译调用、ABI 绑定与测量。

CUDA C++ inventory 包含：

- 23 个 Intent→CUDA structure passes；
- 2 个 CUDA/NVVM target legalization passes；
- 共用的 map emitter、schedule 与 MMA fragment helpers。

23 个 structure passes 覆盖的能力比当前 24-J cohort 更宽：

| 结构范围 | pass translation units 数 | 内容 |
|---|---:|---|
| reduction/update | 6 | row、padded-row、arg-reduce、3 种 Split-K reduce |
| element/address/layout | 9 | elementwise、gather、layout、compact、RNG、sort、pool、upsample、polar |
| dense/conv/attention | 8 | 2 种 conv、3 种 contraction、loop-contraction、Flash MMA、attention backward |
| **总计** | **23** | 当前库的完整能力，不等于 24 个 Kernel 模板 |

### 3.3 RVV 增量

| RVV 口径 | physical LoC | nonblank LoC | 解释 |
|---|---:|---:|---|
| 严格 target-local Python | 2,524 | 2,406 | `compile.py`、`contract.py`、`search.py` |
| RVV-specific C++ pass | 0 | 0 | 当前没有独立 RVV C++ rewrite 目录 |
| 加上实际执行所用的 common C++ passes | 8,060 | 7,523 | 2,524 + 5,536；这是更保守的“可执行 RVV stack”口径 |

RVV consumer 复用全部 12 个 common MLIR passes，包括：

- `ConvertIntentToLinalg`；
- iterator-kind tiling；
- row/mixed-precision reduction vectorization；
- softmax 与 sibling-output fusion；
- paired rotation、row mask、gather chain lowering；
- buffer-result forwarding 与 disjoint-copy lowering。

当前正式集合在 RVV 上是 **23/24 可消费**。FlashAttention forward 因 logical-update contract 尚未被 RVV consumer 支持而 fail closed。因此不能把 RVV 写成 24/24，也不能把 fail-closed 当作免费支持。

## 4. 24 个 J 是否对应手写模板

### 4.1 Intent operation 覆盖

Intent dialect 当前定义 28 个 ops。正式 24 个 J 的 materialized MLIR 实际只使用 15 个计算 ops：

```text
cast, concat, contraction, gather, iota,
loop, map_binary, map_unary, pad, reduce,
scan, scatter, select, transpose, yield
```

`ConvertIntentToLinalg.cpp` 中有 14 个通用 `OpRewritePattern`；`intent.yield` 作为 region terminator 由 `LoopLowering` 一并处理。因此这批 J 没有要求“每个 Kernel 新增一个 Intent op”。

### 4.2 CUDA 的 24→6 多对一映射

| 结构 matcher/lowering | Kernel 数 | 当前 24 中的 Kernel | CUDA implementation LoC |
|---|---:|---|---:|
| complete elementwise / rotary graph | 7 | SwiGLU、GeGLU、DYT、TVD、Liger RoPE、Qwen2VL MRoPE、Llama4 RoPE | 2,363 |
| row normalization/loss/scan graph | 13 | RMSNorm、Fused Add RMSNorm、LayerNorm、Softmax、Sparsemax、KLDiv、PolyNorm、JSD、xFormers RMSNorm、xFormers RMSNorm-add、CrossEntropy、GroupNorm、Flash one-pass LayerNorm | 6,552 |
| padded short row-dot graph | 1 | Flash backward preprocess | 1,129 |
| direct/tensor-core contraction | 1 | Flash Linear | 2,051（两个 lowering TUs） |
| Split-K weighted merge | 1 | SplitK reduce varargs | 1,220 |
| structured attention scan/MMA | 1 | FlashAttention forward | 2,705 |
| **合计** | **24** | **6 个结构类别、7 个 CUDA TUs** | **16,020** |

7 个 active implementation TUs 是：

```text
lib/Intent/Conversion/CUDA/ConvertIntentElementwiseToGPU.cpp
lib/Intent/Conversion/CUDA/ConvertIntentRowReductionToGPU.cpp
lib/Intent/Conversion/CUDA/ConvertIntentPaddedRowReductionToGPU.cpp
lib/Intent/Conversion/CUDA/ConvertIntentContractionToGPU.cpp
lib/Intent/Conversion/CUDA/ConvertIntentContractionToGPUMMA.cpp
lib/Intent/Conversion/CUDA/ConvertIntentSplitKReduceVarargsToGPU.cpp
lib/Intent/Conversion/CUDA/ConvertIntentFlashMMAToGPU.cpp
```

另有 482 行共用 CUDA headers，故当前 24 实际触达的 implementation TUs/headers 上界为 **16,502 physical LoC**。这是 translation-unit 上界：大型 `RowReduction` 文件还包含并非每个 row Kernel 都会选择的 topology 和 numeric states，因此不能把 16,502 除以 24 当作“每 Kernel 688 行手写代码”。

最关键的复用事实是：

```text
7 个 elementwise/rotary J  -> 1 个 Elementwise pass
13 个 reduction/normalization/loss J -> 1 个 RowReduction pass
```

即 **20/24（83.3%）只复用两个 structure passes**。Sparsemax 的 sorted-prefix、normalization 的多统计量、loss 的 epilogue 等是同一个结构 matcher 内由 IR 图、shape、dtype、B facts 与 pass options 决定的分支，不是按 Kernel 名查表。

### 4.3 为什么这不是按 Kernel 名发射

CUDA selection API 只接收：

```text
Intent MLIR + target capability + pass/schedule override
```

它不接收 provider verdict、dataset tag、frontend source path 或 source Kernel 名。每个注册 pass 都必须真实运行 C++ probe；只有当 pass 消费完整 graph、无残留 `intent.*` 且产生 launch graph 时才被准入。

静态审计也没有发现 `fn.getName()`/`getSymName()` 与 Liger、xFormers、Flash、Qwen 或 Llama 名称进行比较。C++ 中的 `fn.getName()` 用于保留/重命名 MLIR public symbol；出现这些项目名称的地方是注释、描述字符串或历史 fast-path ID，不参与 matcher 判定。

这项证据证明的是 **source-name independence**，不是“完全没有结构特定代码”。RowReduction、Split-K 和 Flash MMA 仍然是实质性的 target implementation。

## 5. 新增能力到底要改多少

必须把三种情况分开，否则“新增一个 Kernel 的成本”没有定义。

### 5.1 新 J，已有 operations、contracts 和结构

预期 production consumer 变化：

```text
Intent dialect:        0
common op lowering:    0
target C++ pass TU:    0
target matcher:        0 或少量边界泛化
workload/measurement:  需要，但不计入 consumer LoC
```

当前最直接的证据是 13 个不同的 normalization/loss/scan J 共用一个 RowReduction TU，以及 7 个 elementwise/rotary J 共用一个 Elementwise TU。

### 5.2 真正新增一个 Intent operation

在当前架构中，最小 production 改动通常是：

1. 在 `IntentOps.td` 增加 closed op 定义；
2. 必要时在 `IntentOps.cpp` 增加 verifier；
3. 在 `ConvertIntentToLinalg.cpp` 增加一个 target-neutral rewrite；
4. 更新 emitter/materializer 的 typed mapping；
5. 只有通用 lowering 性能不足时，才修改某个 target 的结构 pass。

历史上三个简单 operation 首次端到端接入的 production diff 为：

| 新 operation | production 文件数 | additions | deletions | 是否新增 CUDA pass TU |
|---|---:|---:|---:|---|
| `intent.select` | 6 | 173 | 28 | 否 |
| `intent.iota` | 5 | 114 | 12 | 否 |
| `intent.scan` | 4 | 305 | 1 | 否 |
| **简单 op 中位数** | **5** | **173** | **12** | **否** |

这是包含首批 consumer 支持的端到端历史改动，不是只计算 TableGen 声明，所以可看成较保守的实际成本。更复杂的 `gather + scatter` 两-op 接入为 6 个 production 文件、476 additions/1 deletion，并新增了一个 277 行的 CUDA pass；这说明“新 op”是否需要 target-specific implementation 是工程成本的真正分界。

### 5.3 新的高性能执行结构

如果 J 虽然只使用已有 ops，但性能需要一种当前不存在的 topology，例如：

- full attention scan + MMA；
- ordered paired rotation；
- 新的 multi-stage update；
- 新硬件上的 async-copy / warpgroup 路径；

那么成本来自新的结构 matcher、lowering、schedule domain 和 target legality，而不是来自 Kernel 名或 frontend。

当前分阶段 intake 的 target-local production diff 可以量化这一点：

| intake 阶段 | 新纳入 J | CUDA C++ | CUDA Python | RVV core Python | 新 C++ TU | 解释 |
|---|---:|---:|---:|---:|---:|---|
| 首批 typed-contract intake | 14 | +0/-0 | +16/-11 | +1/-1 | 0 | 主要建设共享 typed-contract 基座，target pass 未按 14 个 J 复制 |
| 第二批结构扩展 | 8 | +482/-6 | +70/-27 | +0/-0 | 0 | 扩展 3 个已有 CUDA TUs，没有新增 8 个模板 |
| Llama4 RoPE + Flash forward | 2 | +881/-117 | +148/-7 | +186/-14 | 1 个 common TU | paired rotation 与完整 attention 确实引入新结构成本 |

最后一行还增加了 627/-2 行 common C++，其中新的 `LowerPairedRotationToLoops.cpp` 是 536 行。这是重要的反例约束：复杂新结构并不免费，但它可以被后续所有满足相同 Intent/B 前提的 Kernel 和 frontend artifact 复用。

## 6. 对 N×M 论证的正确解释

### 6.1 当前已经证明的内容

- target pass 不读取 frontend source；
- CUDA 同一结构 matcher 消费多个不同 J；
- RVV 从相同 Intent graph 进入 common Linalg/vector pipeline；
- 新增 J 不必新增同名 target implementation；
- C 只提供 schedule/search prior，不是 Kernel recipe；
- 24 个 J 的 CUDA 映射为 6 类结构，而不是 24 套模板。

### 6.2 当前尚未证明的内容

- 任意新 Intent op 都能零成本跨 target；
- consumer LoC 只与 target 数有关；
- FlashAttention forward 已在 RVV 完整消费；
- LoC 可以直接替代开发人时或维护成本；
- 6 个结构类别已经覆盖未来所有 workload。

### 6.3 推荐论文表述

不建议写：

> Adding a target requires one backend independent of operation coverage.

建议写：

> IntentIR factors frontend-specific realizations out of target lowering. A target consumer implements reusable operation and graph-structure lowerings once; new kernels expressed by existing structures require no kernel-named backend rule, while genuinely new execution structures still incur target-specific implementation cost.

对应中文含义：

> IntentIR 把 frontend-specific realization 从 target lowering 中分离。每个 target 为可复用的 operation 与图结构实现 lowering；使用已有结构的新 Kernel 不需要同名后端规则，而真正新的执行结构仍会产生 target-specific 工程成本。

## 7. 建议放入论文的两张表

### 7.1 Consumer inventory 表

主文或附录至少报告：

| Consumer | shared/common LoC | target-local C++ LoC | target-local Python LoC | registered structure passes | 24-J active passes | supported formal J |
|---|---:|---:|---:|---:|---:|---:|
| CUDA | 17,424 | 27,577 | 10,876 | 23 + 2 legalization | 7 CUDA TUs / 6 structures | 24/24 |
| RVV | 17,424 | 0 | 2,524 | 12 common passes | 12 common passes | 23/24 |

表注必须说明 shared/common 只计一次，不能把 17,424 重复加到每个 target 后再声称总成本。

### 7.2 Kernel-to-structure amortization 表

建议使用第 4.2 节的 24→6 映射。它比单独报告总代码量更直接地回答：

> 这些 Kernel 是否对应若干手写模板？

答案不是简单的“否”：

- 不是 24 个 kernel-named templates；
- 是 6 类结构 lowering；
- 其中 Flash、Split-K、row reduction 等确实包含手写 target implementation；
- 这些实现按 IR/B/target 结构匹配并被多个 J 复用。

## 8. 可复核验证

### 8.1 当前回归结果

在本文档对应工作树上：

```text
CUDA/RVV/backend Python tests: 402 passed
MLIR lit tests:               156 passed
MLIR RUN invocations:         228
24-J expected CUDA structure probes: 24 passed
```

24-J probe 逐项对 materialized J 运行指定的 C++ whole-graph pass，而不是只检查字符串：13 个 row、7 个 elementwise、1 个 padded-row、1 个 direct contraction、1 个 Split-K、1 个 Flash structured-scan 均完整消费 graph。额外边界检查显示：Flash Linear 的 direct 与 MMA pass 均准入、MMAWS 对当前图 fail closed；Flash forward 的 structured-scan 与 full-axis MMA pass 均准入、`LoopContraction` 对当前图拒绝。这正是第 4.2 节使用 7 个 active CUDA TUs、而不是把所有注册 pass 都算作“24-J active”的依据。

对应命令：

```bash
PYTHONPATH=python pytest -q \
  python/tests/test_backend*.py \
  python/tests/test_new_closed_rope_flash_backend.py

lit -sv build/test
```

### 8.2 LoC 重算

基础计数可用：

```bash
git ls-files 'lib/Intent/Conversion/CUDA/*.cpp' \
  'lib/Intent/Conversion/CUDA/*.h' \
  'include/Intent/Conversion/CUDA/*.td' \
  'include/Intent/Conversion/CUDA/*.h' \
  'lib/Target/CUDA/*.cpp' \
  'include/Target/CUDA/*.td' \
  'include/Target/CUDA/*.h' | xargs wc -l
```

共享 Python production manifest 是以下 16 个文件，而不是整个 `python/intentir/ir`：

```text
python/intentir/backend/{ablation,contract_capabilities,materialize,protocol,search_and_close}.py
python/intentir/frontend/{canonical_j,emit_mlir}.py
python/intentir/ir/{canonical_j,contract_ir,contract_materialization,
  contract_semantics,dense_predicate_materialization,model,paired_rotation,
  typed_boundary_materialization,value_dimension}.py
```

共享 C++/TableGen manifest 为：

```text
include/Intent/{IntentDialect,IntentOps}.{h,td}
lib/Intent/{IntentDialect,IntentOps}.cpp
include/Intent/Conversion/Common/{Passes.h,Passes.td}
lib/Intent/Conversion/*.cpp
```

这样可以明确排除验证专用的 `functional_boundary.py`、`dim_expr.py`、`dimension_relation.py` 与 `intent_op_contract.py`，也不会把旧 matcher 误写成 production membership gate。

RVV target-local Python：

```bash
wc -l python/intentir/backend/riscv/{compile,contract,search}.py
```

Pass 与 op 数量：

```bash
grep -c '^def ' include/Intent/Conversion/Common/Passes.td
grep -c '^def ' include/Intent/Conversion/CUDA/Passes.td
grep -c '^def ' include/Target/CUDA/Passes.td
grep -c '^def Intent_.*Op' include/Intent/IntentOps.td
```

当前结果依次为 `12 / 23 / 2 / 28`。

### 8.3 还应补成自动测试的 reviewer-oriented 检查

以下两项最适合补成论文 artifact test，而不是继续增加文字说明：

1. **symbol-renaming invariance**：把 24 个 materialized module 的函数名统一改成随机无语义符号，确认 admitted pass、candidate space 和结果不变；
2. **leave-one-structure-in reuse**：对 row 与 elementwise 各增加一个只改变 DAG 组合和函数名的合成 J，确认不修改 target code 即可被现有 matcher 消费。

这两项会把“源码名无关”和“新增 J 的零 target-code 增量”从静态审计提升为可执行证据。

## 9. 最终裁决

当前后端工程量并不小，尤其 CUDA 的 row reduction 与 Flash lowering。隐藏这些数字会削弱论文可信度。真正有价值的结果不是“consumer 很小”，而是：

```text
24 个已接受 J
  -> 15 个现有 Intent 计算 ops
  -> CUDA 6 类结构 / 7 个 active conversion TUs
  -> RVV 12 个 common MLIR passes（23/24）
```

因此可以有证据地声称：

> 工作没有按 frontend×target×Kernel 复制；它集中在共享语义基座和 target×新执行结构上，并被多个 J 摊销。

这就是 consumer inventory/LoC 表应当为 N×M 叙事提供的证据边界。
