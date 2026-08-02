# Lowering 层

> **状态:GPU 后端已打通(2026-06-25 验证),RVV 后端仍为后续阶段。**
> 原本"本层在 IR / 前端稳定后再启动"的计划已部分前移:为了证明 IntentIR 不是
> 纸面 IR,本会话把 GPU 这条 lowering 链路一路打到 RTX 5090 D 上**真实执行**
> (见下"GPU 后端已验证")。设计原则不变——lowering 只做硬件下降,**不做语义
> 分解**;但"现在不做任何 pass"已不再成立。

## 链路定位

```
Intent MLIR dialect
  → lower 到标准 MLIR(linalg / vector / scf / gpu / llvm)
  → RVV / GPU 后端代码
```

lower 到标准 dialect 是把"原子 IR 往硬件下降",**不做语义分解**(分解是 LLM 提取阶段的事,见 [../ir/json-and-dialect.md](../ir/json-and-dialect.md))。

## 启动本层时要遵循的原则(先记下来)

- **A/B 绑定,C 可重调**:后端把 Layer A(计算)和 Layer B(约束)当绑定,Layer C 仅作 schedule 指引。
- **freeze vs retune**:freeze-tiling 直接复用源 schedule;retune 在保持 A/B 不变下为目标搜索 schedule 参数,把 Layer C 当先验而非承诺。论文在 RVV 上量化了 retune 的收益。
- **契约引导特化**:契约成立时,后端可生成 in-contract 快路径(特化维度、去边界处理、向量化),并保留契约外的保守 fallback。
- **common lower 基线**：map/reduce/contraction → `linalg`；iota、数据移动、
  gather 与受支持 scatter → `tensor`/`linalg`；select → `arith`；结构化
  semantic loop → `scf`。向量化和设备映射属于各 target 尾部。

## GPU 后端已验证(2026-06-25,RTX 5090 D / sm_120a / 系统 MLIR 20.1.8)

预测的 lower 目标已被硬件证实:`map.unary` / `map.binary` / `reduce` /
`contraction` → `linalg.generic`;结构化 `loop` / scan 递推 → `scf.for`+`iter_args`;
并行轴 → `scf.parallel` → `gpu` grid。完整脊柱:
`intent → linalg/tensor/arith → scf.parallel → gpu → nvvm → PTX → ptxas → cubin
→ CUDA Driver-API launch`。**朴素(unfused)attention 与 fused FlashAttention
(online-softmax,单 kernel)都在 5090 D 上真实执行**。历史 bring-up 曾与现已
归档的 NumPy 解释器对照到 rel ~1e-7。2026-07-20 起，正式
[SCRC Gate2](../verification/scrc-gate2.md) 分别比较 source↔final-J replay 与
source↔backend-neutral scalar MLIR reference；CUDA/ROCm/RISC-V 等 production target lowering 正确性由独立
backend conformance／regression／benchmark 维护，不进入 Gate2 verdict。

CUDA 实现只存在于 `python/intentir/backend/cuda/`；旧的 flat module 已删除，
不得以 facade、module alias 或根包 re-export 的形式恢复。
C++ pass 分为：

- `MLIRIntentConversionCommon`：`lib/Intent/Conversion/ConvertIntentToLinalg.cpp`；
- `MLIRIntentConversionCUDA`：`lib/Intent/Conversion/CUDA/`；
- `MLIRIntentTargetCUDA`：`lib/Target/CUDA/`。

Pass TableGen 与注册 API 同样位于 `include/Intent/Conversion/{Common,CUDA}/`
和 `include/Target/CUDA/`；不保留兼容 umbrella。**血泪规则**
(后续 RVV /其它后端起步前必读):

- **bare-ptr 调用约定要求静态 shape 的 memref**——kernel 边界用运行时 scalar
  参数驱动尺寸,绝不在边界上留 `memref<?x...>`。
- **`exp` 不走 libdevice**:`math.exp` → `__nv_expf` 外部符号是死路(ptxas
  未解析,或集成 serializer 把 kernel DCE 成空 cubin)。C++ pass
  `LowerGpuMathToNVVM` 把标量 f32 `math.exp` 改写成
  `arith.mulf %x, log2e` + `llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"`
  → 原生 `ex2.approx.ftz.f32`,vs torch.exp 相对误差 1.7e-7。
- **设备阶段用 `cubin-format=isa`(出 PTX)+ 自己跑 ptxas**,不要
  `cubin-format=bin`:集成 `bin` serializer 会静默丢掉任何调用 LLVM intrinsic
  的 kernel(空 cubin,driver rc 200)。
- **fused `loop` 的已验证下降形态**:query-OUTER `scf.parallel`(→ GPU grid)
  套 inner `scf.for` + **per-query 标量 carry**(累加器用 `memref.alloca[D]`),
  一个 fused kernel,一线程一 query。**不要**用 trip-OUTER 的 tensor-iter_args
  形态:朴素 `tensor.empty` body 过不了 one-shot-bufferize;destination-passing
  形态能 bufferize,但其 GPU 映射(query 并行在串行 trip 循环内)不是已证的
  单-kernel 路径。`python/intentir/backend/cuda/flash.py` 把这个已验证形态按
  (Sq,Sk,D,BK) 参数化，是 C++ `intent.loop` lowering 应当镜像的 codegen 规格。
- **retune / freeze 已落地为机制**:
  `python/intentir/backend/cuda/flash.py::retune_sweep` 在保持
  A/B 不变下扫 BK∈{1,2,4,8,16},所有候选 bit-一致(最大偏差 1.8e-7),证明
  Layer C 的非绑定-schedule 性质。**perf 未调优、未声明**(一线程/query +
  每调用 host 同步),retune 这里是机制不是性能数字。

## 前端 `intent.loop` 已打通(M8,2026-06-25)

`JSON → intent dialect → 执行` 的循环这条已经接上:
- C++ `intent.loop` + `intent.yield`(TableGen,verifier 生效):variadic scan
  输入 + carry init + body region。flash JSON 经发射器 round-trip 成该 op。
- 发射器 `python/intentir/frontend/emit_mlir.py`:按 op-list 通用遍历(einsum /
  broadcast / reduce-keepdims 形状推导,与解释器一致),从同一份 JSON 发两件
  工件 —— 完整 `intent.loop` 模块 + 可执行的 loop body 函数(标量 `scale` 作
  rank-0 `tensor<f32>`)。
- lowering `intent.loop` → `scf.for`(trip)+ tensor `iter_args` + scan→
  `tensor.extract_slice` + body→linalg(`LoopLowering`,见 `test/lower-loop.mlir`)。
- **执行 = host 驱动 trip 循环**:发射出的 body 每个 K-tile launch 一次
  (`python/intentir/backend/cuda/orchestrate.py::run_program` 已支持多输出返回
  3 个 carry),carry 放在常驻 CUDA buffer 里前递,末尾 `O=Oacc/l`。这就是
  NON-fused tiled flash 的真实跑
  法 —— kernel 在 5090D 上跑,host 控制循环。与 `interpret(JSON)` 一致到 1.3e-7。

**残项(命名的优化,不是前/后端缺口):** 直接从 `intent.loop` 自动发射*fused*
单 kernel(即已证的 `test/flash_fused.mlir` schedule);今天 fused kernel 由
`python/intentir/backend/cuda/flash.py` 发射,尚未由 loop lowering 产出。需要
interchange + carry privatization(把 query/batch 并行维提到 trip 循环外、carry
按 query 私有化)。

## 多后端余量已验证(2026-06-26):同一 IntentIR 也在 CPU 上执行

`intent → linalg` 这一段是**后端中立**的;只有 device 尾巴(此处 GPU 的
`gpu/nvvm/ptxas`)是特化的。为证明这不是嘴上说说，
`python/intentir/backend/cpu/pipeline.py`
把**同一份提取出来的 intent dialect** 走 `intent → linalg →`(one-shot-bufferize
→ buffer deallocation → linalg-to-loops → … → LLVM)`→ shared object`，再通过
ranked-memref C ABI **在隔离 CPU worker 中真实执行**。活动 module 不含 runtime 输入
payload，同一 dtype/shape specialization 可对不同输入复用 content-addressed 编译结果；
一个 public entry 一次调用写回全部逻辑输出。历史测试已覆盖 add / silu / softmax /
rms_norm / gelu_tanh / matmul；当前 Gate2 只通过
`ScalarLoweringSpec` 复用这条链的 backend-neutral scalar/reference 子集，并与 source
做 Gate2 的第二份差分。该用途不授予 CPU production conformance，也不允许 target selection
或 GPU fallback。CPU 作为 production backend 的完整正确性仍由独立回归维护。
`python/tests/test_cpu_backend.py`(确定性,无 GPU)。

旧 `dense<runtime-input>` wrapper、逐输出重复编译、`mlir-runner` 文本打印／解析路径已从
活动实现删除，不保留 compatibility fallback。它们只在下方历史记录中解释曾经触发的
修复，不能作为当前 Gate2 evidence。

这就是"多后端余量":换后端 = 换 `linalg → <target>` 尾巴(CPU 已通,RVV 是下一个
——`intent → linalg → vector`(RVV 向量化)`→ llvm`,A/B 绑定、C 重调;论文在 RVV
上量化 retune 收益)。dialect 设计不被任何单一后端绑架。

### 历史三方 Gate2 为 CPU lowering 带来的增量(2026-07-15，已归档)

以下内容记录旧三方 Gate2 曾触发的 backend 修复，只是历史工程证据，不属于当前
Gate2 裁决协议。旧设计见
[归档 spec](../verification/back/gate2_production_lowering_v1/scrc-gate2-three-way.md)。

首批 FlagGems live artifact 又验证并补齐了三条非解释器路径：

- `const(value=1.0, shape=[N])` 在零输入函数中经 dims 解析、intent/linalg/LLVM
  产生 f32 symbolic-shape 输出；
- CPU wrapper 不再把所有输入常量硬写成 f32，对声明的 i1/i8/i16/i32/i64
  输入生成相同 element type 的 dense tensor；
- emitter 对 integer/i1 与 f32/int 的真实 dtype 变化发射 `intent.cast`，
  `mean_kernel_2` 的 rank-0 i32 `M → f32` 已三方执行通过；full reduction 没有
  surviving axis 时默认 rank 0；C++ `ReduceLowering` 已补
  collapse-all→extract→`tensor.from_elements` 路径，并从源码重新构建
  `intent-opt`、通过 `test/lower-reduce.mlir`，不依赖旧二进制漂移。
- final-stage max/min/prod 与 `sqrt(sum(partials))` 已分别通过真实 FlagGems
  source/replay/CPU-lowering 三方执行；prod 沿用已有 `ReduceLowering` 乘法
  combiner，没有新增 IntentIR op。
- `mlir-runner` 的 `printMemrefF32` 只给有限十进制精度，曾把正确的 max 输出
  `91.181175` 读回为 `91.1812`。CPU wrapper 现在用一个可 bufferize 的
  `linalg.generic` 对输出逐元素 `arith.bitcast f32→i32`，打印 i32 位模式并按
  IEEE-754 原位恢复；rank-0 与任意 rank f32 输出均保持精确位值。
- CPU main 的 f32 dense 输入也统一使用 MLIR 十六进制位字面量；普通数值不再受
  `.8g` 文本舍入，`-0.0/±inf/NaN` 可以合法且逐位注入。Gate2 的 IEEE-special
  维度因此走的仍是真实 Intent→Linalg→LLVM 路径。
- 通用 `ReduceLowering` 的 max/min 已从传播 NaN 的 `arith.maximumf/minimumf`
  改为 `arith.maxnumf/minnumf`，与 Triton reduction TTIR 及 GPU row-reduction
  一致。修复由真实 max Gate2 矩阵的 `source=replay=+inf, lowering=NaN` 三方模式
  触发，并由 max/min IEEE-special GPU+CPU 回归固定。

当前 common `convert-intent-to-linalg` 已实现 iota 与 gather，CPU ranked-memref ABI
也支持经真实回归验证的 f16/f32/f64、i1/i8/i16/i32/i64 storage；BF16 因 NumPy
没有可移植的原生 storage 而明确 unsupported，不会伪装为 f32。它们是否对某个具体
artifact 可用仍由
op shape/dtype、B obligation 与 capability/conformance record 决定；不得再用一条
过时的全局缺口描述掩盖逐能力的 supported/unsupported 结果。

## 自动融合(elementwise 前沿,2026-06-26)——**不是 PK 标准**(自我审查,2026-06-26)

> **重要纠偏(2026-06-26):** 本节原标题"性能已超越 Triton"是**误导**。自动融合
> 跑赢的是"未融合的 Triton 组合"——这是**另一种比较**(利用库没做的跨 op 融合
> 空隙),不是 PK 标准。用户明确否定了这个方向("我们是 pk 人家的算子库……人家
> 不都已经是高性能库了","感觉你又往玩具方向做了")。**真正的 PK 标准**是:对**同
> 一个算子**、与库**自己的高性能 kernel** 头对头,我们后端**生成**的 MLIR 经真实
> 工具链下降后要足够高性能(Forge E1:ours/Triton)。融合前沿保留为"我们能做、
> 库没做的融合"这一事实,但它**不证明 codegen 高性能**(与手写单融合 Triton 打平
> 才证明 codegen 称职)。真正的 PK 见下一节(row-reduction vs 真实 liger)。

flash 那条 ~983x 的差距是**一线程/query** 的 bring-up 粒度造成的(见上)。在
**elementwise 前沿**,自动融合 kernel 跑赢**未融合**的 Triton 组合——靠的是*自动
融合*,不是手写。关键事实(都用 launch 计数 / 物理带宽证过):

- **不加融合 = 没有优势**:`-convert-intent-to-linalg` 对一条 pointwise 链每个 op
  发一个 `linalg.generic`,下降成 **N 个 `gpu.launch_func`**(N 次 HBM 往返,和库
  一样)。加 `-linalg-fuse-elementwise-ops` 把整条链塌成**一个** `linalg.generic`
  → **一个** kernel(读 a,b,c 各一次 + 写 z 一次 = 4 次 HBM,1 次 launch)。
- **默认映射把每个元素映成一个 block**(threads (1,1,1),正是 flash 慢的同款病)。
  `-scf-parallel-loop-tiling=parallel-loop-tile-sizes=1,256` 把最内连续维切成
  **256 线程合并访存(coalesced)的 block**。这是 **Layer-C 的 schedule 选择**
  (tile 尺寸),A/B 不动。落点:
  `python/intentir/backend/cuda/compile.py::compile_all_kernels_fused(tile)` +
  `python/intentir/backend/cuda/orchestrate.py::run_program(compile_all=…)` 钩子。
- **结果**(RTX 5090 D / f32 / 工作集 > 101MB L2 / median-of-5 / 不做 per-iter
  sync):IntentIR 自动融合 kernel **比未融合的 Triton 组合快 1.41–1.53x**(正好踩
  在 6:4 的流量上限),**与手写的单融合 Triton kernel 打平(~1.00x)**——证明这一
  胜来自库没做的融合,不是 codegen 取巧;打平也证明自动生成的 kernel 是称职的
  (GB/s 与手写 Triton copy 同档)。三条链都赢:`(a+b)*c`、SwiGLU(`silu(a)*b+c`)、
  GeGLU(`gelu(a)*b+c`,xformers/vLLM 的 FFN 块)。
  `python/intentir/benchmarks/fused_elementwise.py` +
  `python/tests/test_fused_elementwise.py` + `corpus/reports/PERF_FUSION.md`。
- baseline 用干净的内联 Triton kernel(最强最公平的 Triton;flag_gems 的
  `pointwise_dynamic` 在这些尺寸上撞 Triton-3.6 launcher 参数数 bug)。SwiGLU 的
  未融合 baseline 已经用**融合的** `silu_and_mul` + 单独 `add`(库能做到的最好),
  所以不是稻草人。
- **范围**:这是 elementwise 前沿。flash 的延迟差距
  (`corpus/reports/PERF.md`)是另一条尚未触碰的前沿,**不被本节覆盖**。

## PK 真实算子库:高性能 row-reduction codegen vs 真实 liger(2026-06-26)

这才是用户要的方向——头对头 PK 库自己的高性能 kernel(Forge 20-kernel 数据集,
liger/xformers/flash_attn)。`pip install --no-deps liger_kernel`(0.8.0)直接装上,
其 `@triton.jit` kernel 在 sm_120 上 JIT 真实运行;autograd `Function.apply` 在本
torch 上撞 `torch.distributed.tensor` 崩,但 forward launcher(`rms_norm_forward`
等)可直接调,**PK 的是库的真实 kernel,不是稻草人**。

**两道可行性门(都过):**
1. **block-cooperative reduction 在 MLIR-20 → sm_120a 上能下降且正确**:手写
   `gpu.all_reduce add` row-sum kernel 经 `lower_memref_module_to_kernels` 下降
   (cubin 非空 .text)+ 真实跑通(rel 1.5e-7)。这是 20 个 kernel 里多数(row-wise
   reduction:rms_norm/softmax/layer_norm/losses)所缺的能力——flash 那条只有标量
   carry,没碰过 barrier/shuffle。
2. 真实 liger rms_norm 在 5090D 上跑通(rel 1.4e-7)。

**rms_norm 真实 PK(`python/intentir/backend/cuda/kernel_driver.py` +
`python/tests/test_row_reduction_pk.py`):**
我们 MLIR 生成的 **read-once、block-cooperative**
row kernel(一 block/row;`T=num_warps*32` 镜像 liger `calculate_settings`;每线程
`memref.alloca<Kxf32>` 缓存,只读一次 x;`gpu.all_reduce` 求平方和;rstd 用 NVVM
`rsqrt.approx.f` intrinsic——`math.sqrt`/`math.rsqrt` 走 libdevice `__nv_*` 是死路;
第二趟从缓存写 y)经**真实 mlir 工具链**(mlir-opt → nvvm → ptxas)下降。
- **结果(kernel-only,两侧预分配,clocks pin,中位):PARITY**——(32768,1024)
  ours/liger ~0.97–1.12x(亚毫秒高方差),(16384,4096) ~1.00x。memory-bound 算子
  上,**parity 就是后端的胜利**(同流量、同 geometry);**不声称跑赢**(同 128 线程
  geometry,没有 geometry 空隙;之前 0.77x 是瞬时高频窗口的假象,已撤回)。
- correctness 是主 DoD 且稳:rel ~1e-7 vs fp32 ref AND liger(Y+RSTD),含非 2 幂
  tail-mask。
- **向量化 v4**:`vector.load`/`store` + region 内 `memref.assume_alignment 16` 能
  下降成 `ld.global.v4.f32`,但**无加速**(HBM-bound 非 issue-bound)——保留标量。
- **read-once 是关键**:两趟全局(写时重读 x)= 1.5x 流量 → 慢 ~1.5x。alloca 缓存
  (K=C/T 小,常量 trip 循环被 unroll + 提升进寄存器,PTX 证无 ld.local 溢出)把
  流量压到 读x一次+读w+写y = 与 liger 持平。

**真正的"后端生成"(`06-26-rowreduce-codegen-from-intent`):** 手写 emit 是*目标
规格*,不是后端。当前 `-convert-intent-to-linalg` 对 rms_norm intent 发 **8 个
linalg.generic → ~8 个朴素 kernel**。已证 **Option B 可行**:从 `IntentFunction`
识别 row-reduction 形态(elementwise→reduce(末轴)→broadcast→elementwise)+ Layer-C
`tile.N` hint → **生成那一个高性能 kernel**(1 gpu.launch,rel 1.3e-7)。下一步:把
识别器从 rms_norm 推广到 softmax(max+sum)、layer_norm(mean+var),做成后端
codegen 入口(命中走快路,否则回落朴素路)。

## 现在不做

- RVV / 其它后端的 lowering pass 暂不实现(GPU 这条已打通并作为模板)。
- fused-kernel-from-`intent.loop` 的自动 schedule(见上"残项")暂不做;非 fused
  的 tiled 执行已端到端跑通。
- 不让"lower 到什么"反向影响 Intent dialect 的设计(dialect 先按提取 / 验证的需要设计)。
