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
- **lower 目标**(预期,非当前承诺):map/reduce/contraction → `linalg`;数据移动 / gather/scatter → `tensor`/`linalg`;select → `arith`;结构化 `loop` → `scf`;向量化 → `vector`。

## GPU 后端已验证(2026-06-25,RTX 5090 D / sm_120a / 系统 MLIR 20.1.8)

预测的 lower 目标已被硬件证实:`map.unary` / `map.binary` / `reduce` /
`contraction` → `linalg.generic`;结构化 `loop` / scan 递推 → `scf.for`+`iter_args`;
并行轴 → `scf.parallel` → `gpu` grid。完整脊柱:
`intent → linalg/tensor/arith → scf.parallel → gpu → nvvm → PTX → ptxas → cubin
→ CUDA Driver-API launch`。**朴素(unfused)attention 与 fused FlashAttention
(online-softmax,单 kernel)都在 5090 D 上真实执行**;输出与解释器参考
(`verify/interpret.py`,即 Layer A 语义)一致到 rel ~1e-7(execute==interpret,
见 [../verification/validation.md](../verification/validation.md) 第 3 道检查)。

实现细节固化在 `python/intentir/backend/`(`compile.py` / `launch.py` /
`orchestrate.py`)与 C++ pass `lib/Intent/Transforms/`。**血泪规则**(后续 RVV /
其它后端起步前必读):

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
  单-kernel 路径。`bench/flash_emit.py` 把这个已验证形态按 (Sq,Sk,D,BK) 参数化,
  是 C++ `intent.loop` lowering 应当镜像的 codegen 规格。
- **retune / freeze 已落地为机制**:`bench/flash_emit.py::retune_sweep` 在保持
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
  (`orchestrate.run_program` 已支持多输出返回 3 个 carry),carry 放在常驻
  CUDA buffer 里前递,末尾 `O=Oacc/l`。这就是 NON-fused tiled flash 的真实跑
  法 —— kernel 在 5090D 上跑,host 控制循环。与 `interpret(JSON)` 一致到 1.3e-7。

**残项(命名的优化,不是前/后端缺口):** 直接从 `intent.loop` 自动发射*fused*
单 kernel(即已证的 `flash_fused.mlir` schedule);今天 fused kernel 由
`bench/flash_emit.py` 发射,尚未由 loop lowering 产出。需要 interchange + carry
privatization(把 query/batch 并行维提到 trip 循环外、carry 按 query 私有化)。

## 现在不做

- RVV / 其它后端的 lowering pass 暂不实现(GPU 这条已打通并作为模板)。
- fused-kernel-from-`intent.loop` 的自动 schedule(见上"残项")暂不做;非 fused
  的 tiled 执行已端到端跑通。
- 不让"lower 到什么"反向影响 Intent dialect 的设计(dialect 先按提取 / 验证的需要设计)。
