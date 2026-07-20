# Gate1/Gate2 与 CUDA/MLIR 后端合并准备

日期：2026-07-19

## 结论

本次合并不是回退到旧 Gate 后端，也不是只保留少量 paper fast path。
需要保留的是 rebuttal 分支已经形成的完整编译器能力：IntentIR 表达、
MLIR conversion passes、NVVM/CUDA codegen、PTX/cubin、运行时编排、结构化
variant selection、真实 lowering/PK/performance 回归。需要替换的是 paper
O1--O7/P0--P7 verifier、evidence receipt 和 provider/oracle 编排。

目标数据流为：

```text
Gate1/Gate2 或其他验证器接受 A/B/C IntentIR
                     |
                     v
       verification-neutral lowering API
                     |
          +----------+----------+
          |                     |
   generic Intent->Linalg   direct GPU pass
          |                     |
          +----------+----------+
                     |
             MLIR GPU/NVVM
                     |
               PTX -> cubin
                     |
          guarded dispatch/runtime
```

Gate verdict 可以作为 provenance/diagnostic 信息保存，但不得成为 selector 或
lowering pass 的隐式输入。反过来，paper receipt 也不得成为 Gate lowering 的
前置条件。

## 已建立的独立合并基线

工作树：`/home/kingdom/phdworks/TianchenIR-lowering-merge`

分支：`lowering-merge-prep`

基线：已提交的 `master`，没有修改另一个包含 Gate1/Gate2 未提交工作的
`/home/kingdom/phdworks/TianchenIR` 工作树。

提交按可审阅边界拆分：

1. `3836c67 lowering: port the CUDA MLIR backend`
2. `9a198b5 test: preserve real MLIR lowering regressions`
3. `335cac6 backend: port real CUDA compile and orchestration`
4. `0c49f0b test: cover frontend CUDA compile and runtime orchestration`
5. `936f573 backend: add verification-neutral CUDA lowering bridge`
6. `be7dc34 test: exercise neutral generic and direct CUDA lowering`
7. `3e417e8 test: isolate xformers split-k import shims`
8. `9886391 backend: preserve Layer-C and runtime dispatch contracts`
9. `9cb0a10 test: cover neutral schedules and runtime contracts`
10. `01b4093 backend: preserve neutral schedule variant planning`
11. `5d12cf3 test: cover neutral variant planning and ablation`
12. `de1e41f backend: preserve variant bundles and guarded dispatch`
13. `ce46632 test: cover variant bundles fallback and dispatch`
14. `0fdc820 backend: connect structural schedules to CUDA lowering`
15. `0d76bfd test: cover structural schedule policies and overrides`

中间两次文档提交只记录合并边界与回归结果，不含编译器逻辑。

## 保留的后端能力

### Intent/MLIR/CUDA 编译器

- Intent ODS 与必要的 dialect verifier；
- generic `convert-intent-to-linalg`；
- 22 个 direct GPU conversion pass；
- row reduction 与 elementwise read-once/vector fast path；
- gather/scatter、layout、compact、RNG、arg-reduce、sort；
- pool、upsample、polar、direct conv 与 implicit-GEMM conv MMA；
- contraction、MMA、warp-specialized MMA；
- loop contraction、Flash forward/backward；
- split-K forward/varargs/backward reduction；
- GPU math 到 NVVM 的近似指令和 FMA/codegen 补强；
- generic 与 direct pass 共用的 PTX/cubin 后半段。

### Python 后端

- `intentir.backend.cuda.compile`：generic 编译、direct memref/GPU 编译、PTX 提取、
  `ptxas` cubin；
- `intentir.backend.cuda.orchestrate`：单/多 kernel、destination passing、alias、
  scalar/global、dynamic shared memory 和 CUDA Driver launch；
- `intentir.frontend.emit_mlir`：`IntentFunction` 到真实 Intent dialect MLIR；
- `intentir.backend.cuda.selection.select_variant`：覆盖全部 22 个 direct pass，
  只基于 target、typed MLIR marker 和 C++ whole-graph structural probe；
- `intentir.backend.cuda.pipeline.lower_to_cuda`：generic/direct 自动选择与真实编译；
- `intentir.backend.cuda.pipeline.lower_function_to_cuda`：当前过渡入口，供 Gate
  adapter 提交已接受的 `IntentFunction`；它不是最终 `A+B+ABI` hand-off。
- `intentir.ir.layer_c`、`intentir.backend.cuda.variant_planner`：Layer-C 参数解析、
  候选枚举、backend legality、constraints-on/off 消融；
- `intentir.backend.cuda.runtime_contract`、`variant_manifest`、`dispatch`：
  runtime guard、同 bundle conservative fallback、预调优顺序、prepared dispatch。

### 当前 target-specific API

```python
from intentir.backend.cuda.pipeline import lower_function_to_cuda, lower_to_cuda
from intentir.backend.cuda.selection import LoweringTarget

program = lower_function_to_cuda(
    accepted_intent_function,
    concrete_extents,
    target=LoweringTarget(chip="sm_120a"),
)
```

`lower_to_cuda` 不接收 receipt、P0--P7、O1--O7、Gate status、provider response、
dataset identity 或 kernel/function 名称。schedule/ablation 通过显式
`pass_overrides` 进入，但 override 仍必须由真实 C++ pass 完整消费同一个 MLIR，
否则自动回落 generic。

这个 API 已经与 Gate verdict 解耦，但单独的 `IntentFunction` 仍不足以表达完整
`A+B+ABI`，因此不能把它命名为最终的中立 lowering 合同。

## Gate adapter 必须保留的 A/B/C 信息

旧 Gate2 lowering 只取 `Gate1Artifact.j.intent` 并运行 CPU `Lower2C`，这不是
当前 CUDA 链路。新 adapter 不能只复制 A 层 op graph；至少必须逐字段处理：

- A：`artifact.j.intent` 中的 op、SSA、dtype、shape、axis、常量和输出；
- B：`artifact.j.boundary`、`artifact.interface` 中的 access、mask、bounds、
  alias、mutation、writeback、reduction identity/associativity；
- C：`artifact.schedule` 与具体 `CompileCase` 中的 grid、warp/thread、tile、
  stage、layout 和 target 条件。

最终 adapter 必须生成与验证器无关的 `IntentProgram`，运行时绑定另行放入 lowering
request；不能把 `Gate` 写进 production DTO 名称：

```python
@dataclass(frozen=True)
class IntentProgram:
    function: IntentFunction
    boundary: BoundaryContract
    interface: LogicalABI
    schedule_hint: LayerC | None

@dataclass(frozen=True)
class LoweringRequest:
    program: IntentProgram
    concrete_extents: Mapping[str, int]
    input_bindings: Mapping[str, str]
    output_bindings: tuple[str, ...]
    target: BackendTarget
```

任何未映射的 correctness-relevant B 必须 fail closed，不能静默丢弃后继续 lower。
C 可以被忽略或重调，但不得被重新解释为 B。Gate1/Gate2 如何证明 B 字段可信，
属于 verification 层；字段一旦被接受，编译和 CUDA 执行统一复用本分支后端。

当前 `python/intentir/verify/gate2/cuda_lowering.py::GateIntentBundle` 只是迁移期
Gate adapter；它不得上移到 `ir/` 或成为 backend API。等 `IntentProgram` 接管完整
B/ABI 后，应迁移调用方并删除该类型，不保留 alias/facade。

Gate2 已有 `Gate2Runner(lowering_runner=...)` 注入缝，且当前
`CudaLoweringRunner.lower/execute` 已通过 `CudaLoweringSpec(GateIntentBundle)` 调用
CUDA backend。下一步不是再造 runner，而是把它的输入迁移为中立
`LoweringRequest(IntentProgram, target)`；runner 仍不得读取或复用 paper verifier。

## 不进入共享 lowering core 的内容

- paper O1--O7/P0--P7 verdict 与 orchestrator；
- live-provider prompt/response；
- canonical evidence、oracle/case adapter；
- paper manifest/status 和 `artifacts/paper_pipeline/**`；
- Gate-Paper/SCRC/masked-J 的历史叙事；
- 任何把旧 Gate verdict 自动升级为 CUDA correctness/performance 结论的逻辑。

这些内容可以归档或继续作为实验记录，但不能被 production selector 读取。

## 已执行验证

- RelWithDebInfo (`-O2`) 构建 `intent-opt` 成功；
- 发现并修复 row-reduction 中临时 `ValueRange` 悬空引用：旧 Debug 构建会掩盖，
  `-O2` 下会在 `memref.store` 构造时崩溃；
- MLIR lit：109/109 通过；
- frontend/backend/orchestrate/neutral bridge、schedule、variant/dispatch 与
  split-K PK：114 passed，1 skipped；
- xFormers split-K CUDA PK：10/10 通过；
- 中立 generic 路径真实生成 `add_kernel` ELF cubin；
- 中立 direct elementwise 路径真实完成 structural probe、specialized MLIR、
  NVVM/PTX/ptxas，并生成 ELF cubin。

## 后续语义迁移阶段

1. 实现中立 `IntentProgram` 与 B-obligation consumption ledger，并检查 loop body
   内嵌 Layer-B，不能只遍历顶层 `fn.body`；
2. 让 `CudaLoweringRunner` 消费 `LoweringRequest(IntentProgram, target)`，覆盖
   多输出、alias、writeback 和 side effect；迁移完成后删除 `GateIntentBundle`；
3. 把 correctness-critical B 按 op family typed 化并闭环消费：accumulator/result
   dtype、logical return/destination、alias/effect、reduction identity/order、
   gather/scatter bounds/conflict；
4. 消除 common `intent.loop` lowering 对 `trip.tile_hint` 的语义依赖；无 C 时仍须
   正确 lower，CUDA planning 可以选择或重调 tile；
5. CUDA schedule、materializer、variant manifest、prepared dispatch 继续留在
   `backend/cuda/`，只把真正 target-neutral 的 protocol/contract 放入 `ir/`；
6. 分层运行全部 lit、Python lowering、CUDA PK 与 performance regression。
