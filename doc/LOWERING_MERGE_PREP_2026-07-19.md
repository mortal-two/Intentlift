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

- `intentir.backend.compile`：generic 编译、direct memref/GPU 编译、PTX 提取、
  `ptxas` cubin；
- `intentir.backend.orchestrate`：单/多 kernel、destination passing、alias、
  scalar/global、dynamic shared memory 和 CUDA Driver launch；
- `intentir.frontend.emit_mlir`：`IntentFunction` 到真实 Intent dialect MLIR；
- `intentir.backend.selection.select_variant`：覆盖全部 22 个 direct pass，
  只基于 target、typed MLIR marker 和 C++ whole-graph structural probe；
- `intentir.backend.pipeline.lower_to_cuda`：generic/direct 自动选择与真实编译；
- `intentir.backend.pipeline.lower_function_to_cuda`：供 Gate adapter 直接提交
  已接受的 `IntentFunction`。
- `intentir.backend.layer_c`、`variant_planner`：Layer-C 参数解析、候选枚举、
  backend legality、constraints-on/off 消融；
- `intentir.backend.runtime_contract`、`variant_manifest`、`cuda_dispatch`：
  runtime guard、同 bundle conservative fallback、预调优顺序、prepared dispatch。

### 中立 API

```python
from intentir.backend.pipeline import lower_function_to_cuda, lower_to_cuda
from intentir.backend.selection import LoweringTarget

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

## Gate adapter 必须保留的 A/B/C 信息

旧 Gate2 lowering 只取 `Gate1Artifact.j.intent` 并运行 CPU `Lower2C`，这不是
当前 CUDA 链路。新 adapter 不能只复制 A 层 op graph；至少必须逐字段处理：

- A：`artifact.j.intent` 中的 op、SSA、dtype、shape、axis、常量和输出；
- B：`artifact.j.boundary`、`artifact.interface` 中的 access、mask、bounds、
  alias、mutation、writeback、reduction identity/associativity；
- C：`artifact.schedule` 与具体 `CompileCase` 中的 grid、warp/thread、tile、
  stage、layout 和 target 条件。

建议 adapter 生成中立 bundle：

```python
@dataclass(frozen=True)
class GateIntentBundle:
    function: IntentFunction
    concrete_extents: Mapping[str, int]
    input_bindings: Mapping[str, str]
    output_bindings: tuple[str, ...]
    output_aliases: Mapping[str, str]
    source_schedule: Mapping[str, Scalar]
    source_grid: tuple[int, ...]
    contract_mapping: tuple[ContractMapping, ...]
```

任何未映射的 B/C correctness 字段必须 fail closed，不能静默丢弃后继续 lower。
Gate1/Gate2 如何证明这些字段可信，属于新的 verification 层；字段一旦被接受，
编译和 CUDA 执行统一复用本分支后端。

Gate2 已有 `Gate2Runner(lowering_runner=...)` 注入缝。下一步应实现
`CudaLoweringRunner.run(spec, values)`，内部调用中立 API，而不是修改或复用
paper verifier。

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
- frontend/backend/orchestrate/neutral bridge 及 variant/dispatch：
  104 passed，1 skipped；
- xFormers split-K CUDA PK：10/10 通过；
- 中立 generic 路径真实生成 `add_kernel` ELF cubin；
- 中立 direct elementwise 路径真实完成 structural probe、specialized MLIR、
  NVVM/PTX/ptxas，并生成 ELF cubin。

## 下一合并阶段

1. 在 Gate 工作树自身提交或隔离现有未提交修改后，把上述合并栈合入；
2. 实现 `GateIntentBundle` adapter，并检查 loop body 内嵌 Layer-B，不能只遍历
   顶层 `fn.body`；
3. 实现 `CudaLoweringRunner`，覆盖多输出、alias、writeback 和 side effect；
4. 继续迁移尚未中立化的 `RowSchedule`、`ElementwiseSchedule`、各 specialized
   materializer 和 Linear multi-cubin bundle；guard、fallback、variant planner、
   manifest 与 prepared dispatch 已进入中立 backend；
5. 分层运行全部 lit、Python lowering、CUDA PK 与 performance regression。
