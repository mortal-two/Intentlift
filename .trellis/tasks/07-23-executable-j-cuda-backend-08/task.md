# Task 08：唯一 J 的可执行 CUDA ABI 与 Backend Conformance 闭环

> 状态：已完成
>
> 类型：IntentIR production backend 系统能力重构；不是新实验、Agent 或 Gate。

## 当前断点

Task06 已固定：

```text
response JSON -> Decode/Elaborate -> J -> Intent MLIR
```

Task07 已让 RefEval 与 CUDA production analyzer 共用 `ProjectFunctionalBoundary(J)`。仓库原有 CUDA
pipeline 也已经能把 Intent MLIR 编译为真实 cubin并通过 Driver API 执行，但 `lower_j_to_cuda(...)`
的正式 API 到此仍只返回编译产物；调用者必须绕过 J-facing backend，手工调用 MLIR-level
`execute_cuda_artifact(...)`，并自行猜测 `out0 -> J output handle`。

本任务闭合这一条缺口：

```text
LoweringRequest(J,C,target)
  -> lower_j_to_cuda(...)
  -> CudaJLowering
  -> execute_j_cuda(CudaJLowering, inputs keyed by J logical handles)
  -> outputs keyed by J logical output handles
  -> independent backend conformance against RefEval(J)
```

本机最小真实探针已经确认基础设施可用：RTX 5090 D 上，一个长度 33 的 J-owned `tanh` graph 经
generic production pipeline 得到一个 cubin、真实 launch，输出误差 `1.19e-7`。因此本任务不是重写
CUDA compiler，而是建立正式、严格的 J ABI execution boundary。

## 所有权与接口纪律

- execution 只接收已经生成的 `CudaJLowering` 和 runtime logical inputs，不再接收另一份 J、
  InterfaceView、Harness、π、Source 或 BindingMap；
- input key、参数顺序、dtype/rank/shape 来自该 lowering 持有的唯一 `request.j` 以及由同一 J 发射的
  Intent MLIR；不接受 `arg0` fallback、缺 input 或额外 input；
- output 按 J logical output 顺序把底层 `out0/out1/...` 唯一映射回 J handles；数量不一致 fail closed；
- 执行复用已编译 cubin，不在 execute 阶段重新编译或替换 MLIR；
- 本任务不把 production result 放入 `SCRC_CERTIFIED`，只使用现有独立
  `BackendConformanceRecord`/focused regression 表达 target correctness。

## 实施步骤

1. 在 CUDA J-facing backend 增加最薄的 `execute_j_cuda` API，复用现有
   `execute_cuda_artifact`，但移除其 positional input fallback 在正式 J API 中的可见性；
2. 对 runtime input 集合与底层 output arity做 J-owned fail-closed 校验，并返回 logical-handle
   keyed outputs；shape/dtype/device 由从同一 J 发射的静态 MLIR ABI继续机械校验；
3. 增加无 monkeypatch 的真实 GPU focused test：唯一 J -> real Intent MLIR -> real cubin -> real
   device launch -> named J output；
4. 用 deterministic/non-symmetric input 将真实 production output 与 `RefEval(J)` 比较。首个 exact
   add profile 使用 exact equality；浮点 transcendental/reduction 若加入则只能使用固定 Q；
5. 增加 missing/extra input、wrong shape/dtype 的前置拒绝测试，证明正式 API 不依赖 Harness 猜测。

## 验收

1. 真实测试不 monkeypatch emitter/compiler/launcher，产物是可加载 ELF cubin并在 GPU 上执行；
2. production API 输入输出只使用 J logical handles，`out0` 不泄露到调用者；
3. odd extent（首版 33）与普通 extent均正确，不因 block/tile 形态发生越界；
4. actual CUDA output 与独立 `RefEval(J)` 在声明的 fixed comparison policy 下闭合；
5. missing/extra/wrong ABI 在 launch 前稳定失败，不静默复制、重排或接受 positional fallback；
6. Task07 的 functional projection仍是 ephemeral internal analysis，不升级为 artifact；
7. 只运行 backend/RefEval focused tests，不调用 provider、不重跑 Gate1/2 batch 或全量实验。

## 非目标

- 不新增 Agent、Gate、J wrapper、IR、RuleCard、hash/commitment 或实验“冻结”设施；
- 不让 production CUDA 取代 Gate2 的 J-only RefEval；
- 不在首版支持 alias/inout、集合值语义、一般 affine access或所有 C schedule；
- 不为每个前端建设专用 lowerer或 formal semantics；
- 不通过测试内临时 launcher 绕开 production API。

## 2026-07-23 completion checkpoint

- 新增正式 `execute_j_cuda(CudaJLowering, logical_inputs)`；它复用 lowering 中已经编译的 cubin与
  原始 Intent MLIR，不重新接收/构造 J，不重新编译；
- J-facing API 在 runtime 之前要求 input keys 与 `J.intent.inputs` 精确相等，明确拒绝 missing、
  extra、non-string 及 legacy `argN` positional fallback；底层静态 MLIR ABI继续检查 CUDA device、
  shape 与 dtype；
- 底层 `out0/out1/...` 只在 backend 内部按 J output 顺序映射为 logical handles；output arity/key
  漂移 fail closed，调用者实际得到 `{"t1": tensor}`；
- 新增 focused unit tests锁定 J handle binding、pre-runtime rejection 与 output mapping；
- 新增无 monkeypatch 的真实 GPU regression：同一 accepted J 的 `map_binary add(t0,t0)->t1` 分别以
  extent 32 和 33 经 production Intent MLIR、真实 ELF cubin、CUDA Driver launch执行，并与独立
  scalar CPU `RefEval(J)` 逐元素 exact 相等；wrong shape/dtype 在 launch 前拒绝；
- Task08 新测试单独为 `8 passed`；与 Task07 direct-J backend及既有 pipeline测试合跑为
  `25 passed`。未调用 provider，未运行 Gate1/2 batch或全量仓库回归。
