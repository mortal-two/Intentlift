# Formal Batch V2 05：闭合统一前端后再完成 28 Kernel 证据链

> 状态：进行中
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

## 为什么是新任务

Task04 的 frozen primary budget 已被真实消耗一次并产生不可恢复的 Lift schema failure；其
`recovery_per_case=0` 使 28/28 验收不可达。Task05 不覆盖该失败，而是以新的 manifest
digest 和 Lift wire protocol 启动 recovery epoch，并在所有统计中保留 Task04 first-attempt。

## 最终目标

仍然完成原研究目标，不降低 Gate：

```text
28 pinned real kernels
  -> real source-aware LLM Lift: one canonical J + pure C + direct pi
  -> source-blind LLM Replay
  -> strict normalized compiler-topology Gate1
  -> source/replay/E_J Gate2 relation triangle
  -> deterministic J -> Intent MLIR -> CUDA
  -> full-B conformance
  -> reproducible performance <= 1.10x frozen source median
```

Task04 的失败必须作为 `protocol_v1_first_attempt` 单列；Task05 的 Swiglu 结果标为
`protocol_v2_recovery`，不得合并成 first-attempt success。

## 架构不变量

- Canonical Intent 仍只有 `J(A+B) + pure C`；pi 在 Intent 外，由 Lift LLM 直接产生。
- Replay 看不到源码、真实 path/symbol、compiler diff 或旧 response。
- 不引入 composer、RuleCard/Registry、IntentProgram、LogicalABI、kernel-name rule。
- deterministic 部分只做 schema validation、frontend compilation/normalization、Gate、lowering、
  execution 与 evidence ledger；不得根据 case 名补语义。
- J/C/pi/interface 的同一个 digest 链必须贯穿 Gate1、Gate2、CPU/CUDA backend 与性能。

## 阶段与提交节奏

### 0. 冻结 protocol-v2（不调用 LLM）

1. 冻结新的 28-case manifest/batch id/digest，source pins、specialization、性能门槛保持不变；
2. 明确 Lift wire grammar version，并绑定 schema/prompt/client SHA；
3. 重新生成 28/28 preflight receipt；Task04 receipt 只引用、不复制或修改；
4. formal runner 对 Triton 与 CuTe 都必须具有完整 source/replay observer 后才允许调用。

完成后单独提交。

### 1. 统一 CuTe Replay arm（不调用 LLM）

- `ResponsesReplayWriter` 按 `Frontend` 选择全局 validator，而不是按 kernel/case；
- CuTe replay 固定输出 `@cute.kernel scrc_k0` 和 `@cute.jit scrc_entry`；
- source/replay 使用同一 CuTe 4.4.1 MLIR+PTX strict fingerprint；
- alpha-normalize device/entry interface，保留 raw bulk/TMA、mbarrier/fence/order 差异；
- 静态与真实 compile regression 通过。

完成后单独提交。

### 2. 单一 recovery canary

只运行 `liger_swiglu_forward`：

- 1 次 protocol-v2 Lift；合法后 1 次 normal Replay；normal Gate1 闭合后最多 1 次 masked-J；
- 保存 exact request/response/raw/provider usage；
- 不向 prompt 提供 Task04 response 或 parser diagnostic；
- 若仍为全局 grammar failure，停止批次并修协议，不扩散到其他 27 项；
- 若 Lift 合法但 Gate1 不闭合，诚实记录为表示/IntentOp 健壮性问题，不局部 repair。

完成后提交真实 receipt。

### 3. 四个 7-case 分层

按原四个机制分层逐层执行，每层结束提交一次。任何 case 的 fresh recovery 必须新建 lineage，
不能覆盖 primary；first-attempt、recovery、masked-J 分开统计。

### 4. Gate2 / backend / performance

- 仅 Gate1 `REPLAY_CLOSED` 进入三腿；三腿独立克隆 committed pre-state；
- exact/tolerance/one-of-updates 只读同一个 J relation；
- E_J 必须走 formal J -> Intent MLIR -> CPU，不用 NumPy legacy fallback；
- 缺失 scatter/recurrence/async executor 或 CUDA B consumption 时独立闭合能力并提交；
- correctness 后才测 source/replay/backend，保存 raw CUDA-event samples 与完整环境 identity。

## 验收标准

1. 新 manifest 28/28 source/compiler/replay observer ready；
2. 28/28 有真实 Lift 与 source-blind Replay provenance，Task04 failure 单独保留；
3. 28/28 Gate1 strict hard-close，且 masked-J 不闭合；
4. 28/28 Gate2 三腿按同一 J relation PASS；
5. 28/28 CUDA full-B conformance PASS；
6. 28/28 性能 evidence 完整且达到冻结门槛；
7. 完整报告 first-attempt/recovery/token/墙钟/调用数，不隐藏失败；
8. 全量回归通过，任务归档，再创建后续论文实验任务。

## 当前第一步

先完成阶段 0/1：建立 protocol-v2 manifest，并把已经实现但尚未接入正式 orchestrator 的
CuTe compiler/validator/fingerprint 接到 source-blind Replay。此阶段在线调用必须保持 0。

## 阶段 1 完成记录（2026-07-22）

- `ResponsesReplayWriter` 已按公共 `Frontend` 在同一入口选择 Triton/CuTe validator；CuTe
  输出固定为 `@cute.kernel scrc_k0` 与 `@cute.jit scrc_entry`，不按 case 名分派。
- 正式 runner 已接入 `CuteGate1Runner`：source/replay 共用 CuTe 4.4.1 MLIR+PTX observer，
  对 device/entry positional interface 作 alpha normalization，并执行同一 normal/masked-J
  状态机。
- `opaque` 只表达匿名 positional slot 与公共 DSL dtype；它禁止 rank、tensor_ref、alias、
  读写 mode 与 manifest symbolic value，因而不能成为 J 的隐藏语义输入。
- 修复 CuTe DSL 对 replay `@cute.jit` 使用 `inspect` 时需要真实临时文件的问题；物化发生在
  已验证代码的隔离编译临时目录中，不改变历史 Task04 worker 字节或 preflight receipt。
- 两个冻结 CuTe source 均以仅 alpha-renaming 的 untrusted replay 真实编译并严格闭合；bulk
  与 TMA 交叉比较保留 `cp.async.bulk.shared` / `cp.async.bulk.tensor.1d.shared` 差异并拒绝闭合。
- 定向回归 `42 passed`，全量回归 `655 passed, 4 skipped`；本阶段在线 LLM 调用为 0。

下一步只做阶段 0：冻结自包含 protocol-v2 manifest、wire SHA 与新的 28/28 preflight；在
这些 evidence 提交前不得启动 recovery canary。
