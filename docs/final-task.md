# 最终任务：真实 Kernel 的语义闭包与生产 Lower

本文是当前仓库唯一活动执行记录。它只保留当前合同、未解决阻塞、下一步和真实 Kernel 的闭包事实；实现日记、合成 canary、历史 artifact 复核和机械诊断不属于活动记录。

## 当前合同

```text
完整真实 Source + Harness/compiler view
  -> Lift
  -> canonical (J, C, π)
  -> CandidateCheck / Decode / TypeCheck
  -> BindSource
  -> fresh source-blind Replay
  -> Gate1
  -> Gate2: Source–Replay ∧ Source–J ∧ Replay–J
```

- `J` 完整保存算法的可观测语义：logical tensor、operator、def-use、control/recurrence、domain、access、predicate、effect、alias、结果和 typed dimension relation。它不保存目标名称、Source 路径、launch、layout、memory hierarchy、program/lane 坐标或后端 recipe。
- `C` 是当前 Source/后端流程实际使用的实现选择；它不新增 operator、predicate、effect 或 result，也不能改变 `J` 的语义。真实生产后端继续使用各自生成的 `target_C`；Harness/compiler view 中的真实 C values 必须与该生产绑定严格对应。
- `π` 只供 Gate1 两个独立 observation arm 做已证明的逐 occurrence normalization。它不得进入 Replay packet、生产 Lower 或改变 Source/Replay 的观测。
- Triton/TTIR 使用共享 SCRC runtime 并保留 TTIR/source observer；CUDA、CuTe、TileLang 使用各自的 Source/compiler/observer/Gate runner，不改变原有多前端运行逻辑，也不增加 `consistency-only` 输入路径。
- Replay 校验只能读取当前规范化 `J`、与 Source/Harness 已绑定且经 Gate1 Source schedule 校验的同一份 `artifact.c` 和固定工具环境；Replay 不重新生成独立的 `target_C`，也不把 Replay C 反向当成生产 C。历史 J、历史 packet、Source、`π`、Gate 差异和失败历史不得进入 Replay。

## 当前权威真实闭包证据

`high_performance:liger_softmax` 的正式 continuation 已完成，证据目录为：

`docs/project-guide/real-evidence/retry-02/liger_softmax/run-001/`

- `main_status`: `SCRC_CERTIFIED`
- Gate1: `replay_closed`，hard comparison 通过，Lift 在第 `3` 次调用闭合
- Gate2 case `finite-hidden-1`: `gate2_pass`
- 三条比较全部通过：Source–Replay、Source–J、Replay–J
- 阶段耗时：Lift `211.920 s`，TypeCheck `53.46 ms`，semantic probe `2.352 s`，Replay/Gate1 `65.164 s`，Gate2 `168.577 ms`
- Gate2 分项：Source `9.743 ms`、Replay `122.988 ms`、J reference `18.768 ms`、三边比较 `4.432 ms`
- 本轮 Replay 使用 Source/Harness 已绑定的 `artifact.c`；没有新建独立 `target_C`，生产 CUDA contract projection 对同一 J 的 access guard 也已闭合。

## 真实闭包事实

活动 workload universe 由 [high-performance-kernels.json](../spec/lowering/high-performance-kernels.json) 定义，共 28 个真实 Kernel。当前权威汇总为：

```text
活动 Kernel：28
已闭包：24
待闭包：4
```

待闭包的 case：

1. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`
3. `high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block`
4. `high_performance:flash_attn.ops.triton.linear.kernel_fwd`

每个“已闭包”计数都必须有同一真实 case 的完整 Source、Harness、Lift、CandidateCheck/Decode/TypeCheck、BindSource、fresh Replay、Gate1 和 Gate2 三边 evidence/lineage。清理活动树、Python import、schema 解析、TypeCheck、构建或 target capability 都不是闭包，不能增加上述计数。

## 当前阻塞

- `high_performance:liger_swiglu` 的现有 evidence 位于 `$INTENTIR_EVIDENCE_ROOT/intentir-real-flow-validation-20260908/liger_swiglu/run-001/`；该 case 的第一轮 fresh Lift 在 901.708 秒后收到 provider HTTP 500，未产生 artifact，也未进入 CandidateCheck、BindSource、Replay、Gate1 或 Gate2。这仍是 provider 基础设施阻塞，不是 J/C 语义闭包结论。
- 复杂真实 Source 的 Lift 候选仍会在 CandidateCheck 前后暴露结构不一致：根级 `j/c/pi` 边界、逻辑 tensor 与物理 carrier 的归属、C slot attachment、value/dimension reference、optional scalar 顺序，以及 loop carry/body/yield 必须共同满足同一闭合 grammar。检查器必须保持失败关闭，不能为通过候选放宽合同。
- Replay ingress 仍要求每个 representation-only carrier 都能明确归属为 J ABI、J-derived placeholder 或合法 C attachment；无法证明所有权的 carrier 必须继续拒绝。
- `π` 的 access、reduction、control 和 materialization relation 尚需两臂各自证明到唯一 J anchor 的 physical occurrence mapping。在此前，未证明 relation 保持 hard，不恢复模块级文本放宽。
- native 多前端 Dataset 的完整来源和对应工具链仍需满足 manifest 与前端版本约束；不可用时必须报告 unavailable，不得用仓库内合成 Source 代替真实 workload。
- CUDA、CuTe、TileLang、RISC-V 的独立 compiler/lowering/执行证据仍需分别闭合；它们不能借用 Triton/TTIR 的 observer 或 Gate 结果。

## 下一步

1. 先针对一个未耗尽真实 workload 修复 CandidateCheck 所报告的最小 authoring/contract 断层，保持每个 candidate branch 最多三次逻辑 Lift，并记录各阶段实际耗时。
2. 只有 Lift、CandidateCheck/Decode/TypeCheck、BindSource 和 fresh source-blind Replay 都成功后，执行该 case 的 Gate1；Gate1 闭合后才执行 Gate2 三条比较，并保存完整 evidence/lineage。
3. 对空 C 与非空 C 各完成一条真实闭包路径，再逐一处理四个待闭包 case。满足 Gate1 与 Gate2 三边后，才可以更新 closed-Kernel count。
4. 对 CUDA、CuTe、TileLang、RISC-V 和 CPU reference 分别运行各自的真实 compiler/lowering/执行路径；任何 native 结果都必须保留自己的 Source、observer、Gate 和 target evidence。

## 完成条件

任务完成仅当下列条件同时成立：

- `J`、`C`、`π` 的职责边界完整，任何 target 切换都不改变 canonical `J`；
- Source 与后端的实际 C binding 均完整、强类型并经过各自路径验证；
- `π` 只含逐 occurrence 已证明的 Gate1 relation；
- 每个活动 Kernel 都拥有 Gate1 closure、Gate2 三边通过和完整真实 lineage；
- 各独立 target consumer 的生产 Lower、真实执行和 Source round-trip 不依赖合成 canary、mock 或历史 artifact 复核。
