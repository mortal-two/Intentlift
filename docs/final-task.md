# 最终任务：统一系统骨架与剩余 Kernel 闭包

## 当前合同

```text
Lift(Source) -> (J, C, π)
Replay(J, MaskC(C); U_f) -> R
Gate1 = normalized-observation(Source, π) == normalized-observation(Replay, π)
Gate2 = Source–Replay ∧ Source–J ∧ Replay–J
```

Replay 不可见 Source、`π`、未遮蔽 `C` 值、Gate 差异和失败历史。Observer 不读取 `π`；normalizer 只能依据冻结的带类型关系分别规范化两侧观测。Gate1 必须先闭合，Gate2 三条比较边必须全部通过。TypeCheck、构建、工具调用和文档变化均不计入闭包进展。

## 权威闭包状态

- 活动清单：28 个真实 Kernel。
- 已闭包：24/28。
- 待闭包：4/28。

尚未闭包的精确 case：

1. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`
3. `high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block`
4. `high_performance:flash_attn.ops.triton.linear.kernel_fwd`

这些项目的当前阻塞是实际 Source/Replay 结构观测与冻结 normalizer 关系尚未闭合，或闭合候选尚未通过完整 Gate2；不得以 per-kernel Gate 放宽、手写语义替代、mock 或只通过 TypeCheck 结案。

## 当前真实证据

FlashAttention forward 在统一代码树上完成零模型调用的能力复核：

- case：`high_performance:flash_attn.flash_attn_triton._fwd_kernel`
- 状态：`scrc_certified`
- Gate1：通过
- Gate2：通过，`Source–Replay`、`Source–J`、`Replay–J` 全部通过
- 模型调用：0
- 复用：精确 parent artifact 与精确 parent Replay
- 当前复核证据：`/tmp/intentir-flash-fwd-unified-rerun-20260805/flash_attn_flash_attn_triton_fwd_kernel/gate2-recheck-001`
- parent：`/tmp/intentir-flash-fwd-manual-access-pi-20260801/flash_attn_flash_attn_triton_fwd_kernel/run-001/attempt-04`

该复核重新运行当前 Gate1 编译和完整 Gate2，但不改变 parent verdict。闭包总数仍为 24/28；复核证明统一树没有破坏这一已闭包 Kernel。

## 当前交付物

- 单一活动代码树包含 SCRC、Triton/TileLang/CUDA 前端和 CUDA/RVV 后端核心实现。
- 公共文档只保留架构、构建、工作流与本最终任务。
- 活动树不包含历史测试、Trellis 或旧状态文档。
- 本地忽略目录按 `scrc/`、`frontend-extension/`、`backend/` 保存内部理解与重要结果。
- 合并前 Git refs、脏改动和不可提交资产必须在删除旧 worktree 前重新生成完整安全快照。
- 最终仓库必须是普通独立 Git 仓库，只保留权威主分支，并能在新机器上构建最小系统骨架。

## 未解决阻塞

- 上述 4 个 Kernel 尚无满足当前合同的 Gate1 + Gate2 三边闭包证据。
- 旧 worktree 中的论文敏感文件、真实实验结果、provider 配置和未跟踪核心源码必须先完成校验归档，才能安全删除旧 checkout。

## 下一步

1. 刷新完整 Git bundle、脏状态 patch、未跟踪/ignored 重要资产清单与校验和。
2. 建立独立、单主分支仓库，迁移 `local/` 与安全配置后复验构建、清单和当前真实闭包证据。
3. 删除已归档的旧 worktree；保留一个本地归档入口，不恢复历史测试或 Trellis 到活动树。
4. 对 4 个剩余 Kernel 逐一解决真实 normalizer/Replay 闭包缺口，并只在 Gate1 与 Gate2 三边均通过后更新 24/28 计数。
