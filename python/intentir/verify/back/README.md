# 旧验证实现归档

这里保存已退出当前主线的验证实现，包括 pre-SCRC 代码与旧三方 Gate2 adapter。
它们用于历史复核与已有调用方兼容，但不再参与新的正确性主线。

特别说明：`legacy_v0/replay_repair_prompt.py` **已经归档**。文件仍然存在，是为了
按项目要求保留历史接口和实验复现能力，不代表活动流程仍有内层 replay repair。
它不得被 `gate1/`、`gate2/`、SCRC Lift 或 live/held-out 入口导入。
因此在全仓搜索中仍能看到这段历史实现是预期现象；若它出现在上述活动模块的
import/call graph 中，才属于架构回归，`test_verify_archival.py` 会直接失败。

新的验证只有两个过程：

1. `intentir.verify.gate1`：源码隔离重放、双 TTIR 指纹闭合；
2. `intentir.verify.gate2`：源与重放两腿动态实跑及逐输出比较。

约束：`gate1/`、`gate2/` 和新的 extraction 主路径不得导入本目录。旧路径
`intentir.verify.interpret` 只是兼容转发层。

选择性归档边界：

- `legacy_v0/interpret.py` 保留旧 JSON→NumPy 执行语义；
- `legacy_v0/gate1_result.py` 保留旧
  `Gate1Result(repair_attempts, leakage_ratio)` 报告读取能力；
- `legacy_v0/masked_ratio.py` 保留旧 fuzzy masked-J ratio/threshold 复现实验；
- `legacy_v0/replay_repair_prompt.py` 保留旧的
  `CURRENT_REPLAY_CANDIDATE/CONTROLLED_REPLAY_UPDATE` prompt 复现能力；
- `gate2_production_lowering_v1/` 保留旧 source/replay/CPU-or-CUDA-lowering
  三方 Gate2 adapter、结果模型和统计口径；
- `intentir.verify.interpret` 保留原公开导入路径，但只转发旧符号；
- 历史 corpus/universe/CUDA demo 必须显式导入
  `intentir.verify.back.legacy_v0`，避免被误认为新验证主线；
- `intentir.verify` 包默认只导出 `gate1` 和 `gate2`，导入该包时不应
  自动加载归档解释器。

兼容可执行不等于仍在新主线中；归档实现不得作为 Gate1 或
Gate2 的备用裁决器。

活动 `gate2/`、live/held-out 入口和正式 campaign 不得导入
`gate2_production_lowering_v1`。通用 backend lowering 仍可独立开发和测试，但其
输出不进入当前 Gate2 verdict。

活动 ReplayWriter 接口不接受 `base_candidate`、`control_candidate` 或
`repair_candidate`。normal、masked-J、paired intervention 与 π/C 诊断都只能从各自
本轮 LiftArtifact fresh generation；旧 replay code 只允许留在审计结果中，不能进入
下一次 writer 调用。

当前 normal 失败恢复链严格为：

```text
fresh writer(J,C,π) -> compile/fingerprint -> source-free ReliftEvidence
-> source-aware Lift/Relift 重新生成完整 J/C/π -> 从头 fresh writer
```

连续两轮若 canonical J/C/π/interface 与确定性 hard-diff 同时完全不变，Gate1 以
`NO_CLOSURE + deterministic stagnation` 停止，不再继续 Relift/writer 采样。这避免
把无变化的外层 Relift 偷换成 replay 重试；同时仍要求第二轮先完成一次全新重放，
不会仅因 artifact 相同而提前拒绝可能闭合的结果。

完成 hard compare 后，下一轮 Lift 接收的是
`deterministic_gate1_hard_diff_guidance_v1` 结构化 worklist，而不是 replay code：exact
diff digest、失败/已确认类别、分组后的语义 delta、J 检查面和已注册 strategy family。
纯 provider timeout/quota/model-cooldown 没有 candidate/hard diff，不调用 Relift；这类
失败不能伪装成对 J/C/π 的语义修复。

`intentir.extraction.lifters.scrc` 中仍保留的 `max_repairs` 只控制 source-aware Lift
输出 JSON/schema 不合法时，在同一次 Lift 调用内重发完整 LiftArtifact；它修正的是
J/C/location 的 Lift 响应，从不接收或修改 replay code，不属于本目录所归档的
replay repair。

活动 `intentir.verify.gate1` 已不再导出 `Gate1Result`、`leakage_ratio`、
`masked_ablation_passes`、`leakage_threshold` 或旧 fallback policy alias。历史调用方
必须显式使用 `intentir.verify.back.legacy_v0` 下的归档模块。

旧手工 TTIR probe、strict-v5 capability driver 与历史 live 报告位于仓库
`test/back/`。它们不属于 Python package，也不得被当前 live/held-out 入口导入；
部分 Gate2 单元测试可以显式读取其中的固定 replay fixture。
