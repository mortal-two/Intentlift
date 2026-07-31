# Task 20：J-only Replay 与 J/C Structural Gate1

> 状态：`IMPLEMENTATION_ACTIVE`（2026-07-31）
> Task 11 继续是不停止的 28-Kernel 闭合主线；本任务先纠正 Replay/Gate1 内部边界，再恢复闭合。

## 0. 唯一目标

```text
Lift(Source) -> (J,C,pi)
TypeCheck(J,C,pi)
ReplayAgent(J,MaskC(C);U_f) -> R
Gate1: normalized J/C structural closure at fixed f,p,c0
Gate2: Source-Replay / Source-J / Replay-J
```

`J+C` 是提升、复用与 production 交付对象。π 只是 Gate1 certification-only normalization witness；
不进入 Replay，Gate1 不证明 π 正确/完整，最终结论只谈 J/C。

规范依据：

- `doc/method/IntentIR_J-only_Replay与JC结构Gate1_2026-07-31.md`；
- 四份 2026-07-22 SCRC 权威文档；
- Figure 3 的既有外部流程。

## 1. 已完成的设计纠偏

- [x] 归档 Task 19 Composer/Target-Normal prototype；
- [x] 归档旧 π-centered `CanonPair/coverage/used-pi` Task 20 草案；
- [x] 冻结 `Replay(J,MaskC(C);U_f)`；
- [x] 定义 π-blind raw observer、固定单臂 `N^pi`、Gate1 equality 与 J/C-only 结论；
- [x] 明确现有 π wire 第一阶段不改，不新增 normalization/allowed-diff 字段；
- [x] 不新增/恢复 `python/tests`，只用真实 Kernel closure 验证。

## 2. P0：文档与活动协议收敛

- [x] 四份权威 SCRC 文档删除 Replay 消费 π 的活动公式与义务；
- [x] `.trellis/spec` 统一为 J-only Replay / J/C Structural Gate1；
- [x] π grammar 中的 “Replay must preserve/materialize” 改为 certification witness denotation；
- [ ] Figure 3 保持拓扑，只在论文阶段改 Replay 输入与 Gate1 内部标签；
- [x] production backend 继续只消费 accepted J+C。

## 3. P1：删除 π→Replay 通道

- [x] Replay request/workspace/cache key/continuation/tool allowlist 删除 canonical π、`pi.intent`、π query、
  obligation ledger 与任何 π-derived view；
- [x] `S_J=M(J)` 继续是只读 J-only 编译视图，不生成 target skeleton；
- [x] Replay compiler diagnostics 只能来自自己的 source/compile，不含 Source/Gate1 diff；
- [x] 历史 π-consuming view/nested wire 显式归档，旧 packet/checker 只作历史解释，不能作为新路径 evidence。

## 4. P2：最小 J/C Structural Gate1

- [x] raw `Compile/Observe/Canon` 不读取 π；
- [x] 实现 profile-scoped、确定性、单臂、幂等、fail-closed 的 `N^pi(observation)`；
- [x] 固定 `H^(J,C)` hard projection；候选 π 不得选择 rule/scope/erasable fields；
- [x] 实现候选外 `ProfileOK` 发布合同：H覆盖所有当前 profile 可观察的 J/C-owned categories，未知
  atom 默认 hard；每条 N_pi rule 具有机械 hard-preservation check 或独立 rule audit；
- [x] 第一纵切仅支持 Triton program-axis / mixed-radix linearization；
- [ ] simple placement/materialization 只在真实 residual 需要且 hard preservation 可局部判定时加入；
- [x] carrier/control/sharing/order/numeric/update 默认不 normalize；
- [x] 不实现 `Diff subset pi` acceptance、π completeness/minimality、used/unused π 认证或 per-kernel matcher。
- [x] 首轮分类：`program_axes/program_linearizations=normalizer_required`；其余 relation 当前仅
  `typed_anchor_only` 或 `legacy_replay_only`，均无 Gate1 放宽权限；仅服务旧
  Replay construction 的字段与 totality obligation 在新 wire 版本中 optionalize/retire，减少 Lift
  序列化负担，不静默重写旧 artifact；
- [ ] 记录 candidate 中 J/C/π 的 bytes、Lift time-to-TypeCheck 与 relation rejection，验证简化确实减负。

## 5. P3：失败路由与 freshness

- [x] Replay compile/API 错误只反馈 Replay Agent，在同一 source-blind episode 修复；
- [x] Gate1 完整 diff/有限 frontier 只返回同 branch source-aware Lift owner；
- [ ] owner 区分 Replay 忠实性、J/C hard residual、合法 normalization 缺口与 observer unsupported；
- [ ] π-only 修改复用 exact R 并重跑 TypeCheck/Gate1；
- [ ] J 或 MaskC structure/attachment 修改触发 fresh Replay；
- [ ] c0-only 变更复用源码但重新 Bind/Compile/Observe/Gate；profile 变更必须新开实验并重算 Gate1 evidence。

## 6. P4：真实纵切与闭合恢复

- [ ] 一个已闭合 Kernel：证明迁移前后 Gate2 不退化；
- [ ] 一个 program-linearization Kernel：证明最小 `N^pi` 能去除合法 realization noise；
- [ ] 一个 loop/carrier/control Kernel：证明 unsupported/fail-closed，不靠 π 擦掉 J/C residual；
- [ ] 恢复 Task 11 横向闭合：至少两个 Kernel 同时运行、同 Kernel 多 branch、首个 Gate1 winner；
- [ ] Gate1 closure 后 Gate2 失败先按 Source-Replay / Source-J / Replay-J 三边分流，不盲目 relift；
- [ ] 稳定后 fresh 运行 28 项，报告 current-version closure，不继承旧 evidence。

## 7. P5：多前端与实验

- [ ] Triton 纵切稳定后，CuTe/TileLang/CUDA 分别定义自己的 raw observer、J/C projection 与最小
  unary normalizer；只共享 wire/ownership，不转移 PASS；
- [ ] 禁止 Kernel/source-path/benchmark-specific normalization；
- [ ] 报告 Gate1 提前拒绝率、Gate1/Gate2 成本、feedback 对 closure@k 的收益、hard mutation
  sensitivity、合法 realization 误拒率和提前拦截 crash 数；
- [ ] π applied/unsupported 只作为调试 telemetry，不作为主实验或证书结论。

## 8. 完成条件

- [x] Replay runtime、spec、活动方法文档均不读取 π；
- [x] Gate1 的主定义只有 `Phi`、`N^pi`、definedness/equality 与固定 J/C hard-preservation；
- [x] π 不出现在 Gate1 的结论中，也没有 wildcard/allowed-diff/comparator 权限；
- [ ] 至少 Triton 的真实纵切通过 Gate1/Gate2；其他 frontend 准确报告 capability；
- [ ] 28/28 current-version SCRC 完成，或剩余项给出真实 observer/J/Replay capability 边界；
- [ ] 不以 mock/test/hash/banner 替代真实闭合；提交后工作区干净。

最终裁决：

> Replay 只从 J 构造；π 只在 Gate1 内部消除冻结、窄且可保守识别的 realization 噪声。
> Gate1 的结论是 J/C structural closure，Gate2 才是声明执行域上的三边行为锚点。
