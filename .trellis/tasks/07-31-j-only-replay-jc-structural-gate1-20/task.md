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
- [x] active π wire 压成 versioned normalization witness，只保留 closed `program_axes`；
  旧 realization graph 只读兼容；当前 profile 在无 sound anchor matcher 时保持它 hard；
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
- [x] Replay 自检默认只调用一次 bounded `jc` TTIR 摘要；不得枚举全部 fingerprint，只有具体问题
  才允许一次 targeted raw/find/observer 查询；
- [x] Codex Lift 写出第一份 complete JSON 后立即 submit/check，禁止先做全量 schema/TTIR 自审；
  同一 owner 只按 closed diagnostics 局部 patch；
- [x] closed submit 接受 exact candidate/Replay 后，由 host 恢复并独立复核同一 checkpoint，跳过无证明
  意义的结尾 prose；模型仍负责全部语义与源码，Gate/接受集合不变；
- [x] SourceReconcile 只增加三类保守 one-bit 必要条件：masked access 整体缺失、含非线性算子的
  multi-state recurrence 整体缺失、terminal effect control 整体缺失。它不生成 predicate、trip、carry、
  update、branch 或 π relation，正式 Gate1 仍检查完整结构；
- [x] `source-brief.v3` 在 10KB 固定预算内一次给出完整紧凑 Harness ABI、匿名顶层函数与
  compiler-observed control/time index；只压缩已声明的 kind/mode/dtype/rank/shape contract，并只给
  loop carrier/yield 数、body op/effect kind 与 terminal 两臂 effect 数。`logical_ref`、公式及
  J/C/π 归属仍完全由 Lift 决定，精确字段可按 ordinal 回查；
- [x] 历史 π-consuming view/nested wire 显式归档，旧 packet/checker 只作历史解释，不能作为新路径 evidence。

## 4. P2：最小 J/C Structural Gate1

- [x] raw `Compile/Observe/Canon` 不读取 π；
- [x] 实现 profile-scoped、确定性、单臂、幂等、fail-closed 的 `N^pi(observation)`；
- [x] 固定 `H^(J,C)` hard projection；候选 π 不得选择 rule/scope/erasable fields；
- [x] 实现候选外 `ProfileOK` 发布合同：H覆盖所有当前 profile 可观察的 J/C-owned categories，未知
  atom 默认 hard；每条 N_pi rule 具有机械 hard-preservation check 或独立 rule audit；
- [x] 撤销不 sound 的 Triton `x/y/z -> x` label quotient；`jc-normalizer.v2`对
  `program_axes` 保持 hard，不会掩盖 domain–axis 交换；
- [ ] 只在 physical occurrence 可唯一回指 J domain 时，实现 Kernel-无关的
  program-axis unary normalization 并升新 profile；
- [ ] simple placement/materialization 只在真实 residual 需要且 hard preservation 可局部判定时加入；
- [x] carrier/control/sharing/order/numeric/update 默认不 normalize；
- [x] 不实现 `Diff subset pi` acceptance、π completeness/minimality、used/unused π 认证或 per-kernel matcher。
- [x] 尚无一元实现的 `program_linearizations` 与其余旧 relation 全部退出 active
  schema，不再让 Lift 序列化不可消费的 graph，也不静默重写旧 artifact；
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

当前真实 pilot 已确认：compact π 候选可以稳定通过 Decode/TypeCheck；旧 exact transfer-count
SourceReconcile 会把 compact π 误读成零 occurrence，现已删除。Flash backward 与 Split-K 的首轮
Gate1 没有被 π 放过：前者暴露 terminal control/非线性 recurrence 整体遗漏，后者暴露五状态 online
recurrence 被错误压成 global max/sum。上述 one-bit preflight 的目的正是把这类真实硬错误提前到首次
submit，而不是改变 Gate1 verdict。

2026-08-01 的真实 Liger/Flash 纵切还确认了两件事：

- J-only Replay 可以稳定到达 TypeCheck 和 Gate1，但主要 residual 仍在
  op/dataflow/predicate-effect，不能用 π 擦除；
- 旧 program-axis label quotient 可能掩盖 swapped domain–axis binding，已改为 hard identity
  normalizer 并升级 Gate1/profile evidence 版本。

`S_J` 现在增加只由 exact J 决定的 `boundary_ownership` 索引：它联合已有
domain/access/effect/output 引用并暴露歧义，不选择 target schedule、program axis 或
Source recipe。Replay 单回合超时时，若 exact checkpoint 已通过同一独立 host
recheck，可立即接受该 bytes，不再消耗剩余 episode 时间；Gate1 仍在 episode 外完整执行。

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
