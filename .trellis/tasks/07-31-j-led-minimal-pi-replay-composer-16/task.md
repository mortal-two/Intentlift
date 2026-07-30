# Task 16：J 主导的最小约束 π / Replay 内部 Composer 与 28-Kernel 最终闭合

> 状态：`P0_ACTIVE`（2026-07-31）  
> 接管 Task 15。Task 15 在完成 Compiler-observed Lift、Codex owner 与 23/28 基线后，因 flat-π Replay 作者化接口暴露根本瓶颈而归档；未完成项由本任务继续，不把归档误写成完成。

## 0. 唯一目标

在不改变 SCRC 数学表达、不泄漏 J 语义、不建立固定 lowering/composer 的前提下，重构 π 与 source-blind Replay agent，使 Codex 能像正常 coding agent 一样在单次 episode 内分步构造、编译和修正 Replay；闭合剩余 Kernel，并用原先 22 项反向检验新 composer 的普适性。

设计依据：

- `doc/IntentIR_J主导_最小约束pi与Replay内部Composer重构设计_2026-07-31.md`
- 四份 2026-07-22 SCRC 权威文档。

认证公式不变：

```text
Lift_H(Source) = (J,C,π)
Replay = W_f(J,π,MaskC(C);U_f)
SCRC_CERTIFIED iff TypeCheck(J,C,π) and Gate1 and Gate2
```

## 1. 起点与分母

当前可核验基线是 23/28：Task 15 原有 22 项，加 fresh SplitK backward：

```text
/tmp/intentir-task15-v35-splitkbwd-source-syntax-carriers-sol-xhigh-b3k5-20260730-01/
  xformers_ops_fmha_triton_splitk_kernels_splitk_reduce_varargs_backward/
  fanout-001/candidate-03/report.json
```

该报告为 `accepted_attempt=2`、`main_status=SCRC_CERTIFIED`。

剩余 active queue：

1. Llama4 RoPE；
2. FlashAttention forward；
3. FlashAttention one-column backward；
4. FlashAttention full backward；
5. SplitK forward。

若后续证据证明某个已有 accepted artifact 的模型级输入、参数化或证书版本无效，该项明确退回 active queue；不靠旧计数掩盖。

2026-07-31 的当前增量证据把 one-column backward 从 Gate2 blocker 中移出：同一 legacy-v11
artifact 的 fresh Codex Replay 已 `replay_closed`，Source–Replay 三个 output 误差均为 0；外部
`gate2-q.v2` 按每个 output 的 closed J-only f16-contraction dependency 选择冻结 mixed-precision
profile 后三边通过：

```text
/tmp/intentir-task16-composer-pimin-onecol-fixed-sol-xhigh-20260731-01/attempt-01/gate1.json
/tmp/intentir-task16-onecol-gate2-qv2-recheck-20260731-01.json
```

因此可复核功能基线为 24/28；但它仍须按 P6 生成 current artifact.v12 evidence，不能把 legacy
artifact 的独立 Gate2 recheck 冒充最终 28/28 交付。

## 2. P0：边界清理

- [x] 删除/停用 flat `AUTHORING_INDEX.json` 与重复自然语言 ledger；
- [x] 删除后置可写 reviewer/强制第二次 semantic self-review；
- [x] Replay prompt 缩为输入路径、工具入口、禁止信息和提交条件；
- [ ] 逐项区分 Task 15 未提交改动：保留独立正确的 compiler observation/typed relation，剥离旧 ledger 和无关 Gate2 retry 改造；
- [x] 不新增或恢复 `python/tests`，只运行真实 Kernel closure。

## 3. P1：π_min

- [x] 建立版本化、exact round-trip 的 Replay-only compact π wire；canonical artifact π 不变；
- [ ] 只保留 multiplicity、region/iteration/carrier/route、terminal control、program axis、materialization、sharing/order 与 representation attachment 等必要 realization 关系；
- [ ] operator、formula、predicate、fill、Source/API、C value 和 target recipe 在 schema/parser/TypeCheck 中 fail closed；
- [ ] 只有可由 typed graph 唯一恢复的 identity/default 才允许省略；
- [x] compact codec 只移除重复 JSON object keys，不能自动发明或删除 artifact 字段；
- [x] 旧 accepted π 原样复用，不静默重写旧证书，也不因 packet serialization relift。

## 4. P2：Replay 内部 Composer

- [x] 每个 formal round 创建 fresh source-blind workspace；
- [x] 给 Codex 通用 syntax-only editor：fragment、stable local、route placeholder、render、raw escape；
- [x] 保留 compile、check、inspect-own-TTIR、checkpoint、局部 patch 和 atomic submit；
- [x] composer scratch 仅存在于本 episode，不能进入 artifact/证书或跨轮复用；
- [x] host 不提供 J-op/π-relation 到 target code 的映射、kernel helper 或模板库；
- [x] prompt 文件化、按需读取，禁止反复内嵌几十 KB J/π/手册；
- [x] 资源边界用 wall-time/context/formal round，不用极小工具调用上限打断 coding loop。

真实 legacy-v11 Flash-forward pilot 已证明 Codex 在一个 episode 内作者化 60+ occurrence fragments，
`composer_status/render` 对显式 slot 与 empty-slot dependency routes 达到 PASS，并将完整源码送入正式
Gate1；旧 artifact 仍因 dataflow/op mismatch 未闭合，故该证据只验收 composer 机制，不冒充
Kernel closure：

```text
/tmp/intentir-task16-routecomposer-deps-flashfwd-terra-high-20260731-01/
```

fresh artifact.v12 的 Llama/SplitK 双核 pilot 又暴露并修正两项通用 agent 接口问题：

- Replay startup 原先只前置 J 的 effects/broadcasts，physical
  `boundary.reductions.tile_axis`、access predicate/fill 等既有 J/B authority 留在完整 packet 中，
  导致 Replay 偶尔误用 logical reduction axis。现在 startup 以有界、exact、可续查的
  `boundary_relations` 投影直接呈现这些 authored rows；它不读取 Source、不产生 target recipe；
- 一个 SplitK Replay 在 20 分钟截止前已写出 6.6 KiB 完整 source，却因尚未完成 submit transaction
  被归为 infrastructure failure。Codex Replay 现以三个普通 provider window（xhigh 两个长 window）
  为默认 episode；截止点只在 LLM 原样 bytes 已通过 ingress 与 synthetic compile 双检查时保留，
  否则仍 fail closed。

同一 pilot 也确认 `reduce.attrs.axes` 是 logical axis，而
`boundary.reductions.tile_axis` 是 Source-local physical axis，两者不得被错误加上等值 checker。

## 5. P3：J 主导与无泄漏验收

- [x] Replay request 严格只有 `J+π+MaskC(C)+U_f`；
- [x] production backend 仍只读 `J+C`；
- [ ] Mask-J 不作为主证书，但至少在代表 Kernel 上验证 π 不能独立恢复语义；
- [ ] 做一次 J-authority mutation：相同 realization shell 下改变闭合 J，fresh Replay 必须跟随 J；
- [x] 最终 Gate2 仍为 Source–Replay、Source–J、Replay–J 三条直接比较。

## 6. P4：剩余五项闭合循环

优先顺序不是逐项死磕，而是按能力面横向覆盖：

1. Flash full backward：recurrence、multi-output、conditional post-state；
2. Flash forward：online recurrence 与 program-id/address realization；
3. one-column backward：terminal CFG/observer；
4. SplitK forward：program-id → div/rem/mul/add → addptr 与跨实例 realization；
5. Llama4 RoPE：conditional path、shape/value relation。

执行规则：

- 同时保持至少两个 Kernel 的真实 Lift/Replay 在推进；
- 每个 Kernel 可有 3 个独立 batch，winner-first，不等待败者；
- Gate1 失败先判定 J 缺语义还是 Replay realization 错：前者反馈给同一 Lift owner relift，后者只启动 fresh Replay；
- 最多 3 个 formal Gate1 round 后若仍失败，先提升共性表达/agent能力，再继续，不跨 Kernel 传递语义；
- Terra 优先用于快速横向，注意力/闭合不足时升级 Sol high/xhigh/max；模型升级不是替代根因修复。

## 7. P5：原 22 项重放

剩余五项若能稳定、明显更轻地闭合，原先 22 项全部保留其 accepted `(J,C,π)` 或按新 π version fresh Lift，并统一经过新 Replay composer：

- 不从历史 Replay source、Gate diff 或 winner 学习；
- 统计 packet bytes、首次可编译时间、compile/tool 次数、Gate1 round 与 wall-time；
- 删除只为某个 Kernel 存在的 composer API；
- 若旧 π 无法无损迁移，则 fresh Lift，不伪造新 π 证书。

## 8. P6：交付与完成条件

- [ ] 28/28 均有 current-version、模型级输入正确、参数化闭合的 accepted `(J,C,π)`；
- [ ] 28/28 fresh Replay、Gate1、Gate2 三边成立；
- [ ] π_min 与 internal composer 的 no-leak/J-authority evidence 完整；
- [ ] spec、四份 SCRC 文档的相关段落、方法文档与实现一致；
- [ ] 没有 tests、hash/banner gate、RuleCard、per-kernel composer 或 production π consumer；
- [ ] 提交后 worktree clean。

## 9. 禁止退化

- 不让固定代码从 J 或 π 决定 target semantic code；
- 不把 agent-local composer 变成共享模板库；
- 不把 π 压缩成信息不足、只能靠 Replay 猜测的 token 节省；
- 不用 TypeCheck PASS、旧证书、更多 batch 或更大 token 冒充闭合；
- 不用全量无关审查和 mock tests 替代真实 SCRC；
- 不在本任务实现 production backend search。

最终裁决：

> **不是让 LLM 手搓一整份扁平 π 对应的目标源码，也不是让 host 预写 lowering；而是由 J 掌握语义、π_min 约束 realization、Codex 在 fresh Replay episode 内用通用构造工具自己完成程序。**
