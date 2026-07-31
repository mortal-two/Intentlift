# Task 17：Typed Skeleton / LLM Target Binding Composer 与 28-Kernel 最终闭合

> 状态：`ARCHIVED_SUPERSEDED_BY_TASK18`（2026-07-31）
> 接管 Task 16。Task 16 已证明 v2 grouped composer 能消费 route、但仍要求 LLM 手写完整目标源码，
> Flash-forward 的 dataflow frontier 仅由 176 降至 136，未形成数量级提升。
> 后续 file-native block pilot 虽可编译，但正式 Gate1 dataflow 仍有 232 项差异；因此
> hole/binding/structural-renderer 路线停止，完整交接给 Task 18。

## 0. 唯一目标

以 `typed skeleton + episode-local LLM target binding + fixed structural renderer` 重构 Replay，同时将
π 收缩为 Lift 易于作者化的最小结构化 realization witness；保持 J 主导、source-blind、无语义泄漏，
完成 current-version 28/28 SCRC closure，并 fresh Replay 原 22 项。

设计依据：

- `doc/method/IntentIR_TypedSkeleton_LLMTargetBinding与最小piComposer设计_2026-07-31.md`
- 四份 2026-07-22 SCRC 权威文档。

## 1. 起点

- 严格功能基线：24/28 `SCRC_CERTIFIED`；
- active：Llama4 RoPE、SplitK forward、FlashAttention forward、FlashAttention full backward；
- TypeCheck、Replay compile 或 Gate1 接近通过都不计闭合；
- Task 16 的 v1/v2 composer 仅作为迁移对照，不再作为新主路径继续补丁。

## 2. P0：边界与协议

- [x] 建立 Task 17 与平衡设计文档；
- [x] 将 Task 16 标记为 superseded handoff；
- [ ] 在 spec/四份 SCRC 文档中区分 `Skeleton`、episode binding `B` 与最终 Replay；
- [ ] 保持认证公式、Gate1/Gate2 与 production `J+C` 输入不变；
- [ ] 禁止新增/恢复 `python/tests`，只用真实 closure artifact 验收。

## 3. P1：π_min 与 Lift 减负

- [x] 定义严格双射的 nested region/occurrence wire，减少 repeated id/region 字段；
- [ ] 唯一同-region 1:1 J operand route 由 closed Decode 无损展开；
- [ ] multiplicity、cross-region、branch、carrier、sharing、terminal、materialization 显式保留；
- [x] nested wire 的 canonical `r*/i*` 只按连续 ordinal 机械恢复，不补语义；
- [ ] parser/TypeCheck 对任何歧义 fail closed；
- [ ] π schema 拒绝 operator/formula/predicate/target API/Source/Gate/kernel recipe；
- [ ] 现有 accepted π 可精确迁移则迁移，否则 fresh Lift，不伪造证书。

Flash-forward 实测：canonical π compact JSON 9064 chars → nested wire 4186 chars，Decode
逐对象相等；这只是 wire/agent-context 减负，尚未把唯一 route 省略接入 Lift ingress。

## 4. P2：Typed Skeleton / file-native block scaffold

- [x] 建立 current J/π 的 target-neutral normalized skeleton v1；
- [x] 每个 semantic node、boundary relation、physical occurrence、region、route 与 carrier 可引用追溯；
- [x] 建立按 lexical region 的 exact on-demand workset，避免把跨表 join 全塞入 agent 上下文；
- [x] 将真实 22-node/78-occurrence packet 从重复 join 的约 100 KB 规范化到约 34 KB；
- [ ] 自动管理所有 control 形态的 region tree、scope、dominance、SSA、stable locals、order/sharing；
- [x] 对 semantic 与 realization target choice 产生独立 typed holes；
- [ ] skeleton 不包含 target API、Source hint、Gate diff 或 semantic repair；
- [ ] 每个最终 semantic target node 必须带 J provenance。

## 5. P3：LLM Target Binding

- [x] 建立 closed binding v1 的 generic target-AST、template/application 与全 hole coverage 校验基础；
- [x] 由 fixed manifest 将连续 typed J nodes 与 π refs 绑定为少量 region blocks；
- [x] computation/address/mask/load-store/layout/cast 全由 agent 在普通 `.source` 文件中明确作者化；
- [x] 新 op 直接使用 episode-local raw block，不要求修改固定 semantic compiler；
- [x] manifest/path/provenance 改动、缺失或空 block 均 fail closed；
- [x] binding/helper 不跨 formal round 保存，不进入 artifact 或证书。

## 6. P4：Structural Renderer 与 Agent Loop

- [x] renderer 固定生成 signature alias、program coordinate、region、loop 与 carrier target plumbing；
- [ ] 不内置 `J op -> target` 或 `π relation -> statement` 表；
- [x] target source map 回指 block/J node/π relation/region；
- [ ] compile/check 诊断回到具体 binding，支持局部修正；
- [x] Codex episode 变为 edit `.source` block → render → compile/check → patch block；
- [ ] wall-time 管理，不按普通工具调用次数切断 coding loop；
- [ ] v1/v2 artifacts 可读，但 fresh Replay 默认只走新协议。

当前真实 Flash-forward scaffold：3 blocks（pre-loop 6 J nodes、loop-body 22、post-loop 9），
固定生成 2 个 program axes、1 个 mixed-radix linearization、1 个 loop 与 3 个 carriers；首次
file-native fresh Replay 正在运行，尚不计新增闭合。

## 7. P5：真实横向闭合

- [ ] Flash forward 同一好 J 做 v2/vNext 对照：binding bytes、agent tokens、首次可编译时间、Gate1 frontier；
- [ ] Llama4 RoPE；
- [ ] SplitK forward；
- [ ] FlashAttention forward；
- [ ] FlashAttention full backward；
- [ ] 每项 Gate1 失败先区分 J、π、binding、renderer 或 observer，不用盲目 relift；
- [ ] 同时保持至少两个 Kernel 推进，winner-first，必要时 Sol high/xhigh/max。

## 8. P6：原 22 项与消融

- [ ] 原 22 项全部 fresh Replay；
- [ ] Mask-J：π+composer 不能独立形成 semantic target program；
- [ ] J mutation：相同 realization shell 下 fresh binding/Replay 跟随新 J；
- [ ] 删除仅为单一 Kernel 存在的 matcher、AST node 或 renderer 分支；
- [ ] 记录 hole/binding 数、occurrence 数、source bytes、agent tokens、tool/compile 次数与 wall-time。

## 9. 完成条件

- [ ] 28/28 current-version、模型级输入正确、参数化 `SCRC_CERTIFIED`；
- [ ] 28/28 fresh Replay、Gate1、Gate2 三边成立；
- [ ] π_min/Lift 负担明显下降，binding 规模不随 occurrence 数线性膨胀；
- [ ] no-leak/J-authority evidence 完整；
- [ ] 无 tests、hash/banner gate、per-kernel recipe、历史 winner 或 production π consumer；
- [ ] spec/权威文档/方法文档/实现一致；
- [ ] 提交后 worktree clean。

最终裁决：

> **Composer 是没有 instruction selection 的结构编译器；fresh Replay agent 作者化 instruction
> selection，π 只约束非唯一 realization，J 继续唯一拥有计算语义。**
