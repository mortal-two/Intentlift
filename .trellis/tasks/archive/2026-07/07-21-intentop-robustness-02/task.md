# IntentOp Robustness 02：用真实机制审计最小可组合 J

> 状态：已完成并通过全量验证（615 passed, 4 skipped）
> 优先级：P0，Architecture Reset 01 后的首个语义任务
> 流程：普通 Markdown 工作项；不运行 Trellis workflow

## 目标

不用 kernel-family named op、RuleCard 或 deterministic source recognizer，先以真实源码
验证当前 `J=A+B` 的能力边界，删除已经确定错误的通道，并把尚未定义的语义明确收窄为
unsupported。不得为了在一个任务中宣称“覆盖”而仓促冻结 logical-update/result-relation。

1. copy / reverse / transpose / gather 的 logical index mapping；
2. elementwise 与 reduction fusion 的组合 DAG；
3. dynamic indexing 与 partial write；
4. scan / online-softmax 一类 semantic recurrence；
5. scatter / histogram / duplicate update 背后的 logical update；
6. atomic、async copy、pipeline 作为 realization 时只进入 π/C；
7. 浮点重结合、并发写与 nondeterminism 的允许结果关系。

本任务是 schema robustness audit 与错误能力回收，不运行 28-kernel 正式 LLM 批次，也
不做性能优化；其产出用于约束下一阶段的 logical-map/update/result-relation 设计。

## 真实代表集

每类至少绑定一个仓库或 `/home/kingdom/phdworks/Dataset` 中的真实 source：

- xFormers `k_index_select_cat.py`：data-dependent gather + masked copy；concat 在 wrapper；
- xFormers `k_scaled_index_add.py`：unique-index logical update，不是 duplicate case；
- Liger experimental embedding backward：atomic accumulation；
- Liger MHC Sinkhorn：ordered recurrence + iteration history；`hist` 不是 histogram；
- FlashAttention Triton forward：semantic recurrence 与 fused aggregate；
- FlashAttention/CuTe 或 racecheck sample：async pipeline realization；
- 当前 layer norm：masked-load placement + reduction fusion。

可使用更小的可执行 fixture做 primitive proof，但 capability claim 必须回链到真实 source
path 和 source digest，不能只靠手写 toy JSON。

## 范围 A：清理现有 J 内部漂移

1. 删除 `IntentOp.layer_b` 第二 B 通道；B 只能存在于 canonical J boundary。
2. 删除 structured loop 的 legacy `tile_hint`；semantic trip 只保留逻辑 axis，tiling 在 C。
3. 更新 `ir/model.py` 的活动 spec 引用和 docstring，不再指向已归档 op-signatures。
4. J 的定义必须是 Gate1/Gate2/backend 共用的同一对象；不得创建 adapter/wrapper。

## 范围 B：五类组合构造审计

对 elementwise expression、logical map、aggregate、semantic recurrence、logical update 建立
一张逐样本 capability matrix，分别记录：

```text
represent / strict-parse / E_J execute / Gate1 π reference / target lower
```

不能实现的格子必须明确 fail closed；“有同名 op”不算支持。

## 范围 C：logical update 与允许结果关系的 fail-closed 边界

审计发现现有 `scatter` 还不足以定义 update，因为 J 未区分：

- key/index relation；
- overwrite 或纯语义 combiner；
- duplicate/conflict policy；
- observable ordering；
- 单值语义、tolerance relation 或允许结果集合。

因此本任务不新增半形式化 `conflict_contract`。atomic instruction、CAS、privatization、
warp/CTA 不进入 J；它们只能作为 π realization 或 C 调度。Gate2 scalar 与 direct CUDA
在 typed relation 落地前对 scatter/update 必须 unsupported。relation 的设计与 comparator
消费是下一独立提交。

## 范围 D：测试与证据

新增：

- canonical J 禁止 per-op `layer_b` 和 loop `tile_hint`；
- copy/fused DAG/dynamic gather 的 parser + 正式 scalar E_J execution；
- reverse/scan/recurrence/update 的明确 executor fail-closed；
- atomic/async/pipeline ownership test：语义在 J，机制只在 π/C；
- scalar/CUDA 对未实现 relation 的 fail-closed test；
- 真实 source inventory、digest 与 capability matrix regression。

## 验收标准

1. 活动 J 不存在第二 B 字段或 schedule-in-J compatibility field。
2. 七类真实代表均有 source-grounded decomposition 和 capability verdict。
3. copy/fusion/dynamic gather 有正式 J-only executable proof；reverse/scan/recurrence 的
   未闭合能力不得被同名 op 或历史 interpreter 掩盖。
4. logical update 的缺失项被逐项记录，atomic spelling 从 J 删除，正式执行路径 fail closed。
5. async pipeline 的普通 race-free机制不污染 J；observable order 例外有明确边界。
6. scalar/backend 对未实现 update/relation 一律 fail closed。
7. 不新增 kernel 专名 op、规则卡、composer 或 per-kernel deterministic policy。
8. 全量测试通过，任务完成后归档并独立提交。

## 实际完成结果

- 活动模型/Lift schema/Gate1 strict parser 已硬拒绝 `layer_b`、`out`、J
  `trip.tile_hint`；
- loop tile 改为由 C 的显式 lowering 参数输入；
- J access/effect 删除 `atomic/synchronize`，π 继续允许 source-neutral
  `update_atomic/transfer_async/synchronize` mechanism；
- Gate2 scalar 增加真实 consumer capability boundary；
- direct CUDA 对缺 update relation 的 scatter fail closed；
- 固定 8 份真实 source digest，并纠正 xFormers unique-index 与 Liger history 两个误判；
- 详细 matrix 写入 `doc/IntentOp_健壮性审计_2026-07-21.md`。

## 完成后的下一项工作

先完成 typed logical map/access 与 logical update/result relation，让 Gate2 从同一个 J
消费 relation；随后才冻结 28 个真实高性能 Kernel 的来源清单与 LLM 调用预算。
