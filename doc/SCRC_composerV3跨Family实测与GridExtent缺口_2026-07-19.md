# SCRC composer-v3 跨 Family 实测与 grid-derived extent 缺口

日期：2026-07-19
模型：`gpt-5.6-sol`
effort：`xhigh`
held-out identity：schema 32
TTIR fingerprint：`triton-ttir-3.6-gate1-v3`
composer：`deterministic_composer_v3`

本轮没有使用 Trellis task/workflow。仅更新项目代码、测试、本文档和 verification spec。
π topology 没有优化，也没有因为单个样本出现拓扑泄漏警告。

## 1. 本轮目标

本轮沿既定主线完成：

1. 修复 `fg_eye` 暴露的同一语义节点多条 broadcast 声明无法确定性组装的问题；
2. 保持 J 承载全部 broadcast 语义，π 仍只是一份节点级确定性施工协议；
3. 用真实 `fg_eye` 扩展二维 iota/broadcast/select family；
4. 用真实 `fg_trace` 检验 loop + diagonal stride access + reduce 的 Lift 上限；
5. 用真实 `fg_embedding` 检验间接 gather，并尝试扩大 fixed-C specialization；
6. 对新发现的问题做机械归因，不把 Lift、Gate1、Gate2 的失败混为一类。

## 2. deterministic composer v3

### 2.1 原问题

`Boundary.broadcasts` 的一条记录表达：

```text
(node_ref, operand, axes)
```

多输入节点可以合法拥有多条记录。例如 eye 的比较节点需要分别描述 row operand 和
column operand 的 rank insertion。但 strict π 中的步骤是：

```text
realize_broadcast(node_ref)
```

它是节点级施工动作，没有 operand/axes 槽。旧 composer 为每条 BroadcastSpec 生成一步，
因此同一节点会产生完全相同的 step marker 并被 duplicate-step 检查拒绝。

### 2.2 v3 规则

v3 按 `node_ref` 将不同 BroadcastSpec 合成一个 work item：

- `facts.broadcasts[]` 保留每个 `(operand, axes)` 配对；
- 成员按 canonical JSON 排序，声明顺序不影响 canonical identity；
- 每个成员都逐项满足 RuleCard 的 `operand/axes` 必需字段；
- `j_digest` 哈希完整成员集合；
- operand 生产节点的依赖取并集；
- 一个节点最终只生成一条 `realize_broadcast(node_ref)`；
- rule leaf 和 node_ref leaf provenance 同时绑定该节点的全部 broadcast JSON pointer；
- 完全相同的重复声明 fail-closed，不做静默去重。

因此，J 中第二个 operand 或 axes 发生变化时 `j_digest` 必须变化；若依赖拓扑没有变化，
π 仍可保持同一节点级机制。若 operand 生产依赖变化，π 的确定性顺序也可以随之变化。

### 2.3 版本与恢复边界

- `COMPOSER_VERSION`：`deterministic_composer_v2` → `v3`；
- held-out identity：schema 31 → schema 32；
- identity 显式记录 `composer_version`；
- 旧报告不能按 schema 32 自动 resume。

## 3. composer 回归

新增/收紧测试覆盖：

- 同节点两个不同 broadcast 只生成一个节点级 step；
- iota producer 必须先于 broadcast，broadcast 必须先于 comparison；
- rule/node_ref provenance 覆盖全部声明；
- 声明顺序交换不改变 `j_digest` 和 canonical π；
- 修改第二项 axes 改变 `j_digest`，但不强制改变同一机制的 plan digest；
- 修改第二项 operand 改变 `j_digest`，并允许依赖顺序改变 plan digest；
- 完全重复声明确定性拒绝。

聚焦 composer/held-out 回归：`63 passed`。
Gate1/Gate2/held-out 相关扩大回归：`228 passed`。

完整 `python/tests`：

```text
701 passed, 3 skipped, 3 failed
```

三个失败均为既有非本轮问题：

- 两个 xFormers split-k backward 用例在全套运行中触发 Triton dynamic binder
  `BLOCK_SIZE` 重复绑定；隔离复跑 `2 passed`；
- 旧 universe `count_nonzero` committed manifest 记录 `ok`，当前 saved replay 稳定为
  `lowering_fail`；隔离复跑仍失败，属于历史 manifest 漂移。

后续 Lower2C boundary-broadcast v4 增加 12 个测试后，完整结果更新为
`713 passed, 3 skipped, 3 failed`，失败集合不变；见
`doc/SCRC_Lower2C_BoundaryBroadcast_v4_2026-07-19.md`。

## 4. `fg_eye` 最终真实实验

报告：

```text
/tmp/scrc_schema32_final_eye_20260719/fg_eye.json
```

真实 response：

- Lift：`resp_0ee3e835c3be4360016a5bc61b4e288199951e445927f4b648`；
- normal replay：`resp_047a63e3ca185a1b016a5bc6db1fd4819ba2efc801b4eae188`；
- masked-J replay：`resp_0b9c16eb806eb904016a5bc74db108819b8c5c72733fafd489`。

### 4.1 Gate1

- 首次 Lift 一轮生成合法 artifact；
- normal replay 一轮编译成功并 hard-close；
- fixed-C family 四个 profile 全部 hard-close：
  - `one_by_one`；
  - `block_edge`；
  - `both_axes_tail`；
  - `rectangular_nondivisible`；
- masked-J family 证据完整，四个 profile 均为确定性 hard mismatch；
- 最终 Gate1：`replay_closed`；
- 没有 Relift，因为 normal/family 没有 hard diff，不应人为制造 Relift。

最终 π 中关键顺序为：

```text
realize_iota(n0)
realize_iota(n1)
realize_broadcast(n2)
realize_elementwise(n2)
realize_broadcast(n5)
realize_select(n5)
```

两个 broadcast step 分别绑定 J 中两条 BroadcastSpec 的 rule/ref provenance。

### 4.2 π 消融

- 删除 `materialize_mask_tile(a0)`：`canonicality_rejected`，支持 canonical π 必要性；
- `materialize_index_tile → materialize_pointer_tile`：writer 出现 off-target semantic drift，
  该次消融不构成干净的机制因果证据。

### 4.3 Gate2

schema-32 fresh 当时独立运行 12 个 case，得到 6 个 `lowering_diverged` 和 6 个
`execution_failed`。后续审计确认 J 的 `Boundary.broadcasts` 已有完整语义，只是 Lower2C
没有消费该层。

`boundary_broadcast_v4` 接入现有 axes 后，使用同一份保存真实 artifact/replay 完成无模型
复验，12/12 case 均为 `gate2_pass`。Gate1 仍为 `replay_closed`，四个 fixed-C profiles
和 masked-J family 结论不变。详见
`doc/SCRC_Lower2C_BoundaryBroadcast_v4_2026-07-19.md`。

## 5. `fg_trace` 真实上限实验

报告：

```text
/tmp/scrc_schema32_fresh_trace_20260719/fg_trace.json
```

source preflight 通过。三次 Lift 均未形成合法 artifact：

1. `intent_op:triu` 无 RuleCard；
2. J tensor 使用不存在于 compiler scalar arguments 的 shape symbol `M`；
3. `inp_ptr` 的 tensor_ref rank 与二维 compiler view 不一致。

因此：

- 状态是 `lift_failed`，不是 Gate1 no-closure；
- 没有 replay、Gate1 diff、Relift 或 Gate2；
- 当前证据更支持 Lift 输出合规性失败，而不是已证明 J schema 无法表达；
- 不为了追求“一次成型率”修改 generic initial Lift 的语义目标。

为后续合法实验保留的 fixed-C profiles 已改为同一静态输入 shape 下的：

- `num_diag=1`；
- `num_diag=1024`；
- `num_diag=1025`。

三者 source 均已真实编译通过。它们分别覆盖短 trip、恰好一个 block 和跨 block 尾部，
不会非法修改 J 的静态 tensor extent。

## 6. `fg_embedding` 与 grid-derived extent 缺口

报告：

```text
/tmp/scrc_schema32_fresh_embedding_20260719/fg_embedding.json
```

结果：

- source preflight 通过；
- 第一次 Lift 因 J 使用 shape symbol `K`、但 `K` 不在 scalar arguments 中而修复；
- 第二次 Lift 生成静态 batch extent `17` 的合法 artifact；
- normal replay 编译成功且 hard-close；
- 尝试 `K=1/17/33` fixed-C family 时，setup 以
  `profile 'single_index' changes static J extent 17 to 1` fail-closed；
- 最终状态 `lift_rejected`，不是 normal TTIR 不一致。

### 6.1 根因

该 Triton kernel 的 signature 没有运行时 `K` 参数。program 数由 launch grid 决定：

```text
grid = (K,)
```

当前 J tensor shape symbol 只允许绑定 compiler scalar argument，不能表达：

```text
out.shape[0] == launch_grid[0]
in.shape[0]  == launch_grid[0]
```

现有 `PortableScope(kind=program_instance)` 只描述定义、capture、effect 和 region lifetime，
不提供 launch/grid extent 到 tensor shape 的符号绑定。因此这不是简单测试配置问题，而是
“grid-derived semantic extent”尚无一等表示。

### 6.2 当前处理

- 已撤回 `fg_embedding` 的非法 1/17/33 shape profiles；
- 保留原有随机索引和边界索引 value profiles；
- 不把单一 specialization 伪装成多 specialization；
- 不私自增加新 schema 字段。

### 6.3 计划外候选改动，等待审查

若要真正覆盖该类 kernel，建议另行审查一种最小、frontend-neutral 的 extent binding，
例如让 interface/J 声明某个符号来源于 `grid[axis]`，再允许 tensor shape 使用该符号。
该改动会影响 schema、Lift wire format、specialization binder、composer digest 和 Lower2C，
属于计划外新特性；本轮只报告必要性，不实现。

## 7. 结论

1. TTIR scope-v3、精确 stagnation 和 hard-diff-guided Relift 主线保持不变；
2. composer-v3 已解决同节点多 broadcast 的确定性组装缺口，并由 `fg_eye` fresh 实验验证；
3. `fg_eye` 扩展了二维索引/broadcast/select family，Gate1 已正式闭合；
4. `fg_trace` 暴露的是 Gate1 前 Lift 合规性上限，不能拿来评价 Gate1 false reject；
5. `fg_embedding` 暴露了真实的 grid-derived extent 表达缺口，当前不能合法验证多个
   program-count specialization；
6. Gate1 与 Gate2 继续保持正交；`fg_eye` 已从 Gate1 closed + Gate2 incomplete 推进为
   Gate1 closed + Gate2 pass，两个 verdict 仍分别保存；
7. π topology 暂不优化；当前没有出现跨 family 大量 topology leakage。
