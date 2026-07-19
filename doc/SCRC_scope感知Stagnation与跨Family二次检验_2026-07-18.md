# SCRC scope 感知、Stagnation 与跨 Family 二次检验

日期：2026-07-18
模型：`gpt-5.6-sol/xhigh`
活动协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`
TTIR fingerprint：`triton-ttir-3.6-gate1-v3`

本轮没有运行 Trellis task/workflow；只参考并更新了 verification spec。工作范围严格
限定为：TTIR SSA scope 感知、既定 deterministic stagnation、第一次 Gate1 差分对
下一次 Lift 的语义指导、现有样本二次检验，以及非 reduce 真实 family 扩样。本轮没有
修改 π 拓扑模型；只有未来批量出现 masked-J/full-π hard closure 时才需要重新报警。

## 1. TTIR SSA scope 感知补强

### 1.1 旧缺口

`triton-ttir-3.6-gate1-v2` 已经使用 lexical binding stack 解析 `scf.for` 和
`tt.reduce` region，因此 sibling region 可以合法复用局部 SSA 名。但两条 bounded
normalization 仍在 parser 之前按全函数文本处理：

- `x + 0 -> x`；
- `1 / sqrt(x) -> rsqrt(x)`。

这会产生名称相关的 false reject：两个语义相同的 TTIR，只因一个版本在 sibling
region 复用了 `%same`，另一个版本使用 `%first_same/%second_same`，就可能得到不同
normalization 结果。

### 1.2 当前实现

v3 将 bounded normalization 移到 lexical SSA 解析之后，直接作用于已经带内部 scope
identity 的 producer/consumer 图：

- neutral add 只重定向对应 scope 内的 result uses；
- `div(1, sqrt(x))` 只识别直接 producer 链；
- normalization 专用常量仅在图中已无其他 use 时删除；
- sibling/nested region 的同名 SSA 不会互相覆盖；
- 同一 region 的 duplicate definition 仍然 fail-closed；
- 一个 sibling region 中定义的值不能被另一个 sibling region 捕获；
- 当前未注册的 region-bearing op（例如 `scf.if`）直接报
  `TTIRParseError`，不会静默把未知 scope 展平。

当前正式解析范围仍是实际观察到的 Triton 3.6 `scf.for`/`tt.reduce` region。后续若
真实 TTIR 出现新的 region op，应先增加明确的 scope 规则和测试，再将其加入支持集。

## 2. 既定 deterministic stagnation

活动 policy 保持为 `exact_artifact_and_hard_diff_v1`：

```text
state_i = (
  canonical_artifact_digest,
  exact_effective_hard_difference_digest,
)
```

只有连续两轮都完成 fresh normal replay、编译和可比较 hard diff，并且
`state_i == state_{i-1}` 时停止。它不使用 score、模糊距离、诊断文本、raw TTIR 或
replay source，也不会仅因 hard category 相同而停止。

本轮新增边界回归：即使 exact hard diff 相同，只要 artifact digest 改变，就不得误判
stagnation；`A -> B -> A` 在当前既定 policy 下会耗尽预算，而不会被当作“连续不变”。
compile/parser/provider 等没有 exact hard diff 的失败会打断连续链。

停止后 Gate1 返回结构化 `NO_CLOSURE/deterministic_stagnation`；若存在与最终 artifact
digest 对齐的可执行 replay，Gate2 仍独立执行。Gate1 与 Gate2 不是包容关系。

## 3. 第一次 Gate1 差分对下一次 Lift 的指导

本轮没有修改 generic initial Lift 目标。只有 source/replay 完成第一次确定性 Gate1
hard comparison 后，下一次 source-aware Lift 才收到新鲜 evidence。

结构 policy 升为：

```text
deterministic_scoped_fingerprint_structure_guidance_v2
```

每个 loop witness 现在包含：

- block argument 数量；
- induction variable use count；
- carried argument use counts；
- exact init-to-yield pairs；
- region 外部 captures；
- loop body op multiset；
- loop body memory effects。

这些字段和 reduction、mask、memory effect、dot 坐标一起进入 digest-bound guidance。
它们是帮助 Lift 回到 SOURCE 重建依赖的有界证据，不是要求将 TTIR 拓扑一比一抄入 J。

### `fg_log_softmax` 的重要修正

scope-v3 对保存的真实 TTIR 重检后，source 的两个 `scf.for` 都是：

```text
block_argument_count = 1
induction_use_count = 1
init_values = []
yielded_values = []
carry_pairs = []
```

它们捕获 region 外已经形成的 reduction 值，并在循环体中完成 masked read/write；当前
证据不支持“缺少语义 loop carry”这一判断，更接近 schedule/output traversal loop。
因此下一次 Lift 应优先保持 whole-tensor J，不应仅为匹配 source 的 `scf.for` 数量向 J
添加假的 semantic loop。新 witness 的价值正是提高第一次 Gate1 差分之后的判断质量，
而不是提高最初一次 Lift 的猜测命中率。

## 4. π 拓扑处理

本轮没有修改 π 拓扑、refs、composer 规则选择或 masked-J 阈值。现有监测继续保留：

- normal replay 必须闭合完整 fixed-C family；
- paired intervention 必须形成注册的 exact TTIR delta；
- masked-J/full-π 必须在完整 family 上产生可比较 hard mismatch；
- 若未来多个不同真实 family 批量出现 masked hard closure，再提交 π 拓扑泄漏报告。

单个 Lift/composer 构造失败不能解释成 π 拓扑泄漏。

## 5. 现有真实样本二次检验

当前代码、当前 compiler 和 scope-v3 fingerprint 对保存的真实 Lift/replay evidence 做了
无模型重检；没有重新生成或修复 replay。

| 样本 | Gate1 | fixed-C family | Gate2 | 结论 |
|---|---|---|---|---|
| `fg_ones` | `replay_closed` | 全部通过；masked family hard mismatch | 18/18 PASS | v3 未破坏已闭合 elementwise/generate 样本 |
| `fg_arange` | `no_closure`，`op_multiset + dataflow` | 无 family | 9/9 PASS | Gate1 false reject/结构差异与 Gate2 通过可同时成立 |
| `fg_log_softmax` | `no_closure`，三类 hard diff | 4/4 profile 未闭合 | 6 PASS + 6 `replay_diverged` | scope witness 表明 source loop 无 carry；多 tile replay 仍有真实动态问题 |
| `fg_diag_extract` | `replay_closed` | 3/3 通过；masked family hard mismatch | 9 `execution_failed` | Lower2C 的 `along_axis gather` rank contract 不合法；属于 Gate2 incomplete，不是语义 nonpass |

重检输出位于：

```text
/tmp/scrc_scope_v3_recheck_20260718/
```

## 6. 新增非 reduce 真实 family

### 6.1 `fg_index_select`

这是二维、两轴多 tile、运行时 index tensor 的间接寻址 family，包含：

- `single`；
- `rectangular_tail`；
- `two_axis_multitile`；
- normal/boundary 两种 index 分布；
- 3 个随机 seed，共 18 个 Gate2 cases。

真实 fresh 结果：

- initial Lift：1 个响应，约 250.4 秒；
- normal replay：2 次 writer，2/2 编译成功；
- source-aware Relift：1 次，返回的 J/interface/C/π 与首轮完全相同；
- 两轮 artifact digest：
  `6261664c1b0c444b97d8b74aebfb4d5859f901d927464de3b67f52a3ce04443b`；
- 两轮 exact family hard-diff digest：
  `2e814f52ba386497907af065c380da6df2d8f8f5759144127892e8b8b921e792`；
- hard categories：`op_multiset + dataflow + predicate_effect`；
- 第二次 fresh replay 后触发 deterministic stagnation；
- Gate2：6 `gate2_pass` + 12 `lift_diverged`，三方执行完整。

动态复现显示 `rectangular_tail` 下：

- source 仅有 row 0..11 非零，每行 16 个非零元素；
- replay/Lower2C 有完整 row 0..18，每行 17 个非零元素；
- source↔replay 与 source↔Lower2C 同时失败；
- replay↔Lower2C 精确通过。

安装的 FlagGems source 使用：

```python
out_mask = rows_mask and (cols_offsets < index_len)
```

Triton 3.6 已对非标量 tensor 的 `and/or` 发出弃用警告。该样本的 source 动态行为没有
满足“输入 source 绝对正确”的实验前提。因此本轮保持 trusted-source 裁决：Gate1 正确
不闭合；但不能据此声称 J/replay 已经正确复现 source，也不能为了关闭 Gate1 修改
replay。后续应先决定更换/修复上游 source 版本，或将该样本标记为 source-contract
异常，再用于正式准确率统计。

报告：

```text
/tmp/scrc_scope_v3_fresh_family_20260718/fg_index_select.json
```

### 6.2 `fg_eye`

该 family 的 trusted source 预编译通过，并声明 1x1、block edge、双轴 tail、矩形非整除
四个 profile。但 3 个真实 Lift 响应都在进入 Gate1 前被 deterministic composer 拒绝：

```text
ComposeDependencyError: composer produced duplicate replay step
('realize_broadcast', <node>, ())
```

因此：

- 没有 normal replay；
- 没有 Gate1 hard diff；
- 没有 Relift；
- 没有 Gate2；
- 不能归类为 Gate1 no-closure，也不能归类为 π topology leakage。

这是扩样暴露出的独立 Lift/composer broadcast work-item 去重缺口。根据本轮范围限制，
这里只记录，不修改 π/composer 主线。

报告：

```text
/tmp/scrc_scope_v3_fresh_family_20260718/fg_eye.json
```

## 7. 验证结果

定向测试：

```text
181 passed
```

scope parser 单测：

```text
24 passed
```

全量 `python/tests`：

```text
698 passed, 3 skipped, 3 failed, 77 warnings
```

三项失败均不是本轮回归：

1. 两个 split-k xFormers 用例在全套顺序下触发 Triton binder 重复
   `BLOCK_SIZE`；隔离重跑 2/2 PASS。
2. 旧 universe `count_nonzero` manifest 记录 `ok`，当前重放为
   `lowering_fail`；隔离仍失败，是既有 manifest 漂移。

修改模块已通过 `compileall`。

## 8. 本轮结论

1. TTIR scope 感知现在覆盖 parser 和 bounded normalization，而不是只修 duplicate
   name map。
2. 既定 stagnation stop 已落地并被真实 `fg_arange`、`fg_index_select` 以及边界单测
   验证；它只在第二次完整 fresh replay 后停止。
3. Relift 改进目标已经对准“第一次 Gate1 差分后的下一次 Lift”。scoped witness 能区分
   semantic carry 与 schedule-only loop，避免为关闭 Gate1 乱补 J loop。
4. 没有修改 π 拓扑；当前证据也没有要求这样做。
5. 非 reduce 扩样已覆盖 generate/indexing/gather。新失败主要暴露上游 source contract
   异常和 composer broadcast 去重缺口，而不是继续堆同类 reduce。
