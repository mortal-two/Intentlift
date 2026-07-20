# 验证栈与契约等级

## 与 SCRC Gate1 的关系

[scrc-gate1.md](scrc-gate1.md) 是动态源码执行一致性之前的保守结构门禁：
源码隔离重写、同 C compile-only 和 TTIR 硬指纹闭合产生
`REPLAY_CLOSED`，但不执行 Kernel，也不授予 FULL/PARTIAL 契约。

下面的动态源码执行一致性以及 [architecture.md](architecture.md) 的 L3/L4
仍是 lift 行为 oracle；随机/边界/极端值输入只在该动态层使用。Gate1 的硬
不一致不能被一次动态样例直接覆盖；若离线 Gate2 表明它是假拒绝，应新增
带前置条件、可枚举且有 mutation 单测的规范化规则，再重新跑 Gate1。

## 三道检查

1. **Schema / type 检查**:候选是合法的 IntentIR 工件(算子类型、符号 shape、轴角色、dtype、Layer B 约束、Layer C hint 都格式正确)。在更贵的检查前先挡掉畸形候选。
2. **静态 obligation 检查**:在 evidence 与 LLM 自报 claim 上评估义务(anchor 覆盖、结构化访问、guarded bounds、接口一致、地址独立性、结构化同步、受控 atomic 等)。每个义务返回 **PASS / FAIL / UNKNOWN**。
3. **动态源码执行一致性**:在记录的输入上跑原 kernel,与 IntentIR 解释器(或后续后端代码)输出对比。记录 input shape/dtype/seed/tolerance/源版本。这是离线 / 编译期检查,不在生产快路径上。

> **此检查已在 5090 D 上跑通(2026-06-25)。** FlashAttention 这一类:`verify/back/legacy_v0/interpret.py`(numpy 解释器,即 Layer A 参考语义)与**真实下降到 sm_120a 并执行的 fused kernel** 在多形状 / 多 seed(含 Sk 不整除 BK 的 tail-tile)上对比,execute==interpret 相对误差 ~3.9e-7。注意这条等式是**内部**一致性(解释器 vs 编译执行,两者独立派生),与 vs-torch / vs-Triton 的外部一致性互补——后者证"算法对",前者证"lower 没改变 IR 语义"(certified semantic lifting 的落点)。浮点 reduce 用 tolerance 而非 bit-exact(见 [../ir/atomic-ops.md](../ir/atomic-ops.md) 结合性标注)。

## obligation 的三态语义

- **PASS**:evidence/claim 支持该结构。
- **FAIL**:evidence 与该结构矛盾,或暴露不安全模式。
- **UNKNOWN**:evidence / checker 不足以裁决——**不等于失败**,交给动态执行一致性兜底(见 [dynamic-feedback.md](dynamic-feedback.md))。

## 契约等级(沿用论文术语,不背旧实现)

- **FULL**:动态执行一致 **且** 所有必需义务 PASS。
- **PARTIAL**:动态执行一致,但有 UNKNOWN 义务 / 记录的假设。
- **OOS**:本质不支持的模式、不安全的 FAIL、无法修复的畸形候选、或有界修复后仍无法建立动态执行一致。

> 注:UNKNOWN 不直接降级判死;它和动态结果一起决定最终等级。OOS 只留给**本质边界**(见 [../capability/boundaries.md](../capability/boundaries.md)),不是"静态证不出"的同义词。

**PARTIAL 是完全可发布的契约,不是降级品。** 它表示"动态验过、但有静态未决项",照样可用、可进入后续 lowering。不要让旧系统"kernel 烂在 PARTIAL"的污名复活——那是旧系统缺动态反哺导致的(见 [dynamic-feedback.md](dynamic-feedback.md)),不是 PARTIAL 本身的问题。

**受控 atomic 的 obligation 怎么 discharge**:atomic 不再一律判 FAIL/OOS。若能识别为受控归约(对应 Layer A 的 `reduce`/`scatter` + 结合性标注,见 [../ir/atomic-ops.md](../ir/atomic-ops.md)),obligation 走 PASS / UNKNOWN 并入动态验证(浮点用 tolerance);只有真正不可控的 atomic 才 FAIL → OOS。这填补了"atomic 一等公民"(IR 层)与"检查它的义务"(本层)之间的接口。

## 有界修复

候选失败 → 产出**结构化诊断**(失败的 schema 字段 / 不一致 shape / 失败义务 / 执行不匹配,可带反例),反馈给 LLM 做**有界**修复。超过修复上限 → 记弱化 / 不支持结果,不强行接受。

SCRC Gate1 的重写侧修复更严格：反馈只能包含重放自身的清洗后编译诊断和
固定差异类别，不能包含源 TTIR item/count、SSA、LocationMap 或动态反例；
完整差异只留给可看源码的提升侧决定是否重提 J。

## 不当回事的东西

- metamorphic / mutation 测试是验证栈的**压力测试**(度量 validator 敏感度、提供反例),**不是 per-kernel 证书字段**。
- 不把 certificate 描述成全域形式证明。
