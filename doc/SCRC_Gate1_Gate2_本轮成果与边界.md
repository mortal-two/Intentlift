# SCRC Gate1 / Gate2：本轮成果、False Reject 与原型边界

## 1. 这轮到底做了什么

这轮不是继续修补旧 `verify`，而是把项目验证主线改成了 SCRC 文档规定的
两个过程：

```text
真实 Kernel
  → LLM 提升官产出 J / π / C / 接口 / 定位表
  → Gate1：源码隔离重放 + source/replay 双 TTIR 结构闭合
  → Gate2：source / replay / production lowering 三方实际执行
```

具体落地如下。

### 1.1 旧实现归档

- 旧 NumPy JSON 解释器移动到
  `python/intentir/verify/back/legacy_v0/interpret.py`；
- `intentir.verify.interpret` 只保留兼容转发，已有调用仍能执行；
- 新 Gate1、Gate2 和新的 SCRC extraction 均不导入旧解释器；
- 旧 schema/obligation、FULL/PARTIAL/OOS、L0–L9 验证规范移动到
  `.trellis/spec/verification/back/legacy_v0/`；
- 当前 `verify` 规范只保留 Gate1 和 Gate2。

### 1.2 真实 LLM 提升入口

新增 `intentir.extraction.lifters.scrc.extract_scrc_artifact`。它把真实源码交给
提升官 LLM，要求直接产出完整的：

- J：IntentIR A+B；
- π：规则卡 id + J/B 引用；
- C：槽和值；
- 接口签名；
- 定位表。

该入口只解析 artifact，不用解释器“先验算一遍”来决定正确性。正确性由
后续 Gate1/Gate2 裁决。模型由现有 provider 配置选择，不在代码中绑定 5.5。

当时也做了真实 provider 审计，历史结果必须如实保留：

- 当时使用了不完整模型名 `gpt-5.6`，因此返回 `unknown provider for model`；
- 回退到配置中的 `gpt-5.5` 后，add 的完整 artifact 在有界修复内仍未解析闭合；
- 实际暴露的都是 wire/schema 漂移：`fill→fill_value`、`node_ids` 列表、以及
  access 字段被重复塞入 `effects`；
- 代码已按这些真实失败收紧 prompt，并只兼容这几个已观察到的序列化形态，
  但本轮没有继续无界调用来制造一个“成功样例”。

这段结论只描述当时那次调用。2026-07-14 后续已改用正确名称
`gpt-5.6-sol` 与 `xhigh`，并通过 wire schema + 可信 compiler view 打通真实
add 的 Lift→Gate1→Gate2；最新证据见 `test/scrc_live_add_report.md`。

### 1.3 Gate1 收回 SCRC 主线

保留了方案要求的硬项：

- 重放能否编译；
- 接口签名；
- TTIR 算子多重集与数据流；
- 归约；
- mask/fill/store/effect；
- π 静态泄漏与 masked-J 消融；
- 同一 C 下比较。

移除了不属于 SCRC 语义裁决的硬条件：

- `do_not_specialize` 不再作为接口硬相等；
- compiler/target 的整份 metadata 不再逐字段相等后才能比较；
- source 不再因为调用者没打 `trusted=True` 标签而直接拒绝；
- 默认不再要求两个独立重写候选连续闭合，一次硬闭合即可进入方案规定的
  最终反作弊检查。

### 1.4 规则库完善

Triton 默认规则库由原来的 7 张扩到 14 张，覆盖当前 A/B/C 主线需要的：

- program domain；
- index tile；
- masked/unmasked access；
- elementwise；
- reduce；
- broadcast；
- tiled dot；
- runtime scalar；
- C 槽；
- cast/select；
- loop carry；
- output mapping。

prompt 不再只是塞卡片，而是显式记录：

```text
π step → 检索到的 rule id → 最小规则卡集合 → 重写
```

这才是方案中的“计划—检索—生成”。

### 1.5 两条真实 TTIR 规范化规则

根据上一轮实际 false reject，只加入了两条可枚举规则：

- 冗余 `x + 0`；
- `1 / sqrt(x)` 与 `rsqrt(x)`。

没有顺手写一个通用代数化简器。新规则仍必须来自真实 diff。

### 1.6 Gate2 最小真实原型

新增 `python/intentir/verify/gate2/`：

- source Triton 与 replay Triton 分别加载、分别 launch；
- IntentIR 第三腿直接复用项目已有
  `IntentIR → intent MLIR → linalg → LLVM → mlir-runner`；
- 三路使用同一份输入；
- 按 tolerance 两两比较并机械归属；
- lower 不支持时明确返回 `LOWERING_UNSUPPORTED`，不会退回旧解释器伪装成功。

`python/tests/test_gate2.py` 已在真实 CUDA/Triton 和真实 MLIR lowering 上跑通
add 的三方一致。

## 2. False reject 是什么意思

在能力实验里，用后续实际执行结果作为行为标签：

```text
Gate1 拒绝，但 Gate2 行为通过 = raw false reject
```

它表示 Gate1 没放过一个在当前测试域里表现正确的候选，损失的是覆盖率，
不是“已放行样本的准确率”。相反：

```text
Gate1 闭合，但 Gate2 失败 = false accept
```

这是更危险的问题，因为错误 lifting 被门禁放行。

需要再区分两类 raw false reject：

1. 真正的规范化缺口：例如 `+0`、`1/sqrt`；应该沉淀固定规则；
2. SCRC 有意拒绝：例如 π 偷写 `tl.load`。候选代码可能碰巧行为正确，但它
   破坏了“语义必须住在 J”这一实验前提，不能为了覆盖率放行。

本轮修正后，上一轮的 4 个 raw false reject 中：

- `+0`：已修；
- `rsqrt ↔ 1/sqrt`：已修；
- 缺 `do_not_specialize`：不再拒绝；
- π API 泄漏：仍按 SCRC 有意拒绝。

因此剩余 raw false reject 主要是反作弊边界，不再是 TTIR 表达覆盖问题。

## 3. 现在是不是“只是原型”

是研究原型，但不是纯占位或玩具。

已经真实发生的事情包括：

- 真实 LLM provider 调用链已触达，但完整提升 artifact 尚未闭合；
- 真实 Triton frontend 编译；
- 真实 TTIR 提取和纯代码结构比较；
- 真实 GPU source/replay launch；
- 真实 IntentIR→MLIR→LLVM lowering 与执行；
- 随机、tail、block±1、大动态范围、多 K tile 等边界样例。

所以核心机制已经不是“差得远到只剩接口”。差距主要在覆盖与工程闭环：

- 尚未把数百个真实 Kernel 全部重新走一遍新的 `extract_scrc_artifact`；
- 首个 live add artifact 仍需让 prompt/schema 在有界修复内收敛；
- 规则卡只在五核和当前变异集上证过；
- production lowering 不是所有现有 A/B/C 结构都能走 CPU 腿；
- TileLang/CUDA 是方案后续阶段；
- 还缺数百 Kernel 的批处理、失败分诊与真实重提循环运行证据。

因此最准确的称呼是：

> Gate1 已是可执行的 Triton SCRC MVP；Gate2 已是可执行的三方最小原型；
> 整体还不是数百 Kernel 规模的完整工具。

## 4. 这轮没有做什么

- 没有建立第三套 verifier；
- 没有引入新的 FULL/PARTIAL/OOS 模式；
- 没有用人工/确定性生成 J 作为正式路径；
- 没有把手写 MLIR PK 当成 LLM lifting 的有效证据；
- 没有新增计划外的 IntentIR op；
- 没有把环境版本、缓存策略、整份编译器 metadata 变成新的门禁条件。
