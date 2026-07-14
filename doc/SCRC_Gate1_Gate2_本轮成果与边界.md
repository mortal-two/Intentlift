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
add 的 Lift、正常 replay 和三方执行。2026-07-15 补齐方案要求的不透明
ABI/C 消融后，最新 add 的 masked replay 也完全闭合，因 `L=1.0` 被 Gate1
正确拒绝；此前 Gate2 PASS 仍是 lowering 工程证据，但不再是当前 Gate1 接受
证据。详见 `test/scrc_live_add_report.md` 与
`doc/SCRC_Gate1_MaskedJ_低熵Kernel阻塞报告_2026-07-15.md`。

2026-07-15 同一入口又推进了 row-sum 和 softmax。两者的 normal
source-blind replay 都已达到 TTIR 硬闭合；softmax 还用真实 diff 补了
全局/局部归约轴契约、隐式 broadcast 和 staged pointer tile 规则。但
row-sum 的 masked score 为 `0.778676`，softmax 的 masked replay 又完整
闭合为 `1.0`，所以两者最终仍是 `LEAKAGE_REJECTED`。softmax 的
production lower2c 三方预检已 PASS，但正式 live Gate2 按门禁顺序没有
运行。详见 `test/scrc_live_softmax_report.md`。

RMSNorm 随后也沿用同一入口完成真实 Lift 和 normal replay。本轮补齐
runtime `eps` 到 J `shape=[]` 输入的 `tensor_ref` 连接、`mean` 的语义
除数、`sqrt+recip` 规范形态，并用真实失败收紧 staged pointer card 的
选择说明。normal replay 首候选 score `1.0`，masked replay 也为 `1.0`，
最终仍是 `LEAKAGE_REJECTED`。对这份被拒绝的真实 live artifact 按方案
9.1 离线执行 Gate2，三方 PASS，但不改变门禁结论。详见
`test/scrc_live_rmsnorm_report.md`。

matmul 最后沿同一入口完成五核探针。最终 J 只包含一个
`contraction("mk,kn->mn")`，源码 K-tile loop 被正确还原为 contraction 的
Layer-C/前端实现，而不是 J `loop/reduce`。本轮还根据真实失败补全 staged
二维 pointer association、向量 mask conjunction、K extent/tile/result dtype
规则，并把 source n-gram 扫描纠正为只约束 π，不再误伤依法保存源码边界的
J.B。normal replay 首候选 score `1.0`；masked replay 虽未硬闭合，仍以
`0.804518 > 0.5` 被 `LEAKAGE_REJECTED`。真实 live 被拒绝样本的离线 Gate2
三方 PASS。详见 `test/scrc_live_matmul_report.md`。

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

Triton 默认规则库由原来的 7 张扩到 15 张，覆盖当前 A/B/C 主线需要的：

- program domain；
- integer-offset index tile 与 staged pointer tile；
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
add、collapsed row-sum、softmax、RMSNorm 与 matmul 的三方一致。RMSNorm 还用
`eps=1e-5/1e-2` 两个值证明 production lower2c 真正消费 runtime scalar。
matmul 使用 `M=17,N=13,K=19` 的 f16 输入/f32 输出和三个随机 seed 覆盖
M/N/K 尾块；错误 B 地址会被机械归属为 `REPLAY_DIVERGED`。
这些固定 J 测试是
production lowering 能力证据，不代替每个 live artifact 的 Gate1 接收。

matmul 当前 Lower2C 会消费已经量化为 f16 的输入数值并产生 f32 输出，但其
production MLIR 仍统一发射为 f32；因此这是数值语义第三腿，不是
mixed-f16/f32 dtype-faithful lowering 已完成的证据。

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

累计到本轮，已经修正或明确保留的 raw false reject 来源为：

- `+0`：已修；
- `rsqrt ↔ 1/sqrt`：已修；
- 缺 `do_not_specialize`：不再拒绝；
- source n-gram 误扫 J.B：已改为只扫描 π；
- π API 泄漏：仍按 SCRC 有意拒绝。

因此剩余 raw false reject 主要是反作弊边界，不再是 TTIR 表达覆盖问题。

## 3. 现在是不是“只是原型”

是研究原型，但不是纯占位或玩具。

已经真实发生的事情包括：

- 真实 `gpt-5.6-sol/xhigh` Lift artifact 与 normal replay 已在
  add、row-sum、softmax、RMSNorm、matmul 上结构闭合；最终接收目前被模型
  先验和 π 结构可识别性阻塞；
- 真实 Triton frontend 编译；
- 真实 TTIR 提取和纯代码结构比较；
- 真实 GPU source/replay launch；
- 真实 IntentIR→MLIR→LLVM lowering 与执行；
- 随机、tail、block±1、大动态范围、多 K tile 等边界样例。

所以核心机制已经不是“差得远到只剩接口”。差距主要在覆盖与工程闭环：

- 尚未把数百个真实 Kernel 全部重新走一遍新的 `extract_scrc_artifact`；
- live 五核的 normal 路径已收敛，但一次性 masked-J
  无法稳定区分 π 泄漏和模型先验猜测，且 softmax 已实测出 π
  有序拓扑结合仍公开的结构信息，可成为语义替代编码；matmul 还证明 masked
  无需完全硬闭合也可能超过当前泄漏比阈值；
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
