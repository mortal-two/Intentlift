# SCRC Gate1：源码隔离重放闭合

## 定位

Gate1 只回答 SCRC 方案规定的问题：重写员在看不到源 Kernel 的情况下，能否
仅凭 `J + π + 接口签名 + C 槽 + 检索出的规则卡` 生成可编译重放 Kernel，且
源与重放在同一前端编译所得的中层结构是否闭合。

Gate1 不执行 Kernel，不解释 IntentIR JSON，也不是另一套通用 verifier。
IntentIR 是否忠于源码，最终由 Gate1 结构闭合与 Gate2 三方实跑共同回答。

## 输入与信息边界

提升官可看源码，并由真实 LLM extraction 路径产出：

- `J`：IntentIR A+B；
- `π`：仅含规则卡 id 及对 J/B 标识的引用；
- 接口：entry、参数顺序/类别/dtype/rank/mode、constexpr 槽、grid；
- `C`：调度槽及源实现使用的绑定值；
- 定位表：源码位置到 J/B 标识的映射，只给诊断侧。

重写员只能看到 J、π、接口、C 槽和按 π 检索出的最小规则卡集合。它看不到
源码、定位表、源 TTIR 与 C 的具体值。`do_not_specialize` 等 JIT 策略元数据
不属于 SCRC 的语义接口硬项。

## 主流程

```text
真实 Kernel 源码
  ├─ 同一 Triton frontend compile-only ──> source TTIR
  └─ LLM 提升 ──> J / π / C / 接口 / 定位表
                         │
                         └─ 计划(π) → 检索规则卡 → 源码隔离重写
                                                   │
                                                   └─ 注入同一 C
                                                        └─ compile-only
                                                             └─ replay TTIR

source TTIR / replay TTIR ──> 纯代码指纹与裁判 ──> CLOSED 或差异报告
```

同一 C、同一前端是本次结构实验的控制条件，不延伸成额外的环境锁、版本资格
审查或另一套正确性等级。

## 纯代码裁判

硬比较范围忠于方案：

1. 可调用接口：entry、参数、constexpr、grid；
2. A：算子多重集、数据流、归约；
3. B：mask/fill/store guard 与可观察 effect；
4. C：只报告，不参与闭合裁决。

规范化后精确相等，不使用图相似度阈值，也不让 LLM 覆盖裁判结果。当前已从
真实失败样例沉淀两条有限规则：

- 冗余中性加法 `x + 0`；
- `1 / sqrt(x)` 与 `rsqrt(x)` 的 TTIR 结构归一。

除此之外不预先发明代数规则。新规则必须来自真实 source/replay diff，并用
定向变异样例确认不会掩盖已有错误。

## 规则库

规则库是 Gate1 的正式组成部分，流程是“计划—检索—生成”：

1. π 的每一步只选择一个注册规则 id，并引用 J/B；
2. `RuleRegistry` 按 π 的首次使用顺序检索最小卡集合；
3. prompt 同时记录 step→card 选择轨迹和 registry digest；
4. 重写员从 J 读取具体语义，从规则卡读取前端构造方法。

当前 Triton 库覆盖 program domain、integer-offset index tile、staged pointer
tile、masked/unmasked access、elementwise、reduce、broadcast、tiled dot、
runtime scalar、C 槽、cast、select、loop carry 与 output mapping，共 15 张卡。
其中 runtime scalar 支持通过 `tensor_ref` 连到 J `shape=[]` 输入；reduce
卡的 `mean` 除数取全局 J 轴语义长度，elementwise 卡支持
`sqrt+recip`。
后续只根据真实 LLM 提取 Kernel 的
失败补卡，不按想象中的特性枚举扩张。

## 修复与反作弊

- 重放编译失败或结构未闭合时，J 冻结，重写侧最多小修 2 次；
- 判断 J 本身有遗漏时，差异报告回到可看源码的提升官重提，最多 3～5 轮；
- 始终保留不回退的最优候选；
- π 每轮做静态 API/source 泄漏扫描；
- 最终闭合前按方案执行不透明重命名后的 masked-J 消融，泄漏比阈值沿用主方案。

这些是 SCRC 本身的信息归属证明，不是额外 verifier。

## 当前实现

代码入口：`python/intentir/verify/gate1/`。

已经具备：

- `extract_scrc_artifact` 的真实 LLM lift-agent 接口；
- Responses JSON Schema wire contract，以及源码侧可信 compiler-view 摘要；
- J/π/C/接口/定位表 artifact；
- 计划—检索—生成规则库；
- 源与重放 compile-only、TTIR 提取；
- interface/A/B 纯代码比较；
- 有界小修/重提接口；
- 静态泄漏与 masked-J 消融；
- add、row-sum、softmax、RMSNorm、matmul 的随机/边界变异探针。

当前仍是 Triton 研究原型，不应声称已经覆盖数百 Kernel。正式规模化样本必须
从真实 Kernel 经 LLM extraction 产生 J/π/C；手写 J、手写 Intent MLIR 与固定
PK 只能作为 lowering 单元测试或早期探针，不能算 SCRC 端到端证据。

2026-07-15 最新真实 `gpt-5.6-sol/xhigh` 证据中，add、row-sum、softmax
和 RMSNorm 的 normal source-blind replay 都已达到 TTIR 硬闭合；在完整不透明
ABI/C 消融下，四者最终都是 `LEAKAGE_REJECTED`。RMSNorm 的 runtime
`eps`、`mean`、`sqrt+recip` 和 staged pointer 路径已经过真实 Lift/replay
验证；该被拒绝 live 样本的离线 Gate2 对照三方 PASS，但不改变
Gate1 结论。下一个五核主线样本是 matmul。

旧 NumPy JSON 解释器已归档到 `python/intentir/verify/back/legacy_v0/`，Gate1
不得导入它。
