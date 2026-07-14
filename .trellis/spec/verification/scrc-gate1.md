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

接口中的 `tensor_ref` 必须引用存在的 J tensor，且 dtype 与被引用 J tensor
一致；真实 LLM extraction 还必须与可信 compiler view 中对应实参 dtype 一致。
该交叉约束防止 f16/f32 等类型在 Lift 接缝漂移，不引入第三套 verifier。
tensor pointer 可声明 `rank=0/shape=[]`；它仍是 pointer binding，不得与按值
runtime scalar 混淆。

完整 Python ABI 仍必须与真实 Triton callable 的参数顺序和 constexpr 位置精确
一致。Triton 若把没有 `tensor_ref` 的普通 shape/stride scalar 专门化并从 TTIR
签名消去，Gate1 验证 surviving argument name 是声明 ABI 的有序子序列；pointer/
tensor 或连接 J 的语义 scalar 被消去时仍 fail-closed。该规则只吸收真实前端
specialization，不恢复 `do_not_specialize` metadata 门禁。

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

当前 Triton 库覆盖 native/wide program domain、integer-offset index tile、
staged pointer tile、masked access、staged mask tile、unmasked access、
elementwise、显式轴/flattened reduce、broadcast、K-tiled dot、outer product、
runtime scalar、C 槽、cast、select、loop carry 与 output mapping，共 19 张卡。
其中 runtime scalar 支持通过 `tensor_ref` 连到 J `shape=[]` 输入；reduce
卡的 `mean` 除数取全局 J 轴语义长度，elementwise 卡支持
`sqrt+recip`。tiled-dot 卡从 contraction/J.B 读取 K 语义 extent、K tile
slot/domain、两侧 access/mask/fill、result dtype 与 output tile；普通 matmul
的 schedule-only K loop 不得重复表示成 J `loop/reduce/broadcast`。二维 access
中的 coordinate component 也不得任意重组为完整 offset，向量 predicate 使用
源码声明的 `&`。`m,n->mn` 等没有 contracted label 的 contraction 使用 outer
product 卡通过 rank insertion 与乘法实现；axis-omitted local reduction 使用
flattened-reduce 卡；compiler slot 只在 launch/compile 时注入，不是 kernel
参数或 body 变量。显式轴 reduce 卡 v4 还支持既有 `fn=prod`：重放模块定义纯
sibling `@triton.jit` 乘法 combiner 并交给 `tl.reduce`。安全策略仅允许同样经过
完整 AST 白名单验证、调用图无递归的 sibling JIT helper；任意 Python global/
free name 仍拒绝。
后续只根据真实 LLM 提取 Kernel 的
失败补卡，不按想象中的特性枚举扩张。

## 修复与反作弊

- 重放编译失败或结构未闭合时，J 冻结，重写侧最多小修 2 次；
- 判断 J 本身有遗漏时，差异报告回到可看源码的提升官重提，最多 3～5 轮；
- 始终保留不回退的最优候选；
- π 每轮做静态 API/source 泄漏扫描：frontend API 拼写扫描 J+π，source
  n-gram 只扫描 π；J.B 本来就负责保存源码提取的 bounds、mask 和 index
  expression，不能被 source n-gram 系统性误杀；
- 最终闭合前按方案执行不透明重命名后的 masked-J 消融，泄漏比阈值沿用主方案。

这些是 SCRC 本身的信息归属证明，不是额外 verifier。

## 当前实现

代码入口：`python/intentir/verify/gate1/`。

真实库函数不一定是裸 Triton `JITFunction`。compile worker 允许沿有界 `.fn`
wrapper chain 查找唯一 ABI metadata owner，检测环并 fail-closed；wrapper 的
`warmup()` 若返回 tuple/list，只接受其中唯一拥有 `asm` 的 CompiledKernel。
外层若也公开 `arg_names`，必须与内层 metadata owner 的名称和顺序完全同构；
不得把内层 constexpr 位置套到重排后的外层参数。也不得假定外层 wrapper 自己
保存 constexpr/do-not-specialize，或从任意对象图递归搜索编译产物。

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
- 24 个真实 FlagGems Kernel 的非 Trellis 批处理队列、逐 case 原子报告、按完整
  batch identity 恢复的 resume，以及同时绑定 source/compiler view/model/effort/
  registry/Lift-prompt 的 Lift cache；其中 `fg_dot`、`fg_addr`、`fg_ones`、
  `fg_sum/fg_mean/fg_max/fg_min/fg_prod/fg_vector_l2` final stage 已具备真实
  contract 并运行。batch identity 还绑定 Gate2 seed、distribution、scale 与
  tolerance，输入策略变化后不得 resume 旧结果。

当前仍是 Triton 研究原型，不应声称已经覆盖数百 Kernel。正式规模化样本必须
从真实 Kernel 经 LLM extraction 产生 J/π/C；手写 J、手写 Intent MLIR 与固定
PK 只能作为 lowering 单元测试或早期探针，不能算 SCRC 端到端证据。

2026-07-15 最新真实 `gpt-5.6-sol/xhigh` 证据中，add、row-sum、softmax、
RMSNorm、matmul、FlagGems dot 和 FlagGems addr 的 normal source-blind replay
以及 FlagGems ones/sum-final/mean-final/max-final/min-final/prod-final/L2-final
都已达到 TTIR 硬闭合；在完整不透明 ABI/C 消融下，十四者最终都是
`LEAKAGE_REJECTED`。RMSNorm 的 runtime
`eps`、`mean`、`sqrt+recip` 和 staged pointer 路径已经过真实 Lift/replay
验证；matmul 的单一 `mk,kn->mn` contraction、二维 tail mask、两轮 K tile、
f16 input/f32 output 接缝与 staged pointer 路径也已经过真实 Lift/replay
验证。RMSNorm、matmul 与九个已运行 FlagGems rejected live 样本的离线 Gate2
对照均三方 PASS，但不改变 Gate1 结论。`fg_dot/fg_addr` 触发四张新卡；后续
`fg_ones/fg_sum/fg_mean/fg_max/fg_min/fg_vector_l2` 直接复用 19 张卡，`fg_prod`
只补齐既有 reduce 卡的 prod 映射。24 个真实外部 Kernel 只完成前九例，
仍不能宣称已具备数百 Kernel 覆盖。

旧 NumPy JSON 解释器已归档到 `python/intentir/verify/back/legacy_v0/`，Gate1
不得导入它。
