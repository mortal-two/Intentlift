# SCRC Gate1：源码隔离重放闭合

## 定位

Gate1 只回答 SCRC 方案规定的问题：重写员在看不到源 Kernel 的情况下，能否
仅凭 `J + π + 接口签名 + C 槽 + 检索出的规则卡` 生成可编译重放 Kernel，且
源与重放在同一前端编译所得的中层结构是否闭合。

Gate1 不执行 Kernel，不解释 IntentIR JSON，也不是另一套通用 verifier。
Gate1 与 Gate2 是两类独立证据，不是包含关系：Gate1 检查结构、
信息来源与重放闭包，Gate2 检查有限输入矩阵上的三方动态行为。
Gate2 通过不推导 Gate1 应通过，也不得覆盖 Gate1 拒绝。

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

Gate1 normal 结构判定是一个有界外层循环，不是单次检查：

```text
initial Lift(source) → artifact_0

for artifact evaluation round:
  source-blind replay(artifact_i)
  compile source/replay with the same frontend and C
  deterministic hard comparison

  if every normal hard component closes:
      run final masked-J check
      if masked-J passes: return REPLAY_CLOSED
      else: return LEAKAGE_REJECTED

  run bounded inner replay repair
  if hard differences remain and relift budget remains:
      relift_input = original source
                   + source fingerprint
                   + allowlisted deterministic difference history
                   + current J/pi/location_map
      artifact_{i+1} = source-aware SCRC Relift(relift_input)
      require interface_{i+1} == interface_i
      require C_{i+1} == C_i
      continue

return NO_CLOSURE
```

“差异较大”不用 fuzzy score 定义：只要 interface/A/B 任一硬项不闭合，
内层小修预算耗尽后就进入外层重提。score 只用于最优失败候选排序和
masked-J 量化，不得将硬差异改判为通过。

外层 Lift 可重新阅读原始 source，但 replay writer 仍始终看不到源码。回流
证据不包含 replay 源码或 Gate2 结果。当前 `LEAKAGE_REJECTED` 是独立的
反作弊终止状态，不自动伪装成 normal source/replay 差异进入同一 relift。

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
`sqrt+recip`，并通过受限 Triton libdevice 路径支持已有 `abs/pow`。AST 只允许
`from triton.language.extra.cuda import libdevice` 与 `libdevice.pow`；其他
`libdevice.*` 仍拒绝。tiled-dot 卡从 contraction/J.B 读取 K 语义 extent、K tile
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
- 有界内层小修与真实 source-aware 外层重提；normal 硬差异、
  原始 source 与当前 J/π 会进入下一轮 Lift，interface/C 冻结；
- 外层重提必须以已编译的 source/replay comparison 和 normal 硬差异为前提；
  replay policy、生成或编译失败本身不触发 semantic relift；
- 静态泄漏与 masked-J 消融；
- add、row-sum、softmax、RMSNorm、matmul 的随机/边界变异探针。
- 24 个真实 FlagGems Kernel 的非 Trellis 批处理队列、逐 case 原子报告、按完整
  batch identity 恢复的 resume，以及同时绑定 source/compiler view/model/effort/
  registry/Lift-prompt 的 Lift cache；其中 `fg_dot`、`fg_addr`、`fg_ones`、
  `fg_sum/fg_mean/fg_max/fg_min/fg_prod/fg_vector_l2` final stage 以及
  `fg_vector_l1` first stage 已具备真实 contract 并运行。batch identity 还绑定
  Gate2 seed、distribution、scale 与
  tolerance，输入策略变化后不得 resume 旧结果。

当前覆盖不等于全部情况：

| 层次 | 已有真实端到端证据 | 部分覆盖 | 尚未覆盖 |
|---|---|---|---|
| A | 基础 elementwise/const、sum/mean/max/min/prod、softmax/RMSNorm 固定链、matmul/dot/outer product | cast/dtype、broadcast、mixed precision、常量敏感性 | RNG、scan/sort/top-k、量化/复数/稀疏、多输出、大型 attention |
| B | 基础 1-D/2-D 仿射索引、tail mask、masked load/store、fill/guard、row reduction | 多 program axis、非末轴/多重归约、动态 stride、复合 mask | structured/online loop、scan carry、atomic、gather/scatter、block pointer/descriptor、alias-sensitive effects |
| C/前端 | 单一 reviewed profile 的 constexpr/compiler slots/K tile、Triton/FlagGems compile-only | specialization 影响、grid/编译属性、region/loop 指纹 | 多 C/多 specialization 聚合、TileLang/CUDA 正式 Gate1、跨版本规范化、atomics/内存序 |

早期五核手写 J/π 是机制探针；后续同五个族加 10 个 FlagGems 已重走
真实 `gpt-5.6-sol/xhigh` Lift，15 个 normal replay 硬闭合。但 15/15 最终
都被 masked-J 拒绝，所以真实 LLM artifact 的正式 `REPLAY_CLOSED` 样本仍为
0。24 个 FlagGems 队列只完成 10 个，不得从样本数量推导特性完备性。

当前仍是 Triton 研究原型，不应声称已经覆盖数百 Kernel。正式规模化样本必须
从真实 Kernel 经 LLM extraction 产生 J/π/C；手写 J、手写 Intent MLIR 与固定
PK 只能作为 lowering 单元测试或早期探针，不能算 SCRC 端到端证据。

Gate2 现已能在一个已闭合 replay 上执行严格 shape/value/seed suite，但这不等于
Gate1 已完成多 specialization 结构闭合：当前正式 Gate1 verdict 仍锚定每个
`LiveKernelCase` 的一个 reviewed source/replay compile profile。若要把 Gate1
升级为多 C/多 specialization 的硬门禁，需要单独定义 source/replay TTIR 的逐
profile 聚合和 masked-J 规则；在主方案未审批前，不用 Gate2 matrix 反向改写
Gate1 verdict。候选条件与最小实现范围记录在
`doc/SCRC_Gate1_多Specialization门禁_计划外候选报告_2026-07-15.md`。

2026-07-15 最新真实 `gpt-5.6-sol/xhigh` 证据中，add、row-sum、softmax、
RMSNorm、matmul、FlagGems dot 和 FlagGems addr 的 normal source-blind replay
以及 FlagGems ones/sum-final/mean-final/max-final/min-final/prod-final/L2-final
与 L1-first 都已达到 TTIR 硬闭合；在完整不透明 ABI/C 消融下，十五者最终都是
`LEAKAGE_REJECTED`。RMSNorm 的 runtime
`eps`、`mean`、`sqrt+recip` 和 staged pointer 路径已经过真实 Lift/replay
验证；matmul 的单一 `mk,kn->mn` contraction、二维 tail mask、两轮 K tile、
f16 input/f32 output 接缝与 staged pointer 路径也已经过真实 Lift/replay
验证。RMSNorm、matmul 与十个已运行 FlagGems rejected live 样本的离线 Gate2
对照均三方 PASS，但不改变 Gate1 结论。`fg_dot/fg_addr` 触发四张新卡；后续
`fg_ones/fg_sum/fg_mean/fg_max/fg_min/fg_vector_l2` 直接复用 19 张卡，`fg_prod`
只补齐既有 reduce 卡的 prod 映射，`fg_vector_l1` 只补齐既有 elementwise 卡的
abs/pow 映射与受限 libdevice 安全路径。其 18-case Gate2 matrix 覆盖
`M={1,1023,1024}` 与 runtime `ord={1.0,1.5,2.0}`，18/18 PASS；两次独立
masked-J 为 `0.906140/1.0`，均拒绝。24 个真实外部 Kernel 只完成前十例，
仍不能宣称已具备数百 Kernel 覆盖。

旧 NumPy JSON 解释器已归档到 `python/intentir/verify/back/legacy_v0/`，Gate1
不得导入它。
