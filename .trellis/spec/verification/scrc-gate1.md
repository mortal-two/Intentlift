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
- `π`：完整、按施工依赖排序的严格重放计划；显式列出重放所需的每个规则卡
  实例，但只保存规则卡 id 与指向 J/B/C/接口字段的 typed reference，不复制
  算子、常量、predicate、fill、轴、源码或前端 API；
- 接口：entry、参数顺序/类别/dtype/rank/mode、constexpr 槽、grid；
- `C`：调度槽及源实现使用的绑定值；
- 定位表：源码位置到 J/B 标识的映射，只给诊断侧。

重写员只能看到 J、π、接口、C 槽以及由 π 逐步检索出的规则卡。规则库负责解释
π 已经选择的机制，不得替 π 注入隐藏默认步骤。重写员看不到
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
                         └─ 严格按完整 π 执行 → 源码隔离重写
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
      interventions = deterministic registered J/B interventions(artifact_i)
      if interventions are available:
          source-blind replay each intervened J with interface/C/pi frozen
          require exact registered TTIR delta; no score or ratio
          if any intervention is ignored/wrong/off-target: LEAKAGE_REJECTED
          run blank-J only as diagnostic
          return REPLAY_CLOSED
      else:
          run masked-J fallback
          if masked-J compile/interface validation collapses: return REPLAY_CLOSED
          if any A/B semantic hard component differs: return REPLAY_CLOSED
          else: return LEAKAGE_REJECTED

  if replay generation/compile/interface/hard comparison fails
     and relift budget remains:
      relift_input = original source
                   + source fingerprint
                   + sanitized failure class or deterministic difference history
                   + current J/pi/location_map
      artifact_{i+1} = source-aware SCRC Relift(relift_input)
      require interface_{i+1} == interface_i
      require C_{i+1} == C_i
      restart from static scan + source compile + fresh replay

return NO_CLOSURE
```

“差异较大”不用 fuzzy score 定义：只要 normal replay 的生成、编译、接口或
interface/A/B 任一硬项失败，就进入 source-aware relift。禁止把旧 replay、编译
反馈或 comparator diff 回喂 writer 做原地代码修补。score 只用于最优失败候选
排序和诊断；
不得覆盖 normal 硬差异、paired intervention 或 masked-J fallback 的精确裁决。
masked-J fallback 只检查是否确定性丢失至少一个 A/B 硬分量，score ratio 不参与
verdict。

外层 Lift 可重新阅读原始 source，但 replay writer 仍始终看不到源码。回流
证据不包含 replay 源码或 Gate2 结果；编译失败只提供净化后的失败类别/范围。
paired intervention 的 mutant J 也不进入
normal source-aware relift，避免把人工干预语义写回源 J。当前
`LEAKAGE_REJECTED` 是独立的
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

score 使用 `active_evidence_weighted_v2`：双方都为空的 category 仍视为硬相等，
但标记为 `applicable=false`、component score 为 `null`，不进入固定权重的分子或
分母。这样没有 reduction 的 Kernel 不再凭空获得 0.20；总分只在存在 source 或
replay 证据的 category 上重新归一化。该 score 仍不是闭合阈值。

- 冗余中性加法 `x + 0`；
- `1 / sqrt(x)` 与 `rsqrt(x)` 的 TTIR 结构归一。

除此之外不预先发明代数规则。新规则必须来自真实 source/replay diff，并用
定向变异样例确认不会掩盖已有错误。

## Paired J/B 语义干预

blank-J 会同时测量 π 泄漏、接口/shape 可猜性、通用规则库与模型先验，不能单独
证明重写是否受 J 因果控制。该机制在历史 causal-v3 中引入，并由当前 strict
controlled causal-v5 收紧：normal 硬闭合后优先运行
确定性 paired intervention：

1. judge 从注册表最多选择 3 个类型/rank/arity 保持不变且合同闭合的 J/B 坐标；
2. 只修改该坐标，interface、C、完整 π、规则库、node id、boundary 和 source
   全部冻结；
3. mutation metadata 与 expected delta 只在 judge sidecar 中，绝不进入 replay
   prompt；
4. 每个 mutant 都把同一个 normal 模型生成 replay 作为结构控制交给同一
   source-blind writer；该候选不是 source，也不拥有语义，writer 必须重新读取当前
   J，但不得重构未变化的 π 步骤；
5. provider 408/503 最多进行 3 次有界 generation 尝试，耗尽后仍 fail-closed；
6. 使用同一 writer/模型/前端/C compile mutant replay；
7. 比较 normal replay TTIR 与 mutant replay TTIR 的有向确定性差分。

首批注册 `map_binary` 同类型浮点替换：`add→sub`、`sub→add`、`mul→add`、
`div→mul`。验收要求算子 Counter 恰好删除一个旧 op、增加一个新 op，改变类别
恰好是 `op_multiset + dataflow`；interface、reduction、predicate/effect 与 C
必须完全不变。另注册非归约耦合 masked load 的 typed `fill 0→1`：必须精确删除
一个 typed 0 fill、增加一个同类型 1 fill；若旧 0 被其他 load/accumulator 共享，
只额外允许可枚举的 `arith.constant:+1`，不允许 select 或其他算子。带
`BoundaryReduction.masked_contribution` 的 load 不适用该干预，因为单改 fill 会
迫使重写员插入 select 来继续满足归约合同。该判断沿 J DAG 追踪间接数据流，不只
检查 access 是否直接挂在 reduction node；RMSNorm 的 `x→square→mean` 是真实失败
沉淀出的反例。状态严格区分：

无输入、单节点、单输出的浮点 `const` Kernel 另注册 direct-output constant
`1→0`。该坐标要求 J output mapping 与唯一 store 都指向同一 node；TTIR 指纹在
store effect 中只对“直接由 `arith.constant` 产生的值”暴露精确 typed value，
非 constant store 仍保持原有数据流指纹，因而不会扩大 binary/reduction 的允许
差分。验收要求 direct store typed constant 精确 `1→0`，改变类别恰为
`dataflow + predicate_effect`，算子计数、fill、reduction 与 schedule 全部不变。

归约坐标注册为一个不可拆分的合法复合干预：
`sum/identity=0/masked_contribution=0/fill=0` 同时变为
`max/identity=-inf/masked_contribution=-inf/fill=-inf`。验收要求 TTIR combiner
精确 `arith.addf:-1, arith.maxnumf:+1`，归约 axis/input/result arity 不变，
typed fill 精确 `0→-inf`；只允许因常量共享产生至多一个可枚举的
`arith.constant` 计数变化。

- `pass`：观察差分与注册差分完全相等；
- `ignored_j`：mutant replay 与 normal replay 相同；
- `wrong_operator_delta`：目标变化方向或计数错误；
- `wrong_constant_delta` / `wrong_fill_delta`：typed value 变化方向或位置错误；
- `off_target_drift`：目标正确但额外改变算子或其他硬类别；
- generation/compile/evaluation failure：均 fail-closed。

此过程没有分数、比例或距离阈值。所有适用 intervention 必须通过；当前 registry
找不到合法坐标时才回退到 masked-J，不为覆盖率制造畸形 mutant。后续
operand order 与 predicate intervention 必须分别补充可定位的精确指纹后再注册。

## 规则库

规则库是 Gate1 的正式组成部分，但它是 π 的“机制字典”，不是隐藏的第二份
重放计划。当前流程是“完整 π 选择规则实例—按序检索规则卡—严格生成”：

RuleCard 定义由系统跨 Kernel 共享，Lift 不得为每份源码创建新规则。每个 artifact
独立产生的是 π 的绑定实例：共享 rule id/机制词汇，但 target/context 指向当前
J/B/interface/C。冻结同一 π 实例而修改 J/B 时，writer 必须随 J/B 生成不同代码；
若 J 相同而来源 realization 会造成不同 TTIR，则该来源差异正是 π 应拥有的信息。
J、π、C、interface 均相同时，source-blind writer 不存在另一个可用的“源码形式”
输入，不能要求它可靠地产生两个来源特定结构。

1. π 必须显式列出 program domain、runtime scalar、C slot、J node、boundary
   access、broadcast 和 output mapping 所需的全部规则实例；
2. 每一步通过 `node_ref`，或在没有 `node_ref` 时通过 `refs[0]` 指定 target；
   其余 refs 是有序的上下文绑定；
3. π 的数组顺序属于施工合同，必须保持 Lift 产生的依赖/构造顺序，不得按字典序
   重新排序，也不得把它解释成无序集合；
4. 规则实例只选择“如何实现”，具体算子、常量、operands、归约函数/轴/identity、
   predicate/fill/store guard、effect、output mapping 与调度值仍分别由 J/B/C/
   接口拥有，并通过引用读取；
5. 重写员只获得 π 选中的规则卡，不获得可用于补齐遗漏步骤的隐藏 baseline；
6. 完全重复的 rule/target/context 实例直接拒绝；同一 alternative family 在同一
   target 上选择两个机制也直接拒绝；
7. 非平凡 Kernel 的空 π 不合法。只有 artifact 确实没有任何重放工作时才允许
   `{"steps":[]}`；
8. prompt 记录完整选择轨迹与 registry digest，协议/registry 改变会使旧 cache
   identity 失效。

实现中的 `selection_mode=derived|override` 只是规则卡分类的历史字段：`derived`
表示通常可依据 J/B/C 选择的规范机制，`override` 表示同一 target 上有来源相关的
替代 lowering 机制。它不再表示 derived 卡会被系统自动注入，也不表示 π 只包含
override。两类规则只要适用，都必须由完整 π 显式实例化。

当前 Triton 库共 21 张卡，覆盖 native/wide program domain、integer-offset
index tile、staged pointer tile、masked/unmasked access、staged mask tile、
elementwise、const、显式轴/flattened reduce、broadcast、K-tiled dot、outer
product、indexed access、runtime scalar、C 槽、cast、select、loop carry 与
output mapping。以下 4 张属于有真实样例依据的 alternative lowering family：

- `bind_wide_program_domain`：`program_coordinate_width`；
- `materialize_pointer_tile`：`address_association`；
- `materialize_mask_tile`：`mask_materialization`；
- `realize_flattened_reduce`：`reduction_call_shape`。

例如，普通 elementwise 也必须在 π 中写出 `realize_elementwise(node_ref=n0)`，
但 `add/sub/mul` 的值只从 `J.node[n0].attrs.fn` 读取；归约必须显式选择
`realize_reduce` 或 `realize_flattened_reduce`，但 reducer、axis、identity 与 fill
仍只从 J/B 读取；contraction 必须显式选择 `realize_k_tiled_dot` 或
`realize_outer_product`，但 subscripts 与 operands 不复制进 π。这样既恢复 π 的
完整、严格、可执行属性，又不会让 π 成为 J 的语义副本，paired J intervention
也仍能在 π 冻结时成立。

runtime scalar 支持通过 `tensor_ref` 连到 J `shape=[]` 输入；reduce 卡的 `mean`
除数取全局 J 轴语义长度，elementwise 卡支持 `sqrt+recip`，并通过受限 Triton
libdevice 路径支持已有 `abs/pow`。AST 只允许
`from triton.language.extra.cuda import libdevice` 与 `libdevice.pow`；其他
`libdevice.*` 仍拒绝。tiled-dot 卡从 contraction/J.B 引用 K 语义 extent、K tile
slot/domain、两侧 access/mask/fill、result dtype 与 output tile；普通 matmul 的
schedule-only K loop 不得重复表示成 J `loop/reduce/broadcast`。二维 access 中的
coordinate component 也不得任意重组为完整 offset，向量 predicate 使用源码声明
的 `&`。`m,n->mn` 等没有 contracted label 的 contraction 使用 outer-product
卡；axis-omitted local reduction 使用 flattened-reduce alternative；compiler
slot 只在 launch/compile 时注入。显式轴 reduce 卡 v4 支持 `fn=prod`，并受现有
AST/调用图白名单约束。

后续只根据真实 LLM Lift/source-replay diff 补充规则或 alternative family，不按
想象中的特性枚举扩张，也不得把缺失的 J 语义搬入 π 规避 Gate1。

## 修复与反作弊

- normal replay 生成、编译、接口或结构未闭合时，统一把失败证据回流可看源码的
  Lift，重产 J/π/定位表，最多 3～5 个 artifact 轮次；
- replay writer 不接收失败候选、compiler/comparator repair feedback；下一轮必须
  从静态扫描、源编译和全新 source-blind replay 重新执行；
- 始终保留不回退的最优候选；
- π 每轮先做完整性、规则存在性、target/ref、重复/alternative 冲突与施工顺序
  保留校验，再做静态
  API/source 泄漏扫描：frontend API 拼写扫描 J+π，source
  n-gram 只扫描 π；J.B 本来就负责保存源码提取的 bounds、mask 和 index
  expression，不能被 source n-gram 系统性误杀；
- normal 闭合且存在注册坐标时，paired intervention 是主要硬判据；blank-J 仍
  执行并报告，但只表示可猜性诊断；没有注册坐标时，masked-J fallback 只检查
  是否确定性丢失至少一个 A/B 硬分量，不使用 score ratio threshold；
- normal replay handoff 使用 `opaque_replay_refs_v1`：J/B/π 与 rule-selection 的
  Lift-owned identifiers 统一改成 declaration-order handle，ABI/C handle 只通过
  `reference_bindings` 指向接口/slot 的位置。`a_load/reduce_max_0` 等名称不得进入
  writer packet。

这些是 SCRC 本身的信息归属证明，不是额外 verifier。

## 当前实现

代码入口：`python/intentir/verify/gate1/`。

当前协议身份是 `relift_only_pi_audit_on_strict_opaque_refs_causal_v7`。v4 恢复 π 的
完整、按序、严格属性；v5 把 paired replay 改为受控结构更新，并扩大到最多 3 个
合法 J/B 坐标；v6 收紧 Lift-owned reference id，并把正常 replay view 也统一
不透明化；v7 取消 replay repair 并默认执行最终 π audit。任何 v3/v4/v5/v6 terminal
report 或 cache 只保留为历史证据，不能静默升级或混入 v7 正式结果。

v6 fresh add 已用真实 `gpt-5.6-sol/xhigh` 验证：首个 Lift response 因非 opaque
plan ref `out` 被拒绝，一次修复后得到 `n0/d0/a0..a2`；normal packet target 全为
`p/c/d/a/n/t` handles，Gate1 `REPLAY_CLOSED`、3/3 paired intervention PASS、
Gate2 PASS、0 relift。五个 v5 历史 artifact 的 compatibility probe 也确认旧的
`acc_a/a_x_load/a_load/d_row` 等名字不会进入 v6 writer channel。

π 消融是 diagnostic，不是另一个 verifier，也不修改正常 Gate1 verdict。正式
live/held-out 证据默认开启；只有 quick smoke/纯单元测试允许显式关闭。当前合同
必须同时满足：

- full-π 控制组已经 `REPLAY_CLOSED` 且有 source TTIR 指纹；否则记录
  `pi_ablation_control.valid=false` 并跳过全部消融；
- control 与 ablation 冻结同一 J/B/C/interface，消融阶段不 relift；
- 消融编译必须注入 artifact 的同一组 privileged C 值，禁止使用空 schedule；
- 消融 replay 与 normal replay 一样只做安全解析和编译，不做 AST/ABI/body 代码
  修补；缺参、默认值、opaque handle 或编译错误直接记为 inconclusive；
- 只有生成、编译和 TTIR fingerprint 均成功后，硬分量退化才记为
  `supports_pi_necessity=true`；生成/编译失败均为 inconclusive；
- 消融报告必须保留生成代码和精确硬差分，score 只作诊断。

当前 held-out batch identity 为 schema 17，绑定 relift-only 正常恢复、normal/消融
两条路径均禁用 replay code repair、冻结-C 编译合同和 closed-control 前置条件，旧 schema
不得 resume。

2026-07-16 的真实 `gpt-5.6-sol/xhigh` v6 五核复验中，五个标准 probe 的完整 π
均 `REPLAY_CLOSED`，合法 paired J/B intervention 共 10/10 PASS，normal Gate2
共 5/5 PASS，均为 0 relift。J-only 为 add 1 个闭合、其余 4 个硬退化；step-drop
为 3 个可编译硬退化、2 个生成失败/inconclusive；受控 address mechanism
substitution 为 5/5 一次编译并硬退化。该组 v6 结果中的 13 个可编译消融分支均未
实际触发当时存在的 deterministic adapter；v7 已将该 adapter 整体删除。12 个退化
分支的失败硬类别均为
`op_multiset + dataflow`，reduction 与 predicate/effect 保持通过。该证据支持 π
携带来源相关施工机制而不复制 J 语义，但当前只覆盖首个可替换 address target，
不能证明所有 π step/规则 family 均必要。

额外 `matmul M=7,N=1003,K=19` 边界运行在一次 relift 后仍 `NO_CLOSURE`，但独立
Gate2 diagnostic 的 source↔replay、source↔lowering、replay↔lowering 三对全部
PASS。该样本必须记录为“Gate2 通过、Gate1 不通过”的合法四象限结果；其 full-π
控制无效，所以 π 消融全部排除。

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
- 完整、按序、严格 π + 精确规则检索生成；writer 不注入隐藏 baseline；
- 源与重放 compile-only、TTIR 提取；
- interface/A/B 纯代码比较；
- relift-only source-aware 外循环；normal replay 的 generation/compile/interface/
  hard-diff 失败都会与原始 source、当前 J/π 一起进入下一轮 Lift，interface/C 冻结；
- 每轮重新执行静态扫描、源编译、全新 replay、编译与比较；旧 replay 永不作为
  writer repair base；
- replay compile failure 只向 Lift 回传去路径、去 traceback、去代码行的确定性异常
  摘要；保留失败类别/原因以指导 J/π 修正，但不暴露失败 replay body；
- 静态泄漏、受控多坐标 paired J/B intervention 与 masked-J 诊断/fallback；
- add、row-sum、softmax、RMSNorm、matmul 的随机/边界变异探针。
- fresh v7 `gpt-5.6-sol/xhigh` add 与边界 matmul `7×1003×19` 均以单个、无
  base candidate 的 normal replay 达到硬闭合并通过 Gate2；matmul 未显式传
  `--pi-ablations` 仍自动执行三档 audit。add 的 J-only 闭合、dropout 不可判定、
  mechanism substitution 退化；matmul 三档均退化。两例消融都严格为一次生成/
  一次编译且无 repair。两例 normal replay 首轮闭合，因此不冒充 live relift 证据；
  relift-only 故障路径由 generation/compile/hard-diff 回归测试覆盖。
- 24 个真实 FlagGems Kernel 的非 Trellis 批处理队列、逐 case 原子报告、按完整
  batch identity 恢复的 resume，以及同时绑定 source/compiler view/model/effort/
  registry/Lift-prompt 的 Lift cache；其中 `fg_dot`、`fg_addr`、`fg_ones`、
  `fg_embedding`、`fg_sum/fg_mean/fg_max/fg_min/fg_prod/fg_vector_l2` final stage
  以及 `fg_vector_l1` first stage 已具备真实 contract 并运行。batch identity
  还绑定 Gate2 seed、distribution、逐参数 scale 与
  tolerance，输入策略变化后不得 resume 旧结果。

当前覆盖不等于全部情况：

| 层次 | 已有真实端到端证据 | 部分覆盖 | 尚未覆盖 |
|---|---|---|---|
| A | 基础 elementwise/const、sum/mean/max/min/prod、softmax/RMSNorm 固定链、matmul/dot/outer product | cast/dtype、broadcast、mixed precision、常量敏感性 | RNG、scan/sort/top-k、量化/复数/稀疏、多输出、大型 attention |
| B | 基础 1-D/2-D 仿射索引、tail mask、masked load/store、fill/guard、row reduction、axis-0 take gather | 多 program axis、非末轴/多重归约、动态 stride、复合 mask、一般 along-axis gather | structured/online loop、scan carry、atomic、scatter、block pointer/descriptor、alias-sensitive effects |
| C/前端 | 单一 reviewed profile 的 constexpr/compiler slots/K tile、Triton/FlagGems compile-only | specialization 影响、grid/编译属性、region/loop 指纹 | 多 C/多 specialization 聚合、TileLang/CUDA 正式 Gate1、跨版本规范化、atomics/内存序 |

以下有关 pi-v2/causal-v3 的运行数量属于历史实验口径，用于解释设计演进，不是
strict-v4 的真实端到端覆盖证据。早期五核手写 J/π 是机制探针；后续同五个族加
11 个 FlagGems 曾重走
真实 `gpt-5.6-sol/xhigh` Lift，16 个 normal replay 硬闭合。在 causal-v3 之前，
16/16 都被 blank-J masked 判据拒绝；该结果现在保留为历史可猜性证据，不再
推导“J 没有因果控制力”。历史 causal-v3 曾重新验证 add、row-sum、RMSNorm 与
matmul 为正式 `REPLAY_CLOSED`，softmax 因 off-target intervention drift 仍拒绝。24 个
FlagGems 队列只完成 11 个，其中只有 `fg_ones` 已重跑 causal-v3，不得从样本
数量推导完备性。

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

2026-07-15 pi-v2 历史真实 `gpt-5.6-sol/xhigh` 证据中，add、row-sum、softmax、
RMSNorm、matmul、FlagGems dot 和 FlagGems addr 的 normal source-blind replay
以及 FlagGems ones/embedding/sum-final/mean-final/max-final/min-final/prod-final/
L2-final 与 L1-first 都已达到 TTIR 硬闭合；在完整不透明 ABI/C 消融下，十六者
在当时的 blank-J 主判据下都是 `LEAKAGE_REJECTED`。RMSNorm 的 runtime
`eps`、`mean`、`sqrt+recip` 和 staged pointer 路径已经过真实 Lift/replay
验证；matmul 的单一 `mk,kn->mn` contraction、二维 tail mask、两轮 K tile、
f16 input/f32 output 接缝与 staged pointer 路径也已经过真实 Lift/replay
验证。RMSNorm、matmul 与十一个已运行 FlagGems rejected live 样本的离线 Gate2
对照均三方 PASS，但不改变 Gate1 结论。`fg_dot/fg_addr` 触发四张新卡；后续
`fg_ones/fg_embedding/fg_sum/fg_mean/fg_max/fg_min/fg_vector_l2` 在当时直接复用
19 张卡，
`fg_prod` 只补齐既有 reduce 卡的 prod 映射，`fg_vector_l1` 只补齐既有
elementwise 卡的 abs/pow 映射与受限 libdevice 安全路径。`fg_embedding` 的 J/B
显式保存 data-dependent index expression，因而无需为 gather 发明专用 replay 卡；
其固定静态 J 下 2 个 value profile × 3 seeds 的 Gate2 matrix 为 6/6 PASS，三对
最大误差均为 0。`fg_vector_l1` 的 18-case Gate2 matrix 覆盖
`M={1,1023,1024}` 与 runtime `ord={1.0,1.5,2.0}`，18/18 PASS；两次独立
masked-J 为 `0.906140/1.0`，均拒绝。24 个真实外部 Kernel 只完成前十一例，
仍不能宣称已具备数百 Kernel 覆盖。

### 历史 π v2 真实验证（2026-07-15）

使用 `gpt-5.6-sol/xhigh` 重新执行 add、RMSNorm、matmul 的真实 source-visible
Lift 与 source-blind replay，三例初始 Lift 均一次通过新 π 校验，normal replay
均编译成功且 interface/op multiset/dataflow/reduction/predicate-effect 五个硬分量
全部精确闭合：

- add：`π={"steps":[]}`；
- RMSNorm：π 只有 x/weight/output 三处 `materialize_pointer_tile`；
- matmul：π 只有 A/B/C 三处 `materialize_pointer_tile`，没有
  `realize_k_tiled_dot`、reduce、loop 或 broadcast step。

这组 pi-v2 证据确认新 Lift 不再用 π 复述
elementwise/reduction/contraction 拓扑，且空 π 仍可依靠稳定 derived 卡和 J/B/C
完成重放。但三例的 blank-J masked replay 也都得到五个硬分量完全相同、score=1.0，
当时最终仍为 `LEAKAGE_REJECTED`。因此剩余问题不能归因于旧 π 步骤序列；该结果
直接促成下面的 causal-v3 paired intervention。

对三例 normal hard-closed replay 独立补跑 Gate2，add/RMSNorm/matmul 均为
`gate2_pass`，且 source↔replay、source↔lowering、replay↔lowering 三对全部通过。
这与 Gate1 的三个 `LEAKAGE_REJECTED` 同时成立，直接验证 Gate1/Gate2 不是包含
关系；Gate2 证据不得覆盖 Gate1，Gate1 也不得阻止记录 Gate2 的独立结论。

### 历史 Causal-v3 真实验证（2026-07-15）

协议身份升级为 `normal_no_closure_pi_override_causal_v3`。使用同一
`gpt-5.6-sol/xhigh` 真实 Lift/replay 路径获得：

- add：π 为空；`n0 add→sub` 后 TTIR operator delta 精确为
  `arith.addf:-1, arith.subf:+1`，改变类别恰为 `op_multiset + dataflow`；
  Gate1 `REPLAY_CLOSED`，anti-leakage basis 为 `paired_intervention`，Gate2 PASS；
- RMSNorm：π 仅含三处 pointer association；`n0 mul→add` 后 delta 精确为
  `arith.mulf:-1, arith.addf:+1`，其他硬类别不变；即使 blank-J 仍 score=1.0，
  Gate1 仍依据确定性 paired evidence `REPLAY_CLOSED`，Gate2 PASS；
- softmax：目标 `n1 sub→add` 已发生，但 mutant replay 额外增加
  `arith.constant + arith.select`，因此是 `off_target_drift`，Gate1
  `LEAKAGE_REJECTED`；原 normal replay 的独立 Gate2 diagnostic PASS，不覆盖
  Gate1；
- matmul：A load 的 typed fill `0→1` 后，观察到精确 fill delta；由于 baseline 0
  仍被其他路径共享，只新增 `arith.constant:+1`，其余变化限于
  dataflow/predicate-effect。真实 LLM artifact 按收紧后的有限合同重新裁决为
  `REPLAY_CLOSED`；不得伪造一个不满足 shape 的 contraction operand swap；
- row-sum：首次 fill 实验因冻结 `masked_contribution=0` 而迫使 replay 合法插入
  select，证明该 mutant 合同本身不适用；该坐标已从 fill registry 排除。随后
  使用合法复合干预 `sum/0/0→max/-inf/-inf`，TTIR 精确得到
  `arith.addf:-1, arith.maxnumf:+1`、typed fill `0→-inf`，axis/arity 与其余硬结构
  不变；Gate1 `REPLAY_CLOSED`，anti-leakage basis 为 `paired_intervention`，
  Gate2 PASS。
- FlagGems ones：重新走真实 Lift，π 只有一条 `bind_wide_program_domain`
  override；`n0 output const 1→0` 后 direct store typed constant 精确
  `tensor<1024xf32> 1→0`，改变类别恰为 `dataflow + predicate_effect`，operator、
  fill、reduction 均无差分。Gate1 `REPLAY_CLOSED`，Gate2 三对全部 PASS；其
  blank-J score `0.825` 仅保留为诊断。

因此历史 causal-v3 的真实 LLM 闭合记录为 add、row-sum、RMSNorm、matmul
和 FlagGems ones 五例；softmax 是一例精确 off-target 失败。核心五探针仍是
4/5 闭合，额外 held-out constant 家族新增 1 例闭合。该数量只证明新机制已真实
工作，不代表数百 Kernel 覆盖。

旧 NumPy JSON 解释器已归档到 `python/intentir/verify/back/legacy_v0/`，Gate1
不得导入它。

### Strict causal-v4 / controlled causal-v5 状态（2026-07-15）

v4 在代码、schema、Lift/rewrite prompt 与规则选择路径上恢复了严格 π：

- `ReplayPlan` 保留 Lift 给出的施工顺序，不再做无序集合排序；
- Lift 必须显式生成 canonical 与 alternative 两类全部适用规则实例；
- writer 只接收完整 π 选中的规则卡，不获得隐藏 derived baseline；
- 完整性校验覆盖已注册 IntentIR op、每个 boundary access 的地址与访问步骤、
  broadcast、output mapping、runtime scalar、C slot 和 grid domain；
- 语义值仍只由 J/B/C/接口拥有，π 只保存规则 id 与引用；
- paired intervention 继续冻结完整 π，只修改注册的 J/B 坐标。

v5 不改变上述 π 定义，而是修正因果实验控制：mutant writer 复用 normal 的
模型生成 replay 作为结构起点，未变化 π 步骤不得被自由重构；每条记录保存
`base_generation_id`。live harness 默认最多检查 3 个坐标；held-out batch identity
v5 历史 batch identity 当时为 schema 12；当前 v7 identity 已升为 schema 17。除
intervention、generation 与 π 消融预算外，还绑定
`active_evidence_weighted_v2` score、
`exact_semantic_hard_degradation_v1` masked-J fallback 和
`opaque_replay_refs_v1` reference naming、消融 replay repair 禁用、冻结 C 与 full-π
closed-control 前置条件，防止旧结果静默复用。

当前五核 v5 真实结果如下。add 是 fresh v5 Lift；其余四例显式复用同日刚生成且
通过 strict 校验的 v4 Lift artifact，只重跑 v5 causal/Gate2，报告保留
`reused_from`：

| Kernel | π steps | paired interventions | Gate1 | Gate2 | masked-J |
|---|---:|---:|---|---|---|
| add | 13 | 3/3 PASS | `REPLAY_CLOSED` | PASS | score 1.0 |
| row-sum | 12 | 1/1 PASS | `REPLAY_CLOSED` | PASS | score 1.0 |
| softmax | 18 | 2/2 PASS | `REPLAY_CLOSED` | PASS | score 1.0 |
| RMSNorm | 24 | 2/2 PASS | `REPLAY_CLOSED` | PASS | 接口硬检查失败 |
| matmul | 18 | 2/2 PASS | `REPLAY_CLOSED` | PASS | score 0.5811 |

softmax 的 `sub→add` 与 `div→mul` 都只产生目标 operator/dataflow delta，解决了历史
fresh-mutant generation 引入 `constant/select` 的实验混杂。RMSNorm 的首轮 Lift
因 π 引用未登记中间名 `inv_rms` 被 schema 拒绝，第二次 response 修复；随后
dataflow-aware eligibility 又排除了会破坏 `masked_contribution` 的 `x` fill 假
干预。五例均未触发 semantic relift。

这些结果证明多个已注册 J/B 坐标对 replay 有精确因果控制，但不等于 π 消融已经
完成。当前 v7 live/held-out 正式证据默认开启、只作 diagnostic 的 J-only、确定性
单步 dropout 与 mechanism substitution；quick/unit 才允许显式关闭。原 artifact
先通过 strict-plan validation 和 full-π `REPLAY_CLOSED`，消融 replay 不做任何
代码修复；只有一次生成、编译后丢失硬分量才记为结构级 π necessity，生成/编译
失败为 inconclusive，且消融永不触发 relift。

五核 J-only 中，add 仍硬闭合；row-sum、softmax、RMSNorm、matmul 均硬退化。单步
dropout 中，softmax 硬退化，RMSNorm/matmul 仍硬闭合，add/row-sum 在两次生成内
未形成合法模块。结果证明 J-only/dropout 不能一刀切成为 Gate1 硬门禁：canonical
Kernel 可能由 J/B+规则库唯一确定，同一 rule card 也可能从其他 target 实例被推广
回被删 target。

受控 mechanism substitution 固定 J/B/C/interface，并以 normal 模型生成 replay 为
结构基线，只将首个可用 address association 在 `materialize_index_tile` 与
`materialize_pointer_tile` 之间替换。add、row-sum、softmax、RMSNorm、matmul 五例
均一次编译成功且至少一个 Gate1 硬分量退化。它与“固定 π、修改 J/B”的 paired
intervention 构成两向因果证据；两类诊断都不改变 Gate1 verdict，也不替代独立的
Gate2。当前证据只覆盖每个 Kernel 的首个可用 address mechanism target，尚未覆盖
所有 π step、alternative family 或跨 Kernel π swap，因此不能宣称每张 π 卡均已
证明必要且无冗余。
