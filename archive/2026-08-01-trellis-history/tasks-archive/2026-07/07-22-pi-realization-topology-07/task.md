# Semantic/Realization Residual 07：修复 J/C 分区并补齐匿名实现证据

> 状态：已结束（J/C/π 所有权与 v4 离线合同已闭合；唯一 canary 暴露 π 图外键可生成性缺口）
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow
> 归档原因：Task07 的所有权回收与 residual contract 工作已完成，v4 唯一调用已耗尽并完整留证；
> `semantic_ref`/`instance.id` 命名空间歧义转入 Task08，不在冻结 epoch 内重试或 repair

## 为什么这是独立任务

Protocol-v3 Swiglu canary 已经把失败边界推进了一层：Lift response 通过完整 schema/parser，
normal Replay 也成功生成并编译，但 strict Gate1 因 source/replay TTIR dataflow 不同而
`no_closure`。source 保留 runtime stride 的 row-base pointer topology；Replay 把同一 frozen
specialization 扁平化。两者语义等价，但 Gate1 按设计不允许闭合。

阶段 0 审计纠正了最初的单因归因：当前 schema **已经允许** pi instance 引用 interface
`p3` 和 C handle，但 v3 Lift 没有这样做；更重要的是，v3 把具有逻辑 extent 意义的
`n_cols` 放进 C，并把源码的 row/column 逻辑因子压成一维 J。故该反例不是单纯“再给 pi
加一个 stride 字段”，而是 `J/C/harness` 所有权错误与 realization residual 缺失的组合。

Task07 必须先修正所有权，再判断现有 region/instance/route/order/sharing 是否仍然信息不足。
pi 不能成为遗漏语义的补丁层。

## 最终目标

建立最小的、全局的、source-neutral semantic/realization 分区，使 source-aware Lift LLM
可以直接表达、source-blind Replay LLM 可以直接消费：

```text
J(A+B)       = what is computed / allowed result relation
C            = pure scheduling choices and values
pi           = only the source-realization residual not already owned by J/C
interface    = anonymous positional/public types only
```

目标不是让确定性程序从 J compose kernel，而是消除 Replay 的信息不充分：若两个 source 在严格
Gate1 下具有不同 realization topology，它们的 pi 必须能够不同；若只改变 J 语义而保持 pi，
masked-J 必须不能继续闭合。

## 架构不变量

- Canonical Intent 仍只有 `J(A+B) + pure C`；pi 始终在 Intent 外。
- pi 由 source-aware Lift LLM 直接写出，Replay LLM 直接读取；不引入 composer、RuleCard/
  Registry、IntentProgram、LogicalABI、kernel-name rule 或 deterministic source-to-pi extractor。
- Replay 看不到源码、真实 path/symbol、source TTIR/fingerprint、compiler diff 或历史 response。
- Gate1 保持严格 normalized compiler topology；不得通过弱化 address/dataflow/predicate 比较闭合。
- pi 只使用匿名 J/C/interface/region handles 与 source-neutral realization structure；禁止 source
  identifier、源码片段、frontend helper 名、case 名和自然语言策略。
- pi 不得首次引入任何会改变合法输入下 observable result/access/effect 的依赖。实现级 cast、
  scalar/vector materialization、address staging 只有在 J 已经闭合逻辑访问后才可能进入 pi。
- `n_cols`/shape/bounds 一类语义 extent 不能伪装成 pure-C schedule；`BLOCK_SIZE`/num_warps
  才是合法 C。匿名化 spelling 不改变这个所有权判断。
- 语义隔离最终只保留一次 masked-J 系统检查，其余作为论文消融。
- 新 protocol/manifest/preflight 提交前在线 LLM 调用保持 0。

## 阶段与提交节奏

### 0. 冻结反例并完成所有权/信息流审计（0 次 LLM）

- 固定 v3 source/replay TTIR、fingerprint、artifact 与 receipt SHA；完整列出地址、cast、mask、
  pointer topology 差异，不把 hash-id 噪声当根因；
- 审计 pi model/schema/parser、Lift prompt 和 Replay packet/consumer，明确哪些字段被真正消费；
- 逐项判断 `p3`、`n_cols`、row/column shape、cast/addptr/mask topology 的 J/C/pi/harness
  所有权；不能把语义遗漏归入 pi；
- 构造 source-neutral fixtures：runtime-stride/flat realization，以及 J-owned
  `select -> square -> reduce` 与 pi-owned staged/share 的边界；
- 对 copy/atomic/async/fusion 做需求边界，区分通用表示需求与未来能力需求。

完成后提交。

### 1. 修复 semantic specialization 分区（0 次 LLM）

- pure C 只接收真正不改变 J denotation 的 schedule/compiler option；语义 constexpr/extent
  留在 J 与匿名 interface/harness，不建立 LogicalABI 或第二份 semantic wire；
- Lift 必须保留 source whole-tensor logical rank/axes/access relation，不能因为 compiler fake
  tensor 是 flat allocation 就抹掉 row/column 语义；
- pi 必须覆盖源码中实际参与 realization 的匿名 interface/C handles，但只能引用，不能首次
  定义逻辑 index、predicate、fill 或 value；
- 先用现有 pi 图验证表达力；只有离线反例证明缺少某类 Gate1-observable residual 时，才增加
  最小 source-neutral relation，不增加 kernel/op family 标签。

完成后提交。

### 2. 最小 pi realization contract（0 次 LLM）

- 只增加表达严格 replay 所必需的匿名 realization graph/relations；优先复用现有 region、instance、
  route/order/sharing，而不是再建一个并行 control plane；
- 至少能表达已由 J 规定的 logical access 如何使用 interface/C/program/lane realization、
  scalar/vector decomposition 与必要的 representation-only cast/materialization；
- compute semantic nodes仍由 `semantic_ref` 绑定 J，realization node 不能凭空定义 observable
  tensor math；validator 只做 closed shape、opaque reference 与 scope/def-use closure；
- Lift/Replay prompt 从同一合同消费，不写 Swiglu/LayerNorm 专项规则，不做 response repair。

完成后提交。

### 3. 双向 conformance 与泄漏边界（0 次 LLM）

- typed pi `parse -> to_json -> parse` 幂等，wire/parser 接受域一致；
- offline fixtures 必须区分 row-stride 与 flat realization、J-owned pre/post-mask value DAG、
  direct/bulk/TMA、logical update 与 atomic realization；不要求一个 fixture 覆盖所有 frontend；
- 删除/扰动 realization edge 必须使对应 strict topology fixture 不闭合；
- mask J 的 semantic operands/attrs 后，pi+interface+C 不足以重建原 semantic compute graph；
- 全量回归通过，历史 v1/v2/v3 evidence 一个字节不改。

完成后提交。

### 4. 新 epoch 单一 canary

- 只有离线合同闭合后才冻结完整自包含 protocol-v4 manifest/wire/preflight；
- 绑定 v3 `gate1_failed` receipt，fresh Swiglu 最多 1 Lift + 1 normal Replay；normal Gate1 闭合后
  才允许 1 masked-J；
- normal 必须 strict hard-close，masked-J 必须不闭合；失败立即停止，不扩散其余 27 项。

成功后归档 Task07，并创建 Task08 执行剩余 27 case、Gate2、backend 与 performance。

## 验收标准

1. 对 v3 no-closure 给出可复核的所有权与信息反例，而非归因于随机模型质量；
2. J/C 分区不再把 semantic extent 放入 C，也不让 pi 首次定义逻辑访问；
3. 最小 residual contract 能区分 runtime-stride/flat，并证明 pre/post-mask 的 operator/
   predicate/fill 仍由 J 唯一拥有；
4. Lift 与 Replay 都消费同一全局合同，没有 deterministic composer 或 per-kernel registry；
5. wire/parser/round-trip/negative conformance 全通过；
6. pi masked-J 证明失败：pi 不能独立恢复完整 J 语义；
7. 历史 evidence 不修改，新 epoch 在调用前冻结且 28/28 ready；
8. 唯一 canary 调用链 provenance 完整，结束时提交、归档并创建下一任务。

## 阶段 0--3 完成记录（2026-07-22，0 次 LLM）

- 冻结并重审 v3 TTIR/receipt 后，纠正“只缺 π stride 字段”的错误单因归因：v3 同时存在
  `n_cols` 被误归 C、logical rank 被扁平化以及 π 未记录 p3 realization dependency 三个问题；
  operator/predicate/fill/value DAG 始终归 J，不能靠 π 修复。
- `SCRCCompilerView` 新增中性的 `bound_constexpr_values`：它只是匿名 frozen invocation binding，
  不建立 LogicalABI。Lift 依据源码后果把逻辑 extent/value/predicate 写入 J，把纯表示后果引用到
  π；只有 schedule/compiler option 进入 C。Replay 从同一次 frozen source invocation 按位置复制
  non-C constexpr，不能从 C 获取。
- 对全部 28 个 frozen source constexpr 逐一按源码 use 分类，manifest-v4 用唯一有序 parameter
  list 表达 source ABI 顺序，避免 runtime/constexpr 两张表产生交错顺序漂移。
- 不扩 π JSON capacity，而是冻结现有 `region + instance + route + order + sharing` 的唯一解释；
  离线 fixtures 已区分 row-stride/flat、重复 anchor、多级 chain、producer sharing，并证明
  LayerNorm pre/post mask 的差异仍由 J 唯一拥有。
- Lift/Replay 共用同一个 source-neutral residual contract；没有 composer、RuleCard/Registry、
  kernel-name branch、deterministic source-to-π extractor 或 response repair。
- targeted conformance 通过 `46 passed`；冻结 v4 前全量回归为 `744 passed, 4 skipped`。历史
  v1/v2/v3 evidence 未修改。

## 阶段 4A v4 冻结记录（2026-07-22，0 次 LLM）

- 冻结 `test/real_llm_batch_28_v4.json`，manifest digest
  `0da031b95049acf5884e1adb60126253ab5441d6f7de142a0373e2659ab98b18`；Swiglu logical tensors
  为 `[128,1024]`，`n_cols` 为 non-C binding，`BLOCK_SIZE` 为 pure-C schedule。
- formal runner 只接受精确的 v3 `gate1_failed` predecessor，并校验其一次 Lift、一次 normal
  Replay、0 masked-J、strict `no_closure` 与冻结 SHA，不能把旧 attempt 当 fresh evidence。
- 新增 0-LLM formal preflight CLI，冻结
  `artifacts/real_llm_batch_28_v4/preflight.json`，SHA
  `74862a1402776d68988f3e704b2ceb22d61e89a5849e6a8c1da6c278e585fa30`；26/26 Triton、
  2/2 CuTe、28/28 source observer/replay arm ready，`online_llm_calls=0`。

## 阶段 4B 唯一 canary 结果（2026-07-22）

- Swiglu 只消耗一次 v4 Lift：input `9555`、output `14152`、total `23707` tokens，provider
  attempts `1`；随后本地 parser 返回
  `SchemaError: pi route instance is unknown`，receipt 状态为 `lift_failed`。
- 正确 fail closed 后 normal Replay、masked-J、Gate1 均为 0；没有重试、没有手工修 artifact、
  没有扩散其余 27 case。
- 返回对象已正确保留二维 J、non-C `p4`、pure-C `c0/c1`，也尝试让 `p3` 参与三个 row-base
  realization；失败不是 v3 所有权问题复发。
- 精确错误是 30 条 routes 中有 9 条把 raw `p*/c*` 当 source；后续完整审计还发现 4 条
  sharing 中有 1 条同样以 raw `p3` 为 source。冻结合同要求所有 graph edge endpoint 都是已声明
  的 `i*`，J/C/interface handle 只能作为 instance 的 anchor。当前 prompt 的
  “handle 必须出现并 route”以及研究文档的 `p* -> access` 简写没有显式区分这两个命名空间。
- 这只证明 π wire 的 referential-integrity 合同对 LLM 仍有歧义；尚未得到合法 artifact，故不能
  据此增加 `address_base/address_lane`，也不能把结果写成 Gate1 no-closure。
- 完整 request/raw response/receipt 在
  `artifacts/real_llm_batch_28_v4/recovery/liger_swiglu_forward/`；逐项审计见
  `research/v4-canary-lift-rejection.md`。

## 结束判定

Task07 已完成 J/C/π 所有权回收、non-C constexpr 分区、现有 residual graph 的离线定义与 v4
冻结实验。系统级“normal Gate1 闭合且 masked-J 不闭合”验收没有达成，因为唯一 canary 在进入
Replay 前暴露了一个更窄、可独立处理的 graph foreign-key 可生成性问题。

继续在 Task07/v4 内修改 prompt 或重试会改写冻结实验。后续 Task08 只闭合 π 图节点命名空间、
schema/prompt/parser 可生成合同与 path-specific fail-closed 诊断，再以新 epoch 绑定本次
`lift_failed` receipt 运行一次 fresh canary；在合法 π 仍无法闭合 Gate1 之前，不扩 π capacity。
