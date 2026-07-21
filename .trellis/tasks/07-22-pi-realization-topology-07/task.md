# Semantic/Realization Residual 07：修复 J/C 分区并补齐匿名实现证据

> 状态：进行中
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

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

## 当前第一步

只做阶段 0，不改 schema，不写实现，不调用 LLM。以 v3 Swiglu evidence 为权威，先证明
每项差异的 J/C/pi/harness 所有权；只有排除 J/C 分区错误后，才允许把剩余缺口命名为
pi realization residual。
