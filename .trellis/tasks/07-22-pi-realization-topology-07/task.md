# Pi Realization Topology 07：补齐 source-blind Replay 的匿名实现证据

> 状态：进行中
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

## 为什么这是独立任务

Protocol-v3 Swiglu canary 已经把失败边界推进了一层：Lift response 通过完整 schema/parser，
normal Replay 也成功生成并编译，但 strict Gate1 因 source/replay TTIR dataflow 不同而
`no_closure`。source 保留 runtime stride 的 row-base pointer topology；Replay 把同一 frozen
specialization 扁平化。两者语义等价，但 Gate1 按设计不允许闭合。

当前 artifact 中 `p3` 只存在于匿名 interface，J 只有 whole-tensor logical access，C 只有
schedule values，pi 只有 region、J/access instance、route、order 与 sharing。没有任何字段说明
`p3` 如何参与 row base、pointer add、integer cast 与 mask construction。因此这不是 Replay LLM
“不够聪明”，而是同一 `J + C + pi + interface` 信息上确实存在多个合法 realization。

## 最终目标

建立最小的、全局的、source-neutral pi realization topology，使 source-aware Lift LLM 可以直接
表达、source-blind Replay LLM 可以直接消费：

```text
J(A+B)       = what is computed / allowed result relation
C            = pure scheduling choices and values
pi           = how this source realizes J under C, including topology needed by Gate1
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
- 允许 pi 携带实现算术/placement 的结构，不等于允许它成为语义副本：pi 不得新增 J 中不存在的
  observable value/result relation；语义隔离最终只保留一次 masked-J 系统检查，其余作为论文消融。
- 新 protocol/manifest/preflight 提交前在线 LLM 调用保持 0。

## 阶段与提交节奏

### 0. 冻结反例并证明当前 pi 信息不充分（0 次 LLM）

- 固定 v3 source/replay TTIR、fingerprint、artifact 与 receipt SHA；完整列出地址、cast、mask、
  pointer topology 差异，不把 hash-id 噪声当根因；
- 审计 pi model/schema/parser、Lift prompt 和 Replay packet/consumer，明确哪些字段被真正消费；
- 构造两个 source-neutral topology fixture：runtime-stride row/column 与 flattened linear address，
  证明它们共享当前 J/C/pi 却在 Gate1 下不同；
- 对 LayerNorm `mask -> centered -> square` 与 `square -> reduction mask` 反例、copy/atomic/async
  realization 做需求边界，区分通用表示需求与未来能力需求。

完成后提交。

### 1. 最小 pi realization contract（0 次 LLM）

- 只增加表达严格 replay 所必需的匿名 realization graph/relations；优先复用现有 region、instance、
  route/order/sharing，而不是再建一个并行 control plane；
- 至少能表达 interface/C/program/lane 到 address/predicate/access 的 def-use、scalar/vector
  decomposition、必要 cast width，以及 predicate 在哪个 access/compute realization 前生效；
- compute semantic nodes仍由 `semantic_ref` 绑定 J，realization node 不能凭空定义 observable
  tensor math；validator 只做 closed shape、opaque reference 与 scope/def-use closure；
- Lift/Replay prompt 从同一合同消费，不写 Swiglu/LayerNorm 专项规则，不做 response repair。

完成后提交。

### 2. 双向 conformance 与泄漏边界（0 次 LLM）

- typed pi `parse -> to_json -> parse` 幂等，wire/parser 接受域一致；
- offline replay fixtures 必须区分 row-stride 与 flat address、pre-mask 与 post-mask、direct/bulk/TMA、
  logical update 与 atomic realization；不要求一个 fixture 覆盖所有 frontend；
- 删除/扰动 realization edge 必须使对应 strict topology fixture 不闭合；
- mask J 的 semantic operands/attrs 后，pi+interface+C 不足以重建原 semantic compute graph；
- 全量回归通过，历史 v1/v2/v3 evidence 一个字节不改。

完成后提交。

### 3. 新 epoch 单一 canary

- 只有离线合同闭合后才冻结完整自包含 protocol-v4 manifest/wire/preflight；
- 绑定 v3 `gate1_failed` receipt，fresh Swiglu 最多 1 Lift + 1 normal Replay；normal Gate1 闭合后
  才允许 1 masked-J；
- normal 必须 strict hard-close，masked-J 必须不闭合；失败立即停止，不扩散其余 27 项。

成功后归档 Task07，并创建 Task08 执行剩余 27 case、Gate2、backend 与 performance。

## 验收标准

1. 对 v3 no-closure 给出信息论层面的可复核反例，而非归因于随机模型质量；
2. pi 最小扩展能区分至少 runtime-stride/flat 与 pre-mask/post-mask 两类真实 topology；
3. Lift 与 Replay 都消费同一全局合同，没有 deterministic composer 或 per-kernel registry；
4. wire/parser/round-trip/negative conformance 全通过；
5. pi masked-J 证明失败：pi 不能独立恢复完整 J 语义；
6. 历史 evidence 不修改，新 epoch 在调用前冻结且 28/28 ready；
7. 唯一 canary 调用链 provenance 完整，结束时提交、归档并创建下一任务。

## 当前第一步

只做阶段 0，不改 schema，不写实现，不调用 LLM。以 v3 Swiglu evidence 为权威，审计当前 pi
从 Lift wire 到 Replay packet 的实际信息流，并用最小双 topology fixture 证明缺失的是通用
realization relation，而不是某个 kernel 的提示词或 Gate1 normalization bug。
