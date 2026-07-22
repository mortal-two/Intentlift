# SCRC System Rebuild 01：真实搭建 TypeCheck、Replay、Gate1 与 Gate2 主链

> 状态：待执行
>
> 优先级：P0；新设计序列的第一个且唯一活动任务
>
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis CLI、workflow 或 session
>
> 计划类型：代码系统搭建与可复现实验准备；当前尚未声称实现完成

## 上位设计

本任务只实现下列四份当前文档，不再进行架构辩论：

1. [C：所有权、绑定、消费与三方验证](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
2. [π：依赖类型实现见证与 Replay 安全](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)
3. [Lift、Replay、Gate1、Gate2 最终执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)
4. [SCRC 最终数学、执行流程与论文表述](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)

工程投影见 [活动 spec](../../spec/index.md)。若代码、旧测试或历史 task 与上述文档冲突，修改或
退役代码与派生材料；不得增加兼容旁路来维持旧语义。

## 为什么这是新序列的 Task 01

J、C、π 和最终执行流程已经确定，但活动实现仍处于过渡状态：logical signature 尚未完整收入
J，π 主要仍是自闭合 parser，新的 Gate2 三边比较、统一实验状态和完整 relift runner 也尚未形成
一条真实主链。

因此第一项工作不是继续跑旧 formal batch，也不是继续为某个 Kernel 增加字段，而是一次性搭建：

```text
source-aware Lift
  -> canonical (J,C,pi)
  -> TypeCheck(J,C,pi)
  -> source-isolated Replay
  -> bind the observed c0
  -> observer-relative Gate1
  -> bounded full relift
  -> Q-governed three-edge Gate2
  -> unified certificate/status/report
```

后续规模实验、跨前端扩样、论文统计和 production 性能优化，必须建立在该主链真实闭合之后。

## 最终目标

在活动、可导入的 production path 中实现：

```text
A = Lift_H(S) = (J,C,pi)

TypeCheck(J,C,pi) = PASS

R = W_f(J,pi,MaskC(C);U_f)

c0 = Bind_H(C)

G1 = [Phi_(f,p)(S,c0) == Phi_(f,p)(R,c0)]

G2 = Close_Q(O_S,O_R)
     and Close_Q(O_S,O_J)
     and Close_Q(O_R,O_J)

SCRC_CERTIFIED iff TypeCheck and G1 and G2
```

其中：

- proof artifact 只有 `(J,C,pi)`；
- `p` 是 observer profile，不是 Lift artifact；
- Q 是外部验证政策，不是 Lift artifact，也不进入 Replay 或 RefEval；
- `R` 必须由实际的 `J + pi + MaskC(C) + U_f` request 生成；
- 主证书只覆盖实际 `c0`、声明的 observer profile、Q 和测试域；
- 当前主线只覆盖 deterministic / contract-deterministic Kernel。

## 什么叫“真实搭建完成”

完成不等于 schema 中出现了新字段，也不等于单元测试用手写对象通过。必须同时具备：

1. 一条唯一、活动、可导入的 end-to-end runner；
2. production parser/TypeCheck/Replay/Gate1/Gate2 的真实 consumer；
3. fail-closed 的负例测试和跨模块集成测试；
4. 实际 request/output、observer、Q、c0、测试输入和状态记录；
5. 至少一组真实 Source + 真实 Lift LLM + 全新盲 Replay LLM 的端到端 canary；
6. 全量回归通过，活动代码不存在旧协议入口或静默 fallback。

mock/deterministic provider 可以用于零成本回归，但不能替代真实 canary。

## 架构不变量

### Artifact 与接口

- `Lift_H(S)` 只输出 `(J,C,pi)`。
- J 拥有 logical signature、operator/dataflow、predicate/fill、typed index、effect 与 result contract。
- C 只有匿名 implementation-selection slots 与 Harness 持有的 valuation。
- π 是 `(J,C)` 索引的 typed relational realization witness，不是第二份 J 或程序。
- 不存在独立 interface artifact、`Ids(interface)` 或语义 BindingMap。
- 工程需要接口时只计算 `InterfaceView(J)`，复用 J.signature handles，不独立序列化。

### Replay 与 Harness

- Replay allowlist 严格是 `J + pi + MaskC(C) + U_f`。
- `U_f` 必须版本化且 source-neutral，不能按 Kernel、operator 或失败样本选择。
- Harness 只枚举和绑定已有坐标；不生成 loop、mask、address、reduction、sharing 或 Replay AST。
- Source 与 Replay 使用同一个实际 `c0`；RefEval 只读取 J 与 logical inputs。

### 认证

- TypeCheck 管显式语义所有权、引用、类型和 authorization，不做 SMT 或一般语义等价证明。
- Gate1 管声明 observer 下的 source-blind realization closure，不判断 J 是否正确。
- Gate2 管测试域中的三条直接 `Close_Q`，不依赖近似关系传递性。
- masked-J 只形成独立 `j_necessity_status`，不改变主认证状态。
- Gate2 失败结束本次实验，不把结果回流 Lift 或 Replay。

## 实施阶段与提交节奏

每个阶段完成自己的测试后独立提交。除阶段 7 的真实 canary 外，阶段 0–6 不调用外部 LLM。

### 阶段 0：活动实现基线与断代映射（0 次 LLM）

- 记录当前全量测试基线、活动 import graph 与现有 Gate1/Gate2 入口。
- 列出需要删除或重写的独立 interface、仅 `Parse(pi)`、旧 runner/receipt 和两边 Gate2 call sites。
- 区分可复用基础设施：J semantic core、compiler adapters、Harness build/run、RefEval、observer。
- 明确新入口和删除顺序；不写兼容转换器，不恢复已归档 formal runner。
- 形成 `research/stage0-active-implementation-map.md`，其结论必须引用真实符号和测试，而不是只按文件名判断。

阶段完成条件：基线可复现，迁移目标和可保留模块逐项明确，没有代码改动混入审计提交。

### 阶段 1：Canonical `(J,C,pi)` 与 logical signature 断代（0 次 LLM）

- 将 inputs、outputs、runtime scalars、destinations、dtype/rank/shape、alias/inout/effect 收入 J.signature。
- 删除活动 wire/schema/API 中的独立 interface 输入和命名空间。
- 只保留确定性的 `InterfaceView(J)` 工程投影。
- 固定一个 canonical C：slot schema + Harness valuation；`MaskC(C)` 删除 value 和 private provenance。
- π refs 只允许 `Ids(J) union Ids(C)`，其中 J.signature handles 属于 `Ids(J)`。
- 新格式直接拒绝旧格式；不建立新旧双轨或自动迁移器。

阶段完成条件：artifact round-trip、unknown-field rejection、旧 interface rejection 和 J.signature
consumer 测试全部通过。

### 阶段 2：最小 dependent `TypeCheck(J,C,pi)`（0 次 LLM）

将语法解析与 dependent TypeCheck 分开。首版必须机械检查：

1. J/C/signature ref 全部存在；
2. route operand slot、arity、type、region/order/sharing 合法且无环；
3. value/store/effect-producing observable instance 锚定 J-owned node/access/effect；
4. C/π 不声明 operator、predicate、fill、formula、semantic constant、Source/API、C value 或可执行自由文本；
5. integer width、numeric cast、accumulation dtype、reassociation、atomic form 等 sensitive refinement
   引用 J/B authorization，或属于系统级 closed `GlobalSafeRefinement`；
6. 未知字段、未知 relation 和自由文本扩展 fail closed。

TypeCheck 的结构化错误只可返回 source-aware Lift。正式 claim 只到：

```text
FreshLogicalSemDecl(C,pi | J) = empty
```

不得声称 π 没有统计语义线索，或 TypeCheck 已证明 realization 行为正确。

阶段完成条件：每项检查都有正例、最小负例和错误分类；活动主链不再用 `Parse(pi)` 冒充
dependent validation。

### 阶段 3：Source-isolated Replay 与薄 Harness（0 次 LLM）

- 建立唯一 Replay request builder，字段 allowlist 只有 J、π、`MaskC(C)` 和版本化 `U_f`。
- 明确拒绝 Source/Harness、C values、source name/position、BindingMap、interface sidecar、Gate1 diff、
  source/compiler fingerprint、prior Replay、旧 conversation 与 per-kernel RuleCard。
- Replay 每次输出完整、参数化、可独立构建的当前前端源码；固定代码不从 π 施工源码。
- Harness 只在 Replay 生成完成后绑定同一个 c0，不新增 use-site、不修改 AST。
- RefEval API 在类型层面只接收 J 与 logical inputs；Q 只比较其输出。
- 保存普通 request/output 与工具日志，但日志不回流 Replay，也不成为 hash/commitment gate。

阶段完成条件：packet allowlist/denylist、fresh request、same-c0 binding、RefEval 权限和 source-isolation
集成测试通过。

### 阶段 4：Observer-relative Gate1 与完整 relift（0 次 LLM）

- 统一 frontend adapter 合同：compile、observe、canonicalize、run。
- observer profile 必须记录 frontend、profile/version、compiler/dump stage、canonicalization version。
- Gate1 直接比较同一 c0 下 Source/Replay 的 canonical observation。
- Gate1 diff 只报告确定性的结构差异，不自行判断 J/π/schema/prompt 应怎样修改。
- diff 只返回 source-aware Lift；下一轮必须产生完整新 `(J,C,pi)` 和全新盲 Replay。
- 最多三次 TypeCheck-valid、真正进入 Replay 的正式轮次；三轮不闭合报告 `NO_CLOSURE`。
- TypeCheck 格式/引用修正单独计数并有上限，所有 LLM 调用都进入预算记录。
- masked-J 是可选独立 audit；hard close + legal π 只得到 `NOT_PROVED`，不改写 main status。

阶段完成条件：成功、结构 diff、完整 relift、fresh Replay、三轮终止、tool failure、masked-J
状态分离均有确定性测试。

### 阶段 5：外部 Q 与三边 Gate2（0 次 LLM）

- Q 作为版本化验证政策存在于 runner/config，不由 Lift 输出。
- J/B 只能声明 closed comparison class 与必要合同；Q 决定 exact/atol/rtol/ULP 模板、上限及
  NaN/Inf 处理。
- `Close_Q` 必须确定且对称；若底层 `isclose` 有方向性，协议执行显式对称化或双向检查。
- Source、Replay 和 RefEval(J) 的输出各执行一次，然后直接比较三条边。
- 加入非传递反例：两条 Source-centered 边通过但 Replay–J 超界时，Gate2 必须失败。
- 当前只接收 deterministic / contract-deterministic Kernel。真正集合值 `Accept_{J,Q}` 不在本任务实现。
- 支持固定有限 T 与 iid 隐藏 challenge 两种模式；隐藏输入只在最终候选选定后生成。
- Gate2 结果不回流本次实验；失败立即得到 `GATE2_FAILED`。

阶段完成条件：Q 权限、候选 tolerance 越权、三边 verdict、fixed-T 声明、hidden-input 顺序和
Gate2 terminal behavior 测试全部通过。

### 阶段 6：统一 runner、状态、报告与系统改进边界（0 次 LLM）

建立一条 runner，而不是 Gate1/Gate2 各自维护候选生命周期：

```text
main_status        = SCRC_CERTIFIED | NO_CLOSURE | GATE2_FAILED | TOOL_FAILED | ...
typecheck_status   = PASS | FAIL
j_necessity_status = PROVED | NOT_PROVED | INCONCLUSIVE
```

每份记录至少包含：

- 实际 J/C/π/Replay 与 Replay request/output；
- frontend、observer profile/version、compiler/dump/canonicalization versions；
- Q/protocol version、测试模式、D 或 T 标识、实际 c0；
- TypeCheck 结果、每轮 Gate1 observation/diff、Gate2 三条边结果；
- Lift calls、Replay calls、total LLM calls、失败分类与最终状态。

批量统计至少输出：

```text
first-attempt TypeCheck pass rate
closure@1 / closure@2 / closure@3
NO_CLOSURE failure categories
Gate2 pass rate among Gate1-closed candidates
SCRC_CERTIFIED rate
```

`closure@k` 是截至第 k 个正式轮次的累计闭合率。receipt 标识只用于追溯，不参与候选合法性、
信息隔离或认证结论。

系统改进必须发生在 terminal result 之后：聚合失败，修改通用 J/π/C/Replay/observer/Q/tool
能力，然后从一次新实验重新开始。不得把新系统规则伪装成旧实验的续轮。

阶段完成条件：状态机、预算统计、失败路由和新实验边界由集成测试机械覆盖。

### 阶段 7：零 LLM preflight 与真实 canary

先运行零 LLM preflight：

- 全量单元/集成测试；
- active import、旧 symbol 与 Replay packet 扫描；
- canonical artifact/TypeCheck/runner fixtures；
- frontend compiler/observer、RefEval、Q 和 test generator capability 检查；
- 确认任何 unsupported capability 在 LLM 调用前 fail closed。

preflight 全部通过后，按由简到难的顺序运行真实 canary：

1. exact elementwise：验证三对象、J.signature、Replay isolation、Gate1 和三边 exact Gate2；
2. masked access：验证 predicate/fill、logical index、π realization 与 observer profile；
3. reduction + typed representation：验证 reduction、cast/width/reassociation authorization 与 Q；
4. 若 J/B 与 RefEval 已真实支持，再加入一个 contract-deterministic effect/update；不得为通过 canary
   增加 Kernel-specific schema 或 comparator。

每个 canary 使用真实 Source、真实 source-aware Lift 和全新 source-blind Replay。按统一 runner
执行，最多三轮；任一 terminal failure 按设计停止，不在同一实验中修改系统。前三类至少各有一个
SCRC_CERTIFIED 后，才允许创建规模实验任务；本任务不直接恢复 28-kernel batch。

阶段完成条件：真实 request/output、三边 Gate2、调用预算、失败/成功状态和环境信息完整留证，
并由全量回归确认 canary 没有引入 case-specific 旁路。

## 必须删除或禁止恢复的旧旁路

活动 production path 中不得存在：

- artifact hash/digest/commitment/epoch 认证门或所谓冻结状态机；
- 独立 interface/LogicalABI/BindingMap 语义对象；
- deterministic composer、RuleCard/RuleRegistry、selected card、`C.strategy`；
- diff-guided Replay、Replay patch、跨候选复用 Replay 或旧 conversation 回流；
- Harness/Injector 生成实现结构或修改 Replay AST；
- 仅 `Parse(pi)` 就进入 Replay；
- candidate-defined comparator、任意 tolerance 或“所有输出均可接受”；
- 两条 Source-centered 比较冒充三方两两闭合；
- `masked hard-close -> LEAKAGE_REJECTED`；
- Gate2 failure 后继续 relift；
- 在线 J mutation、J-authority challenge、π swap/dropout 或 family sweep；
- 真正集合值 conformance、production 性能优化或大规模 batch 偷渡进本任务。

通用 cache key、文件校验和 Git 对象可以使用摘要；它们不能决定实验合法性或认证结论。

## 测试矩阵

### Artifact / TypeCheck

- canonical `(J,C,pi)` round-trip；
- J.signature handles 正负引用；
- independent interface/unknown field 拒绝；
- observable C-anchor、semantic atom、API/source/C-value/free-text 拒绝；
- sensitive refinement 有/无 authorization；
- route slot/type/arity、region/order/sharing cycle 与 dangling ref。

### Replay / Harness

- packet allowlist 精确相等；
- 每个禁止输入单独 mutation 后拒绝；
- U_f source-neutral/versioned；
- fresh Replay per candidate；
- same-c0 binding；
- RefEval 无 C/π/Q 接口。

### Gate1 / Runner

- observer profile 缺失、unsupported 或 evidence 不完整 fail closed/inconclusive；
- exact observation closure 与结构 diff；
- diff 只到 Lift；
- relift 输出完整 artifact，Replay 不读旧候选；
- `closure@1/2/3` 与三轮 `NO_CLOSURE`；
- masked-J 三态不改变 main status。

### Gate2

- exact integer、对称 tolerance、NaN/Inf policy；
- candidate tolerance 越权拒绝；
- Source–Replay、Source–J、Replay–J 三条边独立失败；
- 非传递近似反例必须由第三边拒绝；
- fixed T 与 hidden iid 模式分开；
- Gate2 failure 不反馈、不 relift；
- 真正集合值 Kernel 明确 unsupported，而不是静默套用三边近似。

## 总体验收标准

1. 活动 Lift artifact 和 Replay wire 的 proof-relevant 对象严格只有 J、C、π。
2. logical signature 已收入 J；活动 schema/API 不存在独立 interface namespace。
3. production runner 在 Replay 前真实执行 dependent `TypeCheck(J,C,pi)`。
4. Replay 实际 request 精确等于 `J + pi + MaskC(C) + U_f`，且每个候选都生成全新 Replay。
5. Harness 只绑定同一 c0；RefEval 只读取 J 和 logical inputs。
6. Gate1 绑定 observer profile，结构化 diff 只返回 Lift，最多三轮完整 relift。
7. Q 不受候选控制，`Close_Q` 对称且三条 Gate2 边全部直接执行。
8. Gate2 失败结束实验；fixed-T 与 hidden-iid claim 不混用。
9. `main_status`、`typecheck_status`、`j_necessity_status` 独立且机械一致。
10. 实验内只搜索候选；系统能力修改只发生在 terminal result 之后的新实验。
11. 真实 exact、masked、reduction/typed-representation canary 各至少一个获得完整证书。
12. 全量回归、active legacy-symbol scan、packet scan 和 `git diff --check` 通过。
13. 活动路径不存在兼容双轨、case-specific schema、RuleCard/composer、hash gate 或静默 fallback。
14. 每阶段独立提交；任务完成时工作区干净，证据与最终 commit 列表写回本文件。

## 非目标

- 不在本任务中扩充真正集合值语义或并发调度 membership proof。
- 不证明所有 admissible C、所有输入、所有 observer 或所有前端均正确。
- 不做 production backend 性能、retune、cost model 或“击败 Triton”实验。
- 不运行完整 28-kernel/多模型/大规模随机 batch。
- 不新增 Agent、Gate、IR、per-frontend lowerer 或 frontend formal semantics。
- 不恢复 Trellis workflow、task.json、自动 session 或 Material Passport 作为运行依赖。

## 当前第一步

执行阶段 0：在不调用 LLM、不修改 production code 的前提下，建立当前活动实现到七个目标阶段的
逐符号迁移表，记录测试基线，并确定阶段 1 的唯一断代边界。完成审计提交后才开始 J.signature 与
artifact schema 的实质重构。
