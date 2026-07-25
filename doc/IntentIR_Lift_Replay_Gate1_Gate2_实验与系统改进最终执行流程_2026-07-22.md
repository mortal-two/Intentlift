# IntentIR：Lift、Replay、Gate1、Gate2 实验与系统改进最终执行流程

> 状态：最终执行设计（2026-07-22）
>
> 前置设计：[IntentIR C：所有权、绑定、消费与三方验证设计](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
>
> 前置设计：[IntentIR π：依赖类型实现见证与 Replay 安全设计](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)
>
> 数学与 claim 总述：[IntentIR / SCRC 最终数学、执行流程与论文表述](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)
>
> Claim 收紧复审：[最终数学与三份设计的局部收紧建议](source-notes/高级ai对最终数学与三份设计的局部收紧建议_2026-07-22.md)
>
> 纠错与复审：[旧执行流程纠错](source-notes/高级ai对于之前流程的纠错和判断.md)、[三份核心设计复审与四项修正](source-notes/高级ai对三份核心设计的复审与四项修正_2026-07-22.md)

## 1. 这份文档解决什么问题

C 和 π 已经确定了对象所有权、Replay 可见范围以及 Gate1/Gate2 的验证关系。本文件只确定运行方式：

1. 一次实验怎样从 Lift 开始并走到 Gate2；
2. Gate1 失败后怎样进行 relift；
3. 什么算一次正式实验轮次；
4. 什么时候停止实验；
5. 实验结束后怎样修改系统并重新开始。

本文件不引入新的证明对象、额外 Gate、额外 Agent 或第四个 Replay 输入。

最核心的原则是：

> **一次实验运行期间，系统规则保持不变；实验结束后，才根据失败证据修改系统。**

这里的“系统规则”包括 J/C/π schema、Lift prompt、Replay prompt、Gate1 observer profile、比较
政策 Q、编译器和本次实验使用的工具能力。它只表示一次实验中的运行纪律。

## 2. 系统中的对象和职责

### 2.1 Lift

Lift 是 source-aware 的抽取过程：

```text
Lift_H(Source) -> (J, C, π)
```

它可以看到 Source 以及完成 J/C 分类所需的 Harness 机械信息，但不能产生 `(J,C,π)` 之外的证明相关对象。

### 2.2 J

J 是唯一的语义对象，负责表达：

- value 和 state relation；
- logical domain、predicate、fill；
- logical index 和 typed index relation；
- logical signature（inputs、outputs、runtime scalar、dtype、rank、shape、alias/inout）；
- operator、reducer、effect、alias、result relation；
- 必要的数值、宽度、signedness、overflow 或地址语义。

如果一个事实可能改变 J 的允许结果关系，它必须先由 J 或 B 表达，不能只放进 π 或 C。

如果工程实现需要函数接口视图，只能从 J 确定性得到 `InterfaceView(J)`。它不是第四个 Lift
artifact，也不作为 Replay 的额外输入；C slots 仍由 `MaskC(C)` 提供，调用位置由 Harness
机械绑定。

实现中的模型 response 在成为 Lift artifact 前，必须经过唯一的固定代码标准化：

```text
response JSON -> Decode/Elaborate -> (J,C,pi) | deterministic diagnostic
```

Decode/Elaborate 成功后只有一个 typed J，不存在 `J_wire` 与 `CanonicalJ` 两个对象。该步骤将
`k0`/`[k0]`、`k1,k0`/`[k1,k0]` 等 closed surface sugar 归一为 structured axes，将
domain/predicate/index 解析为 ContractExpr AST，并物化 closed op signature 唯一确定的 result type、
消去可机械证明的 identity output mapping。它不读 Source、C/π、Harness concrete value 或 Gate
结果；需要猜 operator、predicate、effect、authorization 或类型选择时必须 fail closed。

因此，LLM 必须完整生成 `(J,C,π)`。就 artifact 构造而言，固定代码只执行 closed
Decode/Elaborate、dependent checking、拒绝与确定性 diagnostics；它不得从 Source、Gate evidence、
其他候选或历史记录推断、合成或自动补写任何 artifact 字段。

### 2.3 C

C 是保持 J denotation 的 implementation-selection 对象：

```text
C = (Σ_C, ν_C)
```

其中：

- `Σ_C` 是匿名 slot schema；
- `ν_C` 是 Harness 持有的 concrete valuation。

C 可以表达 tile、vector width、warp、pipeline、prefetch、buffer 等纯调度和实现选择，但不能保存 J 的语义副本。

### 2.4 π

π 是 source-realization witness：

```text
TypeCheck(J,C,π)
```

π 可以表达 instance、region、route、operand slot、order、sharing、placement、materialization 和受约束 representation relation，但不能拥有独立语义。

如果 π 表达整数宽度、numeric cast、reassociation、accumulation dtype 或其他可能改变结果关系的事实，TypeCheck 必须确认该事实已经由 J/B 授权，或者属于系统已经规定的合法 representation refinement。

### 2.5 Replay

Replay 是全新的 source-blind 生成：

```text
Replay = W_f(J, π, MaskC(C); U_f)
```

它只能看到：

- J；
- π；
- 去掉具体值后的 `MaskC(C)`；
- 与当前 Source 无关的公共前端知识 `U_f`。

它不能看到 Source、Harness、C 的具体值、source parameter name/position、BindingMap、独立
interface sidecar、Gate1 diff、source/compiler fingerprint、旧 Replay、旧候选代码、旧模型
conversation context 或 per-kernel RuleCard。

### 2.6 Harness

Harness 只负责：

- 枚举匿名参数和 compiler slot；
- 保存 concrete inputs 和 C values；
- 将同一份 C 绑定到 Source 和 Replay；
- 建立 compile/run invocation；
- 执行 Gate1/Gate2 所需的编译和运行。

Harness 不负责：

- 判断事实属于 J 还是 C；
- 生成 loop、mask、address、reduction 或 sharing；
- 修改 Replay AST；
- 创建新的 source-conditioned artifact。

## 3. 一次实验和系统改进必须分开

### 3.1 一次实验

一次实验使用当时系统中的同一套：

- J/C/π schema；
- TypeCheck 规则；
- Lift 和 Replay prompt；
- Gate1 observer profile 和比较政策 Q；
- 编译器、前端和工具能力；
- relift 次数上限。

实验运行期间不能针对某一个 Kernel：

- 临时增加 schema 字段；
- 临时增加 `extsi_mode`、`splat_policy` 等专用字段；
- 临时修改 Gate1 observer；
- 临时删除某个 compiler feature；
- 临时让 Replay 看到 diff 或旧代码；
- 临时增加新的 Agent 或 RuleCard。

### 3.2 系统改进

一次实验结束后，才分析失败属于哪一层：

1. Lift 没有使用已有表达能力；
2. Replay 没有正确消费已有 J/C/π；
3. J 的语义表达能力不足；
4. π 的 realization 表达能力不足；
5. C 的所有权或绑定规则不足；
6. Gate1 observer 比较了不应要求恢复的全局 compiler 差异；
7. 前端、编译器或工具链能力不足；
8. Gate2 的 reference 或 comparator 不完整。

系统改进必须是通用修改，不能只为单个 Kernel 添加分支。修改完成后，重新开始一次新的实验。

## 4. 一次实验的完整流程

```text
Source + Harness 机械信息
          │
          ▼
       Lift
          │
          ▼
  response JSON
          │
          ▼
 Decode/Elaborate -> (J,C,π)
          │
          ├─ 失败：Lift 修正；不进入 Replay
          │
          ▼
 TypeCheck(J,C,π)
          │
          ├─ 失败：Lift 修正；达到修正上限则实验失败
          │
          ▼
  source-blind Replay
          │
          ▼
  Harness 绑定 C
          │
          ▼
        Gate1
          │
          ├─ 失败：结构化 diff 只返回 Lift，重新生成完整 (J,C,π)
          │
          └─ 通过：进入 Gate2
                         │
                         ▼
                       Gate2
                         │
                         ├─ 失败：实验失败，进入系统改进
                       └─ 通过：SCRC_CERTIFIED
```

实际候选搜索采用同一 Source、同一 Harness 与同一 `c0` 下的有界并行 fan-out，而不是把不同
Kernel 并发混在一次认证中。branch 数是运行参数 `1 <= B <= 5`；每个 branch 拥有独立 provider
conversation、Lift/Replay runner、Replay writer、完整 `(J,C,pi)`、TypeCheck/Gate1 runner、buffer、
evidence 与 branch-local relift history。branch 之间不得读取、拼接或投票组合 artifact。`B=1` 主要
用于横向诊断 canary；若唯一 branch 通过全部门禁，其 `SCRC_CERTIFIED` 仍然有效，只有失败结论
不能声称标准五分支 `NO_CLOSURE_(5,K)`。最终未闭合项的搜索可以升到 `B=5`。Harness 仍是机械
绑定环境，不变成新的 LLM Agent。

“不混在一次认证中”不等于不同 case 必须 wall-clock 串行。当前 `intentir.scrc.batch.v3` outer batch
protocol 支持同时执行 `W` 个完整 case lifecycle；全局 branch/provider/Gate1 in-flight 上限为 `G`，
hidden Gate2 设备并发上限为 `G2`，必须满足 `W >= 1`、`1 <= B <= 5`、`G >= 1`、`G2 >= 1` 和
`W*B <= G`。`W/B/G/G2` 只作为 proof-irrelevant audit metadata 报告，不进入任何 certificate
predicate。

v3 相对 v2 只新增 batch-level `observer_unsupported` 分类：它表示当前 Source 在声明的 observer
profile 下无法形成可比较的 `F_source`，是 observer 的能力缺口，不是候选非法、模型失败、Gate1
mismatch 或有界搜索 `NO_CLOSURE`。单 branch 只有 Source observation 以该 closed diagnostic
fail closed 时才这样分类；fan-out case 只有所有已完整运行的 branch 都因同一 Source-observer
能力边界停止时才聚合为该类，混合结果仍按 `NO_CLOSURE` 等既有分类报告。Replay 侧解析/编译失败
也不得伪装成 Source observer 缺口。历史 `intentir.scrc.batch.v2` report 保持原始 protocol、分类与
结论，不由 v3 追认或重判。

每个 case/branch 必须分别新建 Harness 可变状态、provider conversation/session、Lift/Replay runner、
Replay writer、TypeCheck/Gate runner、relift history、input/output buffer 和 evidence state/目录。只读
`ProviderConfig`、registry、source 与 stateless adapter 定义可以共享；候选、会话、Replay、diff、
winner、C valuation 和其他可变状态不得跨 case 或 branch 共享。case evidence 目录必须原子、无冲突
分配，不能用并发的 `max(existing)+1`。单 case 失败必须结构化隔离，不能取消或污染 peer；结果
最终按 registry ordinal 稳定汇总，线程完成顺序 proof-irrelevant。Gate2 仅在 same-case winner 选定
后进入独立 `G2` semaphore，其结果不回流 sibling 或其他 case。

live CLI 必须使用重复的显式 case 集或 `--tier` 选择范围，不能无选择地隐式启动全部 28 项；默认
evidence root 为仓库外的 `/tmp/intentir-scrc-task11`，也允许显式覆盖。

live run 默认在 stderr 输出 proof-irrelevant 的 `intentir.scrc.progress.v1` JSONL observability 旁路。
除固定 protocol/event envelope 外，其 closed payload 只允许 case identity、branch、stage、attempt、
formal round、elapsed 与 closed status（实现字段为
`case_id/ordinal/branch_id/stage/attempt/formal_round/elapsed/status`），并约每 60 秒为活跃工作输出
heartbeat。它不得包含 Source、artifact、diff、diagnostic、provider response/conversation 或 hidden
Gate2 input，也不得进入 Replay packet、evidence 或 certificate predicate；旁路自身的任何异常均被
吞并，不能改变 case verdict。`--no-progress` 只关闭观测，不改变最终 stdout JSON 或执行结果。

## 5. 第一步：Lift 生成完整三元 artifact

Lift 的每一次正式输出都必须是完整的：

```text
(J,C,π)
```

不能只输出：

- J 的补丁；
- π 的局部片段；
- 一个额外 BindingMap；
- 一个 Resolver 输出；
- 一个自由文本诊断结论。

如果是第 r 次 relift，Lift 可以看到：

```text
Source
上一轮完整 (J,C,π)
上一轮 Gate1 的结构化 diff
```

但它仍然必须重新输出完整的 `(J,C,π)`，而不是输出对旧 artifact 的 patch。

Lift 不能把 compiler-visible semantic recurrence 压成一个只在最终数学结果上等价的 global formula。
若 Source 的 value dataflow 依赖 loop-carried state，J 必须使用已有 structured `loop` 完整声明
trip/scan、每个 carry/init、body update 与一一对应的 next/yield/result；body contraction 的 canonical
subscripts 和 result-axis orientation 必须匹配 update 真正消费的值，transpose 只在真实 logical
transformation 存在时出现。π 只能记录这些 J carrier 的 realization topology、multiplicity、placement
与 route，不能携带或让 Replay 猜缺失的 initialization、update formula、yield 或 orientation。

同理，一个 Source SSA coordinate/address intermediate 若只定义一次并被多个 access 使用，π 应只有
一个 producer，并以 routes 与 one-producer sharing 连接所有 consumers；多个语法 use 不产生多个
producer instance。只有不同 compiler-visible definition/stage 才允许重复同一 semantic ref。

## 6. 第二步：Decode/Elaborate 与 TypeCheck(J,C,π)

Decode/Elaborate 与 TypeCheck 都不是新的 Gate或证明对象，而是 Replay 之前的确定性闭合 IR 前端。
前者从 response 构造唯一 J并消除合法 surface form 的偶然差异；后者再检查所有权、引用和授权。
形式上：

```text
Decode/Elaborate(response) = (J,C,pi)
and TypeCheck(J,C,pi)
```

Replay packet、RefEval 与 production lowering 只消费这一个 J，不得各自重新解析 response string。
RefEval 与正式 backend 的 MLIR 路径共享同一个 `J -> Intent MLIR` 入口，随后才在 reference
execution 与 target/C lowering 处分叉。解析 AST/axes 可作为 J 的私有缓存，但不序列化、不传递、
不具有独立身份。

### 6.1 J 检查

检查 J 的：

- node、value、tensor、access、effect 引用；
- 类型和 operand arity；
- domain、predicate、fill、index relation；
- result relation 和 update relation；
- typed index、numeric conversion 等语义关系。

TypeCheck 还必须检查 closed result relation 与对应 output 的 dtype、A 依赖图和 B numeric contract
相容。该检查只拒绝不相容声明，不替 Lift 选择 relation。当前版本定义封闭判定
`RequiresTolerance_v1(J,output)`：

1. 从 logical output 经 direct producer、函数级 `returns` 或唯一 `output_mapping` 解析到 A value，
   再沿 op operands 递归得到 `Deps_A(output)`；无法唯一解析时 fail closed；
2. output dtype 为 floating，且依赖闭包含以下任一项时返回 true：
   - floating `reduce`；
   - `map_unary.fn` 属于 `{exp,log,sqrt,rsqrt,tanh,sigmoid,cos,sin,tan,exp2,atan,erf}`；
   - J/B 中 `kind=reassociation` 的 **source-realization authorization**，其 `subject_ref` 位于该闭包；
3. `RequiresTolerance_v1=true` 时，显式或默认的 `exact` 均 fail closed，Lift 必须显式重新声明
   `tolerance`。

v1 没有“更强 exact contract”例外；未来若需要，必须新增版本化 closed vocabulary 与判定，而不能
使用自由文本豁免。这里的 reassociation authorization 只触发 SCRC comparison-class 检查，不授权
production backend 改变 reduction tree。具体 tolerance 阈值继续只由 Q 提供。

`artifact.v4` 引入、当前 v5 保留由 Lift 显式作者的 `J.value_dimension_relations` 与
`J.boundary.consumer_extensions`：前者只表达 integral rank-0 J input 与 declared `x*` 的 closed
`eq`，后者只表达某一 predicate-bearing load 到某一 `sum|max|min` reduce/scan operand 的 typed
two-sided outside identity。`artifact.v5` 再要求完整 `J.boundary.access_guard_relations` 数组；每项把
一个 boundary access 绑定到一个支配它的 typed Layer-A `i1` guard，并显式给出严格递增唯一 axes 与
`positive|negative` polarity。所有数组包括空数组都由 Lift 写出，fixed code 不根据相等 runtime
value、reducer/fill、Source、C、π 或 Gate evidence 生成 relation。

relation-guarded store 不是 B 驱动的隐式控制流：Layer A 必须显式作者三 operand `select`，目标 tensor
必须精确同时为 J input/output 并由唯一 `mode=inout` parameter 绑定。tensor handle 按首次声明连续
编号；output 端复用同一 input `t*`，不新增 handle，`alias_of` 仍只描述物理 `p*` 参数关系。loop 的 init/current/next 与
non-loop 的 tensor pre-state/return/output mapping 必须精确闭合；相同 dtype/shape 的其他 input 不算
pre-state。`access_guard_relations.axes` 是同一 select guard 的唯一 broadcast authority，不能再由
`boundary.broadcasts` 重复。TypeCheck/CandidateCheck 只拒绝和报告这些 dependent error，不插入
select、不升级 inout、不选择 polarity/axes，也不修 candidate。

### 6.2 C 检查

检查 C 的：

- slot 是否匿名且唯一；
- binding kind 和 value type 是否匹配；
- concrete value 是否来自 Harness 已有坐标；
- 当前 v2+ schema 中，可选 `subject_ref` 是否属于 closed `CSubject_v1(J)`，可选 `axis_ref`
  是否满足 C 权威文档定义的 `axis_ref in AxisRefs_v1(J,subject_ref)`，且 `axis_ref` 不得脱离
  `subject_ref`；
- 是否把 predicate、formula、logical shape 或 result semantics 错误放进 C。

上述 attachment 必须由 Lift 显式生成；TypeCheck 只验证，不从 Source、π、Harness、`role` 或唯一
候选位置推导。缺失/含混 attachment 只使 production backend 安全忽略对应 prior，不允许其猜测。

### 6.3 π 检查

检查 π 的：

- 每个 semantic reference 是否存在于 J/C 的声明环境；
- instance、region、route、order、sharing 是否闭合；
- `iterations` 是否只使用 `extent_step|tile_ordinal`，其 loop/tile instance 是否分别锚定同一
  structured J loop 与 attached C tile；显式 relation 引用的 slot 可以使用任意 role，只要它是同一
  loop/trip axis 上的 required positive-integer C slot；
- route slot 是否与被引用 J carrier 相容；
- 是否出现未由 J/B 授权的 observable stage；
- 是否新增 operator、predicate、fill、formula、effect 或 semantic constant；
- representation refinement 是否属于系统封闭 schema，且参数与类型是否合法；
- sensitive refinement 是否引用 J/B 中已有的 authorization，或属于系统安全白名单；
- 是否存在 source text、frontend API、C value 或可执行自由文本。

iteration relation 的 denotation 是全局 closed contract：`extent_step` 为 lower 0、upper J trip、
step C tile；`tile_ordinal` 为 lower 0、upper `ceil_div(J trip,C tile)`、step 1。Decode/TypeCheck 不
根据 Source、Kernel family、具体 value 或 frontend 惯例选择 basis。多个 declared physical loop
region 可以引用同一 pair；一个被 Lift 完全遗漏、没有任何 handle 的 Source region 不在 fixed checker
的量化域内，仍由 Gate1 topology 检出。

`tile_extent` 与 `reduction_tile_extent` 是 closed C role vocabulary 中专门触发机械
missing-totality 的两个保留值：当这类 slot attachment 指向 structured J loop 时，TypeCheck 要求该
loop/slot pair 至少出现一条 iteration relation。这个保留含义不反向限制显式 relation；custom role
只要满足上述 dependent type 仍可被显式引用。loop trip axis 的 `AxisRefs_v1` 两种等价 spelling——
canonical ordinal `0` 与该 loop 显式声明的 `trip.axis=x*`——都合法。若 Lift 把实际 tile 误标成
custom role 并把 relation 完全遗漏，fixed checker 没有可枚举的 missing-totality trigger；该 topology
遗漏仍只能由 Gate1 检出，不能夸写成 TypeCheck 已证明完整。

artifact schema 必须 closed-world：未知字段、未知 relation 和自由文本扩展均 fail closed。若
`SemDecl` 只计直接创建的显式逻辑语义声明，则 TypeCheck 支撑的是：

```text
FreshLogicalSemDecl(C,pi | J) = empty
```

或在合法引用解析回 J 后写成 `SemDecl(J,C,pi)=SemDecl(J)`。它不证明 π 没有统计语义线索，
不证明 J 对 Source 正确，也不证明 realization 保持 J 行为。

TypeCheck 不证明某种 representation 在一般情况下与 J 语义等价。对于 cast、integer width、
reassociation、atomic form 等 sensitive refinement，它只检查两件事：J/B 是否已有明确的 typed/effect
authorization，或者该 refinement 是否属于系统统一维护的安全白名单；两者都不满足就直接失败。

authorization 的 dependent type rule 必须按 kind 定义，不能把所有 `subject_ref` 都只解释为其
result carrier。特别地，对：

```text
n = cast(x : Ts) -> Tt
u = numeric_cast(subject_ref=n, source_dtype=Ts, target_dtype=Tt)
```

TypeCheck 必须检查：

```text
u.kind == numeric_cast
u.subject_ref resolves to a unary A cast node n
u.source_dtype and u.target_dtype are both present
u.source_dtype == Type(n.operand[0])
u.target_dtype == Type(n.result)
```

并继续检查 refinement kind、`auth_ref`、instance `semantic_ref` 与 `u.subject_ref` 三方一致。
authorization 仍必须由 Lift 显式声明；Decode/Elaborate 和 TypeCheck 都不得从 cast node 自动创建它。
每个 instance 至多拥有一个 refinement；重复项必须在 parser/TypeCheck 阶段 fail closed。

这里的显式 `cast` stage 不等于 closed `map_binary` 的 op-internal dtype promotion。后者由 Lift 已声明的
operand/result types 与全局固定 promotion lattice 唯一决定，属于该 J op 的 typed denotation；共享
`J -> Intent MLIR` projection 可把它 materialize 为内部 `intent.cast`，但不得写回 J、创建
authorization 或把 Source/π 信息带入 promotion。若 Source 中 cast 本身是独立 compiler-visible stage，
Lift 仍必须显式生成 `cast` node 和 authorization，不能借 op-internal promotion 吞掉该 stage。

### 6.4 Lift 内的有界 CandidateCheck

同一个 source-aware Lift agent 可以在一次生成动作内，对自己已经写出的完整 candidate 调用只读、
非认证的 `CandidateCheck`。定义显式上下文：

```text
ctx = {
  anonymous_compiler_view,
  schema_version,
  decode_version,
  typecheck_version
}

CandidateCheck_v(candidate_bytes, ctx) -> PASS | diagnostics
```

`ctx` 精确等于本次 Lift 已可见的匿名机械 compiler view 与协议版本；实现禁止读取 ambient host
Harness 状态。CandidateCheck 共享正式 Runner 的同一 source-free
parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行 source-identifier privacy、
receipt/provenance，并在提交后重新运行该 core。禁止维护第二套近似 checker；工具可以在私有内存中
构造 ephemeral parsed/elaborated validation view，但不得合成、
补写、修改或返回新的 candidate/artifact。diagnostics 使用固定 closed schema：

```text
protocol, status, rule_code, path, observed_refs_or_types
```

`diagnostics` 是有界数组，而不是每次 evaluation 只能返回一项。若同一权威 strict Decode/TypeCheck
已经在当前合法前提下判定出多个相互独立的机械错误，CandidateCheck 按声明/字段的稳定顺序一次
展开，统一最多 20 项，每个 path/observed 文本最多 2048 字符。聚合只展开 checker 已有错误：不得
为继续遍历而补默认值、把别名改成 canonical 字段、删除冲突项或返回 repaired artifact。若 JSON、
父对象或参数自身不合法，使 dependent check 的输入尚未形成，则只报告当前可判定项，不能假设某种
修复后再推演错误。

它不返回 recommended value、canonical artifact 或 patch，也不读取 Replay、Gate1、Gate2、RefEval
或 compiler observation，不运行行为测试，不生成 J/C 分类、authorization、result relation 或 π
topology。agent 必须根据诊断自行修改并最终提交完整 `(J,C,pi)`。调用次数必须有界并作为普通 Lift
audit 记录；tool `PASS` 不构成正式 TypeCheck，提交后 Runner 必须用相同 `typecheck_version` 重新
执行权威检查。

提交动作对尚无本 session `PASS` 的精确 candidate bytes 自动执行同一 CandidateCheck；
FAIL 只返回上述 closed diagnostics 而不提交或修补；只有与某次 `PASS` 对应的精确 bytes 完全
相同时才可复用，任何不同 bytes 都必须重新检查。该前置
调用与 worker 显式调用共享同一有界预算，PASS 仍只是当前 episode 的诊断结果。当前 fresh
worker 使用最多 24 个 model turns。CandidateCheck 总 evaluation 上限为 8：普通共享 allowance
为 7，agent 的显式调用与提前 submission preflight 都从中消费；另有 1 次仅 submission 可消费的
reserve，因此典型 7 次显式 evaluation 后仍可检查最终提交，但任何顺序都不能超过 8 次。receipt
分别记录 `shared_budget`、`submission_reserve`，并按 `explicit`/`submission` trigger 记录 calls 与
实际 evaluations。这些是可报告的工程预算，不进入 certificate predicate。

若 worker 在预算内没有提交候选，runner 只可把该 worker 最后一次**实际执行**的 source-free
checker 返回的最多 20 条 `rule_code/path` 摘要带入同一 branch 的下一次 fresh Lift。没有实际
evaluation 就没有摘要；不得携带 observed value、候选修复、Source/Gate/hidden input、conversation
或 sibling 信息。下一 worker 仍完整生成 `(J,C,π)`，固定代码不应用 patch。submission preflight
`PASS` 也不认证 candidate，外部 Runner 必须照常重跑权威 TypeCheck。

该能力只把原本发生在相邻 fresh Lift invocation 之间的机械反馈移入一次有界 Lift episode，不增加
Agent、proof artifact 或 Replay 信息通道，也不改变正式 Gate round 的定义。

### 6.5 格式修正和正式实验轮次

以下错误仍属于 Lift 输出修正：

- JSON 格式错误；
- 必填字段遗漏；
- enum 拼错；
- route 引用不存在的 instance；
- schema 层级错误；
- TypeCheck 能直接指出的引用和类型错误。

这些错误在 Replay 之前处理，不算一次 Gate1 正式轮次。但每次实际 LLM 调用仍应计入普通调用记录，并设置修正次数上限，不能无限重试。

只有 TypeCheck 通过、能够进入全新 Replay 的候选，才算一次正式实验轮次。

接受集合发生变化时必须窄升 TypeCheck protocol，并同步 CandidateCheck context 与 receipt；只把同一
权威错误列表有界展开、不改变接受集合时，`candidate-check.v1` 无需升版。任何新规则都只对带新
version receipt 的 fresh run 生效，不得静默重判历史 artifact。

当前 fresh-run 版本为 `typecheck.v14`。它完整继承下述 v7 及其前序合同，并增加当前 v4/v5 typed
relations 的 source-free dependent checking：value/dimension 与 consumer provenance/identity 必须
闭合；access guard 必须满足支配、i1、唯一 axes authority，以及 guarded-store 的显式 select、精确
inout/pre-state 和 observable merge。checker 仍不构造任何 artifact fact。历史 v7 完整继承 v6；
v6 继承 v5，v5 保留 v4 的 rank-0 closed signature、
Layer-A total typing 与显式 non-trailing broadcast 合同，并新增已知 `reduce` 与 binary `contraction` 的 canonical
fail-early：reduce 必须显式给出合法 `fn + axes + assoc`；`kind` 别名、缺失/冲突字段、非法
reducer/axis/associativity 均在进入 RefEval/emitter 前拒绝。contraction 必须只给出 canonical
`subscripts`；`lhs_contract_axes/rhs_contract_axes` 不会被翻译，并在已有 typed J 可机械决定的范围内
检查 operand rank、label extent 与 result shape。可选 `keepdims` 必须为 boolean，且既有 declared
result type/shape 的 keep/collapse 校验不回退。fixed code 对两类 op 都不翻译或补齐。
v5 同时将 `opaque(mode=value,dtype=none)` 作为 absent ABI carrier 处理，拒绝它成为 π instance
semantic ref、realization authorization subject 或 refinement carrier；checker 不删除这些引用、
不生成 dummy carrier，也不修复 artifact。

v6 进一步共享并前移 J-only functional logical-domain closure：当 J 自身使用可识别的零起点矩形
domain 与 direct access/padding predicate 时，index axis、predicate extent、`boundary.domains` 和
typed tensor extent 必须显式闭合；缺失或不同的 domain 在 CandidateCheck/TypeCheck 直接拒绝，
不得从 Source、C、π、runtime/hidden input 或 shape 猜测补齐。该 necessary condition 不把 RefEval
尚未实现的 general affine/交叉轴 logical predicate 泛化成 J 语义非法，也不删除合法 masked/padded
predicate、fill 或 effect。fixed code 仍然只诊断，完整修订由 Lift LLM 完成。

v7 新增 source-free、仅限 rank-0 signature logical input 的 liveness necessary condition：
`logical_ref=t*` 所创建的 rank-0 J input 必须进入 observable output 的反向 value-dependency 闭包，
或被显式 J boundary/shape semantic relation 使用。direct identity、function `returns`、
`output_mapping` 与合法 boundary-only use 都保留；dead node 和只在 π 中的 realization route 不构成
logical contribution。一般高阶 tensor liveness 暂不纳入 v7，避免把未证明的全图规则当作语义非法。

同一 v7 另以 `padded_mean_count_unproven` 拒绝一个可机械证明的欠约束，但触发域严格限定为：没有
logical guard 的 predicated direct load；其 index 与 `boundary.domains` 构成完整、零起点矩形 domain；
predicate upper bound 与对应 typed tensor extent 相等；并且同一 masked axis 经已建模的 closed
pointwise/broadcast 位置传播后，确实被 `reduce(mean)` 消去。此时 J 又没有显式 logical
valid-count/denominator contract，中性 fill 只能证明分子，不能授权 physical extent 充当除数。
dense/unpredicated mean、guard-bearing predicate、cross-axis/general predicate、domain/extent 无法闭合、
mask 与该 reduction axis 无关或传播进入未知 op 的候选都不触发 TypeCheck 拒绝；后三类留在
functional consumer 边界并明确 unsupported。CandidateCheck/TypeCheck 只诊断、不读取 Source/C/π、
不自动改图；Lift 应按 Source 原有两阶段写出例如 `reduce(sum)` 后显式 division，而不是由 fixed code
把 padded mean 修成 sum/div。

runtime scalar 若只服务 representation/address realization，Lift 应声明
`kind=scalar,mode=value,rank=0,logical_ref=null`，不创建 J input；该 p* carrier 只可进入 π，不可回流
J output/boundary/shape semantics。fixed decoder 只接受这一显式分类，TypeCheck 只验证；两者都不按
Source 名称/位置判断，也不自动解绑、删除或重编号 J。对应通用 Lift/preflight 合同升为
`intentir.lift-knowledge.v15`，并由当前 v17 继承。

该 null-carrier 例外只适用于 runtime scalar，不适用于 `bound_constexpr_values`。后者按 Harness
合同必须已经是逻辑语义 constexpr：若作为 logical value/predicate/result，绑定 dtype-compatible
rank-0 J input；若只表示 integral logical extent，绑定 declared `x*`。只影响 layout/tile/stage 等
implementation selection 的 constexpr 必须在 Harness 分类为 C slot，而不能作为 bound constexpr
留在 π-only 形态；既不具有 J denotation、又不能合法归入 C 的 constexpr 使 Harness/case invalid，
不得交给 Lift 用 `logical_ref=null` 猜测兜底。

当前 RefEval 消费版本为 `j_scalar_cpu_mlir_v12`，shared functional projection 为
`intentir.j-functional-boundary.v11`。它们完整继承 v10/v3；v10 已继承 v9，v9 已继承 v8 对唯一 node-less
logical-input load 的通用消费，并移除旧 single-input/single-output/single-sum padded topology 限制。
对 predicated direct access，J 必须
自行声明每个 axis 的完整零起点 logical domain与显式 load fill，仍携带 masked axis 的 access 使用
同一 full-domain predicate；padded graph 中每个 logical reduce 必须有唯一、显式中性
`masked_contribution`，pointwise operand 的 trailing 或显式 non-trailing broadcast 必须闭合。满足这些
通用规则的 max+sum、多 reduction、broadcast/pointwise DAG 由既有 Intent MLIR scalar executor 执行，
没有 Softmax 或 Kernel 特判。

v10/v3 只为已经按 pre-Gate2 evidence 选定的 winner，在当前 Gate2 invocation 内增加 runtime
predicate tautology proof。主 SCRC runner 负责 winner 选择并把该次 inputs/dimensions 交给
RefEval-only 入口；projection library 不自行选择 winner。该入口要求精确的 J logical-input mapping 与
完整 closed dimensions，只消费 J 自身 rank-0 logical `i1`/integer values，并逐 access 按 guard
极性、direct axis、零下界及 runtime
bound 不小于 closed extent 证明 predicate 对完整 logical domain 恒真。不改 J、不进入 Lift feedback、
不读取 Source/C/π。v3 projection 的 invocation-local proof subreceipt 只有 access id、J predicate
digest、logical input refs 与 domain axes，没有 raw value 或 value digest；这是 audit-data
minimization，不是 confidentiality/privacy proof。production projection 仍无 runtime-value 接口。
缺 relation、上述窄
rectangular provenance 可证明 masked axis 被 `mean` 消去却没有 logical count contract、
update/alias 等继续 fail closed。无法完成该全域证明的 guard-bearing access 仍由 consumer 拒绝；
masked provenance 进入 contraction/其他未建模 op 时也在入口拒绝，避免 dead-use 冒充控制证明或
provenance 静默消失。这不改变 `typecheck.v7` 的接受集合。CPU host 只在共享 Intent-to-Linalg
之后调度 `rsqrt -> 1/sqrt` 的 libm ABI compatibility pass，共享/GPU lowering 不变。旧 v3/v4/v5/v6
TypeCheck evidence、v7/v8/v9 RefEval 与 functional-boundary v1/v2 evidence 以及已落盘的
Softmax/RMS run 不被 v10/v3 静默改写。只有尚未闭合、或确受该能力影响且要声明
v10/functional-boundary.v3 结论的 case 才必须 fresh relift/fresh run；已经成立的 Softmax
v9/v2/v5 证书继续有效并计入覆盖，不要求重跑或重标。

v11/v9 新增的当前消费边界是：逐项验证/消费 v4 value-dimension equality 与 per-consumer outside
identity；每个 v5 access guard 要么形成 receipt，要么显式 unsupported，guarded store 只运行 A 已有
select，guarded load/update 当前不做 access-local pre-execution materialization；同 producer、同完整
observable typed value、单静态轴、无 predicate、常量 unit-width 且无重叠无缺口的多 store 可形成
通用 direct-store partition。另一个 closed structured J-only reference 可整图执行 common scalar
lowering 尚未支持的 mixed typed binary contraction。direct-store 与 mixed contraction 都是 Gate2
consumer capability，不是 production recipe，也不按 Kernel 名称分支。
production J+C 入口当前对任何 access guard 仍以
`access_guard_materialization_not_implemented` fail closed，不能因 RefEval 已能执行显式 select 就
静默擦除该 B relation。

v12/v11 对 J-owned logical inout 作通用消费：同一 `t*` 必须在 input/output 两端复用、由唯一 inout
parameter 绑定，并以完整 load/store/effect/return/output mapping 区分 pre-state 与 post-state；
RefEval 分别读取 input binding、写出 output binding。物理 `alias_of` 和 `boundary.aliases` 仍 fail
closed，固定代码不生成 select/merge、不改 J，也不从 Harness 或 π 推导 alias。

Task 11 随后的旧 `public_frontend_context_v2`、`W=4/B=1` xhigh 横向波中，Softmax、RMSNorm、
attention row reduction 与 linear contraction 均形成完整 `NO_CLOSURE_(1,K)` evidence，且所有正式
candidate 的 TypeCheck 均通过。RMSNorm、Softmax 与 attention row reduction 的共同 Gate1 证据表明，
pointer carrier 已由 π 直接 route 到 `address_base`，v2 Replay 却把 base/lane 错误折成整数 offset 后
只做一次 pointer add，丢失 pointer-carrier dataflow。该通用消费规则已升级为
`public_frontend_context_v3`，MiniSWE Lift knowledge 同步升级为 `intentir.lift-knowledge.v11`；这些
版本修改只作用于新的 fresh run，不静默追认或重判旧 `NO_CLOSURE` verdict。

同一 linear contraction wave 还暴露了两个彼此独立的 Replay 合同缺口：单个 contraction
`realize` 被展开为 outer-product multiply + `tl.sum`，以及锚定同一 domain `d*` 的第二个 instance
被无依据映射成 `program_id(1)`。该 checkpoint 当时的下一 fresh run 使用
`public_frontend_context_v6`（继承 v5
direct-contraction/program-axis 合同）：在 rank-2、单一
contracted label、公开 `tl.dot` 支持的 floating tile/dtype/accumulation 条件下，单个 stage 使用一个
direct `tl.dot`；同一 `d*` 的多个 instance 保留 multiplicity 但复用同一 program-axis identity，
只有 distinct domain anchor 才能分配下一条 axis。Lift knowledge 升级为
`intentir.lift-knowledge.v13`，要求 direct primitive 与显式 decomposition 的 π stage multiplicity
由 Source-aware Lift 完整提出。它不读取 Source 给 Replay、不自动改 artifact，也不对不支持的
contraction 强用 `tl.dot`；旧 v3/v11 run 保持原 verdict。

## 7. 第三步：source-blind Replay

TypeCheck 通过后，Replay 只接收：

```text
J + π + MaskC(C) + U_f
```

Replay 必须：

1. 从 J 读取 operator、predicate、fill、index、effect 和 result semantics；
2. 从 π 读取 instance multiplicity、route、slot、order、sharing、placement 和 representation；
3. 从 `MaskC(C)` 读取匿名 slot schema，不读取 concrete values；
4. 根据 `U_f` 选择合法的 frontend API spelling；
5. 输出完整、参数化、source-blind 的 frontend source。

Replay 不得：

- 根据自己的猜测覆盖 J 语义；
- 丢弃 π 中的显式 stage；
- 折叠 π 要求保留的 materialization 或 sharing；
- 从旧 Replay 或 Gate1 diff 推断修补方案；
- 等待 Harness 在生成后补出新的 use-site。

`U_f` 必须是版本化、source-neutral 的公共前端知识，不能按当前 Kernel、operator 或失败样本选择。
当前版本组合是 `intentir.scrc.replay-input.v1`、Triton
`public_frontend_context_v7` 与 source-aware MiniSWE
`intentir.lift-knowledge.v17`；fresh evidence 必须保存实际 receipt，旧版本 run 不能被新组合追认。
runner 可以保存实际 request/output 和 compiler metadata 供追溯，但这些内容不能回流 Replay packet；
不需要为此增加 hash、commitment 或签名门。

当前新 Lift response 使用 `intentir.scrc.artifact.v5`，固定 decode 使用
`intentir.scrc.decode.v6`。v3 历史上引入 required closed `pi.iterations`；v4 引入
`value_dimension_relations`、`consumer_extensions` 等 typed relations；v5 引入
`access_guard_relations`。每次 fresh response 必须完整作者当前 wire，fixed code 不构造新增字段。
`artifact.v1-v4` 保持各自 wire bytes/digest 与字段集合，不静默升级、刷新或追认。

当前 `replay-input.v1` 不包含任何 pad-discharge derived hint。若 J 只有 symbolic input
extent、literal pad 与 static output extent，却没有闭合三者的 dependent equality，U_f 不能读取
Harness valuation 猜等式，也不能让 Replay 省略 pad。下一次 Source-aware Lift 必须给出可检查的
J relation或不生成该冗余 pad；fixed consumer 不为一份旧 artifact 在 packet 内增加 shape/slot
theorem prover。

## 8. 第四步：Harness 绑定 C

Replay source 生成并完成后，Harness 执行：

```text
Bind_H(Source,C)
Bind_H(Replay,C)
```

绑定必须使用同一个 `ν_C`。

记本次实际绑定为 `c0 = Bind_H(C)`。当前证书只覆盖这个 observed c0；即使 C 的设计合同要求
所有 admissible choice 保持 J，单次 Gate1/Gate2 也不能外推成全配置证明。多 C 测试只能按实际
覆盖的有限配置集合另行报告。

Harness 可以把同一个 C slot 绑定到 Source 和 Replay 的不同参数名或不同调用位置，但这只是机械调用关系，不得修改 Replay AST。

如果 C slot 无法绑定，实验在 Gate1 之前失败，原因归入 C/接口绑定问题；不能靠 Injector 新增使用点解决。

## 9. 第五步：Gate1

设当前 frontend 与 observer profile 为 `(f,p)`，其中 p 固定 observer/version、compiler/dump stage
与 canonicalization version：

```text
F_source = F_(f,p)(Compile_H(Source,c0))
F_replay = F_(f,p)(Compile_H(Replay,c0))
```

Gate1 为：

```text
G1 = [F_source == F_replay]
```

Gate1 的职责是检查 source-blind Replay 是否恢复了当前 observer 要求的 frontend realization。

Gate1 不负责：

- 运行语义等价证明；
- 在当前 Kernel 上临时判断某个 op 能否忽略；
- 用范围分析或 SMT 临时放宽比较；
- 判断 J 是否正确；
- 替 Lift 做语义分类。

### 9.1 canonicalization 的规则

Gate1 可以使用协议中已经确定的全局 canonicalization，例如：

- SSA 名归一；
- 参数名归一；
- source location 和路径归一；
- 无意义的声明顺序归一；
- 其他已经明确、全局适用的 frontend presentation noise。

如果未来要把某类 compiler transformation 全局 NormalizeAway，必须在实验结束后的系统改进阶段统一修改 observer。不能在某个 Kernel 失败时临时忽略。

当前 Triton 3.6 profile 的一个明示全局决策是：`tt.load` 上的 closed
`cacheModifier in {ca,cg,cv}` 与 `tt.store` 上的 `{wb,cg,cs,wt}` 是 target cache-policy hint，不改变 J denotation、memory
effect、predicate/fill、address/dataflow 或 transfer topology，且 closed J/C/π 不携带 frontend cache
spelling，因此自 `triton-ttir-3.6-gate1-v4` 起统一 NormalizeAway。v8 同样归一 closed
`evictionPolicy in {evict_first,evict_last}`：π 仍严格拥有 transfer 数量与顺序，Replay 只可把 closed
eviction spelling 当作防止 compiler CSE 的 public frontend mechanism；具体 spelling 不进入 hard
fingerprint。未知或方向不合法的 cache/eviction policy、mask、fill、type、volatility 及其他 operation
attributes 仍被观察；不得借此归一任意 memory difference。

当前 active profile 是 `triton-ttir-3.6-gate1-v8`。它保留既有对已建模 op 的 cache-policy 规则与
从 public entry 可达、静态已知且无环的 `tt.call` interprocedural comparison：按 formal-to-actual
substitution 展开 private helper，并将其返回 SSA、op/def-use、effect 与当前已支持的 structured region
纳入同一 observation；public entry ABI 仍是比较边界，helper 名称本身不是新的语义 artifact。

v6 新增 typed、single-block、encodingless quoted-generic `tt.scan`：同 shape ranked tensor、
axis/reverse、`2N` ordered combine args、唯一 typed `tt.scan.return` 与 recurrence/region topology 都
是 hard evidence。operation/effect classification 同时改为全局 closed profile；unknown op 不在 root、
`scf.for` 或 combiner 中默认 pure。malformed generic/body line、动态/未知 callee、递归/cyclic call、
signature/return 不一致、reachable arbitrary multi-block CFG/region 与未支持 control/region 均 fail
closed。若缺口发生在 Source observation，batch v3 记为 `observer_unsupported`，不伪造成 hard diff
或 `NO_CLOSURE`。这不声称 v5→v6 acceptance 单调；历史 v5 evidence 不由 v6 重判。它也不是
observer 完备性或整个 TTIR 的形式证明，
只是当前实现与测试覆盖的 compiler contract。

因此，Gate1 的准确含义是：

```text
在当前 observer 和当前 canonicalization 规则下精确比较。
```

每份证书必须记录 `frontend`、observer profile/version、compiler/dump stage 和 canonicalization
version。mutation test 可以在 CI 或消融中评估 observer 能抓住哪些错误，但不进入主 Gate，也
不能被表述成 observer 完备性证明。

## 10. 第六步：Gate1 diff 与 relift

Gate1 失败后，Comparator 只输出确定性的结构化事实：

```text
Diff = Diff(F_source,F_replay)
```

例如：

```json
{
  "missing_ops": [
    {"name": "arith.extsi", "count": 2}
  ],
  "unexpected_ops": [],
  "dataflow_differences": [],
  "predicate_differences": [],
  "effect_differences": []
}
```

Comparator 不直接下结论：

- “这是 J 的错”；
- “这是 π 的错”；
- “这个差异语义无关”；
- “应该修改 prompt”。

这些判断由 source-aware Lift 在下一轮处理，或者在实验结束后由系统改进阶段处理。

### 10.1 relift 的输入

第 r+1 轮 Lift 可以使用：

```text
Source
Artifact_r = (J_r,C_r,π_r)
D_r
```

它必须输出：

```text
Artifact_(r+1) = (J_(r+1),C_(r+1),π_(r+1))
```

接着重新执行：

```text
TypeCheck
→ source-blind Replay
→ Harness 绑定 C
→ Gate1
```

旧 Replay 只作实验记录，不进入下一轮 Replay 输入。

### 10.1.1 固定 artifact 的 diagnostic recheck

`intentir.scrc.gate1-diagnostic-recheck.v1` 可以对 parent evidence 中一份已落盘 artifact 做独立只读
诊断：保持 artifact bytes/digest 不变，以当前 canonical Decode、`typecheck.v14`、一个 fresh normal
Replay 与 Gate1 写入单独的原子 `recheck-NNN` 目录。该路径不调用 Lift、不运行 masked-J 或 Gate2、
不参加 candidate/winner 选择，并显式记录 `diagnostic_only=true`、`enters_gate2=false`、
`changes_prior_verdict=false`。它只能判断当前固定 consumer/context 如何消费旧 artifact，不能把
parent verdict 改成当前协议的正式结论。provider 调用前必须核对 selected case/current prepared
request、parent source projection/source bytes，以及 parent Gate1 receipt 中的 artifact digest；错
case、错 artifact、错 Source 或缺失旧 lineage 字段均 fail closed。protocol/context/knowledge 改变后
的认证仍必须 fresh rerun。

### 10.1.2 固定 Gate1-closed artifact 的 Gate2 capability recheck

若 parent 已有 `replay_closed` 的精确 artifact 与 Replay，而失败仅来自通用 RefEval consumer 缺口，
修复系统后可执行 `intentir.scrc.gate2-capability-recheck.v1`。该路径核对 artifact 位于 parent evidence
中、Replay bytes/digest、case 与 Source lineage，随后以 `model_calls=0` 复用 exact artifact/Replay，
只重跑当前 Gate2 三腿并写入独立 `gate2-recheck-NNN` 目录。receipt 必须记录
`reuses_exact_gate1_artifact=true`、`reuses_exact_gate1_replay=true` 与
`changes_parent_verdict=false`。它不 relift、不重新生成 Replay、不重做 Gate1，也不改写 parent
verdict；新的结果只属于记录的当前 RefEval/Q/test/environment capability evidence。

### 10.2 branch 数与 relift 预算

当前搜索把 branch 数参数化为 `1 <= B <= 5`。每个 branch 最多进行 `K=3` 次能够进入 Replay 的
TypeCheck-valid 正式 Gate1 round，`K` 包含首次候选；schema/TypeCheck 的 pre-Replay rejection
另按既定上限计数，并必须报告 logical 与 physical provider calls。`B=1` 主要用于横向诊断 canary；
若唯一 branch 通过全部门禁，其 `SCRC_CERTIFIED` 与搜索宽度无关，仍然有效。只有失败结论至多是
本次 `NO_CLOSURE_(1,K)`，不能声称标准五分支 `NO_CLOSURE_(5,K)`。最终未闭合项的搜索可在全局
`G` 允许时升到 `B=5`。

若没有 Gate1-closed winner，只有全部 `B` 个 branch 都在各自预算内完整终止为 `NO_CLOSURE`、
且每个 branch evidence 完整时，case aggregate 才能报告：

```text
experiment_status = NO_CLOSURE_(B,K)
```

只要 branch statuses 混有 `infrastructure_failed`、`superseded`、`incomplete` 或其他非完整
`NO_CLOSURE` 状态，aggregate 必须报告 `infrastructure_failed`，并保留全部 branch statuses、
diagnostics 与 evidence 指针；不能把不完整搜索伪装成 `NO_CLOSURE`。

`NO_CLOSURE_(B,K)` 只表示：该次有界并行搜索在所使用的 J/C/π 表达能力、prompt、Replay 能力和
Gate1 observer 下没有找到闭合候选。

它不自动说明：

- J 一定错误；
- π 一定泄漏；
- Replay 模型一定无能力；
- Gate1 observer 一定设计错误。

这些要到实验结束后的系统改进阶段分类。

## 11. 第七步：Gate2

只有 Gate1 通过的候选才进入 Gate2。

对测试输入 `x`：

```text
O_S = Exec(Source, c0, x)
O_R = Exec(Replay, c0, x)
O_J = RefEval(J, x)
```

Gate2 为：

```text
G2 = Close_Q(O_S,O_R)
     and Close_Q(O_S,O_J)
     and Close_Q(O_R,O_J)
```

其中：

- Source 和 Replay 使用同一 C；
- `RefEval(J,x)` 不接收 C；
- `RefEval` 不读取 π；
- B 只声明封闭 comparison class 与必要语义合同，外部比较政策 Q 据此构造 `Close_Q`。

### 11.1 比较关系的来源和范围

B 只能声明系统支持的 comparison class 和必要语义合同，例如 exact 或受限浮点 tolerance。Q 是
实验协议或 benchmark correctness policy 预先确定的可信政策，根据经验证的 class、dtype、
reduction extent、reassociation 权限等构造 comparator，并检查 tolerance 上限、NaN/Inf 合同等
全部参数；不支持的类别或非法参数直接 fail closed。Q 不是 Lift artifact，也不进入 Replay。

comparison class 在进入 Replay 前已经由 TypeCheck 做结构相容性检查；Q 只实例化一个已接受的
class，不负责把错误的 `exact` 自动改成 `tolerance`。同样，Gate2 的 tolerance 不能反向授权 J 或
production backend 改变 reduction order、accumulation dtype、FMA 或 transcendental 实现。

Lift 不能自由写入任意 Python comparator，也不能通过任意大的 `rtol/atol` 或“所有输出均可接受”
来放宽自己的 Gate2。B 保留对语义比较类别与必要合同的所有权，但候选不能自己决定及格线。

`Close_Q` 必须是确定性、对称的比较谓词；若底层 `isclose` 有方向性，Q 必须显式对称化或执行
双向检查。第三条 Replay–J 边不增加一次执行，只比较三个已经产生的输出；它防止近似关系不传递
时，两条 Source-centered 边都通过但 Replay 与 J 彼此超出范围。

当前主线只覆盖 deterministic / contract-deterministic Kernel：内部执行顺序可以变化，但 J/B
在协议可观察层面必须给出唯一合同结果。真正集合值的语义不因“非确定性”一词自动降低 claim，
而是留给未来 `Accept_{J,Q}(x,o)` 的逐观察 conformance；有限次 membership 不能宣称完整
`Beh(P,x) subseteq Spec_J(x)`。

### 11.2 Gate2 输入的生成顺序

令 `tau_pre` 表示 `B` 个 branch 的 Lift、TypeCheck、Replay 与 Gate1 记录，但不含任何 Gate2 输入或
结果。当前实现等待这一 search wave 完成，然后使用预先确定的选择函数：

```text
a* = sigma(tau_pre)
sigma = 按 (branch_id, formal_round) lexicographic 固定顺序选择第一个 Gate1-closed 候选
```

只有这个完整候选 `a*=(J,C,pi,Replay)` 被选定；随后才生成 Gate2 输入，本次 Gate2
期间不再修改该候选。如果采用隐藏随机输入模式，必须从预先声明的分布 D 独立同分布地抽取输入，
并对 Lift/Replay 隐藏这些输入；Gate2 结果也不回流本次实验。令 `g2_Q(a*,x)` 表示三条边在 x
上全部通过，只有当固定候选的 `Pr_D[not g2_Q(a*,x)] >= epsilon > 0` 时，才能报告 m 个
challenge 全部通过的上界 `(1-epsilon)^m`。Source–J 错误质量是其中的充分特例；“错误 J”本身
不自动保证在 D、Q 下有可检测 disagreement。

如果只使用预先给定的有限测试集合 `T`，则只报告：

```text
for every x in T:
    Close_Q(Source(x),Replay(x))
    and Close_Q(Source(x),RefEval(J,x))
    and Close_Q(Replay(x),RefEval(J,x))
```

不能在只使用固定 `T` 时同时引用隐藏随机测试的概率结论。两种输入模式都不需要新的证明对象
或新的运行阶段。

未选 branch 即使 Gate1 闭合也不执行 Gate2。Gate2 失败后，不 fallback 到 sibling candidate，
不把结果返回任何 Lift，也不在同一次运行中继续局部修补。运行结束，进入通用系统改进，再开始
新的完整 fan-out。

## 12. masked-J 的位置

masked-J 不是主认证条件，也不是每次实验必须增加的额外 Gate。

如果研究需要，可以在 Gate1 通过后单独记录：

```text
J_NECESSARY = masked-J hard mismatch
```

它只说明：在当前 J、π、C 和 Replay 模型下，删除 J 后是否仍能完成相同的 Gate1 closure。
即使状态记录为 `PROVED`，结论也只相对于记录的模型、解码配置、Replay packet 和 masked-J
干预成立，不是对所有生成器的全局不可替代性证明。

masked-J hard-close 不直接等同于语义泄漏，也不自动推翻：

```text
TypeCheck(J,C,π) + Gate1 + Gate2
```

因此，Task 11 当前以 MiniSWE xhigh 启动的六项 fresh v3 横向波可以使用 `--skip-masked-j`：这只
省略非认证性的 J-necessity 干预，不省略 TypeCheck、Gate1、winner 后唯一 Gate2，也不降低
`SCRC_CERTIFIED` 条件。该波的 audit 参数为 `W=6/B=1/G=6/G2=1`；“已启动”不是终态 evidence，
运行中的局部状态不能写成最终 verdict。

如果需要进一步检查 J 的语义权威，可以另行运行离线 J-authority challenge；它不自动触发，
结果独立记录，也不改变主流程的对象边界。

## 13. `arith.extsi` 的处理方式

在一次实验中，若 Gate1 报告 Source 存在 Replay 缺少的 `arith.extsi`：

1. 不在 Gate1 中临时删除该差异；
2. 不新增 `extsi_count` 之类的 Kernel 专用字段；
3. 将结构化 diff 返回 source-aware Lift；
4. Lift 检查 J 是否已经表达对应的 typed conversion relation；
5. 如果 J 已经授权该 relation，π 表达显式 materialization、placement、order 和 sharing；
6. Replay 根据 J、π 和公共 frontend 知识生成相应实现；
7. 再次执行 TypeCheck、Replay 和 Gate1。

如果整数宽度、signedness 或 overflow 可能改变 J 的允许结果，那么这些内容必须由 J/B 拥有，π 只能表达 realization 关系。

如果当前 J/π schema 无法表达该 relation，各 branch 达到 `K=3` 且本次所选 `B` 的完整 fan-out
仍未闭合后结束本次运行；横向诊断可以从 `B=1` 开始，最终闭合搜索可以升至 `B=5`。之后在系统
改进阶段决定：

- 扩充 J 的 typed index/conversion 能力；
- 扩充 π 的 generic representation relation；
- 修改 Replay 对 representation relation 的消费方式；
- 或通过跨案例证据决定该 compiler difference 是否应由 Gate1 全局归一。

不能在同一实验里临时改变其中任何一项。

## 14. 系统改进阶段的修改方法

### 14.1 Lift 能表达但没有抽取出来

修改：

- Lift prompt；
- Lift 输入说明；
- Lift 的公共示例；
- source-neutral schema query 与有界 `CandidateCheck` 诊断工具；
- 结构化 diff 的表达方式。

不修改：

- 单个 Kernel 专用字段；
- Replay 的额外输入；
- Harness AST 修补逻辑。

### 14.2 π 能表达但 Replay 没有消费

修改：

- Replay prompt；
- π 消费顺序说明；
- instance/route/order/sharing 的公共解释。

不把旧 Replay 代码提供给下一次 Replay。

### 14.3 J 能力不足

修改：

- J schema；
- J 的类型和语义关系；
- Gate2 的 J-only reference。

任何可能影响结果关系的表示属性，都先判断是否属于 J/B。

### 14.4 π 能力不足

修改：

- generic representation relation；
- placement/materialization/order/sharing 的表达；
- `TypeCheck(J,C,π)` 的依赖检查。

不按 kernel family 增加 `softmax_stage`、`row_address`、`extsi_mode` 等专用 ontology。

### 14.5 Gate1 observer 不合适

只有在多个实验结果和人工/自动变异证据都说明某类差异不是应该恢复的 realization 时，才修改 observer 的全局规则。

修改后重新开始实验，不能对旧实验结果静默重判。

### 14.6 Gate2 或前端工具不足

如果 Source、Replay 和 J 的行为比较不完整，或者前端编译/执行工具失败，应修复对应通用工具，再重新实验。

分类时先保留 Gate1-closed exact artifact：若 J 本身及 Replay 仍可信、失败只是 RefEval 对一个已声明
relation/op 的通用消费缺口，修复 consumer 后执行 10.1.2 的 Gate2-only recheck；只有证据表明 J
语义/类型/关系本身错误时才 fresh relift。consumer 修复不得按 Kernel 写死 J→reference recipe。

不能用 Gate1 通过代替 Gate2，也不能用 production lowering 代替 `RefEval(J)`。

## 15. 实验结果记录

每个实验至少记录：

- outer batch protocol 与 `W/B/G/G2` proof-irrelevant audit metadata；
- Source、J、C、π、Replay 的文本或结构结果；
- 每个 case/branch 的 status、diagnostic 与 evidence 指针；
- TypeCheck 是否通过；
- 每次正式轮次的 Gate1 结果；
- 每轮结构化 diff；
- Gate2 结果及失败关系；
- LLM 调用次数和失败原因；
- 最终状态。

这些是实验记录，不是新的证明对象，也不是 Replay 输入。

当前 `intentir.scrc.triton-workload-adapter.v3` 在 28 项 registry 中声明全部 28 项：原六项代表 adapter；
full FlashAttention forward/backward、fused-add RMSNorm、Sparsemax、one-column-block backward；以及
Qwen2-VL MRoPE、GroupNorm、JSD、CrossEntropy、GEGLU、PolyNorm、Llama4 RoPE、FlashAttention
LayerNorm、RoPE、LayerNorm、KLDiv、TVD、xFormers RMSNorm/RMSNorm-add 与三个 SplitK entry。ordinal
24/25 使用 `intentir.triton.fixed-c-source-slice.v3` 从显式 root 对实际引用的唯一 undecorated
plain helper 及其
closed dependencies 做源码内联，不 import package，也没有 xFormers/`rsqrt` 特判；两项均已通过真实
Triton CUDA frontend compile。v3 另只按精确 `VAR_ARGS_ARRAY` annotation、显式 bounded arity 与
`range(len(array))` 展开 positional/local arrays，任何动态、foreign 或残留 marker 都 fail closed；
marker 只允许出现在 selected top-level entry 的当前 lexical scope，nested function/class/lambda 不被
遍历或改写；nested marker、跨 scope 捕获以及 marker positional 自身带 default 均拒绝，普通 trailing
default 则保持不变。它不调用外部 `unroll_varargs`、`exec` 或 monkey patch。ordinal 26 使用 8 个真实 runtime `None`
carrier、arity 1 和非退化 split grid；ordinal 27/28 使用 arity 2，展开后 ABI 分别为 38/51
参数。`None` 在 J.signature 中只能是 `opaque(mode=value,dtype=none)`，无 logical_ref/J
tensor/node/effect/π instance，不得换成 dummy pointer 或 scalar；加入 canonical contraction 与
direct/decomposed stage multiplicity 合同后，fresh MiniSWE Lift knowledge 当前升为
`intentir.lift-knowledge.v17`。v17 继承 v15 的 generic functional logical-domain 闭合提示、有界
CandidateCheck 预算、跨 fresh Lift 机械摘要，以及 rank-0 logical-data/representation-only scalar
分类合同；并加入 value/dimension equality、consumer extension 与 access guard 的完整 authored
合同；全部 28 项的 static prepare 与真实 CUDA frontend compile 已通过；
`declared` 只表示 Harness/source ABI 有闭合配置，不构成 TypeCheck、Gate 或 SCRC 终态。

对上述 absent carrier，TypeCheck 分别拒绝其成为 π instance 的 `semantic_ref`、J/B realization
authorization 的 `subject_ref` 或 refinement carrier；本地 CandidateCheck 只复用同一 source-free
checker 提前返回这些机械诊断。两处都不能删除错误字段、生成 dummy carrier、补造 π/authorization
或据此改写候选 verdict；修正仍由下一次 fresh Lift 完整重交 `(J,C,π)`。

建议分别报告：

```text
first-attempt TypeCheck pass rate
Lift 格式修正次数
closure@1 / closure@2 / closure@3
Lift call count / Replay call count / total LLM call count
NO_CLOSURE failure categories
Gate2 pairwise Source/Replay/J pass rate among Gate1-closed candidates
SCRC_CERTIFIED rate
```

`closure@k` 定义为截至第 k 个正式 Gate1 轮次的累计闭合任务比例，不是第 k 轮的独立成功率。
证书绑定实际生成的 Replay，而不是不报告搜索成本的抽象 `exists R`。

这样能够区分：

- 一次 Lift 就能抽取完整关系；
- 经过 Gate1 反馈后才闭合；
- 语义上能够闭合但 Gate2 失败；
- schema 或模型能力确实不足。

## 16. 主结论

一次实验的完整逻辑是：

```text
1 <= B <= 5 个独立 branch 并行：
  Lift response → Decode/Elaborate → (J,C,π) → TypeCheck(J,C,π)
       → source-blind Replay(J,π,MaskC(C);U_f)
       → Harness 绑定同一 c0 → Gate1
       → Gate1 diff 只返回本 branch 的 Lift 并 relift
→ pre-Gate2 wave barrier
→ sigma 仅根据 pre-Gate2 evidence 按 (branch_id, formal_round) lexicographic 顺序选择 winner
→ 只为 winner 生成隐藏输入并在 G2 semaphore 下执行唯一 Gate2；失败不 fallback
```

运行期间不改变系统规则；达到 branch/relift 预算仍失败就结束本次搜索。系统改进只能在完整运行
结束后进行，并且必须以通用能力修改的方式解决：

```text
Lift 能力问题       → 改 Lift
Replay 消费问题     → 改 Replay
J 语义能力问题      → 改 J
π realization 问题  → 改 π
C 所有权问题        → 改 C/绑定规则
观察器问题          → 改 Gate1 observer
语义验证问题        → 改 Gate2/reference
工具问题            → 改通用 frontend/toolchain
```

最终认证条件只有：

```text
SCRC_CERTIFIED_(f,p,Q,T,c0)(Source,J,C,pi,Replay)
```

其中 p 是 observer profile，c0 是本次实际绑定。为便于实现可继续使用下面的简写，但论文不得
把它外推到其他 observer、比较政策、C 配置或全输入域：

```text
SCRC_CERTIFIED
    iff
TypeCheck(J,C,π) = PASS
and Gate1 = PASS
and Gate2 = PASS
```

这套流程保持 C 和 π 的对象边界，不引入第四通道，也不把一次实验中的失败修补误认为系统已经获得新的表达能力。
