# Task 15：模型级输入到 Compiler-Observed Lift / Closed J / Source-Blind Replay 的最终闭合系统

> 状态：`P1_CONTEXT_REDUCTION / P2_DIAGNOSTIC`（2026-07-29）。通用语义边界已实现；formal
> pilot 暴露出首轮 Lift 的 workspace/π authoring attention 瓶颈，先完成不改变数学对象的上下文减法，
> 再继续十项 active queue。
>
> 定位：Task 11 的下一次能力建设与最终十项闭合波。Task 14 已结束归档；其18份可保留证书、十项
> active queue与中断 evidence 是本任务的历史输入，不是本任务已经完成的结果。
>
> 执行后端：只使用 Codex agent。MiniSWE 暂时退出本任务；production backend 继续独立工作，SCRC
> 之后只消费 `J+C`，不读取 Source、TTIR 或 `π`。

## 0. 权威边界

本任务服从以下四份权威设计，不重新发明数学对象：

- `doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md`
- `doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md`
- `doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md`
- `doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md`

方法入口为：

- `doc/method/IntentIR_Lift_匿名Source_TTIR工具化提取与CandidateCheck边界_2026-07-29.md`

唯一认证公式保持：

```text
Lift_H(Source) = (J,C,π)

Replay = W_f(J,π,MaskC(C);U_f)

SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1(Source(C),Replay(C))
  and Gate2(Source(C),Replay(C),RefEval(J))
```

证明相关对象仍严格只有 `(J,C,π)`。`P_cert`、Source-side TTIR、agent workspace、compiler receipt、
CandidateCheck diagnostic和Gate diff都是 proof-irrelevant sidecar；它们不能进入 Replay、证书谓词或
production lowering。

## 1. 当前基线与本任务分母

### 1.1 保留的18项

Task 14 已逐份读取 accepted J，确认以下18项的 source-grounded logical extents 已由 `x*` 及必要
value/dimension或constexpr binding参数化，ABI/variant没有已知错误，因此本任务不机械重跑：

```text
SwiGLU, RMSNorm, Fused Add RMSNorm, GeGLU, LayerNorm,
Softmax, DYT, Sparsemax, KLDiv, PolyNorm, JSD, TVD,
xFormers RMSNorm, xFormers RMSNorm-add,
Flash backward preprocess, CrossEntropy, GroupNorm,
Flash one-pass LayerNorm
```

只有后续证据证明某份 J 的 logical identity、ABI role、dtype或selected variant确实错误时，才将对应
case移回fresh Lift；不能因为schema新增了一个该artifact不需要的relation而全量重跑。

### 1.2 本任务active queue：10项

1. Liger RoPE
2. Qwen2VL MRoPE
3. Llama4 RoPE
4. Flash linear
5. Flash forward
6. Flash one-column backward
7. Flash full backward
8. SplitK forward
9. SplitK reduce
10. SplitK backward

其中SplitK reduce/backward的旧证书只覆盖`vararg_arity=2`；正式模型profile选择`split=4`、
`vararg_arity=4`，属于不同compile-time program specialization，必须fresh Lift。

### 1.3 Task 14中断批次只作诊断

2026-07-29的adapter-v6模型输入批被人工停止，不产生新的最终certified或完整
`NO_CLOSURE_(B,K)`：

- Liger RoPE candidate 2在formal round 3取得Gate1 closure；Source–Replay的q/k误差均为0。通用
  dynamic-extent/index-bound consumer缺口修复后，exact artifact暴露出真正的J缺失：同一tensor存在
  多个不同的node-less load view，却没有声明各自Layer-A consumer。当前TypeCheck明确拒绝这一含混
  ownership；旧Gate1 evidence保留，但该case必须fresh Lift，不能让RefEval猜consumer映射；
- Llama4一个fresh Replay episode耗尽1800秒；其前置Lift约10分钟，receipt累计约90.5万input tokens，
  说明当前主要性能问题是重复上下文与Replay构造工作空间，而不是TypeCheck；
- Qwen2VL、Flash forward/full backward已有部分Gate1 no-closure或未完成round；
- Linear、one-col和SplitK只在停止过程中短暂启动，均不构成能力证据。

## 2. 从真实输入到证书的完整数据流

```text
registry direct symbol + upstream/model invocation
                    |
                    v
     reviewed P_cert / adapter-v6 input_profile
     - selected compile-time specialization
     - ordered ABI + input/output/inout
     - dtype/rank/concrete valuation
     - source-grounded shape identity h*
     - runtime scalar / constexpr / C coordinate
                    |
                    v
      可编译的匿名 Source + 匿名 Harness机械view
                    |
          +---------+---------+
          |                   |
          v                   v
     exact Source阅读     Compile/Observe得到匿名TTIR
          |          （按需查询CFG/SSA/effect/time）
          +---------+---------+
                    v
       一个持久、source-aware Codex Lift owner
                    |
             完整作者化 (J,C,π)
                    |
       CandidateCheck（非认证、本地preflight）
                    |
             外部正式TypeCheck
                    |
                    v
     一个全新、source-blind Codex Replay episode
        只见 J + π + MaskC(C) + U_f
                    |
       自己编辑Replay、编译、检查、再编辑
                    |
             Harness绑定同一个c0
                    |
                   Gate1
           fail |             | pass
                v             v
   diff只回同一Lift owner   固定exact J/Replay
      完整relift + fresh R          |
                                  Gate2三边
                                    |
                              SCRC_CERTIFIED
```

匿名化的目的，是删除Source名字、路径和kernel-family暗示，并保持request isolation；它不是删除程序
结构。匿名Source必须在同一个冻结调用specialization下可真实编译。Lift本来就是source-aware，因此
它读取该Source编译出的匿名TTIR不会形成泄漏；关键边界是TTIR绝不进入Replay。

## 3. 输入与 `P_cert` 计划

### 3.1 `P_cert`只记录事实，不生成J

每项在启动Lift前必须具备：

```text
direct symbol / source snapshot / upstream provenance
selected compile-time specialization
ordered callable ABI
buffer/scalar/None + input/output/inout + dtype + rank
observable outputs与inout pre/post state
concrete model/direct-tile valuation
shape_contract h*、runtime-scalar equality、constexpr equality
C coordinate与grid realization
external Q policy lineage
```

Harness可以枚举匿名坐标和声明source-grounded identity，但不得判断事实属于J还是C，不得给Lift预写
operator、domain、consumer access、loop、effect或π topology。`h*`、profile token与grid都不进入
`(J,C,π)`；Lift必须自己把每个`h*`双射为`x*`。

### 3.2 十项正式profile

| family | primary profile |
|---|---|
| Liger RoPE | FP16，`B=1,T=2048,Hq=32,Hkv=8,D=128` |
| Qwen2VL MRoPE | FP16，`B=1,T=2048,Hq=28,Hkv=4,D=128`，sections `16/24` |
| Llama4 RoPE | FP16，`B=1,T=2048,Hq=32,Hkv=8,D=128` |
| Linear | FP16 boundary，`M=N=K=4096` |
| Flash forward/full backward | FP16，`B=4,H=16,S=8192,D=128` |
| Flash one-col backward | upstream direct-tile：`Sq=128,Sk=64,D=64`，`BLOCK_M=64` |
| SplitK forward/reduce/backward | `B=1,Mq=1,G=1,H=16,KV=4096,D=256,split=4`；reduce/backward arity=4 |

模型级shape可以很大，但J参数化身份与单次Gate2 valuation必须区分。direct-tile symbol不虚构完整模型
tensor；compile-time selector/vararg family是不同程序，不是J中的runtime branch，也不是普通Harness
valuation。

## 4. Compiler-Observed Codex Lift owner

### 4.1 生命周期

每个branch只有一个持久Lift owner。它从首个Source阅读、草稿构造、CandidateCheck修正一直存活到该
branch Gate1闭合或formal `K`耗尽。Gate1失败时，Runner把上一份完整artifact与closed diff作为真实
continuation交给同一owner；owner在自己的workspace里修改J/C/π并再次提交完整三元组。

禁止恢复：

- `planner -> builder -> reviewer`多作者交接；
- reviewer重写已通过CandidateCheck的artifact；
- sibling/cross-case best parent、semantic cache或candidate合并；
- host根据diff自动选择patch或relation。

同一owner可保存working draft与exact checked checkpoint，后续修改失败时回到自己的checkpoint；这只是
branch-local事务，不是跨formal round替模型选择“最佳J”。

### 4.2 Lift可用的Compiler观察工具

只建立一组通用、前端最小接口：

```text
Compile_f + Run_f + Observe_f
```

对当前Triton，Lift workspace开放以下只读能力：

1. 在冻结匿名Harness binding下编译匿名Source并缓存exact TTIR；
2. 分页读取TTIR原文，并按token或单个`%SSA`查询局部def-use上下文；
3. 查看function/block/control/loop/memory/reduction等syntax outline；
4. 按路径读取同一Source的Gate1 observer fingerprint，用于理解当前observer实际比较的
   CFG/dataflow/effect/time事实。

这些查询只展示compiler已经产生的事实，不输出“应写哪个J op/π relation”、不从TTIR生成artifact、
不建立每kernel parser/RuleCard或frontend专用DFG语义器。新增前端仍只实现其真实
`Compile/Run/Observe`，不实现`Source -> J` lowerer。

### 4.3 轻量上下文

Lift首包只包含：

- 简短所有权合同与精确工具清单；
- `source.py`、匿名Harness view、schema bundle和feedback文件路径；
- 必要的小文件原文与路径；
- 当前formal round与exact artifact路径。

Source、TTIR、完整schema、历史artifact和recursive Gate diff留在本地文件，按需读取；同一大对象不在
每个model turn重复嵌入。工具结果保留路径和必要的小片段，exact内容始终可从同一 workspace 重读，
不由 host 生成语义摘要或 hash receipt。

目标不是人为限制模型调用次数，而是让一次Codex式回合只做少量“读—改—查”动作。必须记录：

```text
time-to-first-complete-candidate
model turns / physical calls / tool calls
prompt、cached与output tokens
Source/TTIR/schema重复注入bytes
CandidateCheck calls
formal closure@round与总wall time
```

90万input-token、30分钟Replay停滞属于需要消除的工程失败信号；TypeCheck PASS不是能力进展。

### 4.4 CandidateCheck的唯一含义

CandidateCheck只是同一个Lift owner对自己完整draft调用的Lift-local TypeCheck preflight：

```text
candidate bytes
  -> strict parse
  -> narrow Decode/Elaborate
  -> anonymous ABI/compiler-binding view consistency
  -> 与外部Runner共享的source-free TypeCheck core
  -> PASS或closed rule_code/path diagnostics
```

它不读取或比较Source/Source TTIR，不判断J是否正确，不生成handle/type/relation/patch，不修改artifact，
不进入证书。PASS后外部Runner仍对exact submitted bytes重跑正式TypeCheck。构造阶段不设低
tool/check次数上限，由active wall time、provider安全和exact stagnation治理。

### 4.5 LLM-facing workspace 与 π authoring 边界（新增）

2026-07-29 的 Linear/SplitK 首轮轨迹显示：Codex 已能自行浏览文件，但当前 workspace 同时挂载并
重复暴露 Source、Harness、minimal example、schema、lift contract、preflight、IntentOps、TTIR 和
observer projection；首轮还会通过自定义分页/查询协议试错。Linear 单次 Lift 的累计输入达到约
`1.19M` token（其中约`1.02M`为cached input），而实际 Source 只有百余行。这是上下文/序列化负担，
不是 Source 语义或模型深度不足。

因此 Codex 首包只给路径和极短工作合同；workspace 只暴露 exact Source、Harness、TTIR、一个
source-neutral closed wire reference、continuation feedback 占位及 CandidateCheck。完整Schema仍由
host checker权威消费，但模型侧只看到从同一Schema机械去掉长description后的结构合同与十条简短
所有权规则；README/contract/preflight/IntentOps/minimal-example不再同时挂载。首包不得再次嵌入
Source/Harness/outline/observer的大块投影，也不要求模型学习额外分页协议。该减法不改变
`(J,C,π)`、TypeCheck、Replay输入或SCRC谓词。

π 的 `regions`、`instances`、`routes`、`order`、`sharing`、`iterations`、
`physical_iterations`、`physical_loop_carriers` 暂不删除或合并。它们分别承载词法作用域、实例身份、operand-slot wiring、
显式执行顺序、单一producer复用、逻辑循环与物理循环/载体等不可互相推导的 witness。当前问题是
扁平 wire 对 LLM 的序列化负担，而不是这些数学关系本身错误。若后续要做 region-centric authoring
syntax，必须是一对一的闭合表示、同时更新 Decode/TypeCheck/Replay/RefEval/production consumers；
不得让 fixed code 从 Source 或 valuation 猜造 π/J/C 语义。

首轮能力验收只比较 `time-to-first-complete-candidate`、首次 Gate1 frontier 和累计输入，不把
TypeCheck PASS 当作语义进展；先用 Linear、SplitK、Flash 三个 profile 做小规模 A/B。

Linear 的首个路径清单 A/B 已给出反例：仅删除首包中的 ABI/outline/observer 投影不足以降低成本。
旧首轮为 `683s / 1.19M input tokens / 5 model segments / Gate1 score 0.687`；路径清单版本为
`846s / 1.84M input tokens / 10 model segments / Gate1 score 0.627`。新 owner 随后主动读取完整
minimal example、IntentOps、lift contract、多个 preflight 区段、接近完整的83KB JSON Schema和
TTIR片段，共产生约257KB shell输出并做8次CandidateCheck。因此“把大对象从prompt移到文件”不是
能力改进；只要重复的authoring surface仍对模型可见，它就会在不确定时重新全部读取。

下一次减法必须真正收敛对Agent可见的authoring surface：一个短而权威的closed wire reference、
exact Source/Harness/TTIR和CandidateCheck即可；完整Schema继续由host checker消费，不作为默认可浏览
教程，重叠的README/contract/preflight/IntentOps/minimal-example不得同时要求模型核对。该变化仍不
删除π关系，也不由fixed code作者化任何J/C/π字段。

该最小surface现已作为Codex专用投影实现；MiniSWE的既有workspace不受影响。模型侧reference约
26KB，来自当前artifact.v11 Schema的机械去description投影，并保留所有required、closed field、
enum、pattern、dependent authoring入口；它不是typed skeleton、composer或候选生成器。下一步只用
同一个Linear `B=1/K=1`、Sol xhigh、单候选/单Gate1做对照，先看首次Lift的累计输入、model
segments、CandidateCheck调用、墙钟与Gate1 frontier，再决定是否扩展到SplitK/Flash。

## 5. J / C / π的能力计划

### 5.1 不做大重构

后端新的消费方式不要求推翻A/B/C。稳定边界继续是：

```text
J = Σ + A + B：唯一显式语义
C：保持J denotation的implementation selection
π：依赖(J,C)的Source realization witness，SCRC后停止
```

`p*`只是J logical signature中的匿名callable ABI carrier handle，不是额外数学变量、输入域或第四对象。
compile-time variant由Harness先选择一个具体程序，再对该程序Lift；J不表示整个variant family。

### 5.2 需要补的最小J能力：dimension relations

现有`value_dimension_relations`只能表达integral rank-0 `t* == x*`，不能表达source-grounded的维度间
派生关系。新增一个closed、typed、J-owned relation，建议wire为：

```json
"dimension_relations": [
  {"result": "x3", "expr": {"op": "mul_const", "arg": "x2", "value": 2}},
  {"result": "x5", "expr": {"op": "round_up_const", "arg": "x4", "value": 128}}
]
```

首版grammar只允许：

```text
identity
add_const
mul_const
ceil_div_const
round_up_const
```

操作数只能是已声明`x*`与受限整数字面量；禁止任意`x*y`、自由字符串、SMT formula或从当前valuation
猜关系。典型用途：

- Llama4：`full_head_dim = 2 * head_dim_half`；
- Flash：`rounded_sequence = round_up(sequence, 128)`；
- SplitK：在Source/wrapper确有合同的情况下表达derived capacity/chunk extent。

关系必须由Lift从Source完整作者化。Decode/TypeCheck只检查closed grammar、正值约束、类型、引用、
acyclicity与可机械判定的冲突，不从Harness/grid/C/π补造。

该relation只有在以下消费者同一版本全部落地后才可进入fresh证书：

```text
JSON Schema / Decode / TypeCheck / CandidateCheck
Replay packet与source-blind obligation view
RefEval / functional-boundary / Gate2
共享J -> Intent MLIR materializer与production projection
spec和protocol receipts
```

production target不必模仿Source realization，但必须能消费该J-owned logical identity或明确fail closed；
不能形成“只让Gate2懂、backend不懂”的半闭合relation。

### 5.3 现有能力的所有权

| 维度 | 表达位置 |
|---|---|
| consumer-specific index/predicate/outside identity | J/B consumer access与extensions |
| logical recurrence、carry、next/yield/result | J structured loop |
| physical loop、carrier placement/order | π physical time relations |
| branch value/effect | J `value_if/effect_if` |
| terminal arm vs lexical continuation | π `terminal_controls`，依赖已有J effect_if |
| multi-output、inout pre/post-state | J signature/A/B |
| route-local physical C extent partition | π `route_extent_transforms`，不是J dimension equation |

Compiler-observed pilot先判断剩余kernel是否真的超出这些能力。不能仅凭一个no-closure就新增
conditional-post-state、multi-result control或新mechanism。

明确不增加：`allow_fma`、`allow_tf32`、`reduction_order`、accumulator permission、kernel template、
算子名提示或自由formula。执行域/Q闭合不需要把后端实现选择伪装成J permission。

## 6. Source-Blind Codex Replay agent

### 6.1 生命周期与输入

每个TypeCheck-valid formal round创建一个全新的Replay episode。该episode内部是持久、单一writer，
但formal round结束即销毁；下一轮不得继承旧Replay、旧conversation或旧compiler error。

它只见：

```text
J + π + MaskC(C) + U_f
```

其中`MaskC(C)`保留anonymous slot schema及source-neutral `subject_ref/axis_ref`，删除concrete values、
Source name/position与private binding。Replay不知道`c0`；完整源码写完后才由Harness把同一个C value
绑定给Source与Replay。

### 6.2 Codex式工作空间能力

Replay owner可以：

1. 按需读取exact J、π、MaskC与公共`U_f`；
2. 查询从同一packet无损重索引的ABI、node、region、instance、route、order、sharing、time/control
   obligation ledger；
3. 创建并局部patch自己的Replay源码；
4. 多次运行source-blind construction compile；
5. 查看自己源码的compiler errors与自己编译出的TTIR；
6. 调用ReplayCheck检查已作者化obligation是否被自己的源码消费；
7. 对exact passing bytes原子提交，并在episode内保存last-compilable checkpoint。

它不能看到Source、Source TTIR、P_cert/Harness values、Gate1 diff、prior Replay、sibling或case名称提示。
obligation ledger只重排packet已有事实，不建立relation到statement的映射、不推荐API、不生成Replay。
construction compile与ReplayCheck只是source-blind拒绝/诊断，不是Gate1 oracle。

正常大小packet只exact嵌入一次；超限时给compact manifest和按需分页。provider单turn与episode总active
wall time分离：单turn保持可恢复的有限上限，复杂Replay episode默认允许累计约3600秒；低tool-call
或“必须三步完成”不作为capability cutoff。只有当前 exact output 连续没有变化时才按 stagnation 停止。

当单个provider thread因累计上下文过大或timeout失去效率，而当前workspace已有Replay draft时，可在
同一个fresh episode内滚动到新transport thread；新thread只重读exact `packet.json`、当前Replay bytes与
host closed diagnostic，不接收语义摘要、旧episode或Gate信息。episode的单writer、总wall time和
source-blind输入边界均不改变。

## 7. Gate反馈与责任判断

### 7.1 formal round

只有外部TypeCheck通过并实际启动fresh Replay，才消耗一个formal Gate1 round。JSON、dependent ref、
CandidateCheck、Replay自身语法/编译修正都记录成本，但不冒充formal closure次数。

Gate1 diff只返回产生该candidate的同一Lift owner。Replay永远看不到diff。owner可以修改J、C或π，
也可以明确保持 exact artifact；无论哪种选择，下一formal round都使用fresh Replay。

### 7.2 失败所有权

| 证据 | 下一动作 |
|---|---|
| pre-Replay construction invalid | 同一Lift owner修完整artifact，不消耗K |
| J/B或π遗漏/错误 | Gate1 diff回同一owner，完整relift，fresh Replay |
| exact artifact被owner保留、Replay疑似漏消费 | fixed-artifact diagnostic fresh Replay；不改parent verdict，不把diff给Replay |
| Source observer unsupported | 修通用observer并升version，不记NO_CLOSURE |
| provider/runtime中断 | same-thread安全恢复或记infrastructure；不消耗语义K |
| Gate2 `reference_unsupported` | 判断是否为通用J consumer缺口；若是，修consumer后零模型exact Gate2 recheck |
| Gate2 Source–J/Replay–J数值分歧且RefEval已完整消费 | J错误，下一次实验fresh Lift |
| Gate2 Source–Replay失败 | 检查Replay/Harness/frontend执行；不能用重Lift掩盖 |

Gate1 PASS后必须原样保存exact `(J,C,π,Replay)`。Gate2失败不自动丢弃Gate1成果，也不fallback到
sibling；hidden Gate2输入和结果绝不回流本次Lift/Replay。Liger RoPE现有candidate先走上述
`reference_unsupported`判断树。

## 8. 搜索与并行合同

- 主模型：Codex Sol xhigh；若同一source-grounded语义frontier在xhigh下稳定重复且系统表达已充分，
  才对该case使用Sol max。Terra不作为最终难核主证据；
- 每case默认`B=3`，branch互不共享artifact、conversation、diff或Replay；
- 每branch默认`K=3`个TypeCheck-valid formal rounds。K只约束Gate1 relift，不限制branch内部Codex工具
  循环；若三轮重复暴露同一系统缺口，先结束运行并修能力，不盲目把K加长；
- 首个被coordinator原子观察到的Gate1 closure成为winner，siblings cooperative stop；
- hidden Gate2只对winner执行，设备并发`G2=1`；
- capability pilot稳定后从`W=2,B=3,G=6`开始，provider/host稳定且首轮延迟可控时升到
  `W=3,B=3,G=9`。一个case堵塞不能阻断其他ready case；
- TypeCheck/CandidateCheck/ReplayCheck通过数不计semantic progress；进度只报告Gate1 frontier/closure、
  Gate2三边与SCRC verdict。

## 9. 执行阶段

### P0：设计与接口收敛

1. 将本Task与`.trellis/spec`对齐；
2. 列出当前Source compile/TTIR query、Lift runtime、Replay runtime、dimension relation和RefEval的
   capability matrix；
3. 明确所有protocol升级点与历史artifact兼容规则；
4. 不调用模型、不跑formal closure。

### P1：通用能力实现

1. 向Lift workspace开放匿名Source-side TTIR的exact按需查询；
2. 实现J `dimension_relations`端到端消费者；
3. 将Replay整理为单writer、可编辑、可重复编译、可看自身TTIR/错误的轻量coding-agent episode；
4. 消除Source/J/π/schema在provider turns中的重复大上下文；
5. 修复Liger暴露的dynamic-extent RefEval consumer能力，或以证据判定其J确实缺失；
6. 不新增或恢复`python/tests`；真实pilot artifact是唯一验收入口。

2026-07-29实现状态：上述1--6已落地。真实DYT匿名Source编译观察为`pass`并得到6029字符TTIR；
`artifact.v11`的closed `dimension_relations`已贯穿Decode/TypeCheck、Replay packet、RefEval、ContractIR
与production materializer；Codex Replay已可查询自己的synthetic TTIR，并按3600秒episode总预算运行。
formal pilot 随后证明编译观察能帮助 Linear 首轮恢复多物理 carrier，但原始 TTIR/低层 π 与重复
startup context 仍造成显著 attention/serialization 成本，故 P1 重新打开为上述 context-reduction
工作，不引入新数学对象。

### P2：四个真实pilot

能力提交且工作区干净后，才依次/并行启动：

1. Liger RoPE：复用exact Gate1 evidence做consumer诊断；若J错则fresh Lift；
2. Linear：检验logical loop/carrier与Compiler-observed Lift；
3. Flash full backward：检验online recurrence、physical time与control；
4. SplitK reduce：检验`split=4` ABI与derived dimension relation。

pilot必须使用第3.2节正式profile，不再使用小capture seed。每个pilot结束即按第7.2节归因；不能一边
运行一边改同一experiment的schema/prompt/observer。

### P3：最终十项闭合

pilot证明系统接口稳定后，对10项按`2--3 cases × B=3`持续保持并行。每次完整运行结束后横向聚类
真实frontier，只实现source-neutral能力，再fresh rerun受影响case，直到：

```text
10 active cases -> SCRC_CERTIFIED
18 retained cases + 10 new cases -> final 28-case accepted corpus
```

### P4：收尾

- 对winner执行独立masked-J/J-necessity消融；
- 汇总正式profile、closure@round、Gate1 frontier演化、Gate2三边、wall time与agent telemetry；
- 更新四份权威文档、第五份backend边界文档、spec与Task 11事实；
- 提交并保持工作区干净。

## 10. 交付物

1. reviewed `P_cert`与adapter-v6 formal profile ledger；
2. 轻量匿名Source/TTIR query method与Codex Lift workspace；
3. 持久single-owner Lift runtime及closed CandidateCheck接口；
4. J `dimension_relations`从wire到Replay/RefEval/shared materializer的端到端实现；
5. fresh source-blind Codex Replay coding episode及自身compile/check工具；
6. Gate1/Gate2所有权判断与fixed-artifact recheck路径；
7. 10项新证书、18项保留证书和最终28项stable corpus；
8. 与实现一致的spec、方法文档、Task 11状态与干净提交。

## 11. 完成条件

1. 新增前端仍只需`Compile_f + Run_f + Observe_f`，没有per-kernel extractor/lowerer；
2. Lift在可编译匿名Source与可查询TTIR上完整作者化`(J,C,π)`，fixed code不生成语义；
3. Replay request严格只有`J+π+MaskC(C)+U_f`，Replay可以像轻量Codex一样迭代自己的源码与编译错误；
4. 所有source-grounded logical/derived extent由J closed relation表达，grid/C/valuation不补语义；
5. 十项active queue均取得current-version TypeCheck、fresh Replay、Gate1与三边Gate2；
6. 最终28项均有可供production backend消费的accepted J+C；π不越过SCRC；
7. 没有新增tests、kernel模板、RuleCard、composer、跨branch信息共享或hidden Gate2反馈；
8. 文档/spec/代码/evidence一致，提交后worktree clean。

## 12. 非目标与禁止退化

- 本任务不实现backend search、性能优化或论文实验；
- 不恢复MiniSWE主路径；
- 不恢复planner/builder/reviewer多agent写作；
- 不让Compiler/TTIR工具生成J/C/π或Replay；
- 不把Source TTIR、Harness、C value、Gate diff或旧Replay交给Replay；
- 不建立deterministic Replay composer、kernel-specific prompt/template或每frontend语义规则库；
- 不用hash/signature gate、全量重复审查或mock test代替真实闭合；
- 不用TypeCheck PASS、更多token、跨轮best-parent或无限K冒充能力提升；
- 不因后端搜索新增FMA/TF32/reassociation permission；
- 不让production读取π，也不让Gate2 consumer变成候选合法性的membership oracle。

最终原则：

> **Lift通过Source与真实compiler observation理解开放世界；J完整承载语义；π只承载Source realization；
> Replay以一个新鲜、source-blind的coding-agent episode消费typed artifact；小型固定核心只检查、隔离和
> 比较。先把这条链做轻、做强、做通，再闭合剩余十项。**
