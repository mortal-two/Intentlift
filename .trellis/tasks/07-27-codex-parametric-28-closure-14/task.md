# Task 14：Source-Grounded Certification Profile 下的 Codex 28-Kernel Fresh SCRC 最终闭合

> 状态：执行中（2026-07-28）；六组已横向 fresh 覆盖，当前 14/28 已取得可用闭合 evidence
>（11 个原运行直接认证，另 3 个复用各自 exact Gate1-closed artifact/Replay 经零模型 Gate2-only
> capability recheck 认证）；下一步按 Gate1 frontier 修复通用 Lift/Replay 消费并继续复跑未闭合项
>
> 类型：Task 11 的最终 fresh corpus执行波；不是 production backend、`J_dev`能力建设、性能搜索、
> 论文实验或历史证书重判。
>
> 历史基线：18/28只说明旧版本各自曾取得证书；本任务对active registry的28个direct symbols全部
> fresh运行，不用旧证书替代当前参数化/runtime版本。

## 0. 本任务的权威输入与边界

本任务继续服从四份SCRC权威文档及当前`.trellis/spec`。另吸收后端工作树的输入/输出、dtype和
config审计结论：

```text
/home/kingdom/phdworks/TianchenIR-backend-certified-j-search/
  doc/IntentIR_后端能力建设与最终实验协议_2026-07-27.md
backend commit = 46e1288dabfcca46d783de87fbcfaf4ca49ded95
document sha256 = f9dd1ff1ea4879f664de3fa5c3b86e5ed9c558c511197255e32344a3fbbf9fda
```

该后端文档只提醒前端先核对被认证程序、Harness ABI与concrete specialization；它不能覆盖SCRC数学，
也不能要求J迎合某个target/backend recipe。后端的capability/paper config、Target--Ref receipt、
MLIR search和`intentir.backend.dev-j-variant.v1`均不进入本任务。

当前没有仍在运行的闭合进程。Sol-high Wave 6 已结束：SplitK reduce 在 branch 2 formal round 2
Gate1 closed；Source--Replay 通过，旧 RefEval 只因两个唯一 node-less predicated logical-input load
缺少显式 fill 而 unsupported。`j_scalar_cpu_mlir_v18 / intentir.j-functional-boundary.v17` 只从 J
组合“唯一 whole-input load + 唯一 direct pointwise consumer + 同 predicate observable store”后，原
artifact/Replay 以 `model_calls=0` 的 exact Gate2-only recheck 认证。CrossEntropy、Flash forward、
xFormers RMSNorm、SplitK backward 为完整 NO_CLOSURE；Flash full backward 与 SplitK forward 混有
Replay-before-Gate1 infrastructure branch，不能伪装成完整搜索失败。
CrossEntropy v9 canary 证明 `terminal_controls` 已被 Replay 正确消费，剩余失败是 arm 内 op/dataflow
生成错误，不再误判为 terminal relation 缺失。所有 formal run 继续只从 clean committed stack 启动，
运行中不修改 immutable input。

## 1. 第一性原理目标

为active registry的28个真实direct kernel各自建立一个source-grounded primary certification
specialization，并使用Task13完成的Codex single-owner runtime fresh形成：

```text
Source snapshot + selected compile-time program variant
  + reviewed anonymous Harness/compiler view P_cert
  -> complete LLM-authored (J,C,pi)
  -> fresh source-blind Replay
  -> Gate1 closure
  -> Source--Replay / Source--J / Replay--J Gate2
  -> current-version SCRC certificate + immutable evidence
```

旧artifact、Replay、Gate verdict与digest保持原样，仅作历史对照。Task14不只是“再跑一次旧adapter”：
若输入/输出role、dtype、shape identity、selected variant或observable outputs不对，必须先修正Source/Harness
边界，再fresh Lift；不能让LLM为错误Harness闭合，也不能在Gate后偷偷改ABI。

## 2. 每个case的P_cert：Harness sidecar，不是第四个artifact

每个case进入Lift前必须有一份可审计的`P_cert`记录。它可以由registry、adapter、source slice与
upstream test/wrapper provenance组合产生，但只属于Harness/task sidecar：

```text
case_id / registry ordinal / direct symbol
source location + source snapshot digest + upstream provenance
selected compile-time variant and active/inactive flags
ordered callable ABI
  - buffer/value/None parameter
  - input/output/inout mode
  - concrete boundary dtype and rank
  - concrete capture valuation
  - shape_contract h* / runtime-scalar equality / constexpr equality
observable outputs, inout pre/post state, auxiliary statistics and scratch effects
runtime scalars / semantic constexprs / C slots / grid realization
external Q policy lineage（J result relation仍由Lift独立作者化）
review status and factual evidence pointers
```

`P_cert`不进入`(J,C,π)`、Replay packet、Gate predicate或production backend，不选择Intent op、不推导
relation，也不生成J。Lift LLM仍完整作者化J/C/π；fixed code只可检查P_cert与Source ABI、当前valuation
和LLM输出是否一致并fail closed。

`P_cert`尤其不得预填J `consumer_extensions`、access/control/result relation，或π
`iterations/loop_carriers/physical_iterations/physical_loop_carriers/terminal_controls/routes/order`；这些仍由Lift从Source
完整作者化并由Gate1检验。TMP等scratch的input/output/inout与可观察读写属于ABI/effect审查，但其
realization topology不属于P_cert。backend `ContractView` consumed/missing facts、Target--Ref correctness、
target object与performance receipts也只属于backend evidence，不是P_cert或SCRC predicate。

### 2.1 事实所有权

| 事实 | Task14中的所有者/处理 |
|---|---|
| boundary tensor/scalar dtype、显式cast、reference op dtype | J/A；本次certificate为concrete specialization |
| literal shape、symbolic axes、layout、observable ABI、input/output/inout | J/A+B，由Source/P_cert约束 |
| concrete tensor values、symbol valuation、runtime stride/grid | Harness valuation，必须满足J |
| schedule tile/threads/vector/pipeline | C或后端；不改变J合法空间 |
| target内部FMA/TF32/approx/accumulator choice | 后端/Q；不是Task14输入dtype |

Harness不得把`f32` J改成BF16，也不得把FP32 internal accumulator误写成boundary FP32要求。当前J没有
type variable，所以每种额外boundary dtype都是另一份Source specialization与fresh certificate，不是同一J的
Harness valuation。

### 2.2 capture valuation、参数化J与natural workload分账

- `capture seed`可以小，但必须source-grounded、非退化并真正执行selected variant的关键路径；
- tensor `shape`只是本次valuation；Source/launcher唯一确定的logical identity由Harness-only `h*`表达，
  最终`x*`/value-dimension equality仍由Lift作者化；
- 一个parametric J说明其logical shape identity，不把本次Gate2自动推广为所有natural/paper shapes的
  动态测试证书；
- backend capability/natural/paper config若没有对应fresh accepted J，只能是独立`J_dev`/backend
  evidence，不计入Task14；
- 每个direct symbol先确定一个primary certification specialization。额外dtype/compile-time variant可以
  独立认证，但不改变28项主分母。

每个P_cert还必须列出逐coordinate shape ledger：

```text
h* / current valuation
grounded tensor axes
grounded runtime scalar or semantic constexpr equality（若存在）
grid-only realization（若存在）
expected Lift obligation = dimension-only | rank-0 value eq | selected literal
current closed relation support = supported | unexpressible
```

该ledger不为Lift分配`x*`或生成relation，只把“哪些extent确有Source依据、哪些必须保持literal”从
主观判断变成逐case可审计输入。完成时必须逐项核对J作者化结果。

## 3. P0：28项输入/输出、dtype、variant与shape审查

P0按case并行执行；一个case的P_cert一旦READY即可进入closure queue，不要求等28项形成全局barrier。
每项必须对照actual direct Source、selected upstream wrapper/test与adapter，而不是只看当前JSON数值。

### 3.1 审查结果分类

```text
PROFILE_READY
ADAPTER_ABI_FIX_REQUIRED
SOURCE_VARIANT_AMBIGUOUS
SHAPE_RELATION_UNEXPRESSIBLE
UPSTREAM_PROVENANCE_MISSING
```

非`PROFILE_READY`不得启动正式Lift。修复只能保留Source-grounded事实：不得按Kernel名称生成J、增加
Replay模板、从backend headline config反推语义，或把相等valuation合并成同一extent。

### 3.2 已知必须重点复核的case（不是预判其一定错误）

| case/family | P0必须核对 |
|---|---|
| Flash full backward / one-col backward | Q/K/V/DO与DQ/DK/DV destination的真实wrapper dtype；DQ是inout还是output；FP32 accumulator与boundary grad carrier不能混淆 |
| Flash forward/backward Bias | `BIAS_TYPE` selected variant下Bias是否active、其dtype/shape/role是否可观察 |
| Flash one-pass LayerNorm | inactive ROWSCALE/SEEDS/mask carriers、Mean/Rstd FP32与main output dtype |
| GroupNorm | Y/Mean/RSTD carrier、W/B axis、groups/channels-per-group与tensor shape identity |
| CrossEntropy | X early-return/clear导致的inout事实；loss、optional weight、I64 label及关闭的z-loss/accuracy/token outputs |
| JSD/TVD | `HAS_LABEL=false`时I64 label ABI carrier是否inactive且无假语义 |
| RoPE/MRoPE | q/k真实inout；cos/sin broadcast batch、head counts、full/half head dimension；不得伪造乘积关系 |
| xFormers RMSNorm/RMSNorm-add | FP32 capture是否作为primary specialization有上游依据；X inout与Y same-dtype output |
| SplitK forward/reduce/backward | Q/K/V、FP32 partial Out/LSE、final Out caller dtype、split outputs及vararg/None ABI；selected variant/product/capacity保持literal时给出依据 |

### 3.3 参数化允许与禁止

允许要求fresh Lift作者化：

- row-wise rows/cols；
- Flash/Linear/GroupNorm中由tensor shape、runtime scalar或显式bound constexpr唯一绑定的batch、
  sequence、head、head-dim、rounded capacity；
- Qwen2VL MRoPE、Liger RoPE中唯一对应tensor axes的batch/sequence/head/head-dim；
- Llama4 RoPE中的batch/sequence/head-count/head-dim-half。

禁止擅自提升：

- Llama4 `full_head_dim=2*head_dim_half`在当前closed relation vocabulary中没有乘积关系时保持
  selected-variant literal；
- SplitK产品、vararg、derived capacity、split count与compile-time family保持selected variant，直到
  Gate与production共同消费的closed relation存在；
- inactive/dummy ABI carrier、static coefficient axis、stride/layout metadata不得因数值类似logical
  extent而提升；
- grid prose、当前valuation或backend natural config不得自动生成`h*`/`x*`。

若Lift把P_cert已声明的source-grounded `h*`固化成concrete literal，candidate在正式Replay前拒绝；
fixed code不得替Lift修成`x*`。

## 4. 当前执行栈

- frontend commit起点：`460aaed1c6af0b267a4a9da9ca497aea6e66ad40`；
- artifact/decode/typecheck：`artifact.v9 / decode.v13 / typecheck.v22`；
- Lift：Codex `openai.codex-cli.intentir-lift.v4 / intentir.lift-knowledge.v59`；
- Replay：Codex `openai.codex-cli.intentir-replay.v11`，MiniSWE `minisweagent.intentir-replay.v16`；
- Replay packet/check：`replay-input.v1 / public_frontend_context_v12 / replay-check.v7`；
- Harness：Triton workload adapter v5；
- Gate1 observer：`triton-ttir-3.6-gate1-v11`；outer batch：`intentir.scrc.batch.v3`；
  candidate fan-out：v4 `first_observed_gate1_closure_cooperative_stop`；
- Gate2：三条直接比较，threshold只来自外部Q。
- Gate2 RefEval：`j_scalar_cpu_mlir_v18 / intentir.j-functional-boundary.v17`。

### 4.1 当前 fresh closure 账本

- Wave 1（ordinal 1--7）：SwiGLU、RMSNorm、Fused Add RMSNorm、GeGLU、LayerNorm certified；
  CrossEntropy、RoPE 未闭合。
- Wave 2（ordinal 8--14）：Softmax、KLDiv、DYT、Sparsemax certified；GroupNorm、Qwen2VL MRoPE、
  JSD 未闭合。
- Wave 3（ordinal 15--21）：PolyNorm certified；Llama4 RoPE、TVD、Linear 未闭合；Flash forward、
  one-col backward、full backward 的本次 aggregate 被上述 immutable-input infrastructure 污染。
- Wave 4（ordinal 22--28）：Flash backward preprocess 直接 certified；Flash one-pass LayerNorm 由同一
  Gate1-closed J/Replay 的零模型 Gate2-only recheck certified；xFormers RMSNorm、SplitK forward/reduce
  未闭合；RMSNorm-add 的 J signature overlap 由 typecheck.v22 fail closed；SplitK backward 混有
  Replay wall-time infrastructure。
- Wave 5（ordinal 5/19/24--28）：RMSNorm-add Gate1 closed，并由同一 exact artifact/Replay 的零模型
  Gate2-only recheck certified；RMSNorm、Flash forward、SplitK reduce/backward 未闭合；CrossEntropy
  与 SplitK forward 各有一个 Replay infrastructure branch、其余 branch 未闭合。
- Wave 6（ordinal 5/19/21/24/26--28，Sol high）：SplitK reduce 在 branch 2 formal round 2
  Gate1 closed；v18/v17 通用 consumer 修复后，以原 artifact/Replay、`model_calls=0` 的 exact
  Gate2-only recheck certified。CrossEntropy、Flash forward、xFormers RMSNorm、SplitK backward
  为 NO_CLOSURE；Flash full backward与SplitK forward因至少一个 Replay-before-Gate1
  infrastructure branch 聚合为 infrastructure_failed。
- Wave 7（剩余14项，commit `d40e37ef`，Codex Sol high）进行中：JSD 在 branch 3 formal round 1、
  TVD 在 branch 2 formal round 2 fresh Gate1 closure 后通过三边 Gate2，均直接 certified；其余项继续
  执行，不提前写终态。
- 当前可用闭合总计 `16/28`：13 个原运行直接认证，3 个 exact-artifact capability recheck 认证。
  TypeCheck PASS 不计进度；只有 Gate1 已闭合且当前 Gate2 三边通过才计入。

CrossEntropy 曾暴露一个通用表达缺口：J 已作者化 early-return `effect_if`，旧 π 无法区分 terminal
arm 与 lexical continuation。current `artifact.v9` 只新增 closed
`pi.terminal_controls{terminal_instance,continuation_instance}`；relation 只能连接同一 J `effect_if` 的
sibling arm-scope instances，不复制 predicate/polarity/return/Source/API，且 production 不读 π。这是
source-neutral terminal CFG topology，不是 CrossEntropy hint 或 deterministic composer。

v9 canary 已确认该 relation 从 packet、prompt、ReplayCheck 到生成代码完整消费；当前 terminal_fork
差异来自 Replay 对两个 arm 内 J op/effect/dataflow 的遗漏或改写。因此后续增强目标是 source-free
Replay obligation consumption，不再扩张 terminal relation，也不把 Gate1/Source 反馈泄给 Replay。

Codex Replay v8 增加 `intentir.scrc.replay-workspace-view.v1`：同一 fresh session 先读取一次紧凑、
lossless 的匿名 ABI/J graph/J boundary/MaskC/U_f 与 per-instance π obligation ledger，不再先全文 dump
packet 再重复读取完整 π focused view。ledger 只重排当前 packet 已声明的 route/order/sharing/time/
refinement/terminal facts，不生成 op、predicate、地址公式或 schedule。

本轮 Codex Replay v11 使用该 v2 view，只把 J 中已经作者化的 reduction operand/result type、
semantic axes、对应 boundary reduction、C attachment 与 π instance 无损并置成 reduction obligation；
它不选择 physical rank、tile、reshape、loop 或 frontend spelling。`replay-check.v7` 在公共 synthetic
compile 后按 π 已作者化的 `transfer_direct` occurrence 核对 compiled load/store multiplicity，并按
structured J loops 加 `pi.physical_iterations` 核对 compiled `scf.for` multiplicity，再按对应
`pi.loop_carriers/physical_loop_carriers` 核对 block-argument arity；`pi.iterations`只 realize 已存在的
J loop，不重复计数。检查防止 CSE 折叠 transfer、Replay 遗漏已声明物理时间或 carrier；仍不读取
Source、Harness value 或 Gate1，也不补 loop/bound/carrier/transfer。
Codex Lift v4 与 Replay v11 对单次 provider turn wall-time 只做同 thread/session 续跑，并且只可恢复
本次 invocation/episode 内由同一 LLM 写出且已经通过对应 closed check 的 exact bytes；固定代码不
合并、修复或跨正式轮次选择 candidate。Lift v4 另要求 LLM 在提交前逐项 reconcile syntax-only
Source control-flow outline，但分类与完整 J/C/π 仍由 LLM 作者化。MiniSWE v16 共享同一
replay-check.v7 合同，但当前 28 项
fresh closure 仍以 Codex 为主执行路径。

实现升级后只fresh rerun受影响case，不重判旧receipt。Task14期间每次通用能力提交都必须在后续
evidence中记录commit/protocol；不能让一个长期脏工作区承载多个不可区分的实验版本。

## 5. 不可改变的Agent生命周期与信息边界

1. 每branch只有一个source-aware Codex Lift owner，贯穿该branch全部formal Gate1 relift；closed
   feedback只回到这个owner，每轮重新提交完整`(J,C,π)`。
2. 每个formal round创建全新的source-blind Replay episode。Replay只见
   `J + π + MaskC(C) + U_f`，不见Source、P_cert/Harness value、Gate diff、旧Replay、sibling或其他case。
3. LLM完整作者化artifact与Replay。工具只提供exact current bytes、lossless J/π view、closed checker、
   exact submit/receipt；不得补relation、猜axis、修J或生成Kernel recipe。
4. Gate1 diff只进入产生它的Lift owner；绝不进入Replay或sibling。不同branch/case不共享candidate、
   notes、best parent或semantic cache。
5. compile-time variant是不同程序/证书；一次branch不跨variant。
6. π在SCRC后停止；Task14不实现backend，也不让backend读取Source/π。

Codex为主路径；MiniSWE只在隔离provider/runtime问题时做同合同交叉核验。MiniSWE若承担正式branch，
必须在branch创建前被选为该branch backend，并遵守同一persistent Lift owner、fresh Replay与formal K；
它不能在一次Codex branch中途接管thread/candidate/feedback，诊断交叉核验也不能混入该branch预算或
verdict。provider transient retry属于
transport：只有transport确认可resume时才在同thread/workspace上有界重试并留下receipt，不计formal
K；无法安全resume时归为infrastructure failure。first-winner cancellation绝不能被误当transient error。

## 6. 并行、branch与预算

- 主模型：Terra high；Sol high/xhigh仅作明确记录的独立对照；
- 同case默认`B=3`，三个branch彼此隔离；formal Gate1上限默认`K=5`，首个Gate1 closure立即成为
  winner并cooperative stop siblings；
- agent内部tool/model steps没有“三次”capability上限，由active wall-time、transport safety与exact
  stagnation管理；
- 第一执行波建议`W=4,B=3,G=12`；provider/host稳定后提升到`W=6,B=3,G=18`。`G2`按GPU资源单独
  限制，不让Gate2阻塞其他case的Lift/Replay；
- readiness queue混合多个语义族；一个慢case、P0待修case或provider错误不能让其他READY case空闲；
- report按registry ordinal稳定汇总，与完成顺序无关。

winner必须按fan-out v4定义为host首先原子观察到、且完整Gate1-closed evidence已经落盘的branch；
同一临界区只登记一个winner receipt。sibling看到winner token后停止后续Lift/Replay，保存
`SUPERSEDED + winner_branch_id`，已经消耗的调用与bytes/hash不丢弃。wall-clock完成顺序本身没有
认证含义。

`K=5`是上限而非必须跑满。只有全部B个branch带完整`NO_CLOSURE` evidence耗尽其声明K，case才能
报告`NO_CLOSURE_(B,K)`；infrastructure/observer/profile invalid不能伪装成NO_CLOSURE。

TypeCheck、CandidateCheck、ReplayCheck和synthetic compile只定位构造阶段，不作为能力指标。能力
进展只看Gate1 frontier/closure、Gate2三边与最终SCRC verdict。

## 7. 横向执行与真实问题解决

### Wave 0：P_cert与runtime acceptance并行

1. 对28项横向生成/复核P_cert，READY case持续进入queue；
2. 一个历史简单case fresh形成真正参数化的Gate1/Gate2；
3. 一个Flash/Linear复杂case验证`consumer_extensions`、control、
   `physical_iterations/physical_loop_carriers`、TMP inout与完整π消费；
4. marker、路径、mailbox、provider或工具发现失败只算orchestration/infrastructure，不记语义失败。

### Wave 1：28项第一遍fresh覆盖

- 每项保存P_cert、完整branch/round/evidence；
- Gate1 PASS后固定exact artifact/Replay再进入Gate2；
- Gate1-closed但Gate2未过的候选单独保存，不丢弃、不自动relift；
- 多个语义族持续并行，不能围绕一个高分J长期碰运气。

### Wave 2：按所有权修通用能力并fresh rerun受影响case

失败必须先归入一个主类：

```text
P_cert / adapter / Source projection invalid
Lift artifact construction
J/B semantic or result-relation omission
pi realization / consumer-context / time / control omission
Replay packet consumption or source generation
Gate1 observer unsupported
Gate2 RefEval / Q / frontend consumer gap
provider / sandbox / runtime infrastructure
```

Gate1 category本身不自动证明责任层；必须沿Source、J/B、π、Replay四份现有证据逐层定位：

| Gate1 frontier | 首先核对 | 可能责任；需证据后才能确定 |
|---|---|---|
| interface | P_cert callable ABI、J signature、Replay signature | profile/adapter；Lift J；Replay ABI消费 |
| op_multiset / reduction | J logical A ops/reduce/contraction是否覆盖Source，再看Replay | Lift J；π physical realization；Replay consumer |
| dataflow | J operands/results/access、π routes/sharing/carriers、Replay SSA/address graph | Lift J/B；Lift π；Replay consumer |
| predicate_effect | J domain/predicate/access/effect/control、π order、Replay branches/load/store | Lift J/B；Lift π；Replay consumer |
| schedule-only / realization | π iteration/placement/order与Replay物理结构 | Lift π；Replay consumer |
| observer unsupported | observer closed profile与Source/Replay feature | 固定observer能力，不是candidate NO_CLOSURE |

混合diff可以记录多个secondary categories，但每次通用修复必须给出“最早缺失事实”的artifact path、
Gate witness与责任层；fixed code不得根据分数或类别名称自动修改J/π/Replay。

修复原则：

- P_cert/adapter错误：修Source-grounded Harness事实后fresh Lift，旧attempt保持原样；
- pre-Replay construction错误：同一Lift owner使用closed diagnostics修自己完整artifact，不消耗formal K；
- J/B或π错误：Gate1 closed diff回同一Lift owner，完整relift，下一轮fresh Replay；
- Replay消费错误：不得把Gate1 diff或Source泄给Replay。先用fixed-artifact diagnostic recheck定位并修
  generic source-blind consumer，再对受影响case作fresh formal run；
- observer unsupported：扩展固定observer并记录新版本，不把候选记成NO_CLOSURE；
- Gate2 consumer/Q缺口：若exact J与Replay正确，零模型复用Gate1-closed bytes做Gate2-only recheck；
- J result relation本身错误（例如浮点路径无依据声明exact）：必须fresh Lift，不能靠Q容忍错误J；
- 只修通用能力，不加入case/source-symbol专用prompt、模板J、hard-coded Replay或跨case cache。

Task14不会在第一遍`NO_CLOSURE`后结束；它持续横向聚类、实现通用能力并fresh rerun，直到28项闭合，
或出现经权威设计审查确认的表达边界。后者保持Task活跃并明确列出需要的closed relation/consumer，
不得以“模型不行”或更多token代替分析。

### Wave 3：最终证书、necessity与汇总

- 形成当前版本28/28`SCRC_CERTIFIED` corpus；
- 对winner运行既有Masked-J/necessity消融；
- 汇总P_cert、formal rounds、Gate1 frontier演化、Gate2三边误差、wall-time、model/tool telemetry与失败
  修复谱系；
- 不报告TypeCheck pass rate作为semantic capability。

## 8. Gate1 / Gate2判断树

```text
P_cert READY?
  no  -> fix Source/Harness contract; no Lift
  yes -> Lift complete (J,C,pi)
           |
       construction valid?
           no -> same owner repairs; not a formal Gate1 round
           yes -> fresh Replay -> Gate1
                    |
          Gate1 FAIL/UNSUPPORTED
             |- J/B or pi wrong -> same Lift owner relift
             |- Replay wrong    -> generic Replay fix, then fresh run
             |- observer gap    -> observer capability/version fix
             `- profile mismatch-> invalidate P_cert and fresh restart
                    |
          Gate1 PASS -> preserve exact artifact + Replay -> Gate2
                    |
          Gate2 FAIL
             |- J/result relation wrong -> new fresh Lift
             |- Harness/profile invalid -> repair P_cert, fresh restart
             `- RefEval/Q/consumer gap  -> zero-model Gate2-only recheck
```

任何hidden Gate2 input/result不得回流Lift、Replay或winner selection。Gate2失败后不fallback到另一个
sibling；若问题不是J，不能浪费token重做已经闭合的Gate1。

## 9. Evidence与进度报告

每个case至少保存：

```text
P_cert review + source/adapter/profile digests
branch id / owner session id / model / effort / commit / protocol versions
每轮完整 artifact digest、Replay digest、Gate1 frontier与receipt
first-winner / sibling stop / retry / infrastructure receipts
Gate2三条直接边、Q policy、result relation与误差
masked-J / necessity结果
最终状态与通用能力修复谱系
```

live进度必须至少报告：P0 review、Lift、Replay、Gate1、Gate2当前阶段，当前已certified数量、
Gate1-closed但待Gate2数量、各未闭合主因与活跃W/B/K；不能用TypeCheck PASS制造进度感。

## 10. 完成条件

1. active high-performance registry准确为17 Liger + 6 FlashAttention + 5 xFormers direct symbols；两个
   `scrc_canary`不进入28分母；
2. 28项P_cert全部reviewed，每项primary input/output/inout、boundary dtype、selected variant、observable
   outputs、逐coordinate shape ledger与upstream/source provenance可追溯；
3. 28项都有当前版本fresh、case-local accepted`(J,C,π)`、fresh Replay、Gate1与三边Gate2 evidence；
4. 每个应参数化logical extent由J作者化为`x*`/合法equality；selected-variant literal有依据，不把
   capture valuation冒充natural/paper shape证书；
5. 28项均为`SCRC_CERTIFIED`；历史18/28、diagnostic recheck和任何`J_dev`只保留为独立对照；
6. evidence按registry ordinal稳定汇总，first-winner、sibling stop、formal K、fresh Replay、provider/
   runtime receipts及失败修复谱系完整；
7. 代码、四权威文档、spec、Task11状态与最终corpus事实一致，测试通过，提交后工作区干净。

## 11. 非目标

- 不实现/评估production backend search、C-Fixed/Schedule-Only/IntentIR三臂或target性能；
- 不在Task14创建/认证backend `J_dev`；capability/natural/paper config没有fresh accepted J时不计SCRC；
- 不要求一个J覆盖FP16/BF16/FP32；当前dtype是concrete specialization；
- 不从Forge、backend catalog或旧artifact复制当前证书；
- 不增加FMA/TF32/reassociation permission字段替代执行域闭合；
- 不恢复planner/builder/reviewer多作者流水线；
- 不用TypeCheck、更多token、更多K、best-parent碰运气或Kernel特化prompt冒充Gate1/Gate2能力。
