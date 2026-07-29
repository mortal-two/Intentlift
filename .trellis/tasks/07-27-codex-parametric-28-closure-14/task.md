# Task 14：Source-Grounded Certification Profile 下的 Codex 28-Kernel Fresh SCRC 最终闭合

> 状态：执行中（2026-07-29）；六组已横向 fresh 覆盖，当前 20/28 已取得可用闭合 evidence
>（15 个原运行直接认证，另 5 个复用各自 exact Gate1-closed artifact/Replay 经零模型 Gate2-only
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

Wave 16 的旧六核 Sol-high 诊断批与 Wave 17 的 GroupNorm/SplitK-backward Sol-xhigh 双核批正在仓库外
`/tmp` 独立运行；运行中局部 TypeCheck/Replay 状态不计闭合进度。Wave 14 曾从 clean commit
`16904c1f` 对剩余 11 项启动 Codex
Sol-high、`B=3,K=5` fresh run；若干 branch 已进入 1--3 个 formal Gate1 round 并形成 no-closure
frontier，但运行期间 host 更新了共享 `tools/context_query.py`。旧 runtime 在 continuation 时把仓库
当前工具字节与 owner 创建时的 immutable workspace 比较，因而使受影响 branch 聚合为
infrastructure failure。后启动 branch 实际可以使用增强后的只读工具正常继续；把其工具版本元数据
不整齐误判为数学证书无效并停止整个 batch 是一次过度保守的调度错误。两个 batch 已人工停止，
已有文件保留 partial frontier/orchestration 证据，不构成完整 `NO_CLOSURE_(B,K)`、不改变
`17/28`。v7/v61/v59 现把 Source/Harness/schema/knowledge/tools 固定为 branch-owner 创建时的私有
快照，只原子刷新 closed feedback（Codex 另刷新 CandidateCheck 结果）；同 owner 的 Source/Harness
若变化仍 fail closed。下一次正式闭合必须从该修复的 clean commit 重启。

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
- artifact/decode/typecheck：`artifact.v10 / decode.v15 / typecheck.v27`；
- Lift：Codex `openai.codex-cli.intentir-lift.v11 / intentir.lift-knowledge.v69`，MiniSWE `minisweagent.intentir-lift.v3 / intentir.lift-knowledge.v65`；
- Replay：Codex `openai.codex-cli.intentir-replay.v26`，MiniSWE `minisweagent.intentir-replay.v30`；
- Replay packet/check：`replay-input.v1 / public_frontend_context_v17 / replay-check.v11`；
- Harness：Triton workload adapter v5；
- Gate1 observer：`triton-ttir-3.6-gate1-v11`；outer batch：`intentir.scrc.batch.v3`；
  candidate fan-out：v4 `first_observed_gate1_closure_cooperative_stop`；
- Gate2：三条直接比较，threshold只来自外部Q。
- Gate2 RefEval：`j_scalar_cpu_mlir_v22 / intentir.j-functional-boundary.v20 / intentir.j-contract-indexed-cpu-reference.v1`。

### 4.1 当前 fresh closure 账本

- Wave 1（ordinal 1--7）：SwiGLU、RMSNorm、Fused Add RMSNorm、GeGLU、LayerNorm certified；
  CrossEntropy、RoPE 未闭合。
- Wave 2（ordinal 8--14）：Softmax、KLDiv、DYT、Sparsemax certified；GroupNorm、Qwen2VL MRoPE、
  JSD 未闭合。
- Wave 3（ordinal 15--21）：PolyNorm certified；Llama4 RoPE、TVD、Linear 未闭合；Flash forward、
  one-col backward、full backward 的本次 aggregate 被上述 immutable-input infrastructure 污染。
- Wave 4（ordinal 22--28）：Flash backward preprocess 直接 certified；Flash one-pass LayerNorm 由同一
  Gate1-closed J/Replay 的零模型 Gate2-only recheck certified；xFormers RMSNorm、SplitK forward/reduce
  未闭合；RMSNorm-add 的 J signature overlap 由当时的 typecheck.v22 fail closed；SplitK backward 混有
  Replay wall-time infrastructure。
- Wave 5（ordinal 5/19/24--28）：RMSNorm-add Gate1 closed，并由同一 exact artifact/Replay 的零模型
  Gate2-only recheck certified；RMSNorm、Flash forward、SplitK reduce/backward 未闭合；CrossEntropy
  与 SplitK forward 各有一个 Replay infrastructure branch、其余 branch 未闭合。
- Wave 6（ordinal 5/19/21/24/26--28，Sol high）：SplitK reduce 在 branch 2 formal round 2
  Gate1 closed；v18/v17 通用 consumer 修复后，以原 artifact/Replay、`model_calls=0` 的 exact
  Gate2-only recheck certified。CrossEntropy、Flash forward、xFormers RMSNorm、SplitK backward
  为 NO_CLOSURE；Flash full backward与SplitK forward因至少一个 Replay-before-Gate1
  infrastructure branch 聚合为 infrastructure_failed。
- Wave 7（剩余14项，commit `d40e37ef`，Codex Sol high）已完成：JSD 在 branch 3 formal round 1、
  TVD 在 branch 2 formal round 2、xFormers RMSNorm 在 branch 2 formal round 4 fresh Gate1 closure
  后通过三边 Gate2，均直接 certified；RoPE、CrossEntropy、GroupNorm、Linear 已完整耗尽
  `B=3,K=5` 为 NO_CLOSURE。Qwen2VL MRoPE 的一个 Lift branch、Llama4 RoPE 的一个 Replay branch、
  Flash forward 的一个 Lift branch在旧 Codex runtime v2 首轮 wall-time 后因 thread id 未保存而成为
  infrastructure failure，不能据此归为语义失败；one-col backward、SplitK backward 为
  NO_CLOSURE，full backward、SplitK forward 也各含一个旧 runtime Replay infrastructure branch。
  批次终态为 `3 certified / 6 no_closure / 5 infrastructure_failed`。
- Wave 8：Flash forward 在 branch 3 formal round 3 fresh Gate1 closure，但初次 Gate2 同时暴露
  Source `tl.dot(trans_b=True)` 的当前 Triton execution compatibility 缺口与 RefEval 对多阶段 TMP
  store→reload 的消费缺口。两项通用能力修复后的零模型 exact-artifact recheck 已证明
  Source–Replay 三个输出误差均为 0；随后 RefEval 正确发现闭合 J 将 `[x0,x1,x2]` loop results
  直接映射到 `[x0,x2,x5]` 的 LSE/TMP，属于真实 axis-permutation J 错误。typecheck.v23 现以前置
  `output_binding_type_mismatch` 拒绝，旧 J 不认证，必须 fresh relift 显式 transpose/output mapping。
- Wave 11（commit `29216507`，Codex Sol high）：Flash forward 三个branch均完整耗尽`K=5`为
  NO_CLOSURE。fresh J 已显式作者化旧artifact缺失的transpose/output mapping；其中一个branch又在
  J中把softmax scale重复应用两次，Replay只是忠实消费错误J，Gate1正确拒绝。另两个branch仍停在
  op/dataflow topology，不得弱化Gate或写Flash recipe。
- Wave 12（commit `29216507`，Codex Sol high）：Linear、RoPE、Qwen2VL MRoPE、Llama4 RoPE均完整
  `B=3,K=5`为NO_CLOSURE；共同frontier为`op_multiset + dataflow`，部分还含
  `predicate_effect`。Replay receipt均为fresh/current且多数public compile成功，故不能归因于Terra、
  TypeCheck或提交协议。GroupNorm与CrossEntropy在相同frontier证据饱和后人为停止未终局branch；
  它们只记cancelled infrastructure，不伪装成NO_CLOSURE。该wave没有新增认证。
- Wave 13（commit `e9aa4299`，Codex Sol high）只对 Linear 做 bounded-Replay-query pilot。三个branch均
  已至少一次进入Gate1且前两轮仍为`op_multiset + dataflow` no-closure；约1503秒时为避免继续串行
  消耗而人工停止，branch 1 正在formal round 3 Replay、branch 2正开始第3次Lift、branch 3正进行
  第2次Lift。该run只记人为cancelled infrastructure/diagnostic evidence，不冒充完整NO_CLOSURE，也无
  新认证。它证明v15查询transport稳定，但单独减小Replay首轮上下文尚未提升closure。
- Wave 14（commit `16904c1f`，Codex Sol high）对剩余 11 项分成 3+8 两个 batch 启动；RoPE、
  Qwen2VL MRoPE、Llama4 RoPE、GroupNorm、CrossEntropy、Linear、Flash forward 等部分 branch 已进入
  formal round，后四个复杂 case 也开始首轮。运行中共享工具文件升级触发旧 persistent-input
  comparison；部分既有 owner 因此 infrastructure fail，但后启动 owner 使用增强只读工具本可继续。
  全波因过度保守判断被人工停止，保留 infrastructure/partial-frontier evidence；没有新增认证，也没有任何 case 获得完整
  `NO_CLOSURE_(B,K)` 结论。
- 2026-07-28 通用能力批没有增加 kernel recipe 或新 semantic artifact：J typed constant 现在可精确
  作者化 `inf/-inf/nan`；`physical_iterations` 可在同 subject/axis 上锚定 access/domain，以表达
  reduction 之后的 output epilogue/transfer traversal；`typecheck.v24` 在 Replay 前拒绝没有 pointer
  signature-carrier route 的 transfer；Replay startup 将 dimension carrier、special constant 与 physical
  iteration 作 exact bounded 重索引，ReplayCheck 则按 route reachability 而非 ref spelling 计数
  pointer-producing address stages。所有新增规则都只消费或拒绝 LLM 已作者化事实，不从 Source/Gate
  推导 route、constant、loop、bound 或代码。
- Wave 16（commit `7272bc13`，六个未闭合项，Codex Sol high，`B=3,K=5`）已结束：RoPE、
  CrossEntropy、GroupNorm、Flash one-col backward、SplitK backward 均完整 NO_CLOSURE；Flash full
  backward 的三个 branch 都在 fresh Replay 生成阶段 infrastructure_failed，未消耗完整 formal K。
  该 wave运行的是 v25/v14 前旧代码，只保留为 frontier/latency evidence，不能评价下述新 relation。
- Wave 17（GroupNorm/SplitK backward，Codex Sol xhigh，`B=3,K=3`）未闭合，但最佳分支共同收敛到
  Source 省略 reduction axis 所产生的两次 compiler-visible `tt.reshape`；其中 SplitK 的
  reduction 与 predicate/effect 已闭合，dataflow mismatch 从 460 降至 46。系统因此新增通用、
  source-neutral π `reduction_linearization`：它只由 Lift 作者化全轴线性化 stage及其唯一 reduce
  route；TypeCheck v25 验证、U_f v14消费、ReplayCheck v10只做 source-blind
  `tt.reshape -> tt.reduce(axis=0)` dataflow 下界检查。
  这不是 GroupNorm/SplitK recipe，也不让 fixed code从 Gate1补 relation。
- Wave 18（commit `e379e10f`，GroupNorm/SplitK backward，Codex Sol xhigh，`B=3,K=3`）中，
  SplitK backward 在 branch 2 formal round 2 fresh Gate1 closure 后直接通过三边 Gate2，正式
  certified。GroupNorm 三个 branch 共 9 轮仍为 dataflow NO_CLOSURE；新增 linearization relation 已令
  `tt.reshape`、op multiset 与 predicate/effect 闭合，最佳 62 条差异进一步压缩为一个 realization
  placement：π 已把 `n11` address-base instance 放入第二 loop region，Replay 却把对应乘法 hoist 到
  parent，导致整个 region fingerprint 连锁变化。Replay v25/v21 因此把已有“region 是精确词法 scope”
  合同显式化，并以 source-blind `critical_facts.v2` 精确重索引 loop/branch instance membership 与
  cross-region order；construction compile 前同一 Replay owner 还必须把这张 ledger 与当前 exact
  编号源码并置自审，任何 byte 修改都使审阅失效。工具不读取 Gate、不建立 instance↔statement 映射、
  不生成 stage 或修复 code。
- Wave 18 四核批中，CrossEntropy branch 1 formal round 3 已 Gate1 closure；旧 Gate2 先后暴露
  intermediate gather interval、条件 branch full-store 与同 tensor 多 access padding provenance 三个
  deterministic consumer缺口。integer-range 现可在外部 index interval 已闭合时保守传播 gather
  result；functional-boundary v19 将 padding 绑定到 `(node_ref,operand)` edge，并只对匹配
  `index_bounds` 的独立 gather 点访问豁免另一 padded stream；structured reference v9 只执行 J/B 已
  作者化的 `effect_if` arm full store。精确 artifact/Replay 的零模型 Gate2-only recheck 已三边通过：
  `/tmp/intentir-task14-wave31-final-consumer-rechecks-20260728/liger_cross_entropy/gate2-recheck-001`。
- Wave 25 对 Wave18 GroupNorm 最佳 J 做两个独立 Codex Sol-xhigh fixed-artifact Replay；一个仍
  NO_CLOSURE，另一个在 `critical_facts.v2 + replay-obligation-audit.v1` 下 fresh `replay_closed`，证明
  exact lexical region/self-audit 能修掉既有 address-base hoist，但高 effort并非单调保证。该闭合 Replay
  的 Gate2 又暴露并修复通用 consumer 缺口：dynamic `iota(x*)` 与 nonzero signed `div` 的保守 integer
  interval、node-less padded reduction load 与后续 node-bound reload 的 edge ownership、以及匹配
  `index_bounds` 的动态 gather consumer-footprint。零模型三边 Gate2最终通过：
  `/tmp/intentir-task14-wave31-final-consumer-rechecks-20260728/liger_group_norm/gate2-recheck-001`。
- Wave 32（commit `11354e56`，六核 Codex Sol-xhigh，`B=3,K=3`）中，SplitK backward 与
  GroupNorm 都由 branch 3 在 formal round 3 fresh Gate1 closure 后直接三边 certified。SplitK 的
  relift 轨迹先把两个 `tt.reshape` 误作者化为 `identity_materialization`，第 3 轮由同一 Lift owner
  自行改成已有 π `reduction_linearization` 后才闭合；fixed code没有从 Gate diff补 relation。
  CrossEntropy branch 2 同样在 round 3 Gate1 closure，Source--Replay 的 `x/loss` 误差均为 0；旧
  structured RefEval只接受 equal-rank `along_axis` gather，不能消费 J 已闭合的
  `data=[x0,x1], index/result=[x0], axis=1` compressed form，原 batch 因而保持 Gate2 failed。structured
  reference v9 与既有 JSON/Intent MLIR emitter 合同对齐，只在 index rank=data rank-1 且 shape精确
  等于全部非轴 extent 时机械 expand/squeeze；复用同一 artifact/Replay、`model_calls=0` 的 Gate2-only
  recheck 已 certified：
  `/tmp/intentir-task14-wave33-crossentropy-compressed-gather-recheck-20260729/liger_cross_entropy/gate2-recheck-001`。
  RoPE 与 one-col backward 各有两个 branch完整耗尽 3 轮 NO_CLOSURE、一个 Replay infrastructure；
  full backward 三个 branch最终均为 Replay-before-Gate1 infrastructure，故三者都不能记完整
  `NO_CLOSURE_(B,K)`。18 路同时 xhigh 使首次 Lift约 9--26 分钟，后续改为两核、每核 3B 聚焦调度。
- Wave 35（起点 commit `544c358d`）聚焦 one-col backward 与 Liger RoPE。one-col 使用 Codex
  Sol-xhigh、`B=3,K=3`，三个 branch 均耗尽三轮而未闭合；最佳最终分支的 interface/reduction/
  predicate-effect 已通过，只剩 4 条 op-multiset 与 286 条 dataflow 差异。另一个分支的 J 已明确
  `t6/t7: rank-1` 及两条显式 broadcast，fresh Replay 却直接载入 `tensor<128x1>`，遗漏两个
  `expand_dims`，说明当前主要是 Replay 对 authored shape/broadcast obligation 的消费缺口。
  RoPE 使用 Lift Sol-max、Replay Sol-xhigh；branch 3 三轮仍为
  `op_multiset=3/dataflow=283/predicate-effect=14`，Replay 仍以 16-wide arange 实现 Source 的 8-wide
  half-dimension。branch 1 被运行中短暂 Python import 回归污染，branch 2 在 Replay owner wall-time
  前失败，所以 aggregate 只记 infrastructure/diagnostic evidence，不记完整 NO_CLOSURE 或认证。
  该 wave 启动后主工作区发生过工具修改，全部结果只用于定位，修复后必须从 clean commit fresh
  重跑。由此新增的通用 `decode.v14/typecheck.v26` 只消费 J direct `x* == 1`/`1 == x*` 作为显式
  broadcast 的 singleton 证据；不从 valuation/Source/Gate 猜事实，也不构造 broadcast。
  Replay v27/v23 与 context v16 又把 authored broadcast 的 operand closed type、matching access、named
  consumer/operand ordinal 加入 exact `critical_facts.v3`，并在存在该 obligation 时要求
  `replay-obligation-audit.v2` 对 owner 当前编号源码自审。该工具不映射 relation 到 statement、不选择
  `expand_dims` 次数或代码、不读取 Source/Gate；完整 Replay 仍由 source-blind LLM 作者化。
  三次 fixed-artifact v23 xhigh recheck 均消除了旧 `128x1` load、缺失 expand_dims 与全部
  predicate/effect 差异；最佳分支 interface/op/reduction/predicate-effect 全过，只剩 dataflow 226。
  non-recursive frontier 把它压缩为同一 physical loop 的 5 个 carrier tuple置换：旧 J/π 把两个
  accumulator relation排在三个 pointer relation之前，Source恰好相反，而 Replay忠实消费旧顺序。
  因此 v66/v62 与 Replay v28/v24 只明确现有数组 declaration order 是 shared carrier-tuple ordinal；
  fixed code不重排旧 artifact，后续必须 fresh relift。
- Codex Lift v11 / knowledge v65 将 closed frontier 已有的 interface、reduction、predicate/effect 与
  op-multiset direct categories 一次并列展开，recursive dataflow 保持按需分页。它只改善同一 owner
  对已有 Gate1 evidence 与 π grammar 的注意力；不选择 refinement、不改 J/C/π，也不把 compiler op
  当成 J semantics。针对性回归 `143 passed`。
- Wave 37（起点 commit `ff136192`，one-col backward，Codex Sol-xhigh，`B=3,K=3`）三个 branch
  均完整耗尽三轮 NO_CLOSURE，无 infrastructure failure。branch 3 最终 interface/op-multiset/
  reduction/predicate-effect 全过，只剩 20 条 dataflow 差异；Source helper 的 terminal true arm 按
  `DV -> DK` 写入，而 authored arm `scope.effects` 与 π transfer topology 按 `DK -> DV`，Replay 忠实
  消费该顺序。它是 fresh Lift authorship frontier，不是 fixed-artifact Replay 可修的问题。首轮 xhigh
  Lift约 10--15 分钟，persistent relift约 2.8 分钟，说明 owner continuation 有效但首稿 attention
  仍昂贵。`source-outline.v2` 因此只新增 syntax-only loop-preexisting-write 与 reachable-helper literal
  memory-order checklist；fixed code不分类carrier/effect、不选择顺序、不写J/C/π。该单一稳定 frontier
  允许下一次 fresh Lift 升至 Sol-max，Replay继续fresh/source-blind Sol-xhigh。
- RoPE 的 16-wide/8-wide frontier 证明旧 wire 无法同时保持 Source/Harness raw `pad_hd=16` 与某条
  realization use 的 `pad_hd//2`；把 C 改存 8 会破坏 Source/Replay 同一 valuation，让 Replay 从 J 猜
  `/2` 又是欠定。`artifact.v10/decode.v15/typecheck.v27` 因此只新增 closed
  `pi.route_extent_transforms{source,target,slot,divisor>1,rounding=floor|ceil}`：source 是 attached
  required positive-int C，target 是匹配 J domain realize，且 exact route slot 必须已存在。它不创建
  J half-domain/shape/predicate、不读取 concrete value、不传播 sibling use；Replay v29/v25 将 relation
  放入 exact critical/audit/ingress，production仍在π前停止。修复后 RoPE 必须 fresh relift，旧 artifact
  不得由 fixed code 补 relation，也尚未新增认证。
- Wave 38（clean commit `483bf697`，两核 `B=3,K=3`）中，one-col backward 使用 Lift Sol-max、
  Replay Sol-xhigh，由 branch 2 formal round 2 fresh Gate1 closure；Liger RoPE 同配置由 branch 1
  round 2 fresh Gate1 closure。one-col 的 Source 与 Replay 在 Gate2 同时因 shared memory
  `108544 > 101376` 无法执行，不能把共同 OOR 记成数值认证；RoPE 的 Source--Replay q/k 均为 0
  误差，但旧 RefEval 先停在 symbolic singleton/half-domain consumer capability。
- Wave 39/40 对上述 exact artifact/Replay 做零模型诊断 recheck。v21/v20 已以 J direct `x5==1`
  消费 constant coordinate 0，并以 sibling `effect_if` full-store control partition 修通 one-col 的 J leg；
  one-col 仍只剩共同 OOR。v22 的通用 ContractIR indexed pointwise reference 随后完整执行 RoPE J
  已写出的 index/predicate/fill/store：Source--Replay 仍为 0，而 Source--J/Replay--J 在 q/k 上分别
  出现约 `0.248206 / 0.206278` 最大绝对误差。这证明旧 RoPE J 缺 consumer-local half-view access，
  不是继续修 Gate2 的理由，必须 fresh relift。
- one-col 的 reviewed P_cert/C valuation 已把 `BLOCK_M` 从 128 改为 64。Source 内部 loop 仍遍历完整
  `Sq=128`，grid、logical shape 与算法不变；实机 Source 可执行，相对独立 PyTorch 梯度参考的
  dQ/dK/dV 相对误差约 `2.46e-6 / 2.01e-6 / 3.62e-6`。旧 C=128 artifact 不追认，下一轮在 C=64
  profile fresh SCRC。RoPE/one-col 下一轮默认 Sol-xhigh；若同一明确语义 frontier 稳定重复，再升
  Sol-max，不能把 effort 升级替代系统能力判断。
- 当前可用闭合总计 `20/28`：Wave32 将 GroupNorm 从 exact-artifact capability evidence 升级为 fresh
  直接认证后，现为 16 个原运行直接认证、4 个 exact-artifact capability recheck 认证。
  TypeCheck PASS 不计进度；只有 Gate1 已闭合且当前 Gate2 三边通过才计入。

对Wave12六核的Source/J/π/Replay横向审查得到以下责任边界：Linear缺K-loop/physical-carrier
realization作者化；Liger RoPE缺旋转分解与分支内effect拓扑；Qwen2VL MRoPE缺共享位置组合到q/k
双输出的route/effect；Llama4 RoPE缺bounds control下q/k sibling stores；GroupNorm缺cast/address/
predicate/physical-loop realization；CrossEntropy的`terminal_controls`已闭合，剩余是continuation arm
内部reduction/argmax carrier/masked effects。函数多输出、任意多J loop carrier与terminal relation
本身都已存在；当前未发现真实“两个runtime分支各产多个value并在join后共同消费”的case，因此不
预先扩multi-result `value_if`或conditional post-state schema。

Codex Lift v7 / knowledge v61 与MiniSWE knowledge v59提供等合同的只读authoring views：
`harness-abi`仅并置Harness已声明的p*/c*、bound non-C constexpr及h*；`candidate-control`仅并置
当前LLM candidate的control/reduction、multi-output/effect、branch arm，以及π
regions/instances/time/control/route/order；`feedback-category`只返回frontier已命名的一个closed
Gate1类别。三者均不生成、补齐或推荐字段。MiniSWE continuation 不再重复嵌入已保存在同一
workspace的完整candidate；Codex/MiniSWE owner均固定消费创建时的私有输入快照。CandidateCheck surface diagnostics另显式报告
`construction_semantic_constexpr_in_c`，但仍只拒绝错误candidate；完整语义分类和修订继续由同一
source-aware Lift owner完成。

CrossEntropy 曾暴露一个通用表达缺口：J 已作者化 early-return `effect_if`，旧 π 无法区分 terminal
arm 与 lexical continuation。`artifact.v9` 当时只新增 closed
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

本轮 Codex Replay v14 使用该 v2 view，只把 J 中已经作者化的 reduction operand/result type、
semantic axes、对应 boundary reduction、C attachment 与 π instance 无损并置成 reduction obligation；
它不选择 physical rank、tile、reshape、loop 或 frontend spelling。`replay-check.v10` 在公共 synthetic
compile 后按 π 已作者化的 `transfer_direct` occurrence 核对 compiled load/store multiplicity，并按
structured J loops 加 `pi.physical_iterations` 核对 compiled `scf.for` multiplicity，再按对应
`pi.loop_carriers/physical_loop_carriers` 核对 block-argument arity；`pi.iterations`只 realize 已存在的
J loop，不重复计数；并只在 J.signature carrier kind 与 π address routes 完全闭合时核对 compiled
`tt.addptr` occurrence。检查防止 CSE 折叠 transfer、Replay 遗漏已声明物理时间/carrier/address
stage。当前完整闭包按 authorship route reachability 判定：每个 transfer 必须从绑定同一 boundary
tensor 的 pointer/tensor signature carrier 可达，每个 pointer-producing address stage 必须流向至少
一个 transfer；address 与 transfer 不要求复用相同 semantic-ref spelling，并允许额外非 pointer 的
integer base/lane contribution。真实 GroupNorm branch 3 因而从“跳过”变为闭合要求
11 个 `tt.addptr`，旧 Replay 为 13，Gate1 Source 也为 11。该结论中的 11 由 π 决定，Source 只作为
事后 Gate1 evidence，不进入 ReplayCheck。检查仍不读取 Source、Harness value 或 Gate1，也不补
loop/bound/carrier/transfer/address formula。
Codex Lift v5 与 Replay v14 使用 Codex agent runtime v3：若首轮被 wall-time 中断，runtime 只从
CLI 已经发出的唯一 `thread.started` JSONL 事件恢复原 thread identity，再由同一个 owner/episode、
同一个 workspace 和 closed diagnostics 继续；不会接受未通过检查的 candidate/Replay，也不会恢复
或合成未完成输出。对单次 provider turn wall-time 仍只做同 thread/session 续跑，并且只可恢复
本次 invocation/episode 内由同一 LLM 写出且已经通过对应 closed check 的 exact bytes；固定代码不
合并、修复或跨正式轮次选择 candidate。Lift v5 另要求 LLM 在提交前逐项 reconcile syntax-only
Source control-flow outline，但分类与完整 J/C/π 仍由 LLM 作者化。

Replay v15/v19只改变proof-irrelevant agent消费界面：MiniSWE首轮从完整packet注入改为约500字符的
digest/size receipt，再由现有bounded `intentir_packet_query`按需读取exact authored facts；Codex
file mailbox升级为支持最多四个有界字符串参数的v3 bridge，先给约3KB manifest，再用同一query
语法分页读取ABI/graph/realization/obligation。Flash样本中Codex首轮最大单个view从约32KB降至
9KB以内，首轮四类事实合计约22.6KB。两者都不摘要或生成J/π，不接收Source/Gate，不自动修Replay；
完整Replay仍由source-blind owner作者化并经同一ReplayCheck/public compile fail closed。

Replay v16/v20在该边界内再增加唯一一次 exact authored-requirements view：它只重索引J op的
declaration-order/count、`contraction/effect_if/fma/loop/reduce/value_if`原记录、boundary effects，
以及π已有的iterations/carriers/physical-time/terminal-controls和所引用instance/region。它不读取
Source、Harness或Gate，不给frontend spelling、bound、schedule、地址公式、repair suggestion或代码；
因此解决的是Replay owner遗漏可见packet义务的attention/consumption问题，不是新增FMA permission、
kernel recipe或deterministic composer。旧v15/v19 run保持原receipt，只fresh rerun使用新版本。

Replay v18/v22 撤销该强制查询协议：不超过64 KiB的完整匿名packet只在首次上下文exact embedding
一次；超限manifest直接携带同一requirements清单，并补齐π instance mechanism计数/顺序，focused
query只用于明确截断页；当前版本再并置 dimension carrier、IEEE special constant 与 physical
iteration 的 exact `critical_facts`，不生成实现。当前六个难核packet实测最大约47 KiB，正常路径可直接写完整Replay。
Codex Lift v9 / knowledge v63同时将exact Harness ABI与closed relift frontier/primary category直接交给
同一持久owner，原有harness/candidate-control/feedback-category工具均退为可选局部自读。两项修改
只减少协议/attention负担，不生成J/C/π、Replay代码、schedule或repair，也不改变任何门禁。

Gate2 Triton Source/Replay runner 共享 source-neutral `triton_static_dot_transpose_v1`：只对静态 boolean
`trans_a/trans_b` 做与 Gate1 相同的 AST 等价归一，并记录 rewrite count/digest；不按 kernel 名改写，
不触及 J/π/C。RefEval v19 / functional-boundary v18 消费同一 inout 上多个 closed store→reload
stage，并把未参与 logical access/predicate 的矩形 J domain保留为 `context_domain` receipt，而不是
将其错误当作输出数值轴或静默删除。只有最终 canonical output binding 对应 store 可观察。

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

- 主闭合模型：Sol high；对经过通用能力修复后仍未闭合的两核重点批升级为 Sol xhigh。只有 xhigh
  已形成稳定、可行动且只剩单一高难语义前沿时才使用 Sol max；max 不替代 schema/consumer 修复，
  也不靠更多思考预算重烧相同 candidate。Terra high只作为独立速度/模型能力对照，不占用主闭合
  branch，也不能用其失败覆盖Sol证据；
- 同case默认`B=3`，三个branch彼此隔离；当前聚焦闭合的 formal Gate1上限为`K=3`，首个Gate1 closure立即成为
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

`K=3`是当前聚焦运行上限而非必须跑满。只有全部B个branch带完整`NO_CLOSURE` evidence耗尽其声明K，case才能
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
